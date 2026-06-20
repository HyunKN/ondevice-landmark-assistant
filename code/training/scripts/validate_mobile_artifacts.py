#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_onnx(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(str(path))
    return {
        "ir_version": int(model.ir_version),
        "opsets": {item.domain or "": int(item.version) for item in model.opset_import},
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
    }


def load_prototypes(path: Path) -> tuple[list[str], np.ndarray]:
    payload = read_json(path)
    items = payload.get("items", [])
    classes = [str(item["landmark_id"]) for item in items]
    matrix = np.asarray([item["embedding"] for item in items], dtype=np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return classes, matrix


def encoder_path(precision_dir: Path, encoder_key: str) -> tuple[Path, str]:
    manifest = read_json(precision_dir / "manifest.json")
    encoder = manifest[encoder_key]
    return precision_dir / encoder["onnx"], str(encoder["output"])


def build_transform(preprocessing: dict[str, Any]):
    image_size = int(preprocessing["image_size"])
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=preprocessing["image_mean"], std=preprocessing["image_std"]),
        ]
    )


def list_sample_images(data_root: Path, classes: list[str], limit_per_class: int) -> list[tuple[str, Path]]:
    samples: list[tuple[str, Path]] = []
    for class_name in classes:
        image_dir = data_root / class_name / "images"
        if not image_dir.exists():
            continue
        files = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        samples.extend((class_name, path) for path in files[:limit_per_class])
    return samples


def run_session(session: ort.InferenceSession, output_name: str, image: np.ndarray) -> np.ndarray:
    return session.run([output_name], {"image": image.astype("float32")})[0]


def top1(embedding: np.ndarray, prototypes: np.ndarray, classes: list[str]) -> tuple[str, float]:
    scores = embedding @ prototypes.T
    index = int(np.argmax(scores[0]))
    return classes[index], float(scores[0, index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FP32/FP16 mobile ONNX artifact parity on real dataset images.")
    parser.add_argument("--artifact-dir", required=True, help="Artifact root containing fp32/ and fp16/")
    parser.add_argument("--data-root", required=True, help="Dataset root with <class>/images folders")
    parser.add_argument("--limit-per-class", type=int, default=1)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    fp32_dir = artifact_dir / "fp32"
    fp16_dir = artifact_dir / "fp16"
    fp32_onnx, fp32_output = encoder_path(fp32_dir, "image_encoder")
    fp16_onnx, fp16_output = encoder_path(fp16_dir, "image_encoder")
    text_fp32_onnx, _ = encoder_path(fp32_dir, "text_encoder")
    text_fp16_onnx, _ = encoder_path(fp16_dir, "text_encoder")
    classes, prototypes = load_prototypes(fp32_dir / "prototype_index.json")
    preprocessing = read_json(fp32_dir / "preprocessing.json")
    transform = build_transform(preprocessing)
    samples = list_sample_images(data_root, classes, max(1, args.limit_per_class))

    fp32_session = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    fp16_session = ort.InferenceSession(str(fp16_onnx), providers=["CPUExecutionProvider"])

    rows = []
    cosines = []
    top1_matches = 0
    expected_hits_fp32 = 0
    expected_hits_fp16 = 0
    for expected, image_path in samples:
        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).numpy()
        emb32 = run_session(fp32_session, fp32_output, image)
        emb16 = run_session(fp16_session, fp16_output, image)
        cosine = float(np.sum(emb32 * emb16) / (np.linalg.norm(emb32) * np.linalg.norm(emb16)))
        pred32, score32 = top1(emb32, prototypes, classes)
        pred16, score16 = top1(emb16, prototypes, classes)
        top1_match = pred32 == pred16
        top1_matches += int(top1_match)
        expected_hits_fp32 += int(pred32 == expected)
        expected_hits_fp16 += int(pred16 == expected)
        cosines.append(cosine)
        rows.append(
            {
                "expected": expected,
                "file": str(image_path),
                "cosine_fp32_fp16": cosine,
                "fp32_top1": pred32,
                "fp32_score": score32,
                "fp16_top1": pred16,
                "fp16_score": score16,
                "top1_match": top1_match,
            }
        )

    total = len(rows)
    report = {
        "artifact_dir": str(artifact_dir),
        "data_root": str(data_root),
        "sample_count": total,
        "limit_per_class": args.limit_per_class,
        "fp32_onnx": inspect_onnx(fp32_onnx),
        "fp16_onnx": inspect_onnx(fp16_onnx),
        "text_fp32_onnx": inspect_onnx(text_fp32_onnx),
        "text_fp16_onnx": inspect_onnx(text_fp16_onnx),
        "summary": {
            "fp16_top1_matches_fp32": top1_matches,
            "fp16_top1_match_rate": top1_matches / total if total else None,
            "fp32_expected_top1_accuracy": expected_hits_fp32 / total if total else None,
            "fp16_expected_top1_accuracy": expected_hits_fp16 / total if total else None,
            "cosine_fp32_fp16_min": min(cosines) if cosines else None,
            "cosine_fp32_fp16_mean": float(np.mean(cosines)) if cosines else None,
        },
        "rows": rows,
    }
    output_path = Path(args.output).expanduser().resolve() if args.output else artifact_dir / "artifact_image_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
