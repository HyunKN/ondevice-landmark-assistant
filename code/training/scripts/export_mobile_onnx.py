#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from landmark_candidate.multitask_model import MobileClipS4MultitaskModel  # noqa: E402
from landmark_candidate.train_multitask import initialize_lazy_modules  # noqa: E402


class ImageEmbeddingExportModule(nn.Module):
    def __init__(self, model: MobileClipS4MultitaskModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image_embedding(image)


class TextEmbeddingExportModule(nn.Module):
    def __init__(self, model: MobileClipS4MultitaskModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text_embedding(text_tokens)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "mobileclip2"


def artifact_prefix(cfg: dict[str, Any]) -> str:
    model_cfg = cfg.get("model", {})
    return slugify(str(model_cfg.get("id") or model_cfg.get("model_name") or "mobileclip2"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def remove_stale_onnx(path: Path) -> None:
    for candidate in (path, path.with_suffix(path.suffix + ".data")):
        if candidate.exists():
            candidate.unlink()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[MobileClipS4MultitaskModel, list[str], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config")
    classes = checkpoint.get("classes")
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint does not contain a config dict")
    if not isinstance(classes, list) or not classes:
        raise ValueError("Checkpoint does not contain a non-empty classes list")

    image_size = int(cfg["training"]["image_size"])
    model = MobileClipS4MultitaskModel(cfg, len(classes)).to(device)
    initialize_lazy_modules(model, image_size, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, [str(item) for item in classes], cfg


def maybe_reparameterize(model: MobileClipS4MultitaskModel) -> bool:
    try:
        from mobileclip.modules.common.mobileone import reparameterize_model
    except Exception:
        return False
    model.clip = reparameterize_model(model.clip)
    return True


def write_common_metadata(
    out_dir: Path,
    *,
    classes: list[str],
    cfg: dict[str, Any],
    checkpoint_path: Path,
    precision: str,
    opset: int,
    ir_version: int,
    onnx_path: Path,
    external_data_path: Path,
    reparameterized: bool,
    precision_policy: str | None = None,
    text_onnx_path: Path | None = None,
    text_external_data_path: Path | None = None,
    tokenizer_context_length: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(cfg["training"]["image_size"])
    preprocessing = {
        "image_size": image_size,
        "image_mean": cfg["training"].get("image_mean", [0.48145466, 0.4578275, 0.40821073]),
        "image_std": cfg["training"].get("image_std", [0.26862954, 0.26130258, 0.27577711]),
        "input_layout": "NCHW",
        "input_dtype": "float32",
    }
    (out_dir / "preprocessing.json").write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "labels_master.json").write_text(
        json.dumps({"classes": classes, "class_count": len(classes)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest = {
        "artifact_version": "landmark-assistant-mobileclip2-multitask-export-v1",
        "precision": precision,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_id": cfg.get("model", {}).get("id"),
        "model_name": cfg.get("model", {}).get("model_name"),
        "class_count": len(classes),
        "embedding_dim": int(cfg["training"].get("embedding_dim", 512)),
        "image_size": image_size,
        "image_encoder": {
            "onnx": onnx_path.name,
            "external_data": external_data_path.name if external_data_path.exists() else None,
            "input": "image",
            "output": "embedding",
        },
        "text_encoder": {
            "onnx": text_onnx_path.name if text_onnx_path is not None else None,
            "external_data": text_external_data_path.name if text_external_data_path is not None and text_external_data_path.exists() else None,
            "input": "text_tokens",
            "output": "text_embedding",
            "context_length": tokenizer_context_length,
        },
        "opset": opset,
        "ir_version": ir_version,
        "reparameterized_mobileone": reparameterized,
        "precision_policy": precision_policy,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer = {
        "model_name": cfg.get("model", {}).get("model_name"),
        "input_name": "text_tokens",
        "input_dtype": "int64",
        "context_length": tokenizer_context_length,
        "note": "Use the same open_clip tokenizer for this model_name before running the exported text encoder ONNX.",
    }
    (out_dir / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False, indent=2), encoding="utf-8")


def write_prototype_index(out_dir: Path, model: MobileClipS4MultitaskModel, classes: list[str]) -> None:
    weights = F.normalize(model.classifier.weight.detach().float(), dim=-1).cpu().numpy()
    rows = [
        {
            "landmark_id": landmark_id,
            "prototype_type": "normalized_classifier_weight",
            "embedding": weights[index].tolist(),
        }
        for index, landmark_id in enumerate(classes)
    ]
    payload = {
        "version": "prototype-index-v1",
        "embedding_dim": int(weights.shape[1]),
        "count": len(rows),
        "items": rows,
    }
    (out_dir / "prototype_index.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def export_fp32(
    model: MobileClipS4MultitaskModel,
    out_dir: Path,
    onnx_name: str,
    image_size: int,
    opset: int,
    ir_version: int,
    device: torch.device,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / onnx_name
    remove_stale_onnx(onnx_path)
    wrapper = ImageEmbeddingExportModule(model).to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=opset,
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        external_data=True,
    )
    graph = onnx.load(str(onnx_path), load_external_data=True)
    graph.ir_version = ir_version
    onnx.save_model(
        graph,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=onnx_path.name + ".data",
        size_threshold=1024,
    )
    return onnx_path


def export_text_fp32(
    model: MobileClipS4MultitaskModel,
    out_dir: Path,
    onnx_name: str,
    sample_texts: list[str],
    opset: int,
    ir_version: int,
    device: torch.device,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / onnx_name
    remove_stale_onnx(onnx_path)
    wrapper = TextEmbeddingExportModule(model).to(device).eval()
    dummy = model.tokenize(sample_texts, device)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["text_tokens"],
        output_names=["text_embedding"],
        opset_version=opset,
        dynamic_axes={"text_tokens": {0: "batch"}, "text_embedding": {0: "batch"}},
        external_data=True,
    )
    graph = onnx.load(str(onnx_path), load_external_data=True)
    graph.ir_version = ir_version
    onnx.save_model(
        graph,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=onnx_path.name + ".data",
        size_threshold=1024,
    )
    return onnx_path


def should_store_initializer_as_fp16(name: str) -> bool:
    # Full MobileCLIP2-S3 FP16 ONNX was not numerically stable on CPUExecutionProvider.
    # This selected policy keeps sensitive visual stages in FP32 while storing safe large
    # tensors as FP16. The graph casts those tensors back to FP32 at runtime.
    if name.startswith("val_"):
        return True
    if "image_projection" in name or "text_projection" in name:
        return True
    if not name.startswith("model.clip.visual"):
        return False
    if "stem" in name or "stages.0" in name or "stages.2" in name:
        return True
    if "stages.3" in name and ".mlp." in name:
        return True
    if "stages.4" in name and ".mlp." in name:
        return True
    if "trunk.head" in name:
        return True
    return False


def export_fp16(fp32_path: Path, fp16_dir: Path, ir_version: int) -> tuple[Path, dict[str, Any]]:
    fp16_dir.mkdir(parents=True, exist_ok=True)
    if fp32_path.name.endswith("_fp32.onnx"):
        fp16_name = fp32_path.name[: -len("_fp32.onnx")] + "_fp16_mixed.onnx"
    else:
        fp16_name = fp32_path.stem + "_fp16_mixed.onnx"
    fp16_path = fp16_dir / fp16_name
    remove_stale_onnx(fp16_path)

    fp16_model = onnx.load(str(fp32_path), load_external_data=True)
    cast_nodes = []
    replacements: dict[str, str] = {}
    converted_count = 0
    estimated_saved_bytes = 0
    converted_names: list[str] = []
    for initializer in fp16_model.graph.initializer:
        if initializer.data_type != TensorProto.FLOAT:
            continue
        if not should_store_initializer_as_fp16(initializer.name):
            continue
        array = numpy_helper.to_array(initializer)
        estimated_saved_bytes += array.nbytes // 2
        initializer.CopyFrom(numpy_helper.from_array(array.astype(np.float16), name=initializer.name))
        cast_output = initializer.name + "_to_fp32"
        replacements[initializer.name] = cast_output
        cast_nodes.append(
            helper.make_node(
                "Cast",
                inputs=[initializer.name],
                outputs=[cast_output],
                name=cast_output.replace("/", "_"),
                to=TensorProto.FLOAT,
            )
        )
        converted_count += 1
        converted_names.append(initializer.name)

    original_nodes = list(fp16_model.graph.node)
    for node in original_nodes:
        for index, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[index] = replacements[input_name]
    del fp16_model.graph.node[:]
    fp16_model.graph.node.extend(cast_nodes + original_nodes)
    fp16_model.ir_version = ir_version
    onnx.save_model(
        fp16_model,
        str(fp16_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=fp16_path.name + ".data",
        size_threshold=1024,
    )
    stats = {
        "precision_policy": "selected_fp16_weight_storage_fp32_compute",
        "converted_initializer_count": converted_count,
        "estimated_saved_bytes": estimated_saved_bytes,
        "converted_initializers": converted_names,
    }
    conversion_name = fp16_path.stem + "_fp16_conversion.json"
    (fp16_dir / conversion_name).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return fp16_path, stats


def inspect_onnx(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(str(path))
    return {
        "ir_version": int(model.ir_version),
        "opsets": {item.domain or "": int(item.version) for item in model.opset_import},
        "inputs": [
            {
                "name": value.name,
                "elem_type": int(value.type.tensor_type.elem_type),
                "shape": [
                    dim.dim_param if dim.dim_param else (int(dim.dim_value) if dim.dim_value else None)
                    for dim in value.type.tensor_type.shape.dim
                ],
            }
            for value in model.graph.input
        ],
        "outputs": [
            {
                "name": value.name,
                "elem_type": int(value.type.tensor_type.elem_type),
                "shape": [
                    dim.dim_param if dim.dim_param else (int(dim.dim_value) if dim.dim_value else None)
                    for dim in value.type.tensor_type.shape.dim
                ],
            }
            for value in model.graph.output
        ],
    }


def run_onnx(path: Path, image: torch.Tensor) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(["embedding"], {"image": image.detach().cpu().numpy().astype("float32")})[0]
    return output


def run_text_onnx(path: Path, text_tokens: torch.Tensor) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(["text_embedding"], {"text_tokens": text_tokens.detach().cpu().numpy().astype("int64")})[0]
    return output


def validate_export(
    fp32_path: Path,
    fp16_path: Path,
    text_fp32_path: Path,
    text_fp16_path: Path,
    image: torch.Tensor,
    reference: np.ndarray,
    text_tokens: torch.Tensor,
    text_reference: np.ndarray,
) -> dict[str, Any]:
    results: dict[str, Any] = {"sample_batch": 2, "image_checks": {}, "text_checks": {}}
    for precision, path in (("fp32", fp32_path), ("fp16", fp16_path)):
        output = run_onnx(path, image)
        cosine = np.sum(reference * output, axis=1) / (
            np.linalg.norm(reference, axis=1) * np.linalg.norm(output, axis=1)
        )
        results["image_checks"][precision] = {
            "onnx_inspection": inspect_onnx(path),
            "embedding_shape": list(output.shape),
            "cosine_to_pytorch_min": float(cosine.min()),
            "cosine_to_pytorch_mean": float(cosine.mean()),
            "max_abs_diff_to_pytorch": float(np.max(np.abs(reference - output))),
        }
    for precision, path in (("fp32", text_fp32_path), ("fp16", text_fp16_path)):
        output = run_text_onnx(path, text_tokens)
        cosine = np.sum(text_reference * output, axis=1) / (
            np.linalg.norm(text_reference, axis=1) * np.linalg.norm(output, axis=1)
        )
        results["text_checks"][precision] = {
            "onnx_inspection": inspect_onnx(path),
            "embedding_shape": list(output.shape),
            "cosine_to_pytorch_min": float(cosine.min()),
            "cosine_to_pytorch_mean": float(cosine.mean()),
            "max_abs_diff_to_pytorch": float(np.max(np.abs(text_reference - output))),
        }
    return results


def copy_metadata(src_dir: Path, dst_dir: Path) -> None:
    for name in ("classes.json", "labels_master.json", "preprocessing.json", "prototype_index.json", "config.yaml", "tokenizer.json"):
        shutil.copy2(src_dir / name, dst_dir / name)


def reset_precision_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MobileCLIP2 multi-task checkpoint to FP32 and FP16 ONNX artifacts.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--output-dir", required=True, help="Output root. Creates fp32/ and fp16/ inside this directory.")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--ir-version", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-reparameterize", action="store_true", help="Skip MobileOne reparameterization before export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    fp32_dir = output_root / "fp32"
    fp16_dir = output_root / "fp16"
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    model, classes, cfg = load_checkpoint(checkpoint_path, device)
    reparameterized = False if args.no_reparameterize else maybe_reparameterize(model)
    image_size = int(cfg["training"]["image_size"])
    prefix = artifact_prefix(cfg)
    image_fp32_name = f"{prefix}_image_encoder_fp32.onnx"
    text_fp32_name = f"{prefix}_text_encoder_fp32.onnx"

    reset_precision_dir(fp32_dir)
    reset_precision_dir(fp16_dir)

    torch.manual_seed(20260613)
    validation_image = torch.randn(2, 3, image_size, image_size, device=device)
    validation_texts = ["Gwanghwamun photo", "광화문 사진"]
    validation_text_tokens = model.tokenize(validation_texts, device)
    with torch.no_grad():
        reference_embedding = model.encode_image_embedding(validation_image).detach().cpu().numpy()
        reference_text_embedding = model.encode_text_embedding(validation_text_tokens).detach().cpu().numpy()

    fp32_path = export_fp32(model, fp32_dir, image_fp32_name, image_size, args.opset, args.ir_version, device)
    text_fp32_path = export_text_fp32(model, fp32_dir, text_fp32_name, validation_texts, args.opset, args.ir_version, device)
    write_prototype_index(fp32_dir, model, classes)
    write_common_metadata(
        fp32_dir,
        classes=classes,
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        precision="fp32",
        opset=args.opset,
        ir_version=args.ir_version,
        onnx_path=fp32_path,
        external_data_path=fp32_path.with_suffix(fp32_path.suffix + ".data"),
        reparameterized=reparameterized,
        precision_policy="fp32",
        text_onnx_path=text_fp32_path,
        text_external_data_path=text_fp32_path.with_suffix(text_fp32_path.suffix + ".data"),
        tokenizer_context_length=int(validation_text_tokens.shape[1]),
    )

    fp16_path, fp16_stats = export_fp16(fp32_path, fp16_dir, args.ir_version)
    text_fp16_path, text_fp16_stats = export_fp16(text_fp32_path, fp16_dir, args.ir_version)
    copy_metadata(fp32_dir, fp16_dir)
    write_common_metadata(
        fp16_dir,
        classes=classes,
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        precision="fp16",
        opset=args.opset,
        ir_version=args.ir_version,
        onnx_path=fp16_path,
        external_data_path=fp16_path.with_suffix(fp16_path.suffix + ".data"),
        reparameterized=reparameterized,
        precision_policy=fp16_stats["precision_policy"],
        text_onnx_path=text_fp16_path,
        text_external_data_path=text_fp16_path.with_suffix(text_fp16_path.suffix + ".data"),
        tokenizer_context_length=int(validation_text_tokens.shape[1]),
    )

    validation = validate_export(
        fp32_path,
        fp16_path,
        text_fp32_path,
        text_fp16_path,
        validation_image,
        reference_embedding,
        validation_text_tokens,
        reference_text_embedding,
    )
    report = {
        "status": "ok",
        "checkpoint": file_info(checkpoint_path),
        "output_root": str(output_root),
        "fp32": {
            "onnx": file_info(fp32_path),
            "external_data": file_info(fp32_path.with_suffix(fp32_path.suffix + ".data")),
            "text_onnx": file_info(text_fp32_path),
            "text_external_data": file_info(text_fp32_path.with_suffix(text_fp32_path.suffix + ".data")),
            "manifest": file_info(fp32_dir / "manifest.json"),
            "prototype_index": file_info(fp32_dir / "prototype_index.json"),
        },
        "fp16": {
            "onnx": file_info(fp16_path),
            "external_data": file_info(fp16_path.with_suffix(fp16_path.suffix + ".data")),
            "text_onnx": file_info(text_fp16_path),
            "text_external_data": file_info(text_fp16_path.with_suffix(text_fp16_path.suffix + ".data")),
            "manifest": file_info(fp16_dir / "manifest.json"),
            "prototype_index": file_info(fp16_dir / "prototype_index.json"),
            "conversion": {
                "image": fp16_stats,
                "text": text_fp16_stats,
            },
        },
        "validation": validation,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "export_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
