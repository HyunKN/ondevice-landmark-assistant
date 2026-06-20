#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import LandmarkImageDataset, read_json
from .train import build_model, evaluate, export_onnx, load_split_records, transforms_for


def _read_fold(run_dir: Path, explicit_fold: int | None) -> int:
    if explicit_fold is not None:
        return explicit_fold
    summary_path = run_dir / "split_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if isinstance(summary, dict) and "fold" in summary:
            return int(summary["fold"])
    raise SystemExit("Missing fold. Pass --fold or keep split_summary.json in the run directory.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved best.pt checkpoint without retraining.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="splits/kfold_seed20260513.json")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config")
    if cfg is None:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    names = checkpoint.get("classes")
    if names is None:
        names = read_json(run_dir / "classes.json")

    fold = _read_fold(run_dir, args.fold)
    _, _, test_records, _ = load_split_records(Path(args.split), fold)
    class_to_idx = {name: idx for idx, name in enumerate(names)}
    image_size = int(cfg["training"]["image_size"])
    _, eval_tf = transforms_for(image_size, cfg)
    test_ds = LandmarkImageDataset(Path(args.data_root), test_records, class_to_idx, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["training"]["batch_size_per_gpu"]), shuffle=False, num_workers=8, pin_memory=True)

    model = build_model(cfg, len(names)).to(device)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, image_size, image_size, device=device)
        model(dummy)
    model.load_state_dict(checkpoint["model"])

    test_metrics = evaluate(model, test_loader, device, names)
    metrics_path = run_dir / "metrics.json"
    final = {}
    if metrics_path.exists():
        final = json.loads(metrics_path.read_text(encoding="utf-8"))
    final.update(
        {
            "test_top1_accuracy": test_metrics["top1_accuracy"],
            "test_top3_accuracy": test_metrics["top3_accuracy"],
            "test_count": test_metrics["count"],
            "test_macro_f1": test_metrics.get("macro_f1"),
            "test_per_class": test_metrics.get("per_class"),
            "test_confusion_matrix": test_metrics.get("confusion_matrix"),
            "recovered_from_checkpoint": True,
        }
    )
    if args.export_onnx:
        onnx_ok = export_onnx(model, image_size, run_dir / "landmark_encoder.onnx", device)
        final["onnx_export_success"] = onnx_ok
        final["onnx_file_mb"] = (run_dir / "landmark_encoder.onnx").stat().st_size / 1024 / 1024 if onnx_ok else None
    metrics_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
