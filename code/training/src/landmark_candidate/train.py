#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from .dataset import LandmarkImageDataset, class_names, read_json


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main() -> bool:
    return rank() == 0


class ArcMarginHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.2, scale: float = 30.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None or self.margin <= 0:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


class TimmLandmarkModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, embedding_dim: int, loss_name: str) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        self.embedding = nn.Sequential(
            nn.LazyLinear(embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        if "arcface" in loss_name:
            self.head = ArcMarginHead(embedding_dim, num_classes)
        else:
            self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        embeddings = F.normalize(self.embedding(features))
        if isinstance(self.head, ArcMarginHead):
            logits = self.head(embeddings, labels)
        else:
            logits = self.head(embeddings)
        return logits, embeddings


class MobileClipLandmarkModel(nn.Module):
    def __init__(self, cfg: dict, num_classes: int, embedding_dim: int, loss_name: str) -> None:
        super().__init__()
        import open_clip

        pretrained = os.environ.get("MOBILECLIP_CHECKPOINT") or cfg["model"].get("pretrained", "dfndr2b")
        self.clip, _, _ = open_clip.create_model_and_transforms(cfg["model"]["model_name"], pretrained=pretrained)
        for param in self.clip.parameters():
            param.requires_grad = False
        if not cfg["training"].get("freeze_image_encoder", False):
            visual = getattr(self.clip, "visual", None)
            if visual is None:
                raise RuntimeError("MobileCLIP model does not expose an image tower at clip.visual")
            for param in visual.parameters():
                param.requires_grad = True
        self.embedding = nn.Sequential(
            nn.LazyLinear(embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        if "arcface" in loss_name:
            self.head = ArcMarginHead(embedding_dim, num_classes)
        else:
            self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.clip.encode_image(x)
        embeddings = F.normalize(self.embedding(features.float()))
        if isinstance(self.head, ArcMarginHead):
            logits = self.head(embeddings, labels)
        else:
            logits = self.head(embeddings)
        return logits, embeddings

    def prepare_for_export(self) -> None:
        from mobileclip.modules.common.mobileone import reparameterize_model

        self.clip = reparameterize_model(self.clip)


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    embedding_dim = int(cfg["training"].get("embedding_dim", 512))
    loss_name = str(cfg["training"].get("loss", "ce"))
    family = str(cfg["model"]["family"])
    if family == "mobileclip2":
        return MobileClipLandmarkModel(cfg, num_classes, embedding_dim, loss_name)
    return TimmLandmarkModel(str(cfg["model"]["model_name"]), num_classes, embedding_dim, loss_name)


def load_split_records(path: Path, fold: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    manifest = read_json(path)
    records = manifest["records"]
    train = [r for r in records if r["split_group"] == "trainval" and int(r["fold"]) != fold]
    val = [r for r in records if r["split_group"] == "trainval" and int(r["fold"]) == fold]
    test = [r for r in records if r["split_group"] == "test"]
    holdout = [r for r in records if r["split_group"] == "holdout_non_confirmed"]
    return train, val, test, holdout


def transforms_for(image_size: int, cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    if cfg["model"]["family"] == "mobileclip2":
        mean = tuple(cfg["training"].get("image_mean", [0.48145466, 0.4578275, 0.40821073]))
        std = tuple(cfg["training"].get("image_std", [0.26862954, 0.26130258, 0.27577711]))
    else:
        mean = tuple(cfg["training"].get("image_mean", [0.485, 0.456, 0.406]))
        std = tuple(cfg["training"].get("image_std", [0.229, 0.224, 0.225]))
    crop_min_scale = float(cfg["training"].get("aug_random_resized_crop_min_scale", 0.8))
    ra_num_ops = int(cfg["training"].get("aug_randaugment_num_ops", 2))
    ra_magnitude = int(cfg["training"].get("aug_randaugment_magnitude", 5))
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(crop_min_scale, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_names_list: list[str] | None = None) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred_top1: list[int] = []
    top3_hits = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = model(images)
        topk = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
        y_true.extend(labels.cpu().tolist())
        y_pred_top1.extend(topk[:, 0].cpu().tolist())
        top3_hits += (topk == labels.view(-1, 1)).any(dim=1).sum().item()
        total += labels.numel()
    top1 = accuracy_score(y_true, y_pred_top1) if y_true else 0.0
    top3 = top3_hits / max(total, 1)
    out: dict = {"top1_accuracy": float(top1), "top3_accuracy": float(top3), "count": int(total)}
    if class_names_list and y_true:
        report = classification_report(
            y_true,
            y_pred_top1,
            labels=list(range(len(class_names_list))),
            target_names=class_names_list,
            output_dict=True,
            zero_division=0,
        )
        per_class = {}
        for name in class_names_list:
            entry = report.get(name, {})
            per_class[name] = {
                "precision": float(entry.get("precision", 0.0)),
                "recall": float(entry.get("recall", 0.0)),
                "f1_score": float(entry.get("f1-score", 0.0)),
                "support": int(entry.get("support", 0)),
            }
        out["per_class"] = per_class
        out["macro_f1"] = float(report.get("macro avg", {}).get("f1-score", 0.0))
        cm = confusion_matrix(y_true, y_pred_top1, labels=list(range(len(class_names_list))))
        out["confusion_matrix"] = cm.tolist()
    return out


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR | None:
    scheduler_name = cfg["training"].get("lr_scheduler", "none")
    if scheduler_name == "none":
        return None
    warmup_epochs = int(cfg["training"].get("warmup_epochs", 2))
    total_epochs = int(cfg["training"]["epochs"])
    min_lr_ratio = float(cfg["training"].get("min_lr_ratio", 0.01))
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(current_step / max(warmup_steps, 1), 0.01)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_weighted_sampler(records: list[dict], class_to_idx: dict[str, int], strategy: str) -> WeightedRandomSampler | None:
    if strategy == "none":
        return None
    class_counts = Counter(class_to_idx[str(r["landmark_id"])] for r in records)
    num_samples = len(records)
    if strategy == "sqrt_weighted":
        class_weight = {cls: 1.0 / math.sqrt(count) for cls, count in class_counts.items()}
    elif strategy == "inverse_weighted":
        class_weight = {cls: 1.0 / count for cls, count in class_counts.items()}
    else:
        return None
    sample_weights = [class_weight[class_to_idx[str(r["landmark_id"])]] for r in records]
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)


def export_onnx(model: nn.Module, image_size: int, out_path: Path, device: torch.device) -> bool:
    model.eval()
    if hasattr(model, "prepare_for_export"):
        model.prepare_for_export()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    try:
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=["image"],
            output_names=["logits", "embedding"],
            opset_version=17,
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}},
        )
        return True
    except Exception as exc:
        print(f"[onnx] export failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="splits/kfold_seed20260513.json")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    if is_dist():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank())
    device = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(args.data_root).expanduser().resolve()
    split_path = Path(args.split)
    split_summary = read_json(split_path).get("summary", {})
    train_records, val_records, test_records, holdout_records = load_split_records(split_path, args.fold)
    names = class_names(train_records + val_records + test_records)
    class_to_idx = {name: idx for idx, name in enumerate(names)}
    image_size = int(cfg["training"]["image_size"])
    train_tf, eval_tf = transforms_for(image_size, cfg)

    train_ds = LandmarkImageDataset(data_root, train_records, class_to_idx, train_tf)
    val_ds = LandmarkImageDataset(data_root, val_records, class_to_idx, eval_tf)
    test_ds = LandmarkImageDataset(data_root, test_records, class_to_idx, eval_tf)

    sampler = None
    balance_strategy = str(cfg["training"].get("class_balance_strategy", "none"))
    if is_dist():
        sampler = DistributedSampler(train_ds, shuffle=True)
    elif balance_strategy != "none":
        sampler = build_weighted_sampler(train_records, class_to_idx, balance_strategy)
    batch_size = int(cfg["training"]["batch_size_per_gpu"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = build_model(cfg, len(names)).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, image_size, image_size, device=device)
        dummy_labels = torch.zeros(2, dtype=torch.long, device=device)
        model(dummy, dummy_labels)
    model.train(was_training)
    if is_dist():
        model = DDP(model, device_ids=[local_rank()], find_unused_parameters=False)

    base_model = model.module if isinstance(model, DDP) else model
    head_params = list(base_model.head.parameters()) + list(base_model.embedding.parameters())
    head_ids = {id(param) for param in head_params}
    body_params = [param for param in base_model.parameters() if id(param) not in head_ids and param.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": body_params, "lr": float(cfg["training"]["learning_rate"])},
            {"params": head_params, "lr": float(cfg["training"].get("head_learning_rate", cfg["training"]["learning_rate"]))},
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    class_weights_tensor = None
    if balance_strategy != "none" and is_dist():
        class_counts = Counter(class_to_idx[str(r["landmark_id"])] for r in train_records)
        if balance_strategy == "sqrt_weighted":
            raw_weights = {cls: 1.0 / math.sqrt(count) for cls, count in class_counts.items()}
        elif balance_strategy == "inverse_weighted":
            raw_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        else:
            raw_weights = None
        if raw_weights is not None:
            total_w = sum(raw_weights.values())
            normalized = {cls: w / total_w * len(raw_weights) for cls, w in raw_weights.items()}
            class_weights_tensor = torch.tensor(
                [normalized.get(i, 1.0) for i in range(len(names))],
                dtype=torch.float32,
                device=device,
            )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
        weight=class_weights_tensor,
    )

    run_name = f"{cfg['model']['id']}_fold{args.fold}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_name
    if is_main():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        (run_dir / "classes.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "split_summary.json").write_text(
            json.dumps(
                {
                    **split_summary,
                    "fold": args.fold,
                    "fold_counts": {
                        "train": len(train_records),
                        "val": len(val_records),
                        "test": len(test_records),
                        "holdout": len(holdout_records),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            import wandb

            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "landmark-assistant-sprint1"),
                name=run_name,
                config={"candidate": cfg, "split_summary": split_summary, "fold": args.fold},
            )
        except Exception as exc:
            print(f"[wandb] disabled or unavailable: {exc}")

    best_score = (-1.0, -1.0)
    best_metrics: dict = {}
    epochs = int(cfg["training"]["epochs"])
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        count = 0
        iterator = tqdm(train_loader, disable=not is_main(), desc=f"epoch {epoch + 1}/{epochs}")
        for images, labels in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, _ = model(images, labels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * labels.numel()
            count += labels.numel()

        if is_main():
            val_metrics = evaluate(base_model, val_loader, device, names)
            train_loss = total_loss / max(count, 1)
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_top1_accuracy": val_metrics["top1_accuracy"],
                "val_top3_accuracy": val_metrics["top3_accuracy"],
            }
            print(json.dumps(metrics, indent=2))
            try:
                import wandb

                wandb.log(metrics)
            except Exception:
                pass
            score = (val_metrics["top1_accuracy"], val_metrics["top3_accuracy"])
            if score > best_score:
                best_score = score
                best_metrics = metrics
                torch.save({"model": base_model.state_dict(), "classes": names, "config": cfg}, run_dir / "best.pt")
        if is_dist():
            dist.barrier()

    if is_main():
        if (run_dir / "best.pt").exists():
            checkpoint = torch.load(run_dir / "best.pt", map_location=device)
            base_model.load_state_dict(checkpoint["model"])
        test_metrics = evaluate(base_model, test_loader, device, names)
        final = {
            **best_metrics,
            "test_top1_accuracy": test_metrics["top1_accuracy"],
            "test_top3_accuracy": test_metrics["top3_accuracy"],
            "test_count": test_metrics["count"],
            "test_macro_f1": test_metrics.get("macro_f1"),
            "test_per_class": test_metrics.get("per_class"),
            "test_confusion_matrix": test_metrics.get("confusion_matrix"),
            "onnx_export_success": None,
            "onnx_file_mb": None,
        }
        (run_dir / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(json.dumps(final, indent=2), flush=True)
        try:
            import wandb

            wandb.log(final)
        except Exception:
            pass

        export_enabled = os.environ.get("EXPORT_ONNX", "1").lower() not in {"0", "false", "no"}
        if export_enabled:
            onnx_ok = export_onnx(base_model, image_size, run_dir / "landmark_encoder.onnx", device)
            final["onnx_export_success"] = onnx_ok
            final["onnx_file_mb"] = (
                (run_dir / "landmark_encoder.onnx").stat().st_size / 1024 / 1024 if onnx_ok else None
            )
            (run_dir / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        try:
            import wandb

            wandb.log(final)
            wandb.finish()
        except Exception:
            pass
        print(json.dumps(final, indent=2), flush=True)

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
