#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from tqdm import tqdm

from .dataset import class_names, read_json
from .losses import HardNegativeLogitMarginLoss, MultiPositiveImageTextContrastiveLoss
from .multitask_dataset import LandmarkMultitaskDataset, MultitaskBatch, is_train_positive, make_multitask_collate
from .multitask_model import MobileClipS4MultitaskModel
from .train import build_scheduler, build_weighted_sampler, load_split_records, transforms_for


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main() -> bool:
    return rank() == 0


def move_batch(batch: MultitaskBatch, device: torch.device) -> MultitaskBatch:
    return MultitaskBatch(
        images=batch.images.to(device, non_blocking=True),
        labels=batch.labels.to(device, non_blocking=True),
        image_ids=batch.image_ids,
        landmark_ids=batch.landmark_ids,
        caption_texts=batch.caption_texts,
        caption_image_indices=batch.caption_image_indices.to(device, non_blocking=True),
        caption_labels=batch.caption_labels.to(device, non_blocking=True),
        hard_negative_indices=batch.hard_negative_indices,
        records=batch.records,
    )


def filtered_train_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if is_train_positive(record)]


def build_multitask_sampler(records: list[dict[str, Any]], class_to_idx: dict[str, int], strategy: str):
    if strategy == "none":
        return None
    return build_weighted_sampler(records, class_to_idx, strategy)


def build_loader(
    data_root: Path,
    records: list[dict[str, Any]],
    class_to_idx: dict[str, int],
    transform,
    cfg: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    sampler=None,
    distributed: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    data_cfg = cfg.get("data", {})
    languages = set(data_cfg.get("caption_languages", ["ko", "en"]))
    max_captions = int(data_cfg.get("max_captions_per_image", 6))
    dataset = LandmarkMultitaskDataset(
        data_root=data_root,
        records=records,
        class_to_idx=class_to_idx,
        transform=transform,
        caption_languages=languages,
        max_captions_per_image=max_captions,
    )
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=int(cfg.get("runtime", {}).get("num_workers", 8)),
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=make_multitask_collate(class_to_idx),
    )


def initialize_lazy_modules(model: MobileClipS4MultitaskModel, image_size: int, device: torch.device) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy_images = torch.zeros(2, 3, image_size, image_size, device=device)
        dummy_labels = torch.zeros(2, dtype=torch.long, device=device)
        dummy_text = model.tokenize(["Gwanghwamun photo", "광화문 사진"], device)
        model(dummy_images, dummy_labels, dummy_text)
    model.train(was_training)


def class_weight_tensor(records: list[dict[str, Any]], class_to_idx: dict[str, int], strategy: str, device: torch.device):
    if strategy == "none":
        return None
    class_counts = Counter(class_to_idx[str(record["landmark_id"])] for record in records)
    if strategy == "sqrt_weighted":
        raw_weights = {cls: 1.0 / math.sqrt(count) for cls, count in class_counts.items()}
    elif strategy == "inverse_weighted":
        raw_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    else:
        return None
    total_weight = sum(raw_weights.values())
    normalized = {cls: weight / total_weight * len(raw_weights) for cls, weight in raw_weights.items()}
    return torch.tensor(
        [normalized.get(idx, 1.0) for idx in range(len(class_to_idx))],
        dtype=torch.float32,
        device=device,
    )


@torch.no_grad()
def evaluate_image_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    names: list[str],
    out_jsonl: Path | None = None,
    low_margin_csv: Path | None = None,
    low_margin_threshold: float = 0.05,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred_top1: list[int] = []
    top3_hits = 0
    total = 0
    hard_total = 0
    hard_top1_hits = 0
    low_margin_rows: list[dict[str, Any]] = []
    jsonl_handle = out_jsonl.open("w", encoding="utf-8") if out_jsonl else None
    try:
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch.images)
            logits = output.logits
            probs = torch.softmax(logits, dim=1)
            topk_count = min(3, logits.shape[1])
            top_scores, top_indices = torch.topk(probs, k=topk_count, dim=1)
            margins = top_scores[:, 0] - top_scores[:, 1] if topk_count > 1 else top_scores[:, 0]
            labels = batch.labels
            y_true.extend(labels.cpu().tolist())
            y_pred_top1.extend(top_indices[:, 0].cpu().tolist())
            top3_hits += (top_indices == labels.view(-1, 1)).any(dim=1).sum().item()
            total += labels.numel()

            for row_idx, record in enumerate(batch.records):
                true_idx = int(labels[row_idx].item())
                pred_idx = int(top_indices[row_idx, 0].item())
                hard_ids = [str(item) for item in record.get("confusing_with", []) if str(item)]
                hard_indices = {names.index(item) for item in hard_ids if item in names}
                if hard_indices:
                    hard_total += 1
                    hard_top1_hits += int(pred_idx == true_idx)
                row = {
                    "image_id": batch.image_ids[row_idx],
                    "landmark_id": batch.landmark_ids[row_idx],
                    "true_label": names[true_idx],
                    "pred_label": names[pred_idx],
                    "correct_top1": pred_idx == true_idx,
                    "top3_hit": any(int(idx.item()) == true_idx for idx in top_indices[row_idx]),
                    "margin": float(margins[row_idx].item()),
                    "top3": [
                        {"landmark_id": names[int(idx.item())], "score": float(score.item())}
                        for idx, score in zip(top_indices[row_idx], top_scores[row_idx])
                    ],
                    "confusing_with": hard_ids,
                }
                if jsonl_handle:
                    jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if row["margin"] < low_margin_threshold:
                    low_margin_rows.append(row)
    finally:
        if jsonl_handle:
            jsonl_handle.close()

    top1 = accuracy_score(y_true, y_pred_top1) if y_true else 0.0
    top3 = top3_hits / max(total, 1)
    metrics: dict[str, Any] = {
        "top1_accuracy": float(top1),
        "top3_accuracy": float(top3),
        "count": int(total),
        "hard_case_count": int(hard_total),
        "hard_case_top1_accuracy": float(hard_top1_hits / hard_total) if hard_total else None,
        "low_margin_count": len(low_margin_rows),
        "low_margin_threshold": low_margin_threshold,
    }
    if y_true:
        report = classification_report(
            y_true,
            y_pred_top1,
            labels=list(range(len(names))),
            target_names=names,
            output_dict=True,
            zero_division=0,
        )
        metrics["macro_f1"] = float(report.get("macro avg", {}).get("f1-score", 0.0))
        metrics["per_class"] = {
            name: {
                "precision": float(report.get(name, {}).get("precision", 0.0)),
                "recall": float(report.get(name, {}).get("recall", 0.0)),
                "f1_score": float(report.get(name, {}).get("f1-score", 0.0)),
                "support": int(report.get(name, {}).get("support", 0)),
            }
            for name in names
        }
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred_top1, labels=list(range(len(names)))).tolist()
    if low_margin_csv:
        with low_margin_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "landmark_id", "true_label", "pred_label", "margin", "top3", "confusing_with"],
            )
            writer.writeheader()
            for row in low_margin_rows:
                writer.writerow(
                    {
                        "image_id": row["image_id"],
                        "landmark_id": row["landmark_id"],
                        "true_label": row["true_label"],
                        "pred_label": row["pred_label"],
                        "margin": row["margin"],
                        "top3": json.dumps(row["top3"], ensure_ascii=False),
                        "confusing_with": json.dumps(row["confusing_with"], ensure_ascii=False),
                    }
                )
    return metrics


@torch.no_grad()
def build_image_prototypes(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> torch.Tensor:
    model.eval()
    sums: torch.Tensor | None = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for batch in loader:
        batch = move_batch(batch, device)
        embeddings = model(batch.images).image_embedding
        if sums is None:
            sums = torch.zeros(num_classes, embeddings.shape[1], dtype=embeddings.dtype, device=device)
        sums.index_add_(0, batch.labels, embeddings)
        counts.index_add_(0, batch.labels, torch.ones_like(batch.labels, dtype=torch.float32))
    if sums is None:
        raise RuntimeError("Cannot build prototypes from an empty loader")
    counts = counts.clamp_min(1.0).view(-1, 1)
    return torch.nn.functional.normalize(sums / counts, dim=-1)


def load_text_queries(data_root: Path, names: list[str]) -> list[dict[str, str]]:
    catalog_paths = [
        data_root / "assets" / "landmark_text_catalog_v2.json",
        data_root / "landmark_text_catalog_v2.json",
    ]
    queries: list[dict[str, str]] = []
    for path in catalog_paths:
        if not path.exists():
            continue
        data = read_json(path)
        rows = data.get("landmarks", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            landmark_id = str(row.get("landmark_id") or row.get("id") or "")
            if landmark_id not in names:
                continue
            for field in ("name_ko", "name_en", "description_ko", "description_en"):
                text = str(row.get(field) or "").strip()
                if text:
                    queries.append({"landmark_id": landmark_id, "text": text, "source": field})
            for field in ("aliases", "keywords", "query_examples", "texts"):
                values = row.get(field) or []
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    if isinstance(value, dict):
                        text = str(value.get("text") or value.get("text_ko") or value.get("text_en") or "").strip()
                    else:
                        text = str(value).strip()
                    if text:
                        queries.append({"landmark_id": landmark_id, "text": text, "source": field})
    return queries


@torch.no_grad()
def evaluate_text_queries(
    model: MobileClipS4MultitaskModel,
    prototypes: torch.Tensor,
    queries: list[dict[str, str]],
    names: list[str],
    device: torch.device,
    out_jsonl: Path | None = None,
) -> dict[str, Any]:
    if not queries:
        return {"query_count": 0, "top1_accuracy": None, "top3_accuracy": None}
    model.eval()
    class_to_idx = {name: idx for idx, name in enumerate(names)}
    top1_hits = 0
    top3_hits = 0
    jsonl_handle = out_jsonl.open("w", encoding="utf-8") if out_jsonl else None
    try:
        for start in range(0, len(queries), 64):
            chunk = queries[start : start + 64]
            texts = [item["text"] for item in chunk]
            tokens = model.tokenize(texts, device)
            embeddings = model.encode_text_embedding(tokens)
            sims = embeddings @ prototypes.t()
            top_scores, top_indices = torch.topk(sims, k=min(3, prototypes.shape[0]), dim=1)
            for row_idx, query in enumerate(chunk):
                true_idx = class_to_idx[query["landmark_id"]]
                pred = [int(idx.item()) for idx in top_indices[row_idx]]
                top1_hits += int(pred[0] == true_idx)
                top3_hits += int(true_idx in pred)
                if jsonl_handle:
                    row = {
                        **query,
                        "top3": [
                            {"landmark_id": names[int(idx.item())], "score": float(score.item())}
                            for idx, score in zip(top_indices[row_idx], top_scores[row_idx])
                        ],
                    }
                    jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if jsonl_handle:
            jsonl_handle.close()
    return {
        "query_count": len(queries),
        "top1_accuracy": top1_hits / len(queries),
        "top3_accuracy": top3_hits / len(queries),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    cfg: dict[str, Any],
    class_weights,
) -> dict[str, float]:
    model.train()
    loss_cfg = cfg.get("loss", {})
    cls_cfg = loss_cfg.get("classification", {})
    contrastive_cfg = loss_cfg.get("image_text_contrastive", {})
    hard_cfg = loss_cfg.get("hard_negative", {})
    contrastive = MultiPositiveImageTextContrastiveLoss(float(contrastive_cfg.get("temperature", 0.07)))
    hard_negative_loss = HardNegativeLogitMarginLoss(float(hard_cfg.get("margin", 0.1)))
    cls_weight = float(cls_cfg.get("weight", 1.0))
    contrastive_weight = float(contrastive_cfg.get("weight", 0.3)) if contrastive_cfg.get("enabled", True) else 0.0
    hard_weight = float(hard_cfg.get("weight", 0.1)) if hard_cfg.get("enabled", True) else 0.0
    label_smoothing = float(cls_cfg.get("label_smoothing", 0.05))

    totals = Counter()
    sample_count = 0
    iterator = tqdm(loader, disable=not is_main(), desc="train")
    grad_accum_steps = max(1, int(cfg.get("runtime", {}).get("grad_accum_steps", 1)))
    optimizer.zero_grad(set_to_none=True)
    for step_idx, batch in enumerate(iterator):
        batch = move_batch(batch, device)
        base_model = model.module if isinstance(model, DDP) else model
        tokenized = base_model.tokenize(batch.caption_texts, device) if batch.caption_texts else None
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(batch.images, batch.labels, tokenized)
            cls_loss = nn.functional.cross_entropy(
                output.logits,
                batch.labels,
                label_smoothing=label_smoothing,
                weight=class_weights,
            )
            if output.text_embedding is not None and contrastive_weight > 0:
                text_labels = batch.caption_labels
                contrast_loss = contrastive(output.image_embedding, output.text_embedding, batch.labels, text_labels)
            else:
                contrast_loss = output.logits.sum() * 0.0
            if hard_weight > 0:
                hard_loss = hard_negative_loss(output.logits, batch.labels, batch.hard_negative_indices)
            else:
                hard_loss = output.logits.sum() * 0.0
            loss = cls_weight * cls_loss + contrastive_weight * contrast_loss + hard_weight * hard_loss

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        should_step = (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        batch_size = batch.labels.numel()
        sample_count += batch_size
        totals["loss"] += float(loss.item()) * batch_size
        totals["classification_loss"] += float(cls_loss.item()) * batch_size
        totals["contrastive_loss"] += float(contrast_loss.item()) * batch_size
        totals["hard_negative_loss"] += float(hard_loss.item()) * batch_size
        if is_main():
            iterator.set_postfix(loss=f"{float(loss.item()):.4f}")
    return {name: float(value / max(sample_count, 1)) for name, value in totals.items()}


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
    train_records_raw, val_records, test_records, holdout_records = load_split_records(split_path, args.fold)
    train_records = filtered_train_records(train_records_raw)
    names = class_names(train_records + val_records + test_records)
    class_to_idx = {name: idx for idx, name in enumerate(names)}
    image_size = int(cfg["training"]["image_size"])
    train_tf, eval_tf = transforms_for(image_size, cfg)

    balance_strategy = str(cfg.get("data", {}).get("class_balance_strategy", "sqrt_weighted"))
    if not is_dist() and balance_strategy != "none":
        train_loader_sampler = build_multitask_sampler(train_records, class_to_idx, balance_strategy)
    else:
        train_loader_sampler = None

    batch_size = int(cfg["training"]["batch_size_per_gpu"])
    train_loader = build_loader(
        data_root,
        train_records,
        class_to_idx,
        train_tf,
        cfg,
        batch_size,
        shuffle=True,
        sampler=train_loader_sampler,
        distributed=is_dist(),
        drop_last=True,
    )
    val_loader = build_loader(data_root, val_records, class_to_idx, eval_tf, cfg, batch_size, shuffle=False)
    test_loader = build_loader(data_root, test_records, class_to_idx, eval_tf, cfg, batch_size, shuffle=False)
    prototype_loader = build_loader(data_root, train_records, class_to_idx, eval_tf, cfg, batch_size, shuffle=False)

    model = MobileClipS4MultitaskModel(cfg, len(names)).to(device)
    initialize_lazy_modules(model, image_size, device)
    if is_dist():
        model = DDP(model, device_ids=[local_rank()], find_unused_parameters=True)
    base_model = model.module if isinstance(model, DDP) else model

    optimizer_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        base_model.trainable_parameter_groups(cfg),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.05)),
    )
    grad_accum_steps = max(1, int(cfg.get("runtime", {}).get("grad_accum_steps", 1)))
    scheduler_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    scheduler_cfg = {"training": {**cfg.get("training", {}), "lr_scheduler": optimizer_cfg.get("lr_scheduler", "cosine")}}
    scheduler = build_scheduler(optimizer, scheduler_cfg, scheduler_steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    class_weights = class_weight_tensor(train_records, class_to_idx, balance_strategy, device)

    run_name = f"{cfg['model']['id']}_fold{args.fold}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_name
    if is_main():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (run_dir / "classes.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "split_summary.json").write_text(
            json.dumps(
                {
                    **split_summary,
                    "fold": args.fold,
                    "fold_counts": {
                        "train_raw": len(train_records_raw),
                        "train_used": len(train_records),
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
                project=os.environ.get("WANDB_PROJECT", "landmark-assistant-sprint2"),
                name=run_name,
                config={"candidate": cfg, "split_summary": split_summary, "fold": args.fold},
            )
        except Exception as exc:
            print(f"[wandb] disabled or unavailable: {exc}")

    best_score = (-1.0, -1.0)
    best_metrics: dict[str, Any] = {}
    epochs = int(cfg["training"]["epochs"])
    low_margin_threshold = float(cfg.get("evaluation", {}).get("low_margin_threshold", 0.05))
    for epoch in range(epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, class_weights)
        if is_main():
            val_metrics = evaluate_image_split(base_model, val_loader, device, names, low_margin_threshold=low_margin_threshold)
            metrics = {
                "epoch": epoch + 1,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                "val_top1_accuracy": val_metrics["top1_accuracy"],
                "val_top3_accuracy": val_metrics["top3_accuracy"],
                "val_macro_f1": val_metrics.get("macro_f1"),
                "val_hard_case_top1_accuracy": val_metrics.get("hard_case_top1_accuracy"),
                "val_low_margin_count": val_metrics.get("low_margin_count"),
            }
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
            try:
                import wandb

                wandb.log(metrics)
            except Exception:
                pass
            score = (float(val_metrics["top1_accuracy"]), float(val_metrics["top3_accuracy"]))
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
        val_metrics = evaluate_image_split(
            base_model,
            val_loader,
            device,
            names,
            out_jsonl=run_dir / "predictions_val.jsonl",
            low_margin_csv=run_dir / "low_margin_val.csv",
            low_margin_threshold=low_margin_threshold,
        )
        test_metrics = evaluate_image_split(
            base_model,
            test_loader,
            device,
            names,
            out_jsonl=run_dir / "predictions_test.jsonl",
            low_margin_csv=run_dir / "low_margin_test.csv",
            low_margin_threshold=low_margin_threshold,
        )
        text_queries = load_text_queries(data_root, names)
        prototypes = build_image_prototypes(base_model, prototype_loader, device, len(names))
        text_metrics = evaluate_text_queries(
            base_model,
            prototypes,
            text_queries,
            names,
            device,
            out_jsonl=run_dir / "predictions_text_queries.jsonl",
        )
        final = {
            **best_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "text_query_retrieval": text_metrics,
            "run_dir": str(run_dir),
            "best_selection": "validation top1, then validation top3",
            "losses": cfg.get("loss", {}),
        }
        (run_dir / "metrics.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
        try:
            import wandb

            wandb.log(final)
            wandb.finish()
        except Exception:
            pass

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
