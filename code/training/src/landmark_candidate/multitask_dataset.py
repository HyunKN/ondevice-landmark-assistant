#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .dataset import IMAGE_EXTS


TRAIN_ROLES = {"train_positive", "low_weight_positive"}


def is_train_positive(record: dict[str, Any]) -> bool:
    if str(record.get("label_status", "confirmed")) != "confirmed":
        return False
    role = str(record.get("training_role") or "train_positive")
    return role in TRAIN_ROLES


def caption_texts(record: dict[str, Any], languages: set[str], max_captions: int) -> list[dict[str, str]]:
    items = record.get("caption_set") or []
    if not isinstance(items, list):
        return []
    output: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        caption_type = str(item.get("caption_type") or "unknown")
        target = str(item.get("target") or "")
        for lang in ("ko", "en"):
            if lang not in languages:
                continue
            text = str(item.get(f"text_{lang}") or "").strip()
            if not text:
                continue
            output.append({"text": text, "language": lang, "caption_type": caption_type, "target": target})
    if max_captions > 0:
        return output[:max_captions]
    return output


def normalize_confusing_with(record: dict[str, Any]) -> list[str]:
    values = record.get("confusing_with") or []
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


@dataclass(frozen=True)
class MultitaskBatch:
    images: torch.Tensor
    labels: torch.Tensor
    image_ids: list[str]
    landmark_ids: list[str]
    caption_texts: list[str]
    caption_image_indices: torch.Tensor
    caption_labels: torch.Tensor
    hard_negative_indices: list[list[int]]
    records: list[dict[str, Any]]


class LandmarkMultitaskDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        records: list[dict[str, Any]],
        class_to_idx: dict[str, int],
        transform=None,
        caption_languages: set[str] | None = None,
        max_captions_per_image: int = 6,
    ) -> None:
        self.data_root = Path(data_root)
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.caption_languages = caption_languages or {"ko", "en"}
        self.max_captions_per_image = max_captions_per_image

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        record = self.records[index]
        image_path = self.data_root / str(record["file_name"])
        if image_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        landmark_id = str(record["landmark_id"])
        return {
            "image": image,
            "label": self.class_to_idx[landmark_id],
            "image_id": str(record.get("image_id") or image_path.stem),
            "landmark_id": landmark_id,
            "captions": caption_texts(record, self.caption_languages, self.max_captions_per_image),
            "confusing_with": normalize_confusing_with(record),
            "record": record,
        }


def make_multitask_collate(class_to_idx: dict[str, int]):
    def collate(samples: list[dict[str, Any]]) -> MultitaskBatch:
        images = torch.stack([sample["image"] for sample in samples], dim=0)
        labels = torch.tensor([int(sample["label"]) for sample in samples], dtype=torch.long)
        image_ids = [str(sample["image_id"]) for sample in samples]
        landmark_ids = [str(sample["landmark_id"]) for sample in samples]
        caption_text_values: list[str] = []
        caption_image_indices: list[int] = []
        caption_labels: list[int] = []
        for image_index, sample in enumerate(samples):
            for caption in sample["captions"]:
                text = str(caption.get("text") or "").strip()
                if not text:
                    continue
                caption_text_values.append(text)
                caption_image_indices.append(image_index)
                caption_labels.append(int(sample["label"]))

        hard_negative_indices: list[list[int]] = []
        for sample in samples:
            label = int(sample["label"])
            hard_negative_indices.append(
                [class_to_idx[item] for item in sample["confusing_with"] if item in class_to_idx and class_to_idx[item] != label]
            )

        return MultitaskBatch(
            images=images,
            labels=labels,
            image_ids=image_ids,
            landmark_ids=landmark_ids,
            caption_texts=caption_text_values,
            caption_image_indices=torch.tensor(caption_image_indices, dtype=torch.long),
            caption_labels=torch.tensor(caption_labels, dtype=torch.long),
            hard_negative_indices=hard_negative_indices,
            records=[sample["record"] for sample in samples],
        )

    return collate
