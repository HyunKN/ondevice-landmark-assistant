#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:  # type: ignore[no-redef]
        pass


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def image_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_record(record: dict, landmark_id: str) -> dict:
    out = dict(record)
    file_name = str(out.get("file_name", "")).replace("\\", "/")
    basename = Path(file_name).name
    out["landmark_id"] = landmark_id
    out["file_name"] = str(Path(landmark_id) / "images" / basename).replace("\\", "/")
    out["image_id"] = str(out.get("image_id") or f"{landmark_id}_{Path(basename).stem}")
    out["label_status"] = str(out.get("label_status") or "confirmed")
    return out


def build_master_records(data_root: Path) -> list[dict]:
    data_root = data_root.resolve()
    records: list[dict] = []
    for landmark_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        labels_path = landmark_dir / "labels.json"
        images_dir = landmark_dir / "images"
        if not labels_path.exists() or not images_dir.exists():
            continue
        data = read_json(labels_path)
        if not isinstance(data, list):
            raise ValueError(f"{labels_path} must contain a list")
        for item in data:
            if isinstance(item, dict):
                record = normalize_record(item, landmark_dir.name)
                if (data_root / record["file_name"]).exists():
                    records.append(record)
    return records


def class_names(records: Iterable[dict]) -> list[str]:
    return sorted({str(record["landmark_id"]) for record in records if record.get("label_status") == "confirmed"})


class LandmarkImageDataset(Dataset):
    def __init__(self, data_root: Path, records: list[dict], class_to_idx: dict[str, int], transform=None) -> None:
        self.data_root = Path(data_root)
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        image_path = self.data_root / str(record["file_name"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[str(record["landmark_id"])]
        return image, label
