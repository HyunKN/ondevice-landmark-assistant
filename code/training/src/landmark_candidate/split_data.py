#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import build_master_records, image_sha1


def add_hashes(data_root: Path, records: list[dict]) -> None:
    for record in records:
        image_path = data_root / str(record["file_name"])
        try:
            record["exact_hash"] = image_sha1(image_path)
        except FileNotFoundError:
            record["exact_hash"] = None


def assign_split_for_class(records: list[dict], rng: random.Random, folds: int, test_ratio: float) -> list[dict]:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = str(record.get("exact_hash") or record.get("image_id"))
        by_hash[key].append(record)

    groups = list(by_hash.values())
    rng.shuffle(groups)

    total = sum(len(group) for group in groups)
    target_test = max(1, round(total * test_ratio)) if total >= 10 else 0
    test_count = 0
    fold_index = 0
    output: list[dict] = []

    for group in groups:
        use_test = test_count < target_test
        for record in group:
            item = dict(record)
            if use_test:
                item["split_group"] = "test"
                item["fold"] = None
                test_count += 1
            else:
                item["split_group"] = "trainval"
                item["fold"] = fold_index % folds
                fold_index += 1
            output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default="splits/kfold_seed20260513.json")
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_path = Path(args.out)
    rng = random.Random(args.seed)

    records = build_master_records(data_root)
    add_hashes(data_root, records)

    confirmed = [r for r in records if r.get("label_status") == "confirmed"]
    holdout = [dict(r, split_group="holdout_non_confirmed", fold=None) for r in records if r.get("label_status") != "confirmed"]

    by_class: dict[str, list[dict]] = defaultdict(list)
    for record in confirmed:
        by_class[str(record["landmark_id"])].append(record)

    split_records: list[dict] = []
    for landmark_id, class_records in sorted(by_class.items()):
        split_records.extend(assign_split_for_class(class_records, rng, args.folds, args.test_ratio))

    split_records.extend(holdout)
    split_records.sort(key=lambda r: (str(r.get("landmark_id")), str(r.get("split_group")), str(r.get("image_id"))))
    fingerprint_payload = [
        {
            "image_id": r.get("image_id"),
            "file_name": r.get("file_name"),
            "landmark_id": r.get("landmark_id"),
            "label_status": r.get("label_status"),
            "exact_hash": r.get("exact_hash"),
        }
        for r in split_records
    ]
    dataset_fingerprint = hashlib.sha1(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    summary = {
        "strategy": "locked_test_plus_stratified_kfold_trainval",
        "seed": args.seed,
        "folds": args.folds,
        "test_ratio": args.test_ratio,
        "data_root": str(data_root),
        "dataset_fingerprint": dataset_fingerprint,
        "total_records": len(split_records),
        "confirmed_records": len(confirmed),
        "class_count": len(by_class),
        "counts": dict(Counter(str(r["split_group"]) for r in split_records)),
        "class_counts": {
            split_name: dict(Counter(str(r["landmark_id"]) for r in split_records if r["split_group"] == split_name))
            for split_name in ["trainval", "test", "holdout_non_confirmed"]
        },
        "policy": {
            "supervised": "label_status == confirmed only",
            "model_selection": "k-fold validation over trainval",
            "final_test": "locked per-class test split; do not tune on it",
            "holdout_non_confirmed": "uncertain/reject/non-confirmed records for calibration smoke only",
            "leakage_control": "exact duplicate SHA1 hashes kept in the same split where detected",
        },
    }

    output = {"summary": summary, "records": split_records}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[split] wrote {out_path}")


if __name__ == "__main__":
    main()
