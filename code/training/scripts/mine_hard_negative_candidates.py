#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PairEvidence:
    true_landmark_id: str
    negative_landmark_id: str
    confusion_count: int = 0
    low_margin_count: int = 0
    nearest_negative_count: int = 0
    total_count: int = 0
    margin_sum: float = 0.0
    negative_score_sum: float = 0.0
    examples: list[dict[str, Any]] = field(default_factory=list)

    def add_example(self, row: dict[str, Any], source: str, max_examples: int) -> None:
        if len(self.examples) >= max_examples:
            return
        self.examples.append(
            {
                "source": source,
                "image_id": row.get("image_id"),
                "landmark_id": row.get("landmark_id"),
                "true_label": row.get("true_label"),
                "pred_label": row.get("pred_label"),
                "margin": row.get("margin"),
                "top3": row.get("top3"),
            }
        )

    @property
    def mean_margin(self) -> float | None:
        if self.total_count == 0:
            return None
        return self.margin_sum / self.total_count

    @property
    def mean_negative_score(self) -> float | None:
        if self.nearest_negative_count == 0:
            return None
        return self.negative_score_sum / self.nearest_negative_count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def read_low_margin_keys(path: Path) -> set[tuple[str | None, str | None]]:
    keys: set[tuple[str | None, str | None]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            keys.add((row.get("image_id"), row.get("true_label") or row.get("landmark_id")))
    return keys


def top3_entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    values = row.get("top3") or []
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError:
            return []
    if not isinstance(values, list):
        return []
    out: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict) and value.get("landmark_id"):
            out.append(value)
    return out


def true_label(row: dict[str, Any]) -> str | None:
    value = row.get("true_label") or row.get("landmark_id")
    return str(value) if value is not None and str(value) else None


def pred_label(row: dict[str, Any]) -> str | None:
    value = row.get("pred_label")
    if value is not None and str(value):
        return str(value)
    entries = top3_entries(row)
    if entries:
        return str(entries[0].get("landmark_id"))
    return None


def nearest_negative(row: dict[str, Any], target: str) -> tuple[str, float | None] | None:
    for entry in top3_entries(row):
        candidate = str(entry.get("landmark_id"))
        if candidate == target:
            continue
        score = entry.get("score")
        try:
            score_value = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_value = None
        return candidate, score_value
    return None


def row_margin(row: dict[str, Any]) -> float | None:
    value = row.get("margin")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def evidence_score(evidence: PairEvidence, weights: dict[str, float]) -> float:
    score = (
        evidence.confusion_count * weights["confusion"]
        + evidence.low_margin_count * weights["low_margin"]
        + evidence.nearest_negative_count * weights["nearest_negative"]
    )
    if evidence.mean_negative_score is not None:
        score += evidence.mean_negative_score * weights["negative_score"]
    if evidence.mean_margin is not None:
        score += max(0.0, 1.0 - evidence.mean_margin) * weights["low_margin_strength"]
    return score


def mine_candidates(
    rows_by_source: list[tuple[str, list[dict[str, Any]]]],
    low_margin_keys: set[tuple[str | None, str | None]],
    low_margin_threshold: float,
    max_examples: int,
) -> dict[tuple[str, str], PairEvidence]:
    pairs: dict[tuple[str, str], PairEvidence] = {}
    for source, rows in rows_by_source:
        for row in rows:
            target = true_label(row)
            if target is None:
                continue
            nearest = nearest_negative(row, target)
            if nearest is None:
                continue
            negative, negative_score = nearest
            key = (target, negative)
            if key not in pairs:
                pairs[key] = PairEvidence(true_landmark_id=target, negative_landmark_id=negative)
            evidence = pairs[key]
            evidence.total_count += 1
            evidence.nearest_negative_count += 1
            if negative_score is not None:
                evidence.negative_score_sum += negative_score
            margin = row_margin(row)
            if margin is not None:
                evidence.margin_sum += margin

            predicted = pred_label(row)
            if predicted == negative and predicted != target:
                evidence.confusion_count += 1
                evidence.add_example(row, f"{source}:confusion", max_examples)

            low_margin_from_file = (row.get("image_id"), target) in low_margin_keys
            low_margin_from_value = margin is not None and margin <= low_margin_threshold
            if low_margin_from_file or low_margin_from_value:
                evidence.low_margin_count += 1
                evidence.add_example(row, f"{source}:low_margin", max_examples)

            evidence.add_example(row, f"{source}:nearest_negative", max_examples)
    return pairs


def write_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "true_landmark_id",
        "negative_landmark_id",
        "candidate_score",
        "confusion_count",
        "low_margin_count",
        "nearest_negative_count",
        "total_count",
        "mean_margin",
        "mean_negative_score",
        "recommendation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine hard-negative landmark pair candidates from evaluation prediction logs."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing predictions_*.jsonl files.")
    parser.add_argument(
        "--prediction-files",
        nargs="*",
        default=["predictions_val.jsonl", "predictions_test.jsonl"],
        help="Prediction JSONL files relative to --run-dir.",
    )
    parser.add_argument(
        "--low-margin-files",
        nargs="*",
        default=["low_margin_val.csv", "low_margin_test.csv"],
        help="Low-margin CSV files relative to --run-dir.",
    )
    parser.add_argument("--low-margin-threshold", type=float, default=0.05)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--out-json", default="hard_negative_candidates.json")
    parser.add_argument("--out-csv", default="hard_negative_candidates.csv")
    parser.add_argument("--confusion-weight", type=float, default=3.0)
    parser.add_argument("--low-margin-weight", type=float, default=2.0)
    parser.add_argument("--nearest-weight", type=float, default=0.5)
    parser.add_argument("--negative-score-weight", type=float, default=1.0)
    parser.add_argument("--low-margin-strength-weight", type=float, default=0.25)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Missing run directory: {run_dir}")

    rows_by_source = []
    for file_name in args.prediction_files:
        path = run_dir / file_name
        rows = read_jsonl(path)
        if rows:
            rows_by_source.append((file_name, rows))

    if not rows_by_source:
        raise SystemExit(
            "No prediction rows found. Expected predictions_val.jsonl or predictions_test.jsonl in the run directory."
        )

    low_margin_keys: set[tuple[str | None, str | None]] = set()
    for file_name in args.low_margin_files:
        low_margin_keys.update(read_low_margin_keys(run_dir / file_name))

    pairs = mine_candidates(
        rows_by_source=rows_by_source,
        low_margin_keys=low_margin_keys,
        low_margin_threshold=args.low_margin_threshold,
        max_examples=args.max_examples,
    )
    weights = {
        "confusion": args.confusion_weight,
        "low_margin": args.low_margin_weight,
        "nearest_negative": args.nearest_weight,
        "negative_score": args.negative_score_weight,
        "low_margin_strength": args.low_margin_strength_weight,
    }
    candidates: list[dict[str, Any]] = []
    for evidence in pairs.values():
        score = evidence_score(evidence, weights)
        if score < args.min_score:
            continue
        candidates.append(
            {
                "true_landmark_id": evidence.true_landmark_id,
                "negative_landmark_id": evidence.negative_landmark_id,
                "candidate_score": score,
                "confusion_count": evidence.confusion_count,
                "low_margin_count": evidence.low_margin_count,
                "nearest_negative_count": evidence.nearest_negative_count,
                "total_count": evidence.total_count,
                "mean_margin": evidence.mean_margin,
                "mean_negative_score": evidence.mean_negative_score,
                "recommendation": "review_for_confusing_with",
                "examples": evidence.examples,
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item["candidate_score"]),
            int(item["confusion_count"]),
            int(item["low_margin_count"]),
            int(item["nearest_negative_count"]),
        ),
        reverse=True,
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank"] = index

    suggestions: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        suggestions[str(candidate["true_landmark_id"])].append(str(candidate["negative_landmark_id"]))

    payload = {
        "run_dir": str(run_dir),
        "prediction_files": args.prediction_files,
        "low_margin_files": args.low_margin_files,
        "low_margin_threshold": args.low_margin_threshold,
        "min_score": args.min_score,
        "weights": weights,
        "candidate_count": len(candidates),
        "usage": {
            "meaning": "These are review candidates. Do not mutate labels.json automatically without human review.",
            "next_step": "For each true_landmark_id, decide whether negative_landmark_id should be added to confusing_with.",
        },
        "label_patch_suggestions": dict(suggestions),
        "candidates": candidates,
    }

    out_json = run_dir / args.out_json
    out_csv = run_dir / args.out_csv
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_csv, candidates)
    print(json.dumps({"candidate_count": len(candidates), "out_json": str(out_json), "out_csv": str(out_csv)}, indent=2))


if __name__ == "__main__":
    main()
