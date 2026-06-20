#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import open_clip


DEFAULT_OOS_QUERIES = [
    {"query": "에펠탑", "language": "ko", "query_type": "out_of_scope"},
    {"query": "부산 바다", "language": "ko", "query_type": "out_of_scope"},
    {"query": "맛집 추천", "language": "ko", "query_type": "out_of_scope"},
    {"query": "Eiffel Tower", "language": "en", "query_type": "out_of_scope"},
    {"query": "beach restaurant", "language": "en", "query_type": "out_of_scope"},
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def language_for_field(field: str) -> str:
    if field.endswith("_ko"):
        return "ko"
    if field.endswith("_en"):
        return "en"
    if field.endswith("_zh"):
        return "zh"
    if field.endswith("_ja"):
        return "ja"
    return "unknown"


def text_type_for_field(field: str) -> str:
    if field.startswith("name_"):
        return "official_name"
    if field.startswith("description_"):
        return "description"
    return field


def infer_alias_language(text: str) -> str:
    if any("\uac00" <= ch <= "\ud7a3" for ch in text):
        return "ko"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    return "en"


def build_catalog_texts(landmark_info: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in landmark_info.get("items", []):
        landmark_id = str(item["landmark_id"])
        parent_id = item.get("parent_landmark_id")
        for field in ("name_ko", "name_en", "description_ko", "description_en"):
            text = str(item.get(field) or "").strip()
            if not text:
                continue
            key = (landmark_id, field, text)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "text_id": f"{landmark_id}_{field}",
                    "landmark_id": landmark_id,
                    "parent_landmark_id": parent_id,
                    "language": language_for_field(field),
                    "text_type": text_type_for_field(field),
                    "text": text,
                    "weight": 1.0 if field.startswith("name_") else 0.85,
                }
            )
        for alias_idx, alias in enumerate(item.get("aliases", [])):
            text = str(alias).strip()
            if not text:
                continue
            key = (landmark_id, "alias", text)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "text_id": f"{landmark_id}_alias_{alias_idx:03d}",
                    "landmark_id": landmark_id,
                    "parent_landmark_id": parent_id,
                    "language": infer_alias_language(text),
                    "text_type": "alias",
                    "text": text,
                    "weight": 0.95,
                }
            )
    return rows


def build_query_set(text_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for row in text_rows:
        queries.append(
            {
                "query_id": row["text_id"],
                "query": row["text"],
                "expected_landmark_id": row["landmark_id"],
                "expected_parent_landmark_id": row.get("parent_landmark_id"),
                "language": row["language"],
                "query_type": row["text_type"],
                "difficulty": "standard",
            }
        )
    for idx, row in enumerate(DEFAULT_OOS_QUERIES):
        queries.append(
            {
                "query_id": f"oos_{idx:03d}",
                "query": row["query"],
                "expected_landmark_id": None,
                "expected_parent_landmark_id": None,
                "language": row["language"],
                "query_type": row["query_type"],
                "difficulty": "out_of_scope",
            }
        )
    return queries


def encode_texts(session: ort.InferenceSession, input_name: str, output_name: str, tokenizer, texts: list[str], batch_size: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(batch).cpu().numpy().astype(np.int64)
        output = session.run([output_name], {input_name: tokens})[0]
        vectors.append(output.astype(np.float32))
    return l2_normalize(np.concatenate(vectors, axis=0))


def build_tokenizer_bundle(tokenizer, fixture_texts: list[str]) -> dict[str, Any]:
    fixtures = []
    for text in fixture_texts:
        fixtures.append({"text": text, "tokens": tokenizer([text])[0].cpu().tolist()})
    merges = [
        {"first": first, "second": second, "rank": rank}
        for (first, second), rank in sorted(tokenizer.bpe_ranks.items(), key=lambda item: item[1])
    ]
    return {
        "version": "open-clip-simple-tokenizer-v1",
        "model_name": "MobileCLIP2-S3",
        "tokenizer_type": "open_clip.SimpleTokenizer",
        "clean": "lower",
        "context_length": tokenizer.context_length,
        "sot_token_id": tokenizer.sot_token_id,
        "eot_token_id": tokenizer.eot_token_id,
        "vocab_size": tokenizer.vocab_size,
        "byte_encoder": {str(key): value for key, value in tokenizer.byte_encoder.items()},
        "encoder": tokenizer.encoder,
        "merges": merges,
        "fixtures": fixtures,
    }


def keyword_score(query: str, text: str, text_type: str) -> float:
    q = query.strip().lower()
    t = text.strip().lower()
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0 if text_type == "official_name" else 0.95
    if q in t or t in q:
        return 0.6
    return 0.0


def evaluate_queries(
    queries: list[dict[str, Any]],
    query_embeddings: np.ndarray,
    text_rows: list[dict[str, Any]],
    text_embeddings: np.ndarray,
    policy: dict[str, Any],
) -> dict[str, Any]:
    semantic_weight = float(policy["semantic_weight"])
    keyword_weight = float(policy["keyword_weight"])
    matched_threshold = float(policy["matched_threshold"])
    ambiguous_margin = float(policy["ambiguous_margin"])
    oos_threshold = float(policy["out_of_scope_threshold"])
    no_keyword_oos_margin = float(policy["no_keyword_oos_margin"])
    no_keyword_match_threshold = float(policy["no_keyword_match_threshold"])
    sims = query_embeddings @ text_embeddings.T
    details: list[dict[str, Any]] = []
    top1_hits = 0
    top3_hits = 0
    supervised_count = 0
    oos_total = 0
    oos_correct = 0
    by_language: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "top1": 0, "top3": 0})
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "top1": 0, "top3": 0})

    for q_idx, query in enumerate(queries):
        per_landmark: dict[str, dict[str, Any]] = {}
        for t_idx, row in enumerate(text_rows):
            landmark_id = row["landmark_id"]
            semantic = float(sims[q_idx, t_idx])
            keyword = keyword_score(query["query"], row["text"], row["text_type"])
            final = semantic_weight * semantic + keyword_weight * keyword
            current = per_landmark.get(landmark_id)
            if current is None or final > current["final_text_score"]:
                per_landmark[landmark_id] = {
                    "landmark_id": landmark_id,
                    "parent_landmark_id": row.get("parent_landmark_id"),
                    "semantic_score": semantic,
                    "keyword_score": keyword,
                    "final_text_score": final,
                    "matched_text": row["text"],
                    "text_type": row["text_type"],
                    "language": row["language"],
                }
        ranked = sorted(per_landmark.values(), key=lambda item: item["final_text_score"], reverse=True)
        top3 = ranked[:3]
        top1 = top3[0] if top3 else None
        top2 = top3[1] if len(top3) > 1 else None
        margin = (top1["final_text_score"] - top2["final_text_score"]) if top1 and top2 else (top1["final_text_score"] if top1 else 0.0)
        if not top1 or top1["final_text_score"] < oos_threshold:
            decision = "out_of_scope"
            reasons = ["top1_below_text_oos"]
        elif top1["keyword_score"] == 0.0 and top1["final_text_score"] < no_keyword_match_threshold:
            decision = "out_of_scope"
            reasons = ["no_keyword_and_score_below_match"]
        elif top1["keyword_score"] == 0.0 and margin < no_keyword_oos_margin:
            decision = "out_of_scope"
            reasons = ["no_keyword_and_margin_low"]
        elif margin < ambiguous_margin:
            decision = "ambiguous"
            reasons = ["text_margin_low"]
        elif top1["final_text_score"] >= matched_threshold:
            decision = "matched"
            reasons = ["text_score_high"]
        else:
            decision = "ambiguous"
            reasons = ["text_score_mid"]

        expected = query.get("expected_landmark_id")
        top_ids = [item["landmark_id"] for item in top3]
        if expected is None:
            oos_total += 1
            oos_correct += int(decision == "out_of_scope")
        else:
            supervised_count += 1
            hit1 = int(top_ids[:1] == [expected])
            hit3 = int(expected in top_ids)
            top1_hits += hit1
            top3_hits += hit3
            by_language[query["language"]]["count"] += 1
            by_language[query["language"]]["top1"] += hit1
            by_language[query["language"]]["top3"] += hit3
            by_type[query["query_type"]]["count"] += 1
            by_type[query["query_type"]]["top1"] += hit1
            by_type[query["query_type"]]["top3"] += hit3
        details.append(
            {
                **query,
                "decision_status": decision,
                "reason_codes": reasons,
                "margin": margin,
                "top3": top3,
                "top1_correct": expected is not None and top_ids[:1] == [expected],
                "top3_correct": expected is not None and expected in top_ids,
            }
        )

    def summarize(groups: dict[str, dict[str, int]]) -> dict[str, Any]:
        return {
            key: {
                "count": value["count"],
                "top1_accuracy": value["top1"] / max(value["count"], 1),
                "top3_recall": value["top3"] / max(value["count"], 1),
            }
            for key, value in sorted(groups.items())
        }

    return {
        "query_count": len(queries),
        "supervised_query_count": supervised_count,
        "out_of_scope_query_count": oos_total,
        "top1_accuracy": top1_hits / max(supervised_count, 1),
        "top3_recall": top3_hits / max(supervised_count, 1),
        "out_of_scope_accuracy": oos_correct / max(oos_total, 1) if oos_total else None,
        "policy": policy,
        "by_language": summarize(by_language),
        "by_query_type": summarize(by_type),
        "failure_cases": [
            item
            for item in details
            if item.get("expected_landmark_id") is not None and not item["top1_correct"]
        ],
        "out_of_scope_cases": [
            item
            for item in details
            if item.get("expected_landmark_id") is None
        ],
        "low_margin_cases": [item for item in details if item["decision_status"] == "ambiguous"],
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate semantic text search artifacts for the Flutter app.")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets/mobile_artifacts_fp16"))
    parser.add_argument("--landmark-info", type=Path, default=Path("assets/landmark_info.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    assets_dir = args.assets_dir
    manifest = read_json(assets_dir / "manifest.json")
    text_info = manifest["text_encoder"]
    tokenizer = open_clip.get_tokenizer(manifest["model_name"])
    session = ort.InferenceSession(str(assets_dir / text_info["onnx"]), providers=["CPUExecutionProvider"])

    landmark_info = read_json(args.landmark_info)
    text_rows = build_catalog_texts(landmark_info)
    queries = build_query_set(text_rows)
    fixture_texts = [
        "Gwanghwamun photo",
        "광화문 사진",
        "돌담 있는 궁궐",
        "palace gate",
        "stream in Seoul",
    ]

    text_embeddings = encode_texts(
        session,
        text_info["input"],
        text_info["output"],
        tokenizer,
        [row["text"] for row in text_rows],
        args.batch_size,
    )
    for row, embedding in zip(text_rows, text_embeddings):
        row["embedding"] = embedding.astype(float).tolist()

    query_embeddings = encode_texts(
        session,
        text_info["input"],
        text_info["output"],
        tokenizer,
        [row["query"] for row in queries],
        args.batch_size,
    )

    policy = {
        "score_type": "semantic_text_fusion",
        "semantic_weight": 0.75,
        "keyword_weight": 0.25,
        "matched_threshold": 0.45,
        "ambiguous_margin": 0.08,
        "out_of_scope_threshold": 0.25,
        "no_keyword_oos_margin": 0.05,
        "no_keyword_match_threshold": 0.60,
    }
    report = evaluate_queries(queries, query_embeddings, text_rows, text_embeddings, policy)
    report_summary = {key: value for key, value in report.items() if key != "details"}

    bundle = build_tokenizer_bundle(tokenizer, fixture_texts)
    write_json(assets_dir / "tokenizer_bundle.json", bundle, compact=True)
    write_json(
        assets_dir / "text_index.json",
        {
            "version": "text-index-v1",
            "model_id": manifest["model_id"],
            "model_name": manifest["model_name"],
            "embedding_dim": manifest["embedding_dim"],
            "count": len(text_rows),
            "items": text_rows,
        },
        compact=True,
    )
    write_json(assets_dir / "text_search_policy.json", policy)
    write_json(
        assets_dir / "text_query_regression_set.json",
        {
            "version": "text-query-regression-set-v1",
            "source": str(args.landmark_info),
            "count": len(queries),
            "items": queries,
        },
    )
    write_json(assets_dir / "text_search_eval_report.json", report_summary)
    print(
        json.dumps(
            {
                "text_index_count": len(text_rows),
                "query_count": len(queries),
                "top1_accuracy": report["top1_accuracy"],
                "top3_recall": report["top3_recall"],
                "out_of_scope_accuracy": report["out_of_scope_accuracy"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
