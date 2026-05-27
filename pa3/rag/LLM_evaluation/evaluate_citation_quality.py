import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from collections import defaultdict

ANSWERS_PATH = "eval_answers/ollama_chat-qwen3-8b_rag_one_shot_bge-m3_50_mmarco_3.json"
K = 3


EVAL_DATASET_PATH = "eval_datasets/evaluation_dataset.json"

def load_answers(answers_path: str) -> dict:
    with open(answers_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw if isinstance(raw, list) else raw.get("examples", raw.get("answers", []))

    answers_by_id = {}
    for item in items:
        eval_id = item.get("eval_id")
        if eval_id:
            answers_by_id[eval_id] = item

    return answers_by_id


def load_eval_dataset(dataset_path: str) -> list:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return raw if isinstance(raw, list) else raw.get("examples", [])


def to_str_set(values: list) -> set[str]:
    return set(str(v) for v in values)


def chunk_sort_key(x):
    x = str(x)
    if x.isdigit():
        return (0, int(x))
    return (1, x)


def score_citation_quality(
    cited_chunk_ids: list[str],
    referenced_chunk_ids: list[str],
    required_chunk_ids: list[str],
    acceptable_chunk_ids: list[str],
    partial_support_chunk_ids: list[str],
    hard_negative_chunk_ids: list[str],
) -> dict:
    cited = to_str_set(cited_chunk_ids)
    referenced = to_str_set(referenced_chunk_ids)

    required = to_str_set(required_chunk_ids)
    acceptable = to_str_set(acceptable_chunk_ids)
    partial_support = to_str_set(partial_support_chunk_ids)
    hard_negatives = to_str_set(hard_negative_chunk_ids)

    acceptable = acceptable | required
    partial_support = partial_support - acceptable
    hard_negatives = hard_negatives - acceptable - partial_support
    all_surfaced = cited | referenced

    hit_at_k = bool(all_surfaced & acceptable)

    precision_at_k = (
        len(all_surfaced & acceptable) / len(all_surfaced)
        if all_surfaced else None
    )

    recall_at_k = (
        len(all_surfaced & acceptable) / len(acceptable)
        if acceptable else None
    )

    required_hit = bool(all_surfaced & required)
    acceptable_hit = bool(all_surfaced & acceptable)
    partial_support_hit = bool(all_surfaced & partial_support)
    hard_negative_hit = bool(all_surfaced & hard_negatives)

    required_recall = (
        len(all_surfaced & required) / len(required)
        if required else None
    )

    acceptable_recall = (
        len(all_surfaced & acceptable) / len(acceptable)
        if acceptable else None
    )

    partial_support_recall = (
        len(all_surfaced & partial_support) / len(partial_support)
        if partial_support else None
    )

    cited_reference_match = cited == referenced

    cited_without_reference = sorted(cited - referenced, key=chunk_sort_key)
    referenced_without_citation = sorted(referenced - cited, key=chunk_sort_key)

    return {
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,

        "required_hit": required_hit,
        "acceptable_hit": acceptable_hit,
        "partial_support_hit": partial_support_hit,
        "hard_negative_hit": hard_negative_hit,

        "required_recall": required_recall,
        "acceptable_recall": acceptable_recall,
        "partial_support_recall": partial_support_recall,

        "cited_reference_match": cited_reference_match,
        "cited_without_reference": cited_without_reference,
        "referenced_without_citation": referenced_without_citation,

        "surfaced_chunk_ids": sorted(all_surfaced, key=chunk_sort_key),
        "cited_chunk_ids": sorted(cited, key=chunk_sort_key),
        "referenced_chunk_ids": sorted(referenced, key=chunk_sort_key),

        "required_chunk_ids": sorted(required, key=chunk_sort_key),
        "acceptable_chunk_ids": sorted(acceptable, key=chunk_sort_key),
        "partial_support_chunk_ids": sorted(partial_support, key=chunk_sort_key),
        "hard_negative_chunk_ids": sorted(hard_negatives, key=chunk_sort_key),
    }


def run_eval(dataset_path: str, answers_path: str):
    eval_dataset = load_eval_dataset(dataset_path)
    answers_by_id = load_answers(answers_path)

    print(f"Loaded {len(eval_dataset)} eval examples, {len(answers_by_id)} answers\n")

    results = []
    missing = []

    for example in eval_dataset:
        eval_id = example.get("id")
        answer = answers_by_id.get(eval_id)

        if not answer:
            missing.append(eval_id)
            continue

        cited = answer.get("cited_chunk_ids", [])
        referenced = answer.get("referenced_chunk_ids", [])

        score = score_citation_quality(
            cited_chunk_ids=cited,
            referenced_chunk_ids=referenced,
            required_chunk_ids=example.get("required_chunk_ids", []),
            acceptable_chunk_ids=example.get("acceptable_chunk_ids", []),
            partial_support_chunk_ids=example.get("partial_support_chunk_ids", []),
            hard_negative_chunk_ids=example.get("hard_negative_chunk_ids", []),
        )

        result = {
            "eval_id": eval_id,
            "query": example.get("query", ""),
            **score,
        }
        results.append(result)

        status = "✓" if score["hit_at_k"] else "✗"
        partial = " ~ PARTIAL" if score["partial_support_hit"] else ""
        neg = " ⚠ HARD NEG" if score["hard_negative_hit"] else ""
        ref_mismatch = " ⚠ REF MISMATCH" if not score["cited_reference_match"] else ""

        print(f"[{status}] {eval_id}{partial}{neg}{ref_mismatch}")

        print(
            f"  hit@{K}={score['hit_at_k']}  "
            f"precision@{K}={score['precision_at_k']:.2f}" if score["precision_at_k"] is not None
            else f"  hit@{K}={score['hit_at_k']}  precision@{K}=N/A"
        )

        if score["recall_at_k"] is not None:
            print(f"  recall@{K}={score['recall_at_k']:.2f}")
        else:
            print(f"  recall@{K}=N/A")

        print(
            f"  required_hit={score['required_hit']}  "
            f"acceptable_hit={score['acceptable_hit']}  "
            f"partial_support_hit={score['partial_support_hit']}"
        )
        print(
            f"  hard_negative_hit={score['hard_negative_hit']}  "
            f"cited_reference_match={score['cited_reference_match']}"
        )

        if score["required_recall"] is not None:
            print(f"  required_recall={score['required_recall']:.2f}")
        else:
            print("  required_recall=N/A")

        if score["acceptable_recall"] is not None:
            print(f"  acceptable_recall={score['acceptable_recall']:.2f}")
        else:
            print("  acceptable_recall=N/A")

        if score["partial_support_recall"] is not None:
            print(f"  partial_support_recall={score['partial_support_recall']:.2f}")
        else:
            print("  partial_support_recall=N/A")

        print(f"  surfaced={score['surfaced_chunk_ids']}")
        print()

        if score["partial_support_hit"]:
            hit_ids = sorted(
                set(score["surfaced_chunk_ids"]) & set(score["partial_support_chunk_ids"]),
                key=chunk_sort_key,
            )
            print("  --- PARTIAL SUPPORT DETAIL ---")
            print(f"  query   : {example.get('query', '')}")
            print(f"  answer  : {answer.get('answer', '')}")
            print(f"  hit ids : {hit_ids}")
            print()

        if score["hard_negative_hit"]:
            hit_ids = sorted(
                set(score["surfaced_chunk_ids"]) & set(score["hard_negative_chunk_ids"]),
                key=chunk_sort_key,
            )
            print("  --- HARD NEGATIVE DETAIL ---")
            print(f"  query   : {example.get('query', '')}")
            print(f"  answer  : {answer.get('answer', '')}")
            print(f"  hit ids : {hit_ids}")
            print()

        if not score["cited_reference_match"]:
            print("  --- CITATION / REFERENCE MISMATCH ---")
            print(f"  cited_without_reference     : {score['cited_without_reference']}")
            print(f"  referenced_without_citation : {score['referenced_without_citation']}")
            print()

    if missing:
        print(f"Missing answers for {len(missing)} examples: {missing}\n")

    n = len(results)

    if n:
        hit_at_k_count = sum(r["hit_at_k"] for r in results)

        precision_at_k_values = [
            r["precision_at_k"]
            for r in results
            if r["precision_at_k"] is not None
        ]

        recall_at_k_values = [
            r["recall_at_k"]
            for r in results
            if r["recall_at_k"] is not None
        ]

        req_hits = sum(r["required_hit"] for r in results)
        acc_hits = sum(r["acceptable_hit"] for r in results)
        partial_hits = sum(r["partial_support_hit"] for r in results)
        neg_hits = sum(r["hard_negative_hit"] for r in results)
        ref_matches = sum(r["cited_reference_match"] for r in results)

        req_recalls = [
            r["required_recall"]
            for r in results
            if r["required_recall"] is not None
        ]

        acc_recalls = [
            r["acceptable_recall"]
            for r in results
            if r["acceptable_recall"] is not None
        ]

        partial_recalls = [
            r["partial_support_recall"]
            for r in results
            if r["partial_support_recall"] is not None
        ]

        print("=" * 50)
        print(f"AGGREGATE  (n={n})")

        hit_at_k = hit_at_k_count / n
        mean_precision_at_k = None
        mean_recall_at_k = None
        
        if precision_at_k_values:
           mean_precision_at_k = sum(precision_at_k_values) / len(precision_at_k_values)
        
        if recall_at_k_values:
            mean_recall_at_k = sum(recall_at_k_values) / len(recall_at_k_values)
        
        print()
        print("MAIN METRICS")
        print(f"  hit@{K}                  : {hit_at_k:.1%}")

        if precision_at_k_values:
            print(
                f"  mean_precision@{K}       : "
                f"{mean_precision_at_k:.2f}"
            )
        else:
            print(f"  mean_precision@{K}       : N/A")

        if recall_at_k_values:
            print(
                f"  mean_recall@{K}          : "
                f"{mean_recall_at_k:.2f}"
            )
        else:
            print(f"  mean_recall@{K}          : N/A")

        print()
        print("RAW / DEBUG METRICS")
        print(f"  required_hit_rate        : {req_hits / n:.1%}")
        print(f"  acceptable_hit_rate      : {acc_hits / n:.1%}")
        print(f"  partial_support_hit_rate : {partial_hits / n:.1%}")
        print(f"  hard_negative_rate       : {neg_hits / n:.1%}")
        print(f"  citation_reference_match : {ref_matches / n:.1%}")

        
        if req_recalls:
            print(f"  mean_required_recall     : {sum(req_recalls) / len(req_recalls):.2f}")

        if acc_recalls:
            print(f"  mean_acceptable_recall   : {sum(acc_recalls) / len(acc_recalls):.2f}")

        if partial_recalls:
            print(f"  mean_partial_recall      : {sum(partial_recalls) / len(partial_recalls):.2f}")

    return results, hit_at_k, mean_precision_at_k, mean_recall_at_k


if __name__ == "__main__":
    results, hit_at_k, mean_prec_at_k, mean_recall_at_k = run_eval(EVAL_DATASET_PATH, ANSWERS_PATH)

    metrics_output = {
        "answers_file": ANSWERS_PATH,
        "hit_at_k": hit_at_k,
        "mean_precision_at_k": mean_prec_at_k,
        "mean_recall_at_k": mean_recall_at_k,
    }

    metrics_output_path = "citation_quality_metrics.json"

    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, ensure_ascii=False, indent=2)

    print(f"Saved citation quality metrics to {metrics_output_path}")