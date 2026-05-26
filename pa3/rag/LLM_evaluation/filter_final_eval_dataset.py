import json
from pathlib import Path
from typing import Any


INPUT_FILE = "generated_queries.json"
OUTPUT_JSON = "evaluation_dataset.json"
OUTPUT_JSONL = "evaluation_dataset.jsonl"

KEEP_CHUNK_TEXT = True
KEEP_HARD_NEGATIVES = True
KEEP_PARTIAL_SUPPORTS = True


def normalize_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id)


def dedupe_preserve_order(items: list[Any]) -> list[Any]:
    seen = set()
    out = []

    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


def clean_chunk(chunk: dict) -> dict:
    cleaned = {
        "chunk_id": normalize_chunk_id(chunk["chunk_id"]),
    }

    if "chunk_index" in chunk:
        cleaned["chunk_index"] = chunk["chunk_index"]

    if KEEP_CHUNK_TEXT:
        cleaned["text"] = chunk.get("text", "")

    return cleaned


def clean_key_facts(key_facts: list[dict]) -> list[dict]:
    cleaned = []

    for fact in key_facts:
        if not fact.get("fact"):
            continue

        cleaned.append({
            "fact": fact["fact"],
            "supported_by": [
                normalize_chunk_id(cid)
                for cid in fact.get("supported_by", [])
            ],
        })

    return cleaned


def clean_source_window(source_window: dict) -> dict:
    return {
        "window_id": source_window.get("window_id"),
        "chunk_ids": [
            normalize_chunk_id(cid)
            for cid in source_window.get("chunk_ids", [])
        ],
        "chunk_indices": source_window.get("chunk_indices", []),
        "window_size": source_window.get("window_size"),
        "position_bucket": source_window.get("position_bucket"),
    }


def clean_example(example: dict, idx: int) -> dict | None:
    if example.get("usable") is not True:
        return None

    required_chunk_ids = [
        normalize_chunk_id(cid)
        for cid in example.get("required_chunk_ids", [])
    ]

    acceptable_chunk_ids = [
        normalize_chunk_id(cid)
        for cid in example.get("acceptable_chunk_ids", [])
    ]

    partial_support_chunk_ids = [
        normalize_chunk_id(cid)
        for cid in example.get("partial_support_chunk_ids", [])
    ]

    hard_negative_chunk_ids = [
        normalize_chunk_id(cid)
        for cid in example.get("hard_negative_chunk_ids", [])
    ]

    if not example.get("query") or not example.get("expected_answer"):
        return None

    if not required_chunk_ids:
        return None

    # Ensure required chunks are always acceptable support.
    acceptable_chunk_ids = dedupe_preserve_order(
        required_chunk_ids + acceptable_chunk_ids
    )

    # Partial supports should not overlap with positives.
    positive_ids = set(acceptable_chunk_ids)
    partial_support_chunk_ids = [
        cid for cid in partial_support_chunk_ids
        if cid not in positive_ids
    ]

    # Hard negatives should not overlap with positives or partial supports.
    non_negative_ids = set(acceptable_chunk_ids) | set(partial_support_chunk_ids)
    hard_negative_chunk_ids = [
        cid for cid in hard_negative_chunk_ids
        if cid not in non_negative_ids
    ]

    cleaned = {
        "id": f"eval_{idx:06d}",
        "page_id": example.get("page_id"),
        "url": example.get("url"),

        "query": example["query"],
        "expected_answer": example["expected_answer"],
        "key_facts": clean_key_facts(example.get("key_facts", [])),

        "question_type": example.get("question_type"),
        "difficulty": example.get("difficulty"),

        "source_window": clean_source_window(example.get("source_window", {})),

        "required_chunk_ids": required_chunk_ids,
        "acceptable_chunk_ids": acceptable_chunk_ids,

        "support_label_status": example.get(
            "support_label_status",
            "llm_verified_pooled_not_exhaustive",
        ),
    }

    if KEEP_CHUNK_TEXT:
        cleaned["required_chunks"] = [
            clean_chunk(chunk)
            for chunk in example.get("required_chunks", [])
            if normalize_chunk_id(chunk.get("chunk_id")) in set(required_chunk_ids)
        ]

        cleaned["acceptable_chunks"] = [
            clean_chunk(chunk)
            for chunk in example.get("acceptable_chunks", [])
            if normalize_chunk_id(chunk.get("chunk_id")) in set(acceptable_chunk_ids)
        ]

    if KEEP_PARTIAL_SUPPORTS:
        cleaned["partial_support_chunk_ids"] = partial_support_chunk_ids

        if KEEP_CHUNK_TEXT:
            cleaned["partial_support_chunks"] = [
                clean_chunk(chunk)
                for chunk in example.get("partial_support_chunks", [])
                if normalize_chunk_id(chunk.get("chunk_id")) in set(partial_support_chunk_ids)
            ]

    if KEEP_HARD_NEGATIVES:
        cleaned["hard_negative_chunk_ids"] = hard_negative_chunk_ids

        if KEEP_CHUNK_TEXT:
            cleaned["hard_negative_chunks"] = [
                clean_chunk(chunk)
                for chunk in example.get("hard_negative_chunks", [])
                if normalize_chunk_id(chunk.get("chunk_id")) in set(hard_negative_chunk_ids)
            ]

    return cleaned


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_examples = data.get("examples", [])

    cleaned_examples = []

    for raw in raw_examples:
        cleaned = clean_example(raw, idx=len(cleaned_examples) + 1)

        if cleaned is not None:
            cleaned_examples.append(cleaned)

    output = {
        "metadata": {
            "source_file": INPUT_FILE,
            "num_input_examples": len(raw_examples),
            "num_clean_examples": len(cleaned_examples),
            "keep_chunk_text": KEEP_CHUNK_TEXT,
            "keep_hard_negatives": KEEP_HARD_NEGATIVES,
            "keep_partial_supports": KEEP_PARTIAL_SUPPORTS,
        },
        "examples": cleaned_examples,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for example in cleaned_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Loaded {len(raw_examples)} usable raw examples")
    print(f"Saved {len(cleaned_examples)} clean examples to {OUTPUT_JSON}")
    print(f"Saved JSONL version to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()