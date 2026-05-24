import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import json
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from retrieval import (
    get_database_connection,
    get_chunks_for_page_id_and_model,
    get_page_by_id,
    get_pages_by_chunk_length_sum_interval,
)


EMBEDDING_MODEL = 9
EMBEDDING_MODEL_DIM = 1024
SAMPLE_SIZE = 50
SEED = 42

WINDOWS_PER_PAGE = 1
WINDOW_SIZE_WEIGHTS = {
    1: 0.50,
    2: 0.35,
    3: 0.15,
}
POSITION_WEIGHTS = {
    "lead": 0.25,
    "middle": 0.50,
    "tail": 0.25,
}


def weighted_choice(rng: random.Random, weights: dict):
    items = list(weights.keys())
    probs = list(weights.values())
    return rng.choices(items, weights=probs, k=1)[0]


def position_bucket_for_start(start_idx: int, num_chunks: int) -> str:
    if num_chunks <= 1:
        return "lead"

    rel = start_idx / max(num_chunks - 1, 1)

    if rel < 0.25:
        return "lead"
    elif rel < 0.75:
        return "middle"
    else:
        return "tail"


def valid_start_indices_for_bucket(num_chunks: int, window_size: int, bucket: str) -> list[int]:
    max_start = num_chunks - window_size

    if max_start < 0:
        return []

    starts = list(range(max_start + 1))

    return [
        i for i in starts
        if position_bucket_for_start(i, num_chunks) == bucket
    ]


def sample_source_window(chunks: list[dict], rng: random.Random) -> dict | None:
    num_chunks = len(chunks)

    if num_chunks == 0:
        return None

    possible_sizes = [
        size for size in WINDOW_SIZE_WEIGHTS
        if size <= num_chunks
    ]

    if not possible_sizes:
        return None

    size_weights = {
        size: WINDOW_SIZE_WEIGHTS[size]
        for size in possible_sizes
    }

    window_size = weighted_choice(rng, size_weights)

    position_bucket = weighted_choice(rng, POSITION_WEIGHTS)
    valid_starts = valid_start_indices_for_bucket(
        num_chunks=num_chunks,
        window_size=window_size,
        bucket=position_bucket,
    )

    if not valid_starts:
        valid_starts = list(range(0, num_chunks - window_size + 1))

    start_idx = rng.choice(valid_starts)
    end_idx = start_idx + window_size

    selected_chunks = chunks[start_idx:end_idx]
    actual_bucket = position_bucket_for_start(start_idx, num_chunks)

    return {
        "window_id": f"idx_{start_idx}_size_{window_size}",
        "chunk_ids": [c["chunk_id"] for c in selected_chunks],
        "chunk_indices": [c["chunk_index"] for c in selected_chunks],
        "window_size": window_size,
        "start_chunk_index": start_idx,
        "end_chunk_index_exclusive": end_idx,
        "position_bucket": actual_bucket,
        "text": "\n\n".join(
            f"[chunk_id={c['chunk_id']}]\n{c['text']}"
            for c in selected_chunks
        ),
    }


def sample_unique_source_windows(chunks: list[dict], rng: random.Random, n: int) -> list[dict]:
    sampled = []
    seen = set()

    attempts = 0
    max_attempts = n * 20

    while len(sampled) < n and attempts < max_attempts:
        attempts += 1
        window = sample_source_window(chunks, rng)

        if window is None:
            break

        key = tuple(window["chunk_ids"])

        if key in seen:
            continue

        seen.add(key)
        sampled.append(window)

    return sampled


if __name__ == "__main__":
    db = get_database_connection()
    rng = random.Random(SEED)

    page_ids = get_pages_by_chunk_length_sum_interval(
        db,
        model_id=EMBEDDING_MODEL,
        min_length=2000,
        max_length=4000,
        dimension=EMBEDDING_MODEL_DIM,
    )

    print(f"Total pages in interval: {len(page_ids)}")

    sampled_pages = rng.sample(page_ids, min(SAMPLE_SIZE, len(page_ids)))
    print(f"Sampled {len(sampled_pages)} pages")

    output = []

    for page_id, chunk_length_sum in sampled_pages:
        raw_chunks = get_chunks_for_page_id_and_model(
            db,
            page_id=page_id,
            model_id=EMBEDDING_MODEL,
            dimension=EMBEDDING_MODEL_DIM,
        )

        page_data = get_page_by_id(db, page_id=page_id)

        chunks = [
            {
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "text": text,
                "char_length": len(text),
            }
            for idx, (chunk_id, text) in enumerate(raw_chunks)
        ]

        if not chunks:
            continue

        source_windows = sample_unique_source_windows(
            chunks=chunks,
            rng=rng,
            n=WINDOWS_PER_PAGE,
        )

        if not source_windows:
            continue

        output.append({
            "page_id": page_id,
            "url": page_data["url"],
            "num_chunks": len(chunks),
            "total_chunk_chars": sum(c["char_length"] for c in chunks),
            "chunk_length_sum_from_db": chunk_length_sum,
            "source_windows": source_windows,
            "chunks": chunks,
        })

    with open("sampled_pages.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output)} sampled pages to sampled_pages.json")