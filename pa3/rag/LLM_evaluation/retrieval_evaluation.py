import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from sklearn.metrics import ndcg_score
from embedding import load_embedding_model, embed_text
from retrieval import get_database_connection, query_similar_chunks_with_id_return, health_check
from reranking import load_reranking_model, rerank_chunks_with_ids

import asyncio
import re
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time


async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=20.0)

EMBEDDING_MODEL_KEY  = 'bge-m3'
RERANKER_MODEL_KEY   = 'ms-marco'
DB_METRIC            = 'cosine'

EMBEDDING_RETRIEVE_N = 50
FINAL_RETURN_N       = 50
SCORE_BATCH_SIZE     = 25

QUERIES_PATH         = 'retrieval_queries.txt'
OPENAI_MODEL         = "gpt-4o-mini"

NO_RERANKER          = False
FINAL_EVAL           = True


def slugify(text, max_len=60):
    text = text[:max_len]
    text = re.sub(r'[^\w\s-]', '', text, flags=re.UNICODE)
    text = re.sub(r'[\s]+', '_', text)
    return text.strip('_')


def retrieve_chunks(embedding_model, dimension, reranker, db, query_string):
    """
    Returns a list of dicts with keys: text, dist, chunk_id, page_id
    and optionally cross_score (if reranker is used).
    This is the canonical chunk format used by the rest of the eval pipeline.
    """
    query_vector = embed_text(embedding_model, query_string)

    raw = query_similar_chunks_with_id_return(
        db,
        query_vector=query_vector,
        dimension=dimension,
        metric=DB_METRIC,
        top_n=EMBEDDING_RETRIEVE_N,
        table_name=f"page_segment_vec{dimension}"
    )
    # raw rows: (id, page_id, page_segment, distance)

    if not raw:
        print(f"[WARNING] No chunks retrieved from DB for query: '{query_string}'")
        return []

    if NO_RERANKER or reranker is None:
        return [
            {
                "chunk_id": row[0],
                "page_id":  row[1],
                "text":     row[2],
                "dist":     row[3],
            }
            for row in raw
        ]

    reranked = rerank_chunks_with_ids(reranker, query_string, raw, top_k=FINAL_RETURN_N)
    # reranked rows: (chunk_id, page_id, text, vector_score, cross_score)
    return [
        {
            "chunk_id":    row[0],
            "page_id":     row[1],
            "text":        row[2],
            "dist":        row[3],
            "cross_score": row[4],
        }
        for row in reranked
    ]


def build_prompt(query, chunks):
    candidates = [
        {
            "index": i + 1,
            "chunk_id": c["chunk_id"],
            "text": c["text"][:2000],
        }
        for i, c in enumerate(chunks)
    ]

    return f"""Ocenjuješ kakovost semantičnega iskanja za evalvacijo embedding modelov.

Tvoja naloga je strogo oceniti semantično relevantnost vsakega odlomka glede na poizvedbo.

Poizvedba:
{query}

Kandidati:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Za VSAK odlomek določi TOČNO ENO oceno relevantnosti.

Lestvica:
0 = nerelevantno ali skoraj brez povezave s poizvedbo
1 = omenja podobno širšo temo, vendar ni dejansko relevanten
2 = delno relevantno, vsebuje nekaj povezanih informacij
3 = jasno relevantno, vendar ni osredotočeno na glavni namen poizvedbe (govori o temi, ki se neposredno navezuje na poizvedbo, vendar ne govori direktno o njej)
4 = zelo relevantno in močno povezano s poizvedbo

POMEMBNO:
- Bodi zelo diskriminativen med ocenami.
- Ne dodeljuj visokih ocen prelahko.
- Ocene 4 naj bodo redke.
- Razlikuj med širšo tematsko povezanostjo, dejansko relevantnostjo in neposrednim ujemanjem s poizvedbo.

ZELO POMEMBNO:
- Vsak odlomek MORA dobiti natanko eno oceno.
- Število ocen mora biti TOČNO {len(chunks)}.
- Vrni samo seznam števil kot veljaven JSON brez razlag.
- Indeks v seznamu "scores" ustreza polju "index" pri vsakem odlomku.

Vrni IZKLJUČNO JSON v tej obliki:
{{
  "scores": [4,2,1,0]
}}
"""


async def score_batch_async(query, chunks, batch_idx, retries=3):
    prompt = build_prompt(query, chunks)
    for attempt in range(retries):
        try:
            start = time.time()
            response = await async_client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Si strokovnjak za ocenjevanje semantičnih iskalnih sistemov in embedding modelov. "
                            "Ocenjuj semantično relevantnost odlomkov glede na poizvedbo. "
                            "Bodi strog: šibka tematska povezanost dobi nizko oceno, visoke ocene samo za močno relevantne odlomke."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            elapsed = time.time() - start
            scores = json.loads(response.choices[0].message.content)["scores"]
            print(f"[OPENAI] Batch {batch_idx} ({len(chunks)} chunks) done in {elapsed:.2f}s")

            if len(scores) != len(chunks):
                print(f"[WARNING] Batch {batch_idx}: expected {len(chunks)} scores, got {len(scores)} — padding with zeros")
                scores = (scores + [0] * len(chunks))[:len(chunks)]

            return scores

        except asyncio.TimeoutError:
            print(f"[TIMEOUT] Batch {batch_idx} attempt {attempt + 1}/{retries} timed out")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Batch {batch_idx} attempt {attempt + 1}/{retries} JSON parse failed: {e}")
            print(f"[DEBUG] Raw response: {response.choices[0].message.content[:200]}")
        except Exception as e:
            print(f"[ERROR] Batch {batch_idx} attempt {attempt + 1}/{retries} failed: {type(e).__name__}: {e}")

        if attempt < retries - 1:
            wait = 2 ** attempt
            print(f"[RETRY] Batch {batch_idx} waiting {wait}s before retry...")
            await asyncio.sleep(wait)

    print(f"[FAILED] Batch {batch_idx} all {retries} retries exhausted — returning zeros for {len(chunks)} chunks")
    return [0] * len(chunks)


async def score_chunks_async(query, chunks, batch_size):
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    print(f"\n[OPENAI] Scoring {len(chunks)} chunks in {len(batches)} parallel batches of ~{batch_size} for query: '{query}'")

    start = time.time()
    tasks = [score_batch_async(query, batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    print(f"[OPENAI] All batches completed in {elapsed:.2f}s")
    return [score for batch_scores in results for score in batch_scores]


def score_chunks(query, chunks):
    return asyncio.run(score_chunks_async(query, chunks, SCORE_BATCH_SIZE))


def compute_ndcg(ranked, relevance_scores, k, no_reranker):
    pred_scores = np.array([
        -chunk["dist"] if (no_reranker or "cross_score" not in chunk) else chunk["cross_score"]
        for chunk in ranked
    ]).reshape(1, -1)

    true_rel = np.array([[(2 ** s) - 1 for s in relevance_scores]])

    return ndcg_score(true_rel, pred_scores, k=k)


def make_output_filename():
    model_slug = EMBEDDING_MODEL_KEY.replace("/", "_").replace("-", "_")
    if NO_RERANKER:
        return f"{model_slug}__no_reranker.json"
    reranker_slug = RERANKER_MODEL_KEY.replace("/", "_").replace("-", "_")
    return f"{model_slug}__reranker_{reranker_slug}.json"


if __name__ == "__main__":

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        QUERIES = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(QUERIES)} queries from {QUERIES_PATH}")

    K_RANGE = range(2, FINAL_RETURN_N + 1)

    if not FINAL_EVAL:
        OUTPUT_DIR = "llm_scoring_no_reranker" if NO_RERANKER else "llm_scoring_reranker"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"EMBEDDING_MODEL_KEY={EMBEDDING_MODEL_KEY} | DB_METRIC={DB_METRIC}")
    print(f"NO_RERANKER={NO_RERANKER} | FINAL_EVAL={FINAL_EVAL} | SCORE_BATCH_SIZE={SCORE_BATCH_SIZE}")

    embedding_model, dimension, model_id = load_embedding_model(EMBEDDING_MODEL_KEY, cache_dir='../models')
    reranker = None if NO_RERANKER else load_reranking_model(RERANKER_MODEL_KEY, '../models')

    db = get_database_connection()
    if not health_check(db):
        print("[FATAL] Database is not reachable. Aborting.")
        sys.exit(1)
    print(f"[INFO] DB connected | embedding dimension={dimension} | model_id={model_id}")

    all_query_results = {}

    for query_idx, QUERY_STRING in enumerate(QUERIES):

        print("\n==================================================")
        print(f"QUERY {query_idx + 1}/{len(QUERIES)}: {QUERY_STRING}")
        print("==================================================")

        ranked_full = retrieve_chunks(embedding_model, dimension, reranker, db, QUERY_STRING)

        if not ranked_full:
            print(f"[SKIP] No chunks for query — filling with NaN")
            all_query_results[QUERY_STRING] = [np.nan] * len(K_RANGE)
            continue

        # When using reranker, top-k is already applied inside retrieve_chunks.
        # Without reranker, we have EMBEDDING_RETRIEVE_N chunks; trim to FINAL_RETURN_N for scoring.
        if NO_RERANKER:
            ranked_full = ranked_full[:FINAL_RETURN_N]

        print(f"[INFO] {'Embedding order' if NO_RERANKER else 'Reranked'}: {len(ranked_full)} chunks going to LLM scoring")

        relevance_scores_full = score_chunks(QUERY_STRING, ranked_full)

        print(f"[DEBUG] Expected {len(ranked_full)} scores, received {len(relevance_scores_full)}")

        if len(relevance_scores_full) != len(ranked_full):
            print("[WARNING] Final score count mismatch — truncating")
            min_len = min(len(relevance_scores_full), len(ranked_full))
            relevance_scores_full = relevance_scores_full[:min_len]
            ranked_full = ranked_full[:min_len]

        if not FINAL_EVAL:
            query_output_dir = os.path.join(OUTPUT_DIR, slugify(QUERY_STRING))
            os.makedirs(query_output_dir, exist_ok=True)

            with open(os.path.join(query_output_dir, "llm_prompt_full.txt"), "w", encoding="utf-8") as f:
                f.write(build_prompt(QUERY_STRING, ranked_full))

            with open(os.path.join(query_output_dir, "relevance_scores_full.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "query": QUERY_STRING,
                        "embedding_retrieve_n": EMBEDDING_RETRIEVE_N,
                        "final_return_n": FINAL_RETURN_N,
                        "scores": relevance_scores_full
                    },
                    f, indent=2, ensure_ascii=False
                )
            print(f"[FILE] Saved intermediate results -> {query_output_dir}")

        ndcg_values = []
        for k in K_RANGE:
            try:
                ndcg = compute_ndcg(ranked_full, relevance_scores_full, k, NO_RERANKER)
                ndcg_values.append(ndcg)
            except Exception as e:
                print(f"[ERROR] NDCG@{k} failed: {e}")
                ndcg_values.append(np.nan)

        all_query_results[QUERY_STRING] = ndcg_values

    avg_ndcg_per_k = {}
    for idx, k in enumerate(K_RANGE):
        vals = [
            all_query_results[q][idx]
            for q in QUERIES
            if not np.isnan(all_query_results[q][idx])
        ]
        avg_ndcg_per_k[f"ndcg@{k}"] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std":  float(np.std(vals))  if vals else None,
            "n":    len(vals)
        }

    if FINAL_EVAL:
        os.makedirs("model_evals", exist_ok=True)
        out_path = os.path.join("model_evals", make_output_filename())
    else:
        out_path = os.path.join(OUTPUT_DIR, "average_ndcg.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model_key": EMBEDDING_MODEL_KEY,
                "reranking_enabled":   not NO_RERANKER,
                "reranking_model_key": None if NO_RERANKER else RERANKER_MODEL_KEY,
                "db_metric":           DB_METRIC,
                "embedding_retrieve_n": EMBEDDING_RETRIEVE_N,
                "final_return_n":      FINAL_RETURN_N,
                "average_results":     avg_ndcg_per_k
            },
            f, indent=2, ensure_ascii=False
        )
    print(f"\n[FILE] Saved average NDCG -> {out_path}")

    avg_plot_values = [avg_ndcg_per_k[f"ndcg@{k}"]["mean"] for k in K_RANGE]
    std_plot_values = [avg_ndcg_per_k[f"ndcg@{k}"]["std"]  for k in K_RANGE]
    k_list = list(K_RANGE)

    plt.figure(figsize=(10, 5))
    plt.plot(k_list, avg_plot_values, marker="o")
    plt.fill_between(
        k_list,
        [m - s for m, s in zip(avg_plot_values, std_plot_values)],
        [m + s for m, s in zip(avg_plot_values, std_plot_values)],
        alpha=0.2
    )
    plt.xlabel("k")
    plt.ylabel("Average NDCG@k")
    plt.title("Retrieval Quality (No Reranker)" if NO_RERANKER else "Retrieval Quality (With Reranker)")
    plt.grid(True)
    plt.show()