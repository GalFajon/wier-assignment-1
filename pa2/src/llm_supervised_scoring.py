from sqlalchemy import create_engine
from openai import OpenAI
from sklearn.metrics import ndcg_score

from ParserSettings import load_settings

from embedding import (
    load_embedding_model,
    load_reranking_model,
    embed_string,
    rerank_candidates,
    load_embedding_model_hf,
    embed_string_pooling
)

from db_api import (
    get_source_table,
    get_model_id,
    query_page_segments
)

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def query_database(model, query_string, settings, top_n):
    query_vector = embed_string(model, query_string, settings)

    engine = create_engine(settings.database_url, pool_pre_ping=True)

    source_table = get_source_table(
        engine=engine,
        schema_name=settings.table_schema,
        table_name=settings.table_name
    )

    source_schema = source_table.schema or settings.table_schema
    model_table = get_source_table(engine, source_schema, "model")

    model_id = get_model_id(
        engine,
        model_table,
        settings.model_name
    )

    if model_id is None:
        raise ValueError("Model Name Not Found In DB")

    print(f"Query DB: top_n={top_n}")
    return query_page_segments(
        engine=engine,
        model_id=model_id,
        metric=settings.distance_metric,
        query_vector=query_vector,
        dimension=settings.embedding_dimension,
        top_n=top_n
    )
    
def query_database_hf(model, tokenizer, query_string, settings, top_n):
    query_vector = embed_string_pooling(model, tokenizer, query_string, settings)
    
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    source_table = get_source_table(
        engine=engine,
        schema_name=settings.table_schema,
        table_name=settings.table_name
    )

    source_schema = source_table.schema or settings.table_schema
    model_table = get_source_table(engine, source_schema, "model")

    model_id = get_model_id(
        engine,
        model_table,
        settings.model_name
    )

    if model_id is None:
        raise ValueError("Model Name Not Found In DB")

    print(f"Query DB: top_n={top_n}")
    return query_page_segments(
        engine=engine,
        model_id=model_id,
        metric=settings.distance_metric,
        query_vector=query_vector,
        dimension=settings.embedding_dimension,
        top_n=top_n
    )



def build_prompt(query, chunks):

    chunk_text = "\n\n".join([
        f"[{i+1}] {c['text'][:2000]}"
        for i, c in enumerate(chunks)
    ])

    return f"""
Ocenjuješ kakovost semantičnega iskanja za evalvacijo embedding modelov.

Tvoja naloga je strogo oceniti semantično relevantnost vsakega odlomka glede na poizvedbo.

Poizvedba:
{query}

Kandidati:
{chunk_text}

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
- Oceni predvsem:
  - semantično relevantnost,
  - informacijsko bližino,
  - kako neposredno odlomek obravnava poizvedbo.
- Ocene 4 naj bodo redke.
- Razlikuj med:
  - širšo tematsko povezanostjo,
  - dejansko relevantnostjo,
  - neposrednim ujemanjem s poizvedbo.

ZELO POMEMBNO:
- Vsak odlomek MORA dobiti natanko eno oceno.
- Ne izpusti nobenega odlomka.
- Število ocen mora biti TOČNO {len(chunks)}.
- Prvi element predstavlja prvi odlomek, drugi element drugi odlomek itd.
- Vrni samo seznam števil.
- Odgovor mora biti veljaven JSON.
- Ne dodajaj razlag, komentarjev ali dodatnega besedila.
- Strogo se drži zahtevane oblike.

Vrni IZKLJUČNO JSON v tej obliki:

{{
  "scores": [5,3,1,0]
}}
"""


def score_chunks(query, chunks):

    print(f"\n[OPENAI] Scoring {len(chunks)} chunks for query: '{query}'")

    start = time.time()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
            Si strokovnjak za ocenjevanje semantičnih iskalnih sistemov in embedding modelov.

            Tvoja naloga je oceniti relevantnost vrnjenih besedilnih odlomkov glede na poizvedbo.

            Ocene bodo uporabljene za izračun metrik rangiranja, kot je NDCG (Normalized Discounted Cumulative Gain), za primerjavo kakovosti različnih embedding modelov.

            Ocenjevati moraš semantično relevantnost odlomkov glede na poizvedbo.

            Pomembna pravila ocenjevanja:
            - Ocenjuj semantično relevantnost, ne kakovosti pisanja.
            - Upoštevaj tematsko povezanost, informacijsko relevantnost in semantično bližino.
            - Bodi strog in jasno razlikuj med ocenami.
            - Ne dodeljuj visokih ocen prelahko.
            - Šibka tematska povezanost mora dobiti nizko oceno.
            - Visoke ocene naj dobijo samo odlomki, ki so močno povezani s poizvedbo.
            """
            },
            {
                "role": "user",
                "content": build_prompt(query, chunks)
            }
        ]
    )

    elapsed = time.time() - start

    print(f"[OPENAI] Response received in {elapsed:.2f}s")

    content = response.choices[0].message.content

    return json.loads(content)["scores"]



if __name__ == "__main__":

    QUERIES = [
        "Vladimir Putin bombardiral Ukrajino",
        "Ruski napadi na Kijev",
        "Ukrajinska protiofenziva proti ruski vojski",
        "Vojaška pomoč Zahoda Ukrajini",
        "NATO odziv na rusko invazijo",
        "Evropska unija sankcije Rusiji",
        "Ruski raketni napadi na civiliste",
        "Ukrajinski begunci zaradi vojne",
        "Ameriška podpora Ukrajini",
        "Mirovna pogajanja med Rusijo in Ukrajino",
        "Zasedba ukrajinskih ozemelj",
        "Ruska vojska v Donbasu",
        "Napadi z droni v Ukrajini",
        "Ukrajinska obramba Harkova",
        "Putinove izjave o vojni v Ukrajini",
        "Energetska kriza zaradi vojne v Ukrajini",
        "Mednarodne reakcije na rusko agresijo",
        "Humanitarna kriza v Ukrajini",
        "Ukrajinski predsednik Zelenski o vojni",
        "Posledice vojne med Rusijo in Ukrajino"
    ]

    NO_RERANKER = False

    DB_N = 200
    K_RANGE = range(2, 51)

    MAX_K = max(K_RANGE)

    OUTPUT_DIR = (
        "llm_scoring_no_reranker"
        if NO_RERANKER
        else "llm_scoring_reranker"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    settings = load_settings()
    print(settings)

    model = load_embedding_model(settings)
    #model, tokenizer = load_embedding_model_hf(settings)

    reranker = None

    if not NO_RERANKER:
        reranker = load_reranking_model(settings)

    all_query_results = {}
    avg_ndcg_per_k = {}

    for query_idx, QUERY_STRING in enumerate(QUERIES):

        print("\n==================================================")
        print(f"QUERY {query_idx + 1}/{len(QUERIES)}")
        print(QUERY_STRING)
        print("==================================================")

        query_folder_name = (
            QUERY_STRING[:60]
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        query_output_dir = os.path.join(
            OUTPUT_DIR,
            query_folder_name
        )

        os.makedirs(query_output_dir, exist_ok=True)

        ndcg_values = []

        # --------------------------------------------------
        # RETRIEVE
        # --------------------------------------------------

        raw_chunks = query_database(
            model=model,
            query_string=QUERY_STRING,
            settings=settings,
            top_n=DB_N
        )
        
        # raw_chunks = query_database_hf(
        #     model=model,
        #     tokenizer=tokenizer,
        #     query_string=QUERY_STRING,
        #     settings=settings,
        #     top_n=DB_N
        # )

        # --------------------------------------------------
        # OPTIONAL RERANKING
        # --------------------------------------------------
        if NO_RERANKER:
            chunks = [
                {
                    "text": row[0],
                    "dist": row[1],
                    "id": row[2]
                }
                for row in raw_chunks
            ]
        else:
            chunks = raw_chunks
        print(f"[INFO] Retrieved {len(chunks)} chunks")

        
        if NO_RERANKER:

            ranked_full = chunks[:MAX_K]

            print("[INFO] Using raw embedding retrieval order")

        else:

            settings.rerank_return_n = MAX_K

            ranked_full = rerank_candidates(
                reranker,
                QUERY_STRING,
                chunks,
                settings
            )

            print(f"[INFO] Reranked to {len(ranked_full)} chunks")

        # --------------------------------------------------
        # LLM LABELING
        # --------------------------------------------------

        full_prompt = build_prompt(
            QUERY_STRING,
            ranked_full
        )

        full_prompt_path = os.path.join(
            query_output_dir,
            "llm_prompt_full.txt"
        )

        with open(full_prompt_path, "w", encoding="utf-8") as f:
            f.write(full_prompt)

        print(f"[FILE] Saved prompt -> {full_prompt_path}")

        relevance_scores_full = score_chunks(
            QUERY_STRING,
            ranked_full
        )

        print(f"[DEBUG] Expected {len(ranked_full)} scores")
        print(f"[DEBUG] Received {len(relevance_scores_full)} scores")

        if len(relevance_scores_full) != len(ranked_full):

            print("[WARNING] Score count mismatch detected")

            min_len = min(
                len(relevance_scores_full),
                len(ranked_full)
            )

            relevance_scores_full = relevance_scores_full[:min_len]
            ranked_full = ranked_full[:min_len]

            print(f"[FIX] Truncated both arrays to {min_len}")

        full_score_path = os.path.join(
            query_output_dir,
            "relevance_scores_full.json"
        )

        with open(full_score_path, "w", encoding="utf-8") as f:

            json.dump(
                {
                    "query": QUERY_STRING,
                    "max_k": MAX_K,
                    "scores": relevance_scores_full
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"[FILE] Saved scores -> {full_score_path}")

        # --------------------------------------------------
        # NDCG LOOP
        # --------------------------------------------------

        for k in K_RANGE:

            ranked = ranked_full[:k]
            relevance_scores = relevance_scores_full[:k]

            retrieval_scores = []

            for chunk in ranked:

                if not NO_RERANKER and "cross_score" in chunk:

                    retrieval_scores.append(
                        chunk["cross_score"]
                    )

                else:

                    retrieval_scores.append(
                        -chunk["dist"]
                    )

            gain_scores = [
                (2 ** s) - 1
                for s in relevance_scores
            ]

            true_rel = np.array([gain_scores])

            pred_scores = np.array([retrieval_scores])

            pred_scores = (
                pred_scores - np.mean(pred_scores)
            ) / (
                np.std(pred_scores) + 1e-8
            )

            try:

                ndcg = ndcg_score(
                    true_rel,
                    pred_scores.reshape(1, -1),
                    k=k
                )

                ndcg_values.append(ndcg)

                #print(f"[RESULT] NDCG@{k} = {ndcg:.4f}")

            except Exception as e:

                print("[ERROR] Failed to compute NDCG")
                print(e)

                ndcg_values.append(np.nan)

        print(f"[DEBUG] Saving results for query: {QUERY_STRING}")

        all_query_results[QUERY_STRING] = ndcg_values

    # --------------------------------------------------
    # AVERAGE NDCG
    # --------------------------------------------------

    for idx, k in enumerate(K_RANGE):

        vals = []

        for query in QUERIES:

            v = all_query_results[query][idx]

            if not np.isnan(v):
                vals.append(v)

        avg_ndcg = (
            float(np.mean(vals))
            if len(vals) > 0
            else None
        )

        avg_ndcg_per_k[f"ndcg@{k}"] = avg_ndcg

    avg_ndcg_path = os.path.join(
        OUTPUT_DIR,
        "average_ndcg.json"
    )

    with open(avg_ndcg_path, "w", encoding="utf-8") as f:

        json.dump(
            {
                "embedding_model": settings.model_name,
                "reranking_enabled": not NO_RERANKER,
                "reranking_model": (
                    None
                    if NO_RERANKER
                    else settings.reranking_model_name
                ),
                "queries": QUERIES,
                "db_n": DB_N,
                "max_k": MAX_K,
                "average_results": avg_ndcg_per_k
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\n[FILE] Saved average NDCG -> {avg_ndcg_path}")

    avg_plot_values = [
        avg_ndcg_per_k[f"ndcg@{k}"]
        for k in K_RANGE
    ]

    plt.figure(figsize=(10, 5))

    plt.plot(
        list(K_RANGE),
        avg_plot_values,
        marker="o"
    )

    plt.xlabel("k")
    plt.ylabel("Average NDCG@k")

    plt.title(
        "Retrieval Quality (No Reranker)"
        if NO_RERANKER
        else "Retrieval Quality (With Reranker)"
    )

    plt.grid(True)

    plt.show()