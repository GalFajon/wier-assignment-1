from sqlalchemy import create_engine


from ParserSettings import ParserSettings, load_settings
from embedding import load_embedding_model, load_embedding_model2, load_reranking_model, embed_string, rerank_candidates, embed_string_resize_vector, embed_string_pooling
from db_api import get_source_table, get_model_id, query_page_segments

import numpy as np
from dotenv import load_dotenv



def query_database(model, query_string, settings: ParserSettings):
    query_vector = embed_string(model, query_string, settings)
  
    n_chunks = settings.query_return_n
    distance_metric = settings.distance_metric
    embedding_model_name = settings.model_name

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    source_table = get_source_table(engine=engine, schema_name=settings.table_schema, table_name=settings.table_name)
    source_schema = source_table.schema or settings.table_schema
    model_table = get_source_table(engine, source_schema, "model")
    model_id = get_model_id(engine, model_table, embedding_model_name)
    
    if model_id == None:
        raise ValueError('Model Name Not Found In DB')

    chunks = query_page_segments(engine, model_id, distance_metric, query_vector, dimension=settings.embedding_dimension, top_n=n_chunks)
    
    return chunks


def retrieve_chunks(model, reranker, query_string, settings):
    chunks = query_database(model, query_string, settings)
    reranked = rerank_candidates(reranker, query_string, chunks, settings)
    return reranked
    
    
    
    
    
if __name__ == '__main__':

    load_dotenv()
    settings = load_settings()

    print("\n" + "=" * 100)
    print("DOCUMENT RETRIEVAL DEMO")
    print("=" * 100)

    model = load_embedding_model(settings)
    reranker = load_reranking_model(settings)

    MAX_LEN = 250
    
    queries = [
        "Vladimir Putin odziv na Ameriko",
        "Trump glede Ukrajine",
        "Rusija in Ukrajina vojna"
        
        # slabo
        "Vladimir Putin",
        "Ob katerih dneh je živalski vrt Zoo odprt",
        "Vojno stanje v Ukrajini v Aprilu leta 2024",

    ]

    for query_idx, query_string in enumerate(queries, start=1):

        print(f"\n[{query_idx}] QUERY: {query_string}")
        print("-" * 100)

        retrieved_chunks = retrieve_chunks(
            model,
            reranker,
            query_string,
            settings
        )

        for chunk_idx, chunk_data in enumerate(retrieved_chunks, start=1):

            text = chunk_data["text"].strip().replace("\n", " ")
            dist = chunk_data["dist"]
            cross_score = chunk_data["cross_score"]

            preview = text[:MAX_LEN]
            if len(text) > MAX_LEN:
                preview += "..."

            print(
                f"{chunk_idx:02d} | "
                f"Cross={cross_score:7.4f} | "
                f"Dist={dist:7.4f} | "
                f"{preview}"
            )

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)