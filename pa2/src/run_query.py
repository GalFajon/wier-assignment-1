from sqlalchemy import create_engine


from ParserSettings import ParserSettings, load_settings
from embedding import load_embedding_model, load_reranking_model, embed_string, rerank_candidates
from db_api import get_source_table, get_model_id, query_page_segments


def query_database(model, query_string, settings):
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

    chunks = query_page_segments(engine, model_id, distance_metric, query_vector, n_chunks)
    
    return chunks



    

if __name__ == '__main__':

    QUERY_STRING = 'Vladimir Putin'
    print(f"Querying with: {QUERY_STRING}")
        
    settings = load_settings()
    model = load_embedding_model(settings)
    reranker = load_reranking_model(settings)

    chunks = query_database(model, QUERY_STRING, settings)
    reranked = rerank_candidates(reranker, QUERY_STRING, chunks, settings)

    for i, final_chunk_data in enumerate(reranked):
        text = final_chunk_data['text']
        dist = final_chunk_data['dist']
        cross_score = final_chunk_data['cross_score']
        print(f'{i+1}: Text={text[:30]}... Cross_score={cross_score}, Distance={dist}')
    