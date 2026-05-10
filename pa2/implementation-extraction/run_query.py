from sqlalchemy import create_engine


from ParserSettings import ParserSettings, load_settings
from embedding import load_embedding_model, load_embedding_model2, load_reranking_model, embed_string, rerank_candidates, embed_string_resize_vector, embed_string_pooling
from db_api import get_source_table, get_model_id, query_page_segments

import numpy as np

def query_database(model, query_string, settings: ParserSettings):
    query_vector = embed_string(model, query_string, settings)
    vector_length = np.linalg.norm(query_vector)
    print(vector_length)
    
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

def query_database2(model, query_string, settings, dimensions, query_return_n, distance_metric, model_name):
    if type(model) == tuple:
        query_vector = embed_string_pooling(model[0], model[1], query_string, settings)
    else:
        query_vector = embed_string_resize_vector(model, query_string, dimensions, settings)
    n_chunks = query_return_n
    distance_metric = distance_metric
    embedding_model_name = model_name

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    source_table = get_source_table(engine=engine, schema_name=settings.table_schema, table_name=settings.table_name)
    source_schema = source_table.schema or settings.table_schema
    model_table = get_source_table(engine, source_schema, "model")
    model_id = get_model_id(engine, model_table, embedding_model_name)
    
    if model_id == None:
        raise ValueError('Model Name Not Found In DB')

    chunks = query_page_segments(engine, model_id, distance_metric, query_vector, dimension=dimensions, top_n=n_chunks)
    
    return chunks



    

if __name__ == '__main__':

    QUERY_STRING = 'Ob katerih dneh je živalski vrt Zoo odprt.'
    print(f"Querying with: {QUERY_STRING}")
        
    settings = load_settings()
    print(settings)
    model = load_embedding_model(settings)
    reranker = load_reranking_model(settings)

    chunks = query_database(model, QUERY_STRING, settings)
    print(chunks)
    for i, chunk in enumerate(chunks):
        
        text = chunk[1]
        dist = chunk[2]
        print(f'{i+1:<2}: Text={text[:160]:<160}{'...' if len(text) > 160 else ''}, Distance={dist:.4}')
        if i+1 >= settings.rerank_return_n:
            break
    # reranked = chunks
    reranked = rerank_candidates(reranker, QUERY_STRING, chunks, settings)

    for i, final_chunk_data in enumerate(reranked):
        text = final_chunk_data['text']
        dist = final_chunk_data['dist']
        cross_score = final_chunk_data['cross_score']
        print(f'{i+1:<2}: Text={text[:160]:<160}{'...' if len(text) > 160 else ''}, Cross_score={cross_score:.4}, Distance={dist:.4}')
    