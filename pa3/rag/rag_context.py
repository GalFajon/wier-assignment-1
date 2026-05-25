from embedding import load_embedding_model, embed_text
from retrieval import get_database_connection, query_similar_chunks, health_check, query_similar_chunks_with_id_return, get_page_url_by_id
from reranking import load_reranking_model, rerank_chunks, format_context_from_chunks, rerank_chunks_with_ids


def retrieve_context(question, num_candidates=10, num_final=3, embedding_model_key="bge-m3", reranker_model_key="mmarco", metric="cosine"):
    try:
        embedding_model, dimension, model_id = load_embedding_model(embedding_model_key)
        reranker = load_reranking_model(reranker_model_key)
        db = get_database_connection()
        if not health_check(db):
            print("Database not available. Using question as context.")
            return question
        question_vector = embed_text(embedding_model, question)
        candidates = query_similar_chunks(
            db,
            query_vector=question_vector,
            dimension=dimension,
            metric=metric,
            top_n=num_candidates,
            table_name=f"page_segment_vec{dimension}"
        )
        if not candidates:
            print("No chunks retrieved from database. Using question as context.")
            return question
        reranked = rerank_chunks(reranker, question, candidates, top_k=num_final)
        if not reranked:
            print("Reranking returned no results. Using first candidate.")
            return candidates[0][0] if candidates else question
        return format_context_from_chunks(reranked)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        print("Using question as context.")
        return question






def format_chunk_context_with_url_references(chunks):
    if not chunks:
        return "Kontekst ni na voljo."

    formatted_chunks = []

    for i, (chunk_id, page_url, text) in enumerate(chunks, 1):
        formatted_chunks.append(
            f"### ODSEK {i}\n"
            f"ChunkID: {chunk_id}\n"
            f"URL: {page_url}\n"
            f"Besedilo:\n{text}"
        )
    return "\n\n---------------\n\n".join(formatted_chunks)

def retrieve_context_loaded_model(question, loaded_model, loaded_dim, loaded_reranker, num_candidates=10, num_final=3, metric="cosine"):
    try:
        embedding_model = loaded_model
        dimension = loaded_dim
        reranker = loaded_reranker
        db = get_database_connection()
        
        if not health_check(db):
            print("Database not available. Using question as context.")
            return question
        
        question_vector = embed_text(embedding_model, question)
        
        candidates = query_similar_chunks_with_id_return(
            db,
            query_vector=question_vector,
            dimension=dimension,
            metric=metric,
            top_n=num_candidates,
            table_name=f"page_segment_vec{dimension}"
        )
        
        if not candidates:
            print("No chunks retrieved from database. Returning empty context.")
            return ''
        
        reranked_chunks = rerank_chunks_with_ids(reranker, question, candidates, top_k=num_final)
        
        if not reranked_chunks:
            print("Reranking returned no results. Returning empty context.")
            return ''
        
        cleaned_reranked_chunks = []
        for chunk_id, page_id, text, _, _ in reranked_chunks:
            
            page_url = get_page_url_by_id(db, page_id=page_id)
            page_url_string = page_url if page_url != None else "URL not found!"
            
            cleaned_reranked_chunks.append((chunk_id, page_url_string, text))
        
        return format_chunk_context_with_url_references(cleaned_reranked_chunks)
    
    except Exception as e:
        print(f"Retrieval failed: {e}")
        print('Returning empty context.')
        return ''
