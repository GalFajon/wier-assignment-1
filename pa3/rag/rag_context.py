from embedding import load_embedding_model, embed_text
from retrieval import get_database_connection, query_similar_chunks, health_check
from reranking import load_reranking_model, rerank_chunks, format_context_from_chunks


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
