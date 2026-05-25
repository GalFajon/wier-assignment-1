from sentence_transformers import CrossEncoder
import torch
from pathlib import Path
from hf_auth import maybe_login

RERANKING_MODELS = {
    "mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "qnli": "cross-encoder/qnli-distilroberta-base",
}

DEFAULT_RERANKER = "mmarco"

def load_reranking_model(model_key = "mmarco", cache_dir = "./models"):
    maybe_login()
    model_name = RERANKING_MODELS[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_path = Path(cache_dir) / model_name.replace("/", "_")
    
    if cache_path.exists():
        model = CrossEncoder(str(cache_path), device=device)
    else:
        print(f"Loading reranker from HuggingFace: {model_name}")
        model = CrossEncoder(model_name, device=device)
    
    return model


def rerank_chunks(reranker, query, chunks, top_k = 3):
    if not chunks:
        return []
    
    texts = [chunk[0] for chunk in chunks]
    #distances = [chunk[1] for chunk in chunks]
    query_text_pairs = [[query, text] for text in texts]
    cross_encoder_scores = reranker.predict(query_text_pairs)
    
    results = []

    for (text, vector_score), cross_score in zip(chunks, cross_encoder_scores):
        results.append((text, float(vector_score), float(cross_score)))
    
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results[:top_k]


def rerank_chunks_with_ids(reranker, query, chunks, top_k = 3):
    if not chunks:
        return []
    
    # 0: id   1: page_id   2: text   3: similarity metric score
    texts = [chunk[2] for chunk in chunks]
    query_text_pairs = [[query, text] for text in texts]
    cross_encoder_scores = reranker.predict(query_text_pairs)
    
    results = []
    for (chunk_id, page_id, text, vector_score), cross_score in zip(chunks, cross_encoder_scores):
        results.append((chunk_id, page_id, text, float(vector_score), float(cross_score)))
    
    results.sort(key=lambda x: x[4], reverse=True)
    
    return results[:top_k]



def format_context_from_chunks(chunks):
    context_parts = []

    for i, (text, vec_score, cross_score) in enumerate(chunks, 1):
        context_parts.append(f"\n{text}")
    
    return "\n\n".join(context_parts)


