from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from pathlib import Path



def load_embedding_model(settings):
    devic_str = settings.model_run_device
    device = devic_str if torch.cuda.is_available() and devic_str == 'cuda' else "cpu"

    model_dir = Path("./models") / settings.model_name.replace("/", "_")

    if model_dir.exists():
        model = SentenceTransformer(str(model_dir), device=device)
    else:
        model = SentenceTransformer(settings.model_name, device=device)
        model.save(str(model_dir))

    return model


def load_reranking_model(settings):
    devic_str = settings.model_run_device
    device = devic_str if torch.cuda.is_available() and devic_str == 'cuda' else "cpu"
    
    model_dir = Path("./models") / settings.reranking_model_name.replace("/", "_")

    if model_dir.exists():
        model = CrossEncoder(str(model_dir), device=device)
    else:
        model = CrossEncoder(settings.reranking_model_name, device=device)
        model.save(str(model_dir))
        
    return model


def embed_chunks(model, chunk_list_raw, settings):
    # print(chunk_list_raw)
    # print('Chunk count: ', len(chunk_list_raw))
    embeddings = model.encode(
        chunk_list_raw,
        batch_size=settings.batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    target_dim = settings.embedding_dimension
    output = []

    for emb in embeddings:
        emb_list = emb.tolist()

        if len(emb_list) < target_dim:
            emb_list = emb_list + [0.0] * (target_dim - len(emb_list))
        elif len(emb_list) > target_dim:
            emb_list = emb_list[:target_dim]

        output.append(emb_list)

    return output




def embed_string(model, string, settings):
    emb = model.encode(
        string,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    target_dim = settings.embedding_dimension

    emb_list = emb.tolist()

    if len(emb_list) < target_dim:
        emb_list = emb_list + [0.0] * (target_dim - len(emb_list))
    elif len(emb_list) > target_dim:
        emb_list = emb_list[:target_dim]

    return emb_list





def rerank_candidates(reranker, query_string, candidates, settings):
    return_n = settings.rerank_return_n
    
    cross_inputs = [[query_string, raw_text] for raw_text, _ in candidates]
    cross_scores = reranker.predict(cross_inputs)
    
    enriched = []
    for (text, dist), cross in zip(candidates, cross_scores):
        enriched.append({
            "text": text,
            "dist": float(dist),
            "cross_score": float(cross)
        })

    enriched.sort(key=lambda x: x["cross_score"], reverse=True)

    return enriched[:min(len(enriched), return_n)]