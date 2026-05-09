from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel



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

def load_embedding_model2(model_name, model_run_device):
    devic_str = model_run_device
    device = devic_str if torch.cuda.is_available() and devic_str == 'cuda' else "cpu"

    model_dir = Path("./models") / model_name.replace("/", "_")

    if model_dir.exists():
        model = SentenceTransformer(str(model_dir), device=device)
    else:
        model = SentenceTransformer(model_name, device=device)
        model.save(str(model_dir))

    return model

def load_embedding_model_hf(settings):
    device_str = settings.model_run_device
    device = device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu"

    model_dir = Path("./models") / settings.model_name.replace("/", "_")

    if model_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        model = AutoModel.from_pretrained(settings.model_name)

        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    model = model.to(device)
    model.eval()

    return model, tokenizer

def load_embedding_model_hf2(settings, model_name):
    device_str = settings.model_run_device
    device = device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu"

    model_dir = Path("./models") / model_name.replace("/", "_")

    if model_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    model = model.to(device)
    model.eval()

    return model, tokenizer


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

def load_reranking_model2(settings, model_name):
    devic_str = settings.model_run_device
    device = devic_str if torch.cuda.is_available() and devic_str == 'cuda' else "cpu"
    
    model_dir = Path("./models") / model_name.replace("/", "_")

    if model_dir.exists():
        model = CrossEncoder(str(model_dir), device=device)
    else:
        model = CrossEncoder(model_name, device=device)
        model.save(str(model_dir))
        
    return model









def embed_chunks(model, chunk_list_raw, settings):
    embeddings = model.encode(
        chunk_list_raw,
        batch_size=settings.batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=settings.normalize_embedding,
    )

    return [emb.tolist() for emb in embeddings]



def embed_chunks_pooling(model, tokenizer, chunk_list_raw, settings):
    device = next(model.parameters()).device
    embeddings = []

    for i in range(0, len(chunk_list_raw), settings.batch_size):
        batch = chunk_list_raw[i:i + settings.batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)

        mean_pooled = summed / counts

        if settings.normalize_embedding:
            mean_pooled = torch.nn.functional.normalize(
                mean_pooled,
                p=2,
                dim=1
            )

        embeddings.extend(mean_pooled.cpu().numpy())

    return [emb.tolist() for emb in embeddings]




def embed_string(model, string, settings):
    emb = model.encode(
        string,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=settings.normalize_embedding,
    )

    return emb.tolist()


def embed_string_resize_vector(model, string, embedding_dimension, settings):
    emb = model.encode(
        string,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=settings.normalize_embedding,
    )

    target_dim = embedding_dimension

    emb_list = emb.tolist()

    if len(emb_list) < target_dim:
        emb_list = emb_list + [0.0] * (target_dim - len(emb_list))
    elif len(emb_list) > target_dim:
        emb_list = emb_list[:target_dim]

    return emb_list


def embed_string_pooling(model, tokenizer, string, settings):
    device = next(model.parameters()).device

    inputs = tokenizer(
        string,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    mean_pooled = summed / counts

    if settings.normalize_embedding:
        mean_pooled = torch.nn.functional.normalize(
            mean_pooled,
            p=2,
            dim=1
        )

    return mean_pooled[0].cpu().numpy().tolist()


def rerank_candidates(reranker, query_string, candidates, settings):
    return_n = settings.rerank_return_n
    
    cross_inputs = [[query_string, raw_text] for raw_text, _, _ in candidates]
    cross_scores = reranker.predict(cross_inputs)
    
    enriched = []
    for (text, _, dist), cross in zip(candidates, cross_scores):
        enriched.append({
            "text": text,
            "dist": float(dist),
            "cross_score": float(cross)
        })

    enriched.sort(key=lambda x: x["cross_score"], reverse=True)

    return enriched[:min(len(enriched), return_n)]




def rerank_candidates2(reranker, query_string, candidates, rerank_return_n):
    return_n = rerank_return_n
    
    cross_inputs = [[query_string, raw_text] for raw_text, _, _ in candidates]
    cross_scores = reranker.predict(cross_inputs)
    
    enriched = []
    for (text, dist, i), cross in zip(candidates, cross_scores):
        # print((text, dist, i))
        enriched.append({
            "text": text,
            "id": i,
            "dist": float(dist),
            "cross_score": float(cross)
        })

    # enriched.sort(key=lambda x: x["cross_score"], reverse=True)
    min_val = min(enriched, key=lambda x: x["cross_score"])["cross_score"]
    max_val = max(enriched, key=lambda x: x["cross_score"])["cross_score"]
    # print(min_val)
    for i in range(len(enriched)):
        enriched[i]["cross_score"] = (enriched[i]["cross_score"] - min_val) / (max_val - min_val)
    return enriched[:min(len(enriched), return_n)]