from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
from hf_auth import maybe_login

EMBEDDING_MODELS = {
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "dimension": 1024,
        "model_id": 1
    },
    "mini-lm": {
        "name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "model_id": 2
    },
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "dimension": 768,
        "model_id": 3
    }
}

DEFAULT_MODEL = "bge-m3"

def load_embedding_model(model_key: str = DEFAULT_MODEL, cache_dir: str = "./models"):
    maybe_login()
    
    config = EMBEDDING_MODELS[model_key]

    model_name = config["name"]    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cache_path = Path(cache_dir) / model_name.replace("/", "_")
    
    if cache_path.exists():
        model = SentenceTransformer(str(cache_path), device=device)
    else:
        print(f"Loading {model_key} from HuggingFace: {model_name}")

        model = SentenceTransformer(model_name, device=device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(cache_path))

        print(f"Model cached to: {cache_path}")
    
    return model, config["dimension"], config["model_id"]


def embed_text(model, text: str) -> list:
    embedding = model.encode([text], convert_to_tensor=False)[0]
    return embedding.tolist()


def embed_batch(model, texts: list) -> list:
    embeddings = model.encode(texts, convert_to_tensor=False)
    return [emb.tolist() for emb in embeddings]
