from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path

def load_model(settings):
    devic_str = settings.model_run_device
    device = devic_str if torch.cuda.is_available() and devic_str == 'cuda' else "cpu"

    model_dir = Path("./models") / settings.model_name.replace("/", "_")

    if model_dir.exists():
        model = SentenceTransformer(str(model_dir), device=device)
    else:
        model = SentenceTransformer(settings.model_name, device=device)
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