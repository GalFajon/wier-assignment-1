
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import re
import unicodedata
import json
import dspy
from dotenv import load_dotenv

load_dotenv()

from reranking import load_reranking_model
from embedding import load_embedding_model

from LLM_templates import run_signature_direct_query, run_signature_rag, RAG_ZeroShot_Signature, RAG_OneShot_Signature


LLM_MODEL = 'ollama_chat/qwen3:32b'

MODEL_ANSWER_MODE = 'direct' # direct, rag_zero_shot, rag_one_shot
EVAL_DATASET_PATH = "evaluation_dataset.json"

EMBEDDING_MODEL_KEY = 'bge-m3'
EMBEEDDING_SIMILARITY_RETURN_N = 50
RERANKING_MODEL_KEY = 'mmarco'
RERANKING_RETURN_N = 3


lm = dspy.LM(LLM_MODEL, api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

embedding_model, embedding_dim, model_db_id = load_embedding_model(model_key=EMBEDDING_MODEL_KEY, cache_dir='../models')
reranking_model = load_reranking_model(model_key=RERANKING_MODEL_KEY, cache_dir='../models')

def get_answer_dict(question):
    if MODEL_ANSWER_MODE == 'direct':
        answer = run_signature_direct_query(question)

        return {
            "answer": answer
        }

    elif MODEL_ANSWER_MODE == 'rag_zero_shot':
        llm_answer = run_signature_rag(
            question,
            RAG_ZeroShot_Signature,
            embedding_model,
            embedding_dim,
            reranking_model,
            num_candidates=EMBEEDDING_SIMILARITY_RETURN_N,
            num_final=RERANKING_RETURN_N,
        )

        if llm_answer is None:
            return {
                "answer": None
            }

        return llm_answer

    elif MODEL_ANSWER_MODE == 'rag_one_shot':
        llm_answer = run_signature_rag(
            question,
            RAG_OneShot_Signature,
            embedding_model,
            embedding_dim,
            reranking_model,
            num_candidates=EMBEEDDING_SIMILARITY_RETURN_N,
            num_final=RERANKING_RETURN_N,
        )

        if llm_answer is None:
            return {
                "answer": None
            }

        return llm_answer

    else:
        raise ValueError(f"Unknown MODEL_ANSWER_MODE: {MODEL_ANSWER_MODE}")


def slugify_filename(value: str, max_length: int = 120) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r'[<>:"/\\|?*\s]+', "-", value)
    value = re.sub(r"[^a-z0-9._-]+", "", value)
    value = re.sub(r"-+", "-", value)
    value = value.strip(".-_")
    if not value:
        value = "unnamed"
    return value[:max_length].strip(".-_")



if __name__ == "__main__":
    
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    eval_dataset = data["examples"]

    print(f"Loaded {len(eval_dataset)} evaluation examples")
    print(eval_dataset[0].keys())
    
    
    answer_dicts = []

    for eval_example in eval_dataset:
        eval_id = eval_example["id"]
        eval_query = eval_example["query"]

        print(f"Running evaluation: {eval_id}")

        answer_dict = get_answer_dict(eval_query)
        answer_dicts.append(answer_dict)


    output_json_path = (
        f"{slugify_filename(LLM_MODEL)}_"
        f"{MODEL_ANSWER_MODE}_"
        f"{EMBEDDING_MODEL_KEY}_"
        f"{EMBEEDDING_SIMILARITY_RETURN_N}_"
        f"{RERANKING_MODEL_KEY}_"
        f"{RERANKING_RETURN_N}.json"
    )

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(answer_dicts, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(answer_dicts)} answer dicts to {output_json_path}")