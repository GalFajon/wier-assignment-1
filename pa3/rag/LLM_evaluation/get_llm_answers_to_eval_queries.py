
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


LLM_MODEL = 'ollama_chat/qwen3:14b'

MODEL_ANSWER_MODE = 'rag_one_shot' # direct, rag_zero_shot, rag_one_shot
UNBIASED_DATASET = False

EMBEDDING_MODEL_KEY = 'bge-m3'
EMBEEDDING_SIMILARITY_RETURN_N = 50
RERANKING_MODEL_KEY = 'bge'
RERANKING_RETURN_N = 9


if UNBIASED_DATASET:
    EVAL_DATASET_PATH = "eval_datasets/evaluation_dataset_unbiased.json"
else:
    EVAL_DATASET_PATH = "eval_datasets/evaluation_dataset.json"

lm = dspy.LM(LLM_MODEL, api_base='http://localhost:11434', api_key='', num_retries=3)
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
        answer_dict['eval_id'] = eval_id
        answer_dicts.append(answer_dict)



    output_json_path = f"{slugify_filename(LLM_MODEL)}"
    
    if MODEL_ANSWER_MODE == 'direct':
        output_json_path += "_direct"
    else:
        output_json_path += f"_{MODEL_ANSWER_MODE}"
        output_json_path += f"_{EMBEDDING_MODEL_KEY}"
        output_json_path += f"_{EMBEEDDING_SIMILARITY_RETURN_N}"
        output_json_path += f"_{RERANKING_MODEL_KEY}"
        output_json_path += f"_{RERANKING_RETURN_N}"
    
    if UNBIASED_DATASET:
        output_json_path += "_unbiased"
    
    output_json_path += ".json"

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(answer_dicts, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(answer_dicts)} answer dicts to {output_json_path}")