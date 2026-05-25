
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


if __name__ == "__main__":
    
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    eval_dataset = data["examples"]

    print(f"Loaded {len(eval_dataset)} evaluation examples")
    print(eval_dataset[0].keys())
    
    evaluation_stats = []
    
    for eval_example in eval_dataset:
        eval_id = eval_example['id']
        source_url = eval_example['url']
        
        eval_query = eval_example['query']
        eval_expected_answer = eval_example['expected_answer']
        key_facts = eval_example['key_facts']
        
        required_chunk_ids = eval_example['required_chunk_ids']
        acceptable_chunk_ids = eval_example['acceptable_chunk_ids']
        hard_negative_chunk_ids = eval_example['hard_negative_chunk_ids']
        
        print(f"Running evaluation: {eval_id}")
        print(required_chunk_ids)
        print(hard_negative_chunk_ids)