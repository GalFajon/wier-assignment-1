
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


EVAL_DATASET_PATH = "evaluation_dataset.json"






if __name__ == "__main__":
    
    lm = dspy.LM('ollama_chat/qwen3:32b', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm)

    embedding_model, embedding_dim, model_db_id = load_embedding_model(model_key='bge-m3', cache_dir='../models')
    reranking_model = load_reranking_model(model_key='mmarco', cache_dir='../models')
    
    
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    eval_dataset = data["examples"]

    print(f"Loaded {len(eval_dataset)} evaluation examples")
    print(eval_dataset[0].keys())
    
    
    
    
    