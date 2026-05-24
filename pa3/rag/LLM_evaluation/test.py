
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import dspy
from dotenv import load_dotenv

load_dotenv()


from reranking import load_reranking_model
from embedding import load_embedding_model

from LLM_templates import run_signature_direct_query, run_signature_rag


if __name__ == "__main__":
    
    lm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm)

    embedding_model, embedding_dim, model_db_id = load_embedding_model(model_key='bge-m3', cache_dir='../models')
    reranking_model = load_reranking_model(model_key='mmarco', cache_dir='../models')
    
    query = 'Who is the US president as of 2026?'
    
    llm_answer = run_signature_direct_query(query)
    print(llm_answer)
    
    llm_answer = run_signature_rag(query, embedding_model, embedding_dim, reranking_model, num_candidates = 100, num_final = 3)
    print(llm_answer)