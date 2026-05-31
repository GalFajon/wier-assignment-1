#!/usr/bin/env python3
import sys
import os
app_dir = os.path.dirname(__file__)
rag_dir = os.path.dirname(app_dir)
sys.path.insert(0, rag_dir)
os.chdir(rag_dir)

import dspy
from dotenv import load_dotenv

from LLM_templates import (
    run_signature_direct_query,
    run_signature_rag,
    RAG_ZeroShot_Signature,
    RAG_OneShot_Signature,
)

from embedding import load_embedding_model
from reranking import load_reranking_model

import torch

LLM_MODEL = os.getenv("LLM_MODEL", "ollama_chat/qwen3:14b")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
EMBEDDING_MODEL_KEY = os.getenv("EMBEDDING_MODEL_KEY", "bge-m3")
RERANKING_MODEL_KEY = os.getenv("RERANKING_MODEL_KEY", "bge")
NUM_CANDIDATES = int(os.getenv("NUM_CANDIDATES", "25"))
NUM_FINAL = int(os.getenv("NUM_FINAL", "5"))

print(LLM_MODEL)

def init_dspy():
    try:
        print(f"[INFO] Configuring DSPy with {LLM_MODEL}\n")
        lm = dspy.LM(LLM_MODEL, api_base=OLLAMA_API_BASE, api_key="")
        dspy.configure(lm=lm)
        return True
    except Exception as e:
        print(f"[ERROR] DSPy init failed: {e}")
        return False


def load_models():
    try:
        print(f"[INFO] Loading {EMBEDDING_MODEL_KEY} embedding model...")
        embedding_model, embedding_dim, _ = load_embedding_model(EMBEDDING_MODEL_KEY)
        print(f"[OK] Embedding model loaded (dim: {embedding_dim})")

        print(f"[INFO] Loading {RERANKING_MODEL_KEY} reranking model...")
        reranking_model = load_reranking_model(RERANKING_MODEL_KEY)
        print("[OK] Reranking model loaded\n")

        return embedding_model, embedding_dim, reranking_model
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None


def display_result(title, result):
    print(f"\n{'='*80}\n  {title}\n{'='*80}\n")
    
    if isinstance(result, str):
        print(f"{result}\n")
        return

    print(f"Answer:\n{result.get('answer', '[No answer]')}\n")
    
    refs = result.get("references", [])
    if refs:
        print("References:")
        for i, ref in enumerate(refs, 1):
            print(f"  [{i}] ChunkID: {ref.get('chunk_id')}, URL: {ref.get('url')}")
        print()
    
    if result.get("citation_reference_match") is not None:
        status = "MATCH" if result["citation_reference_match"] else "MISMATCH"
        print(f"Citation Validation: {status}\n")


def run_all_modes(question, embedding_model, embedding_dim, reranking_model):
    print(f"\n")

    print("\n[1/3] Direct Query...")
    result1 = run_signature_direct_query(question)
    display_result("MODE 1: DIRECT (No Context)", result1)

    print("[2/3] RAG Zero-Shot...")
    result2 = run_signature_rag(
        question, RAG_ZeroShot_Signature, embedding_model, embedding_dim,
        reranking_model, num_candidates=NUM_CANDIDATES, num_final=NUM_FINAL,
    )
    display_result("MODE 2: RAG ZERO-SHOT (Context)", result2 or "[Error]")

    print("[3/3] RAG One-Shot...")
    result3 = run_signature_rag(
        question, RAG_OneShot_Signature, embedding_model, embedding_dim,
        reranking_model, num_candidates=NUM_CANDIDATES, num_final=NUM_FINAL,
    )
    display_result("MODE 3: RAG ONE-SHOT (Context + Examples)", result3 or "[Error]")


def main():
    print("="*80 + "\n")

    if not init_dspy():
        sys.exit(1)

    models = load_models()
    if not models:
        sys.exit(1)

    embedding_model, embedding_dim, reranking_model = models
    query_count = 0

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            query_count += 1
            run_all_modes(user_input, embedding_model, embedding_dim, reranking_model)

        except KeyboardInterrupt:
            print("\n\n[INFO] Interrupted.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    load_dotenv()
    main()
