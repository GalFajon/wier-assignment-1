import dspy
import os
import sys
import threading
from dotenv import load_dotenv
from rag_context import retrieve_context

load_dotenv()

lm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434', api_key='')

dspy.configure(lm=lm)


class SimpleRAG(dspy.Signature):
    context: str = dspy.InputField(desc="Retrieved text context")
    question: str = dspy.InputField(desc="User question")
    answer: str = dspy.OutputField(desc="Answer based on context")

class SimpleQuery(dspy.Signature):
    question: str = dspy.InputField(desc="User question")
    answer: str = dspy.OutputField(desc="Direct answer to question")


default_num_candidates = int(os.getenv("NUM_CANDIDATES", "3"))
default_num_final = int(os.getenv("NUM_FINAL", "3"))

if len(sys.argv) > 2:
    question = sys.argv[1]
    num_candidates = int(sys.argv[2])
    num_final = default_num_final
elif len(sys.argv) > 1:
    question = sys.argv[1]
    num_candidates = default_num_candidates
    num_final = default_num_final
else:
    question = "Kdo je predsednik ukrajine?"
    num_candidates = default_num_candidates
    num_final = default_num_final

# Storage for parallel results
results = {
    'with_context': None,
    'without_context': None
}

def run_with_context():
    try:
        context = retrieve_context(question, num_candidates=num_candidates, num_final=num_final)
        rag_module = dspy.Predict(SimpleRAG)
        print(context)
        response = rag_module(context=context, question=question)
        results['with_context'] = response.answer
    except Exception as e:
        results['with_context'] = f"Error: {e}"

def run_without_context():
    try:
        query_module = dspy.Predict(SimpleQuery)
        response = query_module(question=question)
        results['without_context'] = response.answer
    except Exception as e:
        results['without_context'] = f"Error: {e}"

print(f"Query: {question}")
print("-" * 80)

t1 = threading.Thread(target=run_with_context)
t2 = threading.Thread(target=run_without_context)

t1.start()
t2.start()

t1.join()
t2.join()

print("WITH CONTEXT (RAG):")
print("-" * 80)
print(f"{results['with_context']}")
print()

print("WITHOUT CONTEXT (Direct Query):")
print("-" * 80)
print(f"{results['without_context']}")
print()