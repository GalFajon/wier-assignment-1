import dspy
import sys

lm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class SimpleRAG(dspy.Signature):
    """Answer the user's question based on the context."""
    context:str = dspy.InputField(desc="Text context")
    question:str = dspy.InputField(desc="Question")
    answer:str = dspy.OutputField()

rag_module = dspy.Predict(SimpleRAG)

default_context = "Vladimir Putin is the president of Russia."
default_question = "Who is the president of Russia?"

if len(sys.argv) > 2:
    context = sys.argv[1]
    question = sys.argv[2]
else:
    context = default_context
    question = default_question

print(f"Context: {context}")
print(f"Question: {question}")
print()

response = rag_module(context=context, question=question)
print(f"Answer: {response.answer}")