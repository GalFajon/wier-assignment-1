import dspy
from rag_context import retrieve_context_loaded_model


class RAGSignature(dspy.Signature):
    """You are a precise question-answering assistant.
    Use ONLY the provided context to answer the question.
    If the context does not contain enough information, say 'I don't know based on the provided context.'
    Be concise and factual. Do not make up information."""
    context: str = dspy.InputField(desc="Retrieved text passages relevant to the question")
    question: str = dspy.InputField(desc="The user's question to answer")
    answer: str = dspy.OutputField(desc="A concise, factual answer derived strictly from the context")

def run_signature_rag(question, embedding_model, embedding_dim, reranking_model, num_candidates: int = 3, num_final: int = 3) -> str:
    try:
        context = retrieve_context_loaded_model(
            question, 
            loaded_model=embedding_model,
            loaded_dim=embedding_dim,
            loaded_reranker=reranking_model,
            num_candidates=num_candidates, 
            num_final=num_final
        )
        response = dspy.Predict(RAGSignature)(context=context, question=question)
        return response.answer
    except Exception as e:
        print(f"Error: {e}")
        return None


class DirectQuerySignature(dspy.Signature):
    """Answer the question directly using your own knowledge.
    Be concise and factual."""
    question: str = dspy.InputField(desc="The user's question")
    answer: str = dspy.OutputField(desc="A direct answer to the question")

def run_signature_direct_query(question: str) -> str:
    try:
        response = dspy.Predict(DirectQuerySignature)(question=question)
        return response.answer
    except Exception as e:
        print(f"Error: {e}")
        return None