
from nltk.tokenize import sent_tokenize
import nltk


nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

def chunk_fixed_length(text, chunk_size=50):
    """Fixed length chunking."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_segments(text, max_words=256):
	"""Splits text into sentence-based chunks with a max word count limit."""
	sentences = sent_tokenize(text, language="slovene")  # Split into sentences
	chunks, current_chunk = [], []
	current_length = 0

	for sentence in sentences:
		words = sentence.split()
		#print(f"sentence: {sentence}")
		if current_length + len(words) > max_words:
			chunks.append(" ".join(current_chunk))  # Save current chunk
			current_chunk, current_length = [], 0  # Reset chunk
		current_chunk.append(sentence)
		current_length += len(words)
    
	if current_chunk:
		chunks.append(" ".join(current_chunk))  # Add last chunk

	return chunks
