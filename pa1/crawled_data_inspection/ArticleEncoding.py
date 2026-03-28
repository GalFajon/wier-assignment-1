import os
import yaml
import pickle
import numpy as np
from tqdm import tqdm

from urllib.parse import urlparse


import stanza
from sklearn.feature_extraction.text import TfidfVectorizer

CACHE_DIR = "lemmatized_texts"
os.makedirs(CACHE_DIR, exist_ok=True)

# Stopwords and POS filter
slovenian_stopwords = set([
    "in", "ali", "da", "se", "je", "so", "za", "na", "s", "z", 
    "kot", "ne", "pa", "to", "ki", "ob", "po", "od", "do", "a"
])
accepted_pos = {"NOUN", "VERB", "ADJ", "ADV"}


def load_articles():
    with open("cleaned_pages.yaml", "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.CFullLoader)

def lemmatize_with_cache(corpus):
    nlp = stanza.Pipeline(
        'sl',
        processors='tokenize,pos,lemma',
        use_gpu=False,
        tokenize_no_ssplit=True,
        verbose=False,
        logging_level='WARN'
    )

    lemmatized_texts = []
    for i, text in enumerate(tqdm(corpus, desc="Lemmatizing")):
        cache_file = os.path.join(CACHE_DIR, f"doc_{i}.txt")

        # Use cached file if exists
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                lemmatized_texts.append(f.read())
            continue

        # Otherwise, lemmatize
        doc = nlp(text)
        lemmas = [
            word.lemma.lower()
            for sentence in doc.sentences
            for word in sentence.words
            if word.lemma and word.lemma.lower() not in slovenian_stopwords and word.upos in accepted_pos
        ]
        joined = ' '.join(lemmas)
        lemmatized_texts.append(joined)

        # Cache result
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(joined)

    return lemmatized_texts

def load_articles_and_lemmas_filtered():
    articles = load_articles()

    corpus = [article['title'] + ' ' + article.get('summary') if article.get('summary') is not None else '' + ' ' + article['body'] for article in articles]
    lemmatized_texts = lemmatize_with_cache(corpus)

    #cleaning up articles
    clean_articles = []
    for art in articles:
        raw_text = art['title'] + ' ' + art.get('summary') if art.get('summary') is not None else '' + ' ' + art['body']

        c_art = {
            'id' : art['id'],
            'title' : art['title'],
            'url' : art['url'],
            'raw_text' : raw_text
        }
        clean_articles.append(c_art)

    return clean_articles, lemmatized_texts


def embed_TFIDF(lemmatized_texts, vector_length=6000, use_cache=True, pickle_file='article_encodings/tfidf_encoded_articles.pkl'):
    if use_cache and os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            print(f"Loading pickled TF-IDF embedded articles from: {pickle_file}")
            return pickle.load(f)

    print("Calculating TF-IDF encodings...")
    vectorizer = TfidfVectorizer(max_features=vector_length)
    tfidf_matrix = vectorizer.fit_transform(lemmatized_texts)
    tfidf_encoded_paragraphs = [np.array(tfidf_matrix[i].toarray()[0]) for i in range(len(lemmatized_texts))]

    if use_cache:
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        with open(pickle_file, 'wb') as f:
            pickle.dump(tfidf_encoded_paragraphs, f)
            print(f"Saved TF-IDF encoded articles to: {pickle_file}")

    return tfidf_encoded_paragraphs



if __name__ == "__main__":

    #load_articles_and_lemmas_filtered()

    #encode_TFIDF(vector_length=5000)
    pass