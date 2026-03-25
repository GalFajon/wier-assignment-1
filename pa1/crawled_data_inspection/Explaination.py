from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from keybert import KeyBERT
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path


def extract_cluster_keywords(
    lemmatized_texts,
    labels,
    raw_texts,
    final_count = 6,
    top_n=20,
    use_keybert=True,
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    ngram_range = (1, 3),
    min_df=2,
    max_df=0.85,
    diversity = 0.5,
    deduplicate = True,
    output_json_path=None
):
    #lemma filtering
    test_blacklist = {'prvi niz', 'drugi niz', 'tretji niz', 'as', 'blok'}
    lemmatized_texts = [" ".join([lemma for lemma in text.split() if lemma not in test_blacklist]) for text in lemmatized_texts]

    # Group texts by cluster
    cluster_texts = defaultdict(list)
    for lemma, label in zip(lemmatized_texts, labels):
        if label != -1:
            cluster_texts[label].append(lemma)

    # Build one string per cluster
    cluster_docs = {label: " ".join(docs) for label, docs in cluster_texts.items()}

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df, max_features=15000)
    X = vectorizer.fit_transform(cluster_docs.values())
    terms = np.array(vectorizer.get_feature_names_out())

    cluster_keywords = {}

    print("Extracting TF-IDF keywords...")
    for idx, label in tqdm(enumerate(cluster_docs.keys()), total=len(cluster_docs)):
        tfidf_scores = X[idx].toarray().flatten()
        top_idx = tfidf_scores.argsort()[::-1]
        keywords = terms[top_idx]

        # Optional deduplication
        if deduplicate:
            seen = set()
            filtered = []
            for kw in keywords:
                if not any(kw in s or s in kw for s in seen):
                    seen.add(kw)
                    filtered.append(kw)
                if len(filtered) >= top_n:
                    break
            keywords = filtered
        else:
            keywords = keywords[:top_n]

        cluster_keywords[label] = keywords[:top_n]

    # Optional: Semantic re-ranking with KeyBERT
    if use_keybert:
        print("Refining with KeyBERT...")
        kw_model = KeyBERT(model_name)
        for label in tqdm(cluster_keywords.keys(), total=len(cluster_keywords)):
            raw_cluster_text = " ".join(
                [text for text, lab in zip(raw_texts, labels) if lab == label]
            ) if raw_texts else cluster_docs[label]

            reranked = kw_model.extract_keywords(
                raw_cluster_text,
                keyphrase_ngram_range=ngram_range,
                stop_words=None,
                use_mmr=True,
                diversity=diversity,
                candidates=cluster_keywords[label],
                top_n=top_n
            )
            cluster_keywords[label] = [kw for kw, _ in reranked]

    kw_dict = {str(k): cluster_keywords[k][:final_count] for k in sorted(cluster_keywords)}

    if output_json_path:
        Path(os.path.dirname(output_json_path)).mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(kw_dict, f, indent=2, ensure_ascii=False)

    return kw_dict
