from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import time
from sentence_transformers import SentenceTransformer

DEVICE = "cpu"
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device=DEVICE
)

def embed_BERT(text: str):
    return model.encode(
        text,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

def torch_cosine_distance(a, b):
    return F.cosine_similarity(a, b).item()

def priority_score_BOW(logger, website_html, link, metadata_list, query):
    
    final_score = 0

    # query = classla_nlp(query)
    # print(query)

    for metadata in metadata_list:
        # print(metadata["source"])
        score = 0
        vectorizer = CountVectorizer()
        textst = [query.lower()]

        # if the link is on a relevant page, increase its score
        if metadata["source_title"] != None:
            textst.append(metadata["source_title"])
        else:
            textst.append("")
        
        # add all keywords to one string
        keyword_string = " ".join(metadata["article_keywords"])
        textst.append(keyword_string)

        if metadata["link_title"] != None:
            textst.append(metadata["link_title"])
        else:
            textst.append("")

        if metadata["summary"] != None:
            textst.append(metadata["summary"])
        else:
            textst.append("")


        # extract text from the url
        link_text = ""
        if metadata["section"] != None:
            link_text += metadata["section"] + " "
        if metadata["topic"] != None:
            link_text += metadata["topic"] + " "
        textst.append(link_text)

        if metadata["link_keywords"] != None:
            textst.append(metadata["link_keywords"])
        else:
            textst.append("")


        vectors = vectorizer.fit_transform(textst)
        query_vector = vectors[0]
        source_title_vector = vectors[1]
        keyword_vector = vectors[2]
        link_title_vector = vectors[3]
        summary_vector = vectors[4]
        link_text_vector = vectors[5]
        link_keywords_vector = vectors[6]

        source_title_score = cosine_similarity(query_vector, source_title_vector)
        keywords_score = cosine_similarity(query_vector, keyword_vector)
        link_title_score = cosine_similarity(query_vector, link_title_vector)
        summary_score = cosine_similarity(query_vector, summary_vector)
        link_text_score = cosine_similarity(query_vector, link_text_vector)
        link_keywords_score = cosine_similarity(query_vector, link_keywords_vector)
        # if link_keywords_score[0][0] > 0:
        #     print(link)
        #     print(metadata)
        #     print(f"Source: {source_title_score}, Keywords: {keywords_score}, Link title: {link_title_score} Summary: {summary_score}, Link text: {link_text_score}, Link keywords: {link_keywords_score}")
        score = source_title_score + keywords_score + link_title_score + summary_score + link_text_score + link_keywords_score

        final_score += score

    final_score /= len(metadata_list)
        #if metadata[""]

    return final_score


def priority_score_BERT(logger, website_html, link, metadata_list, query_emb):
    #logger.debug(f"Scoring {link}")

    titles = set()
    summaries = set()
    keywords = set()

    for m in metadata_list:
        title = m.get("link_title", "").strip()
        if title:
            titles.add(title.replace("_", " "))

        summary = m.get("summary")
        if summary:
            summaries.add(summary.replace("...", "").strip())

        topic = m.get("topic", "")
        if topic:
            keywords.add(topic.replace("-", " ").strip())

        section = m.get("section", "")
        if section:
            keywords.add(section.replace("-", " ").strip())

        article_keywords = m.get("article_keywords", [])
        if article_keywords:
            for kw in article_keywords:
                if kw:
                    keywords.add(kw.replace("-", " ").strip())

        link_keywords = m.get("link_keywords", "")
        if link_keywords:
            keywords.add(link_keywords.strip())

        #logger.debug(f"Metadata: {m}")

    if not titles and not summaries and not keywords:
        return 0.0
    
    metadata_parts = []
    if titles:
        metadata_parts.append("Naslovi: " + ". ".join(titles) + ".")

    if summaries:
        metadata_parts.append("Povzetki: " + ". ".join(summaries) + ".")

    if keywords:
        metadata_parts.append("Ključne besede: " + ", ".join(keywords) + ".")

    metadata_str = " ".join(metadata_parts)
    metadata_emb = embed_BERT(metadata_str)
    title_score = float(torch_cosine_distance(query_emb, metadata_emb))

    #logger.info(f'SCORED URL: Score={title_score},     len(m_str)={len(metadata_str)}')
    return title_score
 

def embed_batch(texts):
    return model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True  # important for cosine similarity
    )

def BERT_score_batch(logger, candidates, query_emb):
    if not candidates:
        return []

    t_start = time.perf_counter()

    metadata_strings = []
    for c in candidates:
        metadata_list = c["metadata"]

        titles = set()
        summaries = set()
        keywords = set()

        for m in metadata_list:
            title = m.get("link_title", "").strip()
            if title:
                titles.add(title.replace("_", " "))

            summary = m.get("summary")
            if summary:
                summaries.add(summary.replace("...", "").strip())

            topic = m.get("topic", "")
            if topic:
                keywords.add(topic.replace("-", " ").strip())

            section = m.get("section", "")
            if section:
                keywords.add(section.replace("-", " ").strip())

            article_keywords = m.get("article_keywords", [])
            if article_keywords:
                for kw in article_keywords:
                    if kw:
                        keywords.add(kw.replace("-", " ").strip())

            link_keywords = m.get("link_keywords", "")
            if link_keywords:
                keywords.add(link_keywords.strip())

        if not titles and not summaries and not keywords:
            metadata_strings.append("")
            continue

        parts = []
        if titles:
            parts.append("Naslovi: " + ". ".join(titles) + ".")
        if summaries:
            parts.append("Povzetki: " + ". ".join(summaries) + ".")
        if keywords:
            parts.append("Ključne besede: " + ", ".join(keywords) + ".")

        metadata_strings.append(" ".join(parts))

    t_after_build = time.perf_counter()

    embeddings = embed_batch(metadata_strings)

    t_after_embed = time.perf_counter()

    if query_emb.dim() == 2:
        query_emb = query_emb.squeeze(0)

    scores = F.cosine_similarity(
        embeddings,
        query_emb.unsqueeze(0),
        dim=1
    )

    t_after_score = time.perf_counter()

    if logger:
        logger.info(
            f"BERT_score_batch - BERT batch timing | total={t_after_score - t_start:.4f}s | "
            f"build={t_after_build - t_start:.4f}s | "
            f"embed={t_after_embed - t_after_build:.4f}s | "
            f"cosine={t_after_score - t_after_embed:.4f}s | "
            f"n={len(candidates)}"
        )

        # for i, s in enumerate(scores):
        #     logger.info(
        #         f"SCORED URL: Score={float(s)}, len(m_str)={len(metadata_strings[i])}"
        #     )

    return scores