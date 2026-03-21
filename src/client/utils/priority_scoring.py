from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import classla
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

DEVICE = "cpu"
MODEL_CACHE_DIR = "./model_cache/sloberta"
EMBEDDING_TOKEN_MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(
    "EMBEDDIA/sloberta",
    cache_dir=MODEL_CACHE_DIR
)
model = AutoModel.from_pretrained(
    "EMBEDDIA/sloberta",
    cache_dir=MODEL_CACHE_DIR
)

def embed_BERT(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=EMBEDDING_TOKEN_MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1)

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

        vectors = vectorizer.fit_transform(textst)
        query_vector = vectors[0]
        source_title_vector = vectors[1]
        keyword_vector = vectors[2]
        link_title_vector = vectors[3]
        summary_vector = vectors[4]

        source_title_score = cosine_similarity(query_vector, source_title_vector)
        keywords_score = cosine_similarity(query_vector, keyword_vector)
        link_title_score = cosine_similarity(query_vector, link_title_vector)
        summary_score = cosine_similarity(query_vector, summary_vector)
        if summary_score[0][0] > 0:
            print(link)
            print(metadata)
            print(f"Source: {source_title_score}, Keywords: {keywords_score}, Link title: {link_title_score} Summary: {summary_score}")
        score = source_title_score + keywords_score + link_title_score + summary_score

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

    #logger.debug(f'     Metadata_str="{metadata_str}", Score={title_score},     len(m_str)={len(metadata_str)}')
    return title_score
 

    

    