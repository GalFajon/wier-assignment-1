from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import classla

def priority_score(website_html, link, metadata_list, query):
    
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
