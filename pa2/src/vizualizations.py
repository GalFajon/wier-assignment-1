import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from sqlalchemy import Engine, create_engine
import mplcursors
import string
from umap import UMAP
from embedding import embed_string2, load_embedding_model2, load_reranking_model, load_reranking_model2, rerank_candidates, rerank_candidates2, load_embedding_model_hf2
from run_query import query_database, query_database2
import Stemmer
import stopwordsiso as stopwords
import bm25s
from sklearn.metrics import ndcg_score

cmap = plt.get_cmap("tab20")


model_names = {
        1: "all-MiniLM-L6-v2",
        8: "sentence-transformers/all-mpnet-base-v2",
        9: "BAAI/bge-m3",
        10: "EMBEDDIA/sloberta",
        13: "cjvt/crosloengual-bert-si-nli"
    }

model_dims = {
        1: 384,
        8: 768,
        9: 1024,
        10: 768,
        13: 768
    }

from ParserSettings import load_settings
from db_api import get_random_page_segments, get_segments_by_model, get_segments_by_id, get_random_page_ids, get_page_segment_ids

def get_embedding_pca(embeddings, dim=2):
    pca = PCA(n_components=dim)
    return pca.fit_transform(embeddings), pca

def get_embedding_tsne(embeddings, dim=2):
    tsne = TSNE(n_components=dim, perplexity=30, max_iter=1000)
    return tsne.fit_transform(embeddings), tsne

def get_embedding_umap(embeddings, dim=2):
    u = UMAP(n_components=dim, n_neighbors=60, metric="cosine")
    return u.fit_transform(embeddings), u


def plot_embeddings_2d(embeddings_2d, segments, clusters, colors, query_embedding_2d, result_embeddings_2d):
    fig = plt.figure(figsize=(9, 9))
    plt.title("Document positions in projected embedding space")

    if len(colors) == 0:
        mapped_colors = cmap(clusters)
    else:
        mapped_colors = np.array(colors).take(clusters)

    scat = plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], color=mapped_colors, alpha=0.2)

    # query embedding
    # if query_embedding_2d != None:
    plt.scatter(result_embeddings_2d[:,0], result_embeddings_2d[:,1], color="black", alpha=1, marker="X", s=10)
    plt.scatter(query_embedding_2d[0], query_embedding_2d[1], color="red", alpha=1, marker="X")
    
    cursor = mplcursors.cursor(scat, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        sel.annotation.set_text(f"{segments[i][:200]} \nx={embeddings_2d[i,0]:.2f}\ny={embeddings_2d[i,1]:.2f}")
    
    plt.show()

def plot_embeddings_3d(embeddings_3d, segments, cluster_indices, colors):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(projection='3d')
    # colors_putin = ["red" if "Putin" in seg or "putin" in seg else "tab:blue" for seg in segments]
    colors = cmap(cluster_indices)

    scat = ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], color=colors, alpha=0.2)
    cursor = mplcursors.cursor(scat, hover=mplcursors.HoverMode.Transient)
    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        sel.annotation.set_text(f"{segments[i][:50]} \nx={embeddings_3d[i,0]:.2f}\ny={embeddings_3d[i,1]:.2f}\ny={embeddings_3d[i,2]:.2f}")

    plt.show()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (cosine_similarity(array, np.array([value]))).argmax()
    return array[idx], idx

def visualize_embeddings(engine: Engine, model_id: int, reduction_fun=get_embedding_pca, vector_size=384, queries=[]):

    print("Fetching data")
    segments = get_segments_by_model(engine, model_id=model_id, max_segments=100000, vector_size=vector_size)
    segment_texts, embs = zip(*segments)
    segment_texts = np.array(segment_texts)
    embeddings = np.array([np.fromstring(e[1:-1], dtype=float, sep=",") for e in embs])

    # project to lower dimension
    embeddings_xd, reduction_model = reduction_fun(embeddings)
    print(embeddings.shape)

    # get query embedding
    settings = load_settings()
    model = load_embedding_model2(model_names[model_id], settings.model_run_device)
    embedding_array = embed_string2(model, queries[0], vector_size)
    query_embedding = embedding_array
    query_embedding_xd = reduction_model.transform(np.array([query_embedding]))[0]
    print(query_embedding[:10])


    # get query results
    reranker = load_reranking_model(settings)
    chunks = query_database(model, queries[0], settings)
    
    # reranked = rerank_candidates(reranker, queries[0], chunks, settings)
    result_embeddings = np.array([np.fromstring(c[0] [c[0].index("["):c[0].index("]")+1] [1:-1], dtype=float, sep=",") for c in chunks])
    result_embeddings_xd = np.array(reduction_model.transform(result_embeddings))
    print(result_embeddings_xd)
    # for i, final_chunk_data in enumerate(reranked):
    #     text = final_chunk_data['text']
    #     dist = final_chunk_data['dist']
    #     cross_score = final_chunk_data['cross_score']
    #     print(f'{i+1}: Text={text[:300]}... Cross_score={cross_score}, Distance={dist}')

    # remove outliers with kmeans

    # kmeans = KMeans(n_clusters=2)
    # cluster_labels = kmeans.fit_predict(embeddings_xd)
    # print(f"Data shape: {cluster_labels.shape}")
    # total_sum = np.sum(cluster_labels)
    # if total_sum < len(embeddings):
    #     outlier_mask = cluster_labels==0
    # else:
    #     outlier_mask = cluster_labels==1
    # filtered_embeddings = embeddings[outlier_mask]
    # filtered_embeddings_xd = embeddings_xd[outlier_mask]
    # filtered_segment_texts = segment_texts[outlier_mask]
    # print(f"Removed segments: ${(segment_texts[outlier_mask==False])[:20]}")

    # remove outliers if they are too far from the average position
    avg_embedding_xd = np.average(embeddings_xd, axis=0)
    std_embedding_xd = np.std(embeddings_xd, axis=0)
    print(f"Average: {avg_embedding_xd}")
    print(f"Standard deviation: {std_embedding_xd}")
    outlier_mask = np.all((np.abs(embeddings_xd - avg_embedding_xd) < std_embedding_xd*2.2), axis=1)
    filtered_embeddings = embeddings[outlier_mask]
    filtered_embeddings_xd = embeddings_xd[outlier_mask]
    filtered_segment_texts = segment_texts[outlier_mask]
    print(f"Removed segments: ${(segment_texts[outlier_mask==False])[:20]}")

    for i in range(len(filtered_segment_texts)):
        filtered_segment_texts[i] = filtered_segment_texts[i].translate(str.maketrans('', '', string.punctuation)).lower()

    # split_filtered_segments = [seg.split(" ") for seg in filtered_segment_texts]
    # find clusters with kmeans
    # svd_clusters = TruncatedSVD(n_components=200)
    # kmeans_clusters = KMeans(n_clusters=20)
    # cluster_indices = kmeans_clusters.fit_predict(filtered_embeddings)

    # # find closest segments to each cluster
    # for i, center in enumerate(kmeans_clusters.cluster_centers_):
    #     print(center[:3])
    #     nearest_emb, nearest_idx = find_nearest(filtered_embeddings, center)
    #     print(nearest_idx)
    #     print(f"Cluster {i}: {filtered_segment_texts[nearest_idx][:400]}")

    colors=["lightgrey", "tab:red", "tab:green", "tab:orange", "tab:blue", "tab:purple"]

    # find cluster by most common word from chosen words
    trump_counts = np.array([seg.count("trump") for seg in filtered_segment_texts])
    ukrajin_counts = np.array([seg.count("ukrajin") for seg in filtered_segment_texts])
    nogomet_counts = np.array([seg.count("nogomet") for seg in filtered_segment_texts])
    rusija_counts = np.array([seg.count("rusija") + seg.count("rusk") for seg in filtered_segment_texts])
    iran_counts = np.array([seg.count("iran") for seg in filtered_segment_texts])

    cluster_indices = np.argmax([trump_counts, ukrajin_counts, nogomet_counts, rusija_counts, iran_counts], axis=0) + 1
    sums = trump_counts + ukrajin_counts + nogomet_counts + rusija_counts + iran_counts
    print(sums)
    cluster_indices[sums == 0] = 0

    if embeddings_xd.shape[1]==2:
        plot_embeddings_2d(filtered_embeddings_xd, filtered_segment_texts, cluster_indices, colors, query_embedding_xd, result_embeddings_xd)
    else:
        plot_embeddings_3d(filtered_embeddings_xd, filtered_segment_texts, cluster_indices, colors=colors)

def relevancy_score_keywords(texts: np.ndarray, keywords):
    scores = np.zeros_like(texts, dtype=int)
    for i,t in enumerate(texts):
        for k in keywords:
            if k in t:
                scores[i] += 1
    return scores

def relevancy_score_bm25(texts: list[str], query: str):

    # Tokenize the corpus and only keep the ids (faster and saves memory)
    corpus_tokens = bm25s.tokenize(texts, stopwords=stopwords.stopwords("sl"))

    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25(method="bm25+")
    retriever.index(corpus_tokens)

    # Query the corpus
    query_tokens = bm25s.tokenize(query)

    # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
    # To return docs instead of IDs, set the `corpus=corpus` parameter.
    results, scores = retriever.retrieve(query_tokens, k=len(texts), sorted=False)

    # for i in range(results.shape[1]):
    #     doc, score = results[0, i], scores[0, i]
    #     print(f"Rank {i+1} (score: {score:.2f})")
    return scores[0, :]


def visualize_precision_recall(models, reranking_models, model_to_queries, reranking_model_nicknames, model_to_relevant_seg_ids):
    plt.figure()
    return_n = 40
    plt.title(f"NDCG @ k ({len(model_to_queries[models[0]])} queries, same page relevancy)")
    settings = load_settings()
    # rerankers = [load_reranking_model2(settings, rr) for rr in reranking_models]
    xs = np.arange(1, return_n+1)

    patterns = ["--", ":", "-.", "-"]
    # query the database
    # model_to_chunk_dict = dict()
    cumsums_per_model_per_metric = dict()
    colors=["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]

    model_to_query_to_chunks = dict()

    print ("------- Gettings chunks from retrieval models ------------")
    # get chunks for each model for each query
    for j,m in enumerate(models):
        print(f"Model: {model_names[m]}")
        if m != 10:
            model = load_embedding_model2(model_names[m], settings.model_run_device)
        else:
            model = load_embedding_model_hf2(settings, model_names[m])
        for q,query in enumerate(model_to_queries[m]):
            print(f"Query {q}: {query}")
            chunks = query_database2(model, query, settings, model_dims[m], return_n, settings.distance_metric, model_names[m])
            # print([(c[0][:100], c[1], c[2]) for c in chunks])
            if m not in model_to_query_to_chunks:
                model_to_query_to_chunks[m] = dict()
            if q not in model_to_query_to_chunks[m]:
                model_to_query_to_chunks[m][q] = chunks

    print ("------- Reranking and scoring ------------")
    # reranking and scoring
    for j,m in enumerate(model_to_query_to_chunks.keys()):
        print(f"Model: {model_names[m]}")

        for i, rr in enumerate(reranking_models):
            print(f"Reranking model: {rr}")
            rr_nickname = "No rerank"
            if rr != None:
                reranker = load_reranking_model2(settings, rr)
                rr_nickname = reranking_model_nicknames[i]

            for query_keys in model_to_query_to_chunks[m].keys():
                # print(queries)
                correct_query_key = int(query_keys)
                query = model_to_queries[m][correct_query_key]
                # print(correct_query_key)
                print(f"Query {correct_query_key}: {model_to_queries[m][correct_query_key]}")
                # for item in model_to_query_to_chunks[m].items():
                #     print(item)
                chunks = model_to_query_to_chunks[m][correct_query_key]
                retrieved_seg_ids = [c[2] for c in chunks]
                not_reranked = [{"text": t, "id": cid, "cross_score": (len(chunks)-n)/(len(chunks))} for n,(t,_,cid) in enumerate(chunks)]
                reranked = rerank_candidates2(reranker, query, chunks, return_n)

                texts = [r['text'].lower() for r in reranked]
                reranked_seg_ids_2 = [r['id'] for r in reranked]
                reranked_scores = [r['cross_score'] for r in reranked][1:]
                not_reranked_scores = [r['cross_score'] for r in not_reranked]
                print(not_reranked_scores)
                print(reranked_scores)

                relevancy_bm25_scores = relevancy_score_bm25(texts, query)
                relevancy_bm25 = np.where(relevancy_bm25_scores >= np.average(relevancy_bm25_scores)*1.3, 1.0, 0.0)[1:]
                # print(relevant_seg_ids)
                #print(model_to_relevant_seg_ids[m])
                #print(f"Query seg id: {model_to_relevant_seg_ids[m][correct_query_key][0]}")
                #print(f"Same page seg ids: {model_to_relevant_seg_ids[m][correct_query_key]}")
                # print(chunks)
                #print(f"Retrieved seg ids: {retrieved_seg_ids}")
                #print(f"Reranked seg ids: {reranked_seg_ids_2}")

                relevancy_same_page = np.array([1.0 if c in model_to_relevant_seg_ids[m][correct_query_key] else 0.0 for c in reranked_seg_ids_2])
                print(relevancy_same_page)
                relevancy_score_ideal = relevancy_same_page
                #print(relevancy_score_ideal)
                final_scores = []
                for top_k in range(1, return_n+1):
                    score = ndcg_score(np.array([relevancy_score_ideal]), np.array([not_reranked_scores]), k=top_k)
                    final_scores.append(score)
                final_scores = np.array(final_scores)
                print(final_scores)
                if m not in cumsums_per_model_per_metric:
                    cumsums_per_model_per_metric[m] = dict()
                if rr_nickname not in cumsums_per_model_per_metric[m]:
                    cumsums_per_model_per_metric[m][rr_nickname] = final_scores
                else:
                    cumsums_per_model_per_metric[m][rr_nickname] += final_scores


    for j,m in enumerate(cumsums_per_model_per_metric.keys()):
        for i,rr in enumerate(cumsums_per_model_per_metric[m].keys()):
            cumsum = cumsums_per_model_per_metric[m][rr] / len(model_to_queries[m])
            if len(models) == 1 and len(reranking_models) > 1:
                label = f"{rr}"
                color = colors[1+i]
            if len(models) > 1 and len(reranking_models) == 1:
                label = f"{model_names[m]}"
                color = colors[j]
            plt.plot(xs, cumsum+0.005, patterns[i], color=color, label=label, alpha=0.7, zorder=50)
            

    plt.xlabel("k")
    plt.ylabel("NDCG")
    plt.legend()
    plt.show()
    # rerank results

if __name__ == '__main__':
    settings = load_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    n = 50
    model_ids = [1, 8, 10, 13, 9]
    model_to_page_segment_ids = dict()

    # pick a few random pages
    random_pages = [p[0] for p in get_random_page_ids(engine, n=n)]

    # for each model retrieve segments of those random pages
    for m in model_ids:
        model_to_page_segment_ids[m] = sorted(get_page_segment_ids(engine, random_pages, m, model_dims[m]))
        model_to_page_segment_ids[m] = [(page_id, sorted(seg_ids)) for (page_id, seg_ids) in model_to_page_segment_ids[m]]

    print()
    print(model_to_page_segment_ids)
    print()
    # take any model and get query texts from the first segment of each page
    model_to_query_ids: dict = {m: [seg_ids[0] for (page_id, seg_ids) in model_to_page_segment_ids[m]] for m in model_ids}
    print()
    print(model_to_query_ids)
    print()
    model_to_query_results = {m: get_segments_by_id(engine, model_to_query_ids[m], model_dims[m]) for m in model_ids}
    print()
    print(model_to_query_results)
    print()
    model_to_queries_with_ids = {m: [(q[0], q[1][:128]) for q in model_to_query_results[m]] for m in model_ids}
    print()
    print(model_to_queries_with_ids)
    print()
    model_to_queries = {m: [q[1] for q in model_to_queries_with_ids[m]] for m in model_ids}
    print()
    print(model_to_queries)
    print()
    print()
    for q in range(n):
        for m in model_ids:
            print(model_to_queries[m][q])
    print()
    model_to_relevant_seg_ids = {m: [m2[1] for m2 in model_to_page_segment_ids[m]] for m in model_ids}
    print()
    print(model_to_relevant_seg_ids)
    print()

    # visualize_embeddings(engine, 1, reduction_fun=(lambda x: get_embedding_umap(x, dim=2)), vector_size=384, queries=["Vladimir Putin noče prenehati z vojno kljub pozivom"])
    # visualize_embeddings(engine, 8, reduction_fun=(lambda x: get_embedding_umap(x, dim=2)), vector_size=768, queries=["Vladimir Putin noče prenehati z vojno kljub pozivom"])
    # visualize_embeddings(engine, 10, reduction_fun=(lambda x: get_embedding_umap(x, dim=2)), vector_size=768)
    # visualize_embeddings(engine, 13, reduction_fun=(lambda x: get_embedding_umap(x, dim=2)), vector_size=768)
    # visualize_embeddings(engine, 9, reduction_fun=(lambda x: get_embedding_umap(x, dim=2)), vector_size=1024)
    # reranking_models = [None, "cross-encoder/ms-marco-MiniLM-L6-v2", "BAAI/bge-reranker-v2-m3"]
    # reranking_models_nicknames = ["No rerank", "ms-marco-MiniLM-L6-v2", "bge-reranker-v2-m3"]
    reranking_models = ["cross-encoder/ms-marco-MiniLM-L6-v2"]
    reranking_models_nicknames = ["ms-marco-MiniLM-L6-v2"]
    # "mixedbread-ai/mxbai-rerank-large-v2" is a bit too large
    # visualize_precision_recall([1], ["cross-encoder/ms-marco-MiniLM-L6-v2", "BAAI/bge-reranker-v2-m3"], "ruski predsednik vladimir putin in njegov ukrajinski kolega volodimir zelenski sta sicer med prvim telefonskim pogovorom v začetku meseca govorila o možnosti zamenjave ujetnikov.", ["putin", "zelensk", "ukrajin", "raket"])
    # queries = [
    #     "Vladimir Putin, raketni napad na ukrajino, zelenski odziv",
    #     "ruski predsednik vladimir putin in njegov ukrajinski kolega volodimir zelenski sta sicer med prvim telefonskim pogovorom v začetku meseca govorila o možnosti zamenjave ujetnikov.",
    #     "Vladimir Putin noče prenehati z vojno kljub pozivom",
    #     "Madžari so upravičili vlogo favoritov in zasluženo prišli do zmage. Minimalni poraz malce celo laska naši izbrani vrsti, ki jo je nekajkrat rešil odlični Igor Vekić v vratih. Cesar slabši drugi polčas pripisuje",
    #     "To je bil eden največjih napadov podnevi",
    #     "Prav tako je ponovil, da Evropa potrebuje Ukrajino, saj da ima slednja najmočnejšo vojsko v Evropi, poroča britanski BBC.Trump Zelenskega pozval k večjim naporom za dogovor z Rusijo Rusija želi skleniti dogovor in Zelenski bo moral ukrepati, sicer bo zamudil veliko priložnost, je v petek povedal ameriški predsednik Donald Trump.Zelenski je namreč na dogodku v okviru Münchenske varnostne konference ponovno zavrnil ruske zahteve po prevzemu ukrajinskega Donbasa",
    #     "Deripaskin tiskovni predstavnik je za Bloomberg povedal, da Epsteina osebno ne pozna.Druga Rusinja v Epsteinovem krogu je bila Maša Drokova Bucher, 37-letna podjetnica, ki je bila leta 2017 Epsteinova publicistka in mu pomagala po obsodbi leta 2008 zaradi nagovarjanja k prostituciji. Bucherjeva je bila v Rusiji znana kot članica Naših, proputinovske mladinske skupine",
    #     "Takšnih je bilo več kot 4000 ljudi in njim se godi precej slabše, kot se je meni",
    #     "Putin je na začetku vojne računal na to, da bo Zelenski pobegnil iz države. Že ob prvih udarcih je ruska propaganda sporočala, da je Zelenski že na Poljskem. Mislim, da je bilo to za Putina veliko presenečenje. Nekdanji igralec, komedijant, se je izkazal za dovolj pogumnega, da postane vodja države, ki je v vojni. ",
    #     "Katere države so še posebej ranljive?V vseh državah Kremelj uporablja iste narative, to so problemi migracij, socialni problemi, LGBT in homofobija in tako naprej. Putin povsod deluje na isti princip.",
    #     "Putinov hvalospev vojaškemu izvozu, ki naj bi presegel 15 milijard. Po besedah ruskega predsednika Vladimirja Putina je bila Rusija v lanskem letu na področju vojaške industrije uspešna.",
    # ]
    visualize_precision_recall(model_ids, reranking_models, model_to_queries, reranking_models_nicknames, model_to_relevant_seg_ids)

    # visualize_precision_recall([1, 8, 10, 13, 9], "ruski predsednik vladimir putin in njegov ukrajinski kolega volodimir zelenski sta sicer med prvim telefonskim pogovorom v začetku meseca govorila o možnosti zamenjave ujetnikov.", ["putin", "zelensk", "ujetni"])
    # visualize_precision_recall([9], reranking_models, "Putin, Zelenski in Trump pogovor.", ["putin", "zelenski", "trump"])
    # visualize_precision_recall([1, 8, 10, 13, 9], "Ali bo iran ostal v nogometni ligi?", ["iran", "nogomet", "liga"])

