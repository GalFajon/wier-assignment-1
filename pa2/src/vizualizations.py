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
from embedding import embed_string2, load_embedding_model2, load_reranking_model, rerank_candidates, rerank_candidates2, load_embedding_model_hf2
from run_query import query_database, query_database2

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
from db_api import get_segments_by_model

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

def visualize_precision_recall(models, query):
    plt.figure()
    plt.title(f"Precision & recall, Query={query}")
    return_n = 40
    settings = load_settings()
    reranker = load_reranking_model(settings)
    xs = np.arange(1, return_n+1)
    # query the database
    # model_to_chunk_dict = dict()
    colors=["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
    for j,m in enumerate(models):
        if m != 10:
            model = load_embedding_model2(model_names[m], settings.model_run_device)
        else:
            model = load_embedding_model_hf2(settings, model_names[m])
        chunks = query_database2(model, query, settings, model_dims[m], return_n, settings.distance_metric, model_names[m])
        print(len(chunks))
        texts = [c[0].lower() for c in chunks]
        relevancy = np.where(relevancy_score_keywords(np.array(texts), ["putin", "zelenski", "raket", "ukrajin"]) >= 3, 1, 0)
        # print(texts)
        #for t,r in zip(texts, relevancy):
        #    print(t, r)
        print(relevancy)
        
        cumsum = np.cumsum(relevancy) / xs
        plt.plot(xs, cumsum, color=colors[j], label=f"{model_names[m][:16]}", alpha=0.3)

        reranked = rerank_candidates2(reranker, query, chunks, return_n)
        texts = [r['text'].lower() for r in reranked]
        relevancy = np.where(relevancy_score_keywords(np.array(texts), ["putin", "zelenski", "raket", "ukrajin"]) >= 3, 1, 0)
        cumsum = np.cumsum(relevancy) / xs
        plt.plot(xs, cumsum, color=colors[j], label=f"{model_names[m][:16]} + rerank", alpha=1, zorder=50)
        # model_to_chunk_dict[m] = chunks

    plt.xlabel("# Retrieved results")
    plt.ylabel("% Relevant results")
    plt.legend()
    plt.show()
    # rerank results
    result_embeddings = np.array([np.fromstring(c[0] [c[0].index("["):c[0].index("]")+1] [1:-1], dtype=float, sep=",") for c in chunks])

if __name__ == '__main__':
    settings = load_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    

    # visualize_embeddings(engine, 1, reduction_fun=(lambda x: get_embedding_umap(x, dim=2)), vector_size=384, queries=["Vladimir Putin noče prenehati z vojno kljub pozivom"])
    # visualize_embeddings(engine, 8, reduction_fun=(lambda x: get_embedding_umap(x, dim=2)), vector_size=768, queries=["Vladimir Putin noče prenehati z vojno kljub pozivom"])
    # visualize_embeddings(engine, 10, reduction_fun=(lambda x: get_embedding_umap(x, dim=2)), vector_size=768)
    # visualize_embeddings(engine, 13, reduction_fun=(lambda x: get_embedding_umap(x, dim=2)), vector_size=768)
    # visualize_embeddings(engine, 9, reduction_fun=(lambda x: get_embedding_umap(x, dim=2)), vector_size=1024)

    visualize_precision_recall([1, 8, 9, 13], "Vladimir Putin, rakete nad ukrajino, zelenski odziv")

