import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from sqlalchemy import Engine, create_engine
import mplcursors
from umap import UMAP

cmap = plt.get_cmap("tab20")

from ParserSettings import load_settings
from db_api import get_segments_by_model

def get_embedding_pca(embeddings, dim=2):
    pca = PCA(n_components=dim)
    return pca.fit_transform(embeddings)

def get_embedding_tsne(embeddings, dim=2):
    tsne = TSNE(n_components=dim, perplexity=30, max_iter=1000)
    return tsne.fit_transform(embeddings)

def get_embedding_umap(embeddings, dim=2):
    u = UMAP(n_components=dim, n_neighbors=60, metric="cosine")
    return u.fit_transform(embeddings)


def plot_embeddings_2d(embeddings_2d, segments, clusters):
    fig = plt.figure()
    plt.title("Document positions in projected embedding space")
    colors_putin = ["red" if "Putin" in seg or "putin" in seg else "tab:blue" for seg in segments]
    # colors_ukrajin = ["red" if "Ukrajin" in seg or "ukrajin" in seg else "tab:blue" for seg in segments]
    # colors_ameri = ["red" if "ameri" in seg or "Ameri" in seg or "ZDA" in seg else "tab:blue" for seg in segments]
    # colors_trump = ["red" if "Trump" in seg or "trump" in  seg else "tab:blue" for seg in segments]
    # colors_zelenski = ["red" if "zelensk" in seg or "Zelensk" in seg else "tab:blue" for seg in segments]

    
    colors = cmap(clusters)

    scat = plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], color=colors, alpha=0.2)
    
    cursor = mplcursors.cursor(scat, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        sel.annotation.set_text(f"{segments[i][:50]} \nx={embeddings_2d[i,0]:.2f}\ny={embeddings_2d[i,1]:.2f}")
    
    plt.show()

def plot_embeddings_3d(embeddings_3d, segments, cluster_indices):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(projection='3d')
    # colors_putin = ["red" if "Putin" in seg or "putin" in seg else "tab:blue" for seg in segments]
    colors = cmap(cluster_indices)

    scat = ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], color=colors, alpha=0.2)
    cursor = mplcursors.cursor(scat, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        sel.annotation.set_text(f"{segments[i][:50]} \nx={embeddings_3d[i,0]:.2f}\ny={embeddings_3d[i,1]:.2f}\ny={embeddings_3d[i,2]:.2f}")

    plt.show()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (cosine_similarity(array, np.array([value]))).argmax()
    return array[idx], idx

def visualize_embeddings(engine: Engine, model_id: int, reduction_fun=get_embedding_pca):
    segments = get_segments_by_model(engine, model_id=model_id, max_segments=100000)
    segment_texts, embs = zip(*segments)
    segment_texts = np.array(segment_texts)
    embeddings = np.array([np.fromstring(e[1:-1], dtype=float, sep=",") for e in embs])
    print(embeddings.shape)

    # remove outliers with kmeans
    embeddings_xd = reduction_fun(embeddings)
    kmeans = KMeans(n_clusters=2)
    cluster_labels = kmeans.fit_predict(embeddings_xd)
    print(cluster_labels.shape)
    total_sum = np.sum(cluster_labels)
    if total_sum < len(embeddings):
        outlier_mask = cluster_labels==0
    else:
        outlier_mask = cluster_labels==1
    filtered_embeddings = embeddings[outlier_mask]
    filtered_embeddings_xd = embeddings_xd[outlier_mask]
    filtered_segment_texts = segment_texts[outlier_mask]
    print(f"Removed segments: ${(segment_texts[outlier_mask==False])[:10]}")

    # find clusters with kmeans
    svd_clusters = TruncatedSVD(n_components=200)
    kmeans_clusters = KMeans(n_clusters=20)
    cluster_indices = kmeans_clusters.fit_predict(filtered_embeddings)

    # find closest segments to each cluster
    for i, center in enumerate(kmeans_clusters.cluster_centers_):
        print(center[:3])
        nearest_emb, nearest_idx = find_nearest(filtered_embeddings, center)
        print(nearest_idx)
        print(f"Cluster {i}: {filtered_segment_texts[nearest_idx][:400]}")

    if embeddings_xd.shape[1]==2:
        plot_embeddings_2d(filtered_embeddings_xd, filtered_segment_texts, cluster_indices)
    else:
        plot_embeddings_3d(filtered_embeddings_xd, filtered_segment_texts, cluster_indices)


if __name__ == '__main__':
    settings = load_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    visualize_embeddings(engine, 1, reduction_fun=(lambda x: get_embedding_umap(x, dim=3)))

