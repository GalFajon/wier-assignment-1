import pickle
import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples

def mark_outliers_by_distance(
    embeddings,
    labels,
    threshold_std=2.0,
    use_cache=True,
    verbose=True,
    pickle_file="filtered_labels.pkl"
):
    if use_cache and os.path.exists(pickle_file):
        if verbose:
            print(f"Loading filtered labels from cache: {pickle_file}")
        with open(pickle_file, "rb") as f:
            new_labels = pickle.load(f)
        return new_labels

    if verbose:
        print("Marking outliers by distance...")

    embeddings = normalize(embeddings, norm='l2')
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    new_labels = labels.copy()

    unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]

    for lbl in unique_labels:
        cluster_mask = labels == lbl
        cluster_points = embeddings[cluster_mask]

        if len(cluster_points) < 2:
            continue  # skip tiny clusters

        centroid = cluster_points.mean(axis=0, keepdims=True)
        dists = cosine_distances(cluster_points, centroid).flatten()

        dist_mean = np.mean(dists)
        dist_std = np.std(dists)

        # Find outliers
        outliers = dists > (dist_mean + threshold_std * dist_std)
        cluster_indices = np.where(cluster_mask)[0]
        new_labels[cluster_indices[outliers]] = -1

    if use_cache:
        if verbose:
            print(f"Saving filtered labels to cache: {pickle_file}")
        with open(pickle_file, "wb") as f:
            pickle.dump(new_labels, f)

    return new_labels


def cluster_KMeans(embeddings, k=10, use_cache=True, pickle_file="clustering_cache/kmeans_labels.pkl", verbose=True):
    if use_cache and os.path.exists(pickle_file):
        if verbose:
            print(f"Loading KMeans labels from cache: {pickle_file}")
        with open(pickle_file, "rb") as f:
            labels = pickle.load(f)
    else:
        if verbose:
            print("Running KMeans clustering")

        normalized_embeds = normalize(embeddings, norm='l2') #force cosine dist with normalization
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(normalized_embeds)
        
        if use_cache:
            if verbose:
                print(f"Saving KMeans labels to cache: {pickle_file}")
            with open(pickle_file, "wb") as f:
                pickle.dump(labels, f)

    return labels


def KMeans_get_best_k_silhouette(embeddings, k_range = range(15, 100, 1)):
    print("Finding best Kmeans k param with silhouette")
    normalized_embeds = normalize(embeddings, norm='l2')

    best_score = -1
    best_labels = None
    best_k = None

    for k in k_range:
        
        labels = cluster_KMeans(embeddings, k=k, use_cache = False, verbose = False)
        if len(set(labels)) > 1:
            sil_score = silhouette_score(normalized_embeds, labels)
            if sil_score > best_score:
                best_score = sil_score
                best_labels = labels
                best_k = k
            print(f"tried k={k} with sil={sil_score}")
        
    return best_labels, best_k, best_score

def cluster_silhouette_scores(embeddings, labels):
    normalized_embeds = normalize(embeddings, norm='l2')
    sil = silhouette_samples(normalized_embeds, labels, metric='cosine')
    cluster_scores = {}
    for label in set(labels):
        if label == -1:
            continue
        cluster_scores[label] = np.mean(sil[np.array(labels) == label])

    return sorted(cluster_scores.items(), key=lambda x: x[1])

def relabel_clusters_by_silhouette(silhouette_map, cluster_labels, threshold = 0.1):
    bad_clusters = {label for label, score in silhouette_map if score <= threshold}

    new_labels = []
    label_map = {}
    next_label = 0

    for old_label in cluster_labels:
        if old_label in bad_clusters or old_label == -1:
            new_labels.append(-1)  # mark as noise
        else:
            if old_label not in label_map:
                label_map[old_label] = next_label
                next_label += 1
            new_labels.append(label_map[old_label])

    return np.array(new_labels)
