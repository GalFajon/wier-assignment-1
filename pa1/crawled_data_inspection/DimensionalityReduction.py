import pickle
import os
import numpy as np
import warnings

from sklearn.decomposition import TruncatedSVD
import umap.umap_ as umap


def truncated_SVD_reduction(embeddings, reduce_to_dim=100, use_cache=True, pickle_file='dim_reduction_cache/trunc_SVD_results.pkl'):
    if use_cache and os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            print(f"Loaded reduced embeddings from: {pickle_file}")
            return data['reduced_embeddings'], data['explained_variance']

    print("Performing Truncated SVD dimensionality reduction...")
    embeddings = np.array(embeddings)

    svd = TruncatedSVD(n_components=reduce_to_dim, random_state=42)
    reduced_embeddings = svd.fit_transform(embeddings)
    explained_variance = np.sum(svd.explained_variance_ratio_)

    if use_cache:
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'reduced_embeddings': reduced_embeddings,
                'explained_variance': explained_variance,
                'svd_model': svd
            }, f)
            print(f"Saved reduced embeddings to: {pickle_file}")

    return reduced_embeddings, explained_variance


def UMAP_reduction(embeded_vectors, reduce_to=2, min_dist = 0.1, n_neighbours = 15, spread = 1, use_cache=True, pickle_file='dim_reduction_cache/umap_results.pkl'):
    if use_cache and os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            print(f"Loading cached UMAP results from: {pickle_file}")
            return pickle.load(f)

    print(f"Running UMAP reduction to {reduce_to} dimensions")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        umap_model = umap.UMAP(
            n_components=reduce_to,
            n_neighbors=n_neighbours,
            min_dist=min_dist,
            metric='cosine',
            spread=spread,
            random_state=42
        )
        reduced_points = umap_model.fit_transform(embeded_vectors)

    if use_cache:
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        with open(pickle_file, 'wb') as f:
            pickle.dump(reduced_points, f)
            print(f"Saved UMAP reduced points to: {pickle_file}")

    return reduced_points