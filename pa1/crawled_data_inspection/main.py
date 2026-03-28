import pickle

from ArticleEncoding import *
from DimensionalityReduction import *
from ClusteringPoints import *
from Explaination import *
from Visualization import visualize_clusters_interactive

display_run = False
save_run_for_display = False

if __name__ == "__main__":

    if display_run:
        #runs only the lightweight cached files in directory final_display
        print("Opening cached visualization files")

        with open('final_display/umap.pkl', "rb") as f:
            umap_2d_points = pickle.load(f)

        with open('final_display/cluster_labels.pkl', "rb") as f:
            cluster_labels = pickle.load(f)

        with open('final_display/titles.pkl', "rb") as f:
            titles = pickle.load(f)

        with open('final_display/topics.pkl', "rb") as f:
            topics = pickle.load(f)

        with open('final_display/cluster_keywords.pkl', "rb") as f:
            cluster_keywords = pickle.load(f)

        visualize_clusters_interactive(umap_2d_points, cluster_labels, titles, topics, cluster_keywords)
    else:
        #recompute everything

        

        #embedding
        articles, lemmas = load_articles_and_lemmas_filtered()

        IDS_FILE = "hide_ids.txt"
        with open(IDS_FILE, "r", encoding="utf-8") as f:
            hide_ids = set(line.strip() for line in f if line.strip())

        filtered_articles = []
        filtered_lemmas = []
        for i, article in enumerate(articles):
            if article['id'] in hide_ids:
                continue

            filtered_articles.append(articles[i])
            filtered_lemmas.append(lemmas[i])
        articles = filtered_articles
        lemmas = filtered_lemmas

        tfidf_embedded_lemmas = embed_TFIDF(lemmas, vector_length = 15000, use_cache = False, pickle_file='tmp_caching/0_tfidf.pkl')
        reduced_embeddings, explained_variance = truncated_SVD_reduction(tfidf_embedded_lemmas, reduce_to_dim = 40, use_cache = False, pickle_file='tmp_caching/1_tsvd.pkl')
        print("Explained variance of SVD:", explained_variance)

        #cluster k selection
        cluster_labels, best_k, best_score = KMeans_get_best_k_silhouette(reduced_embeddings, k_range=range(35, 45, 1))
        print(best_k, best_score)

        #clustering
        cluster_labels = cluster_KMeans(reduced_embeddings, k=best_k, use_cache = False, verbose = True, pickle_file='tmp_caching/3_kmeans.pkl')

        #discarding bad clusters/labels
        # cluster_labels = mark_outliers_by_distance(reduced_embeddings, cluster_labels, threshold_std=3.0, use_cache = False, verbose = True, pickle_file="tmp_caching/4_reject_outliers.pkl")

        # remove_bottom_percent = 1
        # cluster_silhouettes = cluster_silhouette_scores(reduced_embeddings, cluster_labels)
        # scores = np.array([score for _, score in cluster_silhouettes])
        # threshold = np.percentile(scores, remove_bottom_percent)
        # cluster_labels = relabel_clusters_by_silhouette(cluster_silhouettes, cluster_labels, threshold = threshold)

        #visualization projection
        umap_2d_points = UMAP_reduction(reduced_embeddings, reduce_to = 2, min_dist = 0.8, n_neighbours = 15, spread = 0.99, use_cache = False, pickle_file='tmp_caching/5_umap.pkl')

        #keyword extraction
        raw_articles = [art['raw_text'] for art in articles]
        cluster_keywords = extract_cluster_keywords(
            lemmatized_texts=lemmas,
            labels=cluster_labels,
            raw_texts=raw_articles,
            final_count = 6,
            top_n=30,
            min_df=2,
            max_df=0.85,
            diversity=0.4,
            output_json_path = None #'tmp_caching/6_cluster_keywords.json'
        )

        

        #display
        titles = [art['title'] for art in articles]
        urls = [art['url'] for art in articles]
        ids = [art['id'] for art in articles]
        visualize_clusters_interactive(umap_2d_points, cluster_labels, titles, urls, ids, cluster_keywords = cluster_keywords)


        # output_file = "hide_clusters.txt"
        # with open(output_file, "w", encoding="utf-8") as f:
        #     for i, label in enumerate(cluster_labels):
        #         if label != 10:
        #             continue
        #         f.write(str(articles[i]['id']) + "\n")
        # print(f"Saved to {output_file}")

        if save_run_for_display:
            with open('final_display/umap.pkl', "wb") as f:
                pickle.dump(umap_2d_points, f)
            
            with open('final_display/cluster_labels.pkl', "wb") as f:
                pickle.dump(cluster_labels, f)

            with open('final_display/titles.pkl', "wb") as f:
                pickle.dump(titles, f)

            with open('final_display/urls.pkl', "wb") as f:
                pickle.dump(urls, f)

            with open('final_display/cluster_keywords.pkl', "wb") as f:
                pickle.dump(cluster_keywords, f)