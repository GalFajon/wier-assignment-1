import json

# from dotenv import load_dotenv
# import dspy
import os 
import numpy as np
# from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
# from sqlalchemy import Engine, create_engine
# import tqdm
# from umap import UMAP

# from run_query import query_database, query_database2
# import stopwordsiso as stopwords
# from sklearn.metrics import ndcg_score

# from embedding import embed_text, load_embedding_model
# from main import SimpleQuery, SimpleRAG
# from reranking import load_reranking_model
# from rag_context import retrieve_context, retrieve_context_loaded_model

cmap = plt.get_cmap("tab10")


# model_names = {
#         1: "all-MiniLM-L6-v2",
#         8: "sentence-transformers/all-mpnet-base-v2",
#         9: "BAAI/bge-m3",
#         10: "EMBEDDIA/sloberta",
#         13: "cjvt/crosloengual-bert-si-nli"
#     }

# model_dims = {
#         1: 384,
#         8: 768,
#         9: 1024,
#         10: 768,
#         13: 768
#     }

# from ParserSettings import load_settings
# from db_api import get_random_page_segments, get_segments_by_model, get_segments_by_id, get_random_page_ids, get_page_segment_ids


# def visualize_precision_recall(models, reranking_models, model_to_queries, reranking_model_nicknames, model_to_relevant_seg_ids):
#     plt.figure()
#     return_n = 40
#     plt.title(f"NDCG @ k ({len(model_to_queries[models[0]])} queries, bm25)")
#     settings = load_settings()
#     # rerankers = [load_reranking_model2(settings, rr) for rr in reranking_models]
#     xs = np.arange(1, return_n)

#     patterns = ["--", ":", "-.", "-"]
#     # query the database
#     # model_to_chunk_dict = dict()
#     cumsums_per_model_per_metric = dict()
#     colors=["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]

#     model_to_query_to_chunks = dict()

#     print ("------- Gettings chunks from retrieval models ------------")
#     # get chunks for each model for each query
#     for j,m in enumerate(models):
#         print(f"Model: {model_names[m]}")
#         if m != 10:
#             model = load_embedding_model2(model_names[m], settings.model_run_device)
#         else:
#             model = load_embedding_model_hf2(settings, model_names[m])
#         for q,query in enumerate(model_to_queries[m]):
#             print(f"Query {q}: {query}")
#             chunks = query_database2(model, query, settings, model_dims[m], return_n, "cosine", model_names[m])
#             # print([(c[0][:100], c[1], c[2]) for c in chunks])
#             if m not in model_to_query_to_chunks:
#                 model_to_query_to_chunks[m] = dict()
#             if q not in model_to_query_to_chunks[m]:
#                 model_to_query_to_chunks[m][q] = chunks

#     print ("------- Reranking and scoring ------------")
#     # reranking and scoring
#     for j,m in enumerate(model_to_query_to_chunks.keys()):
#         print(f"Model: {model_names[m]}")

#         for i, rr in enumerate(reranking_models):
#             print(f"Reranking model: {rr}")
#             rr_nickname = "No rerank"
#             if rr != None:
#                 reranker = load_reranking_model2(settings, rr)
#                 rr_nickname = reranking_model_nicknames[i]

#             for query_keys in model_to_query_to_chunks[m].keys():
#                 # print(queries)
#                 correct_query_key = int(query_keys)
#                 query = model_to_queries[m][correct_query_key]
#                 # print(correct_query_key)
#                 print(f"Query {correct_query_key}: {model_to_queries[m][correct_query_key]}")
#                 # for item in model_to_query_to_chunks[m].items():
#                 #     print(item)
#                 chunks = model_to_query_to_chunks[m][correct_query_key]
#                 retrieved_seg_ids = [c[2] for c in chunks]
#                 not_reranked = [{"text": t, "id": cid, "cross_score": (len(chunks)-n)/(len(chunks))} for n,(e,t,_,cid) in enumerate(chunks)]
#                 reranked = rerank_candidates2(reranker, query, chunks, return_n)

#                 texts = [r['text'].lower() for r in reranked]
#                 not_reranked_texts = [r['text'].lower() for r in not_reranked]
#                 reranked_seg_ids_2 = [r['id'] for r in reranked]
#                 reranked_scores = [r['cross_score'] for r in reranked][1:]
#                 not_reranked_scores = [r['cross_score'] for r in not_reranked][1:]
#                 # print(not_reranked_scores)
#                 # print(reranked_scores)

#                 relevancy_bm25_scores = relevancy_score_bm25(not_reranked_texts, query)/20
#                 print(relevancy_bm25_scores)
#                 cutoff_score = np.sort(relevancy_bm25_scores.flatten())[-10]
#                 print(cutoff_score)
#                 relevancy_bm25 = (np.where(relevancy_bm25_scores >= 0.09, 1.0, 0.0) * relevancy_bm25_scores)[1:]
#                 print(relevancy_bm25)
#                 # print(relevant_seg_ids)
#                 #print(model_to_relevant_seg_ids[m])
#                 #print(f"Query seg id: {model_to_relevant_seg_ids[m][correct_query_key][0]}")
#                 #print(f"Same page seg ids: {model_to_relevant_seg_ids[m][correct_query_key]}")
#                 # print(chunks)
#                 #print(f"Retrieved seg ids: {retrieved_seg_ids}")
#                 #print(f"Reranked seg ids: {reranked_seg_ids_2}")

#                 relevancy_same_page = np.array([1.0 if c in model_to_relevant_seg_ids[m][correct_query_key] else 0.0 for c in reranked_seg_ids_2])
#                 # print(relevancy_same_page)
#                 relevancy_score_ideal = relevancy_bm25
#                 #print(relevancy_score_ideal)
#                 final_scores = []
#                 for top_k in range(1, return_n):
#                     score = ndcg_score(np.array([relevancy_score_ideal]), np.array([not_reranked_scores]), k=top_k)
#                     final_scores.append(score)
#                 final_scores = np.array(final_scores)
#                 print(final_scores)
#                 if m not in cumsums_per_model_per_metric:
#                     cumsums_per_model_per_metric[m] = dict()
#                 if rr_nickname not in cumsums_per_model_per_metric[m]:
#                     cumsums_per_model_per_metric[m][rr_nickname] = final_scores
#                 else:
#                     cumsums_per_model_per_metric[m][rr_nickname] += final_scores


#     for j,m in enumerate(cumsums_per_model_per_metric.keys()):
#         for i,rr in enumerate(cumsums_per_model_per_metric[m].keys()):
#             cumsum = cumsums_per_model_per_metric[m][rr] / len(model_to_queries[m])
#             if len(models) == 1 and len(reranking_models) > 1:
#                 label = f"{rr}"
#                 color = colors[1+i]
#             if len(models) > 1 and len(reranking_models) == 1:
#                 label = f"{model_names[m]}"
#                 color = colors[j]
#             plt.plot(xs, cumsum+0.005, patterns[i], color=color, label=label, alpha=0.7, zorder=50)
            

#     plt.xlabel("k")
#     plt.ylabel("NDCG")
#     plt.legend()
#    plt.show()
    # rerank results



# def eval_dataset(dataset, embedding_model, dimension, reranker_model, num_candidates=10, num_final=3):

#     similarities_no_context = []
#     similarities_context = []

#     rag_module = dspy.Predict(SimpleRAG)
#     query_module = dspy.Predict(SimpleQuery)

#     for i, row in enumerate(dataset):
#         print("----------------------------")
#         print(f"Question {i}: {row["question"]}")
#         context = retrieve_context_loaded_model(row["question"], embedding_model, dimension, reranker_model, num_candidates=num_candidates, num_final=num_final)
#         print("----RAG----")
#         print(f"Context: {context[:256]}")
#         response_rag = rag_module(context=context, question=row["question"])["answer"]
#         print(f"Response: {response_rag[:256]}")

#         response_rag_embedding = np.array([embed_text(embedding_model, response_rag)])
#         answer_embedding = np.array([embed_text(embedding_model, row["answer"])])

#         similarity_rag = cosine_similarity(response_rag_embedding, answer_embedding)
#         print("Response answer similarity (RAG): " + str(similarity_rag))

#         similarities_context.append(similarity_rag[0][0])

#         print("----No Contex----")
#         response_query = query_module(question=row["question"])["answer"]
#         response_query_embedding = np.array([embed_text(embedding_model, response_query)])
#         print(f"Response: {response_query[:256]}")
#         similarity_query = cosine_similarity(response_query_embedding, answer_embedding)
#         print("Response answer similarity (NO CONTEXT): " + str(similarity_query))
#         similarities_no_context.append(similarity_query[0][0])
        

#     plt.figure()
#     plt.bar(np.arange(len(similarities_no_context)) - 0.2, np.array(similarities_no_context), width=0.4, label="No context")
#     plt.bar(np.arange(len(similarities_context))+0.2, np.array(similarities_context), width=0.4, label="with context")
#     plt.legend()
#     plt.show()

def load_answer_quality_results(filename: str, folder_path="/LLM_evaluation/eval_baseline_model_selection/quality_evaluations"):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + f"{folder_path}/{filename}", encoding="UTF8") as f:
        jsonFile = json.load(f)
        return jsonFile
    return None

def plot_answers_quality_comparison(model_names: list[str]):

    # plot_models_semantic_scores(model_names, 10)
    # plot_zero_one_shot_comparison(model_names, 5)
    # plot_key_fact_coverage_score(withRagResults, withoutRagResults) # TODO: fix
    # plot_aggregate_results(model_names)
    # plot_dataset_comparison_aggregate_results()
    plot_precision_at_k()
    # plot_hit_at_k()
    

def plot_models_semantic_scores(model_names, n):
    
    plt.figure()
    plt.title("Overall baseline semantic scores of queries for each model")
    plt.xticks(np.arange(n))
    plt.grid()
    w = 0.6
    for i,m in enumerate(model_names):
        # withRagResults = load_answer_quality_results(f"ollama_chat-{m}_direct_unbiased_ANSWER_QUALITY.json")
        withoutRagResults = load_answer_quality_results(f"ollama_chat-{m}_direct_unbiased_ANSWER_QUALITY.json") # TODO: generate results for without rag
        #plt.bar(np.arange(n) - w/2 + 2*(i-0.5) / len(model_names) * w, [e["overall_semantic_score"] for e in withRagResults["examples"][:n]], width=w/len(model_names), label=f"{m} (no ctx)", color=cmap(i), alpha=0.5)
        plt.bar(np.arange(n) - w/2 + ((i-0.5) + 1)/ len(model_names) * w, [e["overall_semantic_score"] for e in withoutRagResults["examples"][:n]], width=w/len(model_names), label=f"{m}", color=cmap(i))
    plt.ylabel("Quality score")
    plt.xlabel("Query number")
    plt.legend().set_draggable(True)
    plt.show()

def plot_zero_one_shot_comparison(model_names, n):
    
    plt.figure()
    plt.title("Zero vs one shot comparison between models on queries")
    plt.xticks(np.arange(n))
    plt.grid()
    w = 0.3
    zeroShotResults = load_answer_quality_results(f"ollama_chat-qwen3-b14_rag_zero_shot_bge-m3_50_bge_7_unbiased.json", folder_path="/LLM_evaluation/eval_querying_styles/quality_evaluations") # TODO: generate zero and one shot results
    oneShotResults = load_answer_quality_results(f"ollama_chat-qwen3-b14_rag_one_shot_bge-m3_50_bge_7_unbiased.json", folder_path="/LLM_evaluation/eval_querying_styles/quality_evaluations") # TODO: generate zero and one shot results
    plt.bar(np.arange(n) - w/2, [e["overall_semantic_score"] for e in zeroShotResults["examples"][:n]], width=w/len(model_names), label=f"zero shot")
    plt.bar(np.arange(n) - w/2, [e["overall_semantic_score"] for e in oneShotResults["examples"][:n]], width=w/len(model_names), label=f"one shot")
    plt.ylabel("Quality score")
    plt.xlabel("Query number")
    plt.legend().set_draggable(True)
    plt.show()

def plot_precision_at_k():
    
    plt.figure()
    plt.title("Mean citation precision@k")
    plt.grid()
    w = 0.3
    # directResults = load_answer_quality_results(f"ollama_chat-qwen3-14b_direct_CITATION_QUALITY.json", folder_path="/LLM_evaluation/eval_querying_styles/citation_evaluations") # TODO: generate zero and one shot results
    zeroShotResults = load_answer_quality_results(f"ollama_chat-qwen3-14b_rag_zero_shot_bge-m3_50_bge_7_CITATION_QUALITY.json", folder_path="/LLM_evaluation/eval_querying_styles/citation_evaluations") # TODO: generate zero and one shot results
    oneShotResults = load_answer_quality_results(f"ollama_chat-qwen3-14b_rag_one_shot_bge-m3_50_bge_7_CITATION_QUALITY.json", folder_path="/LLM_evaluation/eval_querying_styles/citation_evaluations") # TODO: generate zero and one shot results
    plt.plot([7], zeroShotResults["mean_precision_at_k"], "o", label="zero shot")
    plt.plot([7], oneShotResults["mean_precision_at_k"], "o", label="one shot")
    plt.ylabel("Score")
    plt.xlabel("k")
    plt.legend().set_draggable(True)
    plt.show()

def plot_hit_at_k():
    
    plt.figure()
    plt.title("Mean citation hit@k")
    plt.grid()
    w = 0.3
    # directResults = load_answer_quality_results(f"ollama_chat-qwen3-14b_direct_CITATION_QUALITY.json", folder_path="/LLM_evaluation/eval_querying_styles/citation_evaluations") # TODO: generate zero and one shot results
    zeroShotResults = load_answer_quality_results(f"ollama_chat-qwen3-14b_rag_zero_shot_bge-m3_50_bge_7_CITATION_QUALITY.json", folder_path="/LLM_evaluation/eval_querying_styles/citation_evaluations") # TODO: generate zero and one shot results
    oneShotResults = load_answer_quality_results(f"ollama_chat-qwen3-14b_rag_one_shot_bge-m3_50_bge_7_CITATION_QUALITY.json", folder_path="/LLM_evaluation/eval_querying_styles/citation_evaluations") # TODO: generate zero and one shot results
    plt.plot([7], zeroShotResults["hit_at_k"], "o", label="zero shot")
    plt.plot([7], oneShotResults["hit_at_k"], "o", label="one shot")
    plt.ylabel("Score")
    plt.xlabel("k")
    plt.legend().set_draggable(True)
    plt.show()
    
def plot_key_fact_coverage_score(withRagResults, withoutRagResults):
    plt.figure()
    plt.title("Overall semantic scores for each query")
    plt.xticks(np.arange(withRagResults["aggregate"]["n"], step=2))
    plt.bar(np.arange(withRagResults["aggregate"]["n"])-0.2, [e["key_fact_coverage_score"] for e in withRagResults["examples"]], width=0.4, label="no context")
    plt.bar(np.arange(withoutRagResults["aggregate"]["n"])+0.2, [e["key_fact_coverage_score"] for e in withoutRagResults["examples"]], width=0.4, label="with context")
    plt.ylabel("Quality score")
    plt.xlabel("Query number")
    plt.legend().set_draggable(True)
    plt.show()

def plot_aggregate_results(model_names):
    plt.figure()
    plt.title("Overall scores")
    plt.grid()
    metric_labels = ["Mean overall\nsemantic score", "Mean key fact\n coverage score", "Strict key fact\ncoverage", "Mean answer\nquality score"]
    plt.xticks(ticks=np.arange(len(metric_labels)), labels=metric_labels)
    w = 0.4
    for i,m in enumerate(model_names):
        withoutRagResults = load_answer_quality_results(f"ollama_chat-{m}_direct_unbiased_ANSWER_QUALITY.json") # TODO: generate results for without rag
        plt.bar(np.arange(len(metric_labels)) - w/2 + ((i-0.5) + 1)/len(model_names) * w, [
            withoutRagResults["aggregate"]["mean_overall_semantic_score"],
            withoutRagResults["aggregate"]["mean_key_fact_coverage_score"],
            withoutRagResults["aggregate"]["strict_key_fact_coverage"],
            withoutRagResults["aggregate"]["mean_answer_quality_score"],
        ], width=w/len(model_names), label=f"{m}", color=cmap(i))
    plt.ylabel("Score")
    plt.legend().set_draggable(True)
    plt.show()

def plot_dataset_comparison_aggregate_results():
    plt.figure()
    plt.grid()
    plt.title("Overall baseline scores on biased and unbiased datasets")
    metric_labels = ["Mean overall\nsemantic score", "Mean key fact\n coverage score", "Strict key fact\ncoverage", "Mean answer\nquality score"]
    plt.xticks(ticks=np.arange(len(metric_labels)), labels=metric_labels)
    w = 0.4
    styles=["direct", "rag_zero_shot_bge-m3_50_bge_7", "rag_one_shot_bge-m3_50_bge_7"]
    style_labels=["direct", "zero-shot", "one-shot"]
    for i,s in enumerate(styles):
        unbiasedResults = load_answer_quality_results(f"ollama_chat-qwen3-14b_{s}_unbiased_ANSWER_QUALITY.json", folder_path="/LLM_evaluation/eval_querying_styles/quality_evaluations")
        biasedResults = load_answer_quality_results(f"ollama_chat-qwen3-14b_{s}_ANSWER_QUALITY.json", folder_path="/LLM_evaluation/eval_querying_styles/quality_evaluations")
        plt.bar(np.arange(len(metric_labels)) - w/2 + i * w / len(styles), [
            unbiasedResults["aggregate"]["mean_overall_semantic_score"],
            unbiasedResults["aggregate"]["mean_key_fact_coverage_score"],
            unbiasedResults["aggregate"]["strict_key_fact_coverage"],
            unbiasedResults["aggregate"]["mean_answer_quality_score"],
        ], width=w / len(styles), label=f"unbiased {style_labels[i]}", color=cmap(0), alpha=0.4 + i * 0.3)
        plt.bar(np.arange(len(metric_labels)) + w/2 + i * w / len(styles), [
            biasedResults["aggregate"]["mean_overall_semantic_score"],
            biasedResults["aggregate"]["mean_key_fact_coverage_score"],
            biasedResults["aggregate"]["strict_key_fact_coverage"],
            biasedResults["aggregate"]["mean_answer_quality_score"],
        ], width=w / len(styles), label=f"biased {style_labels[i]}", color=cmap(1), alpha=0.4 + i * 0.3)
    plt.ylabel("Score")
    plt.legend().set_draggable(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]).set_draggable(True)
    plt.show()

if __name__ == '__main__':
    
    
    plot_answers_quality_comparison(["qwen3-4b", "qwen3-8b", "qwen3-14b"])
    # load_dotenv()

    # lm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434', api_key='')

    # dspy.configure(lm=lm)

    # print("Main")
    # embedding_model_key = "bge-m3"
    # reranker_model_key = "mmarco"
    # embedding_model, dimension, model_id = load_embedding_model(embedding_model_key)
    # reranker = load_reranking_model(reranker_model_key)
    # dataset = None
    # with open("dataset.json", encoding="UTF8") as f:
    #     dataset = json.load(f)

    # dataset = dataset[:10]
    # eval_dataset(dataset, embedding_model, dimension, reranker)

