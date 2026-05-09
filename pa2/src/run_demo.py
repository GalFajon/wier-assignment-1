from sqlalchemy import create_engine
from ParserSettings import ParserSettings, load_settings
from embedding import load_embedding_model, load_reranking_model, embed_string, rerank_candidates
from db_api import get_source_table, get_model_id, query_page_segments

import numpy as np


def query_database(model, query_string, settings: ParserSettings):
    query_vector = embed_string(model, query_string, settings)

    vector_length = np.linalg.norm(query_vector)
    print(f"Query vector length: {vector_length:.4f}")

    engine = create_engine(settings.database_url, pool_pre_ping=True)

    source_table = get_source_table(
        engine=engine,
        schema_name=settings.table_schema,
        table_name=settings.table_name
    )

    source_schema = source_table.schema or settings.table_schema
    model_table = get_source_table(engine, source_schema, "model")

    model_id = get_model_id(
        engine,
        model_table,
        settings.model_name
    )

    if model_id is None:
        raise ValueError("Model name not found in DB")

    return query_page_segments(
        engine,
        model_id,
        settings.distance_metric,
        query_vector,
        dimension=settings.embedding_dimension,
        top_n=settings.query_return_n
    )


def print_help():
    print("\nCommands:")
    print("/help              Show commands")
    print("/exit              Exit CLI")
    print("/set_n <value>     Set query_return_n")
    print("/set_k <value>     Set rerank_return_n")
    print()


if __name__ == '__main__':

    settings = load_settings()

    print(settings)

    model = load_embedding_model(settings)
    reranker = load_reranking_model(settings)

    print("\nInteractive Retrieval CLI")
    print("Type /help for commands.\n")

    while True:

        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                split = user_input.split()
                command = split[0].lower()

                if command == "/help":
                    print_help()

                elif command == "/exit":
                    print("Exiting...")
                    break

                elif command == "/set_n":

                    if len(split) != 2:
                        print("Usage: /set_n <value>")
                        continue

                    try:
                        settings.query_return_n = int(split[1])
                        print(f"query_return_n = {settings.query_return_n}")

                    except ValueError:
                        print("Invalid integer")

                elif command == "/set_k":

                    if len(split) != 2:
                        print("Usage: /set_k <value>")
                        continue

                    try:
                        settings.rerank_return_n = int(split[1])
                        print(f"rerank_return_n = {settings.rerank_return_n}")

                    except ValueError:
                        print("Invalid integer")

                else:
                    print("Unknown command. Type /help")

                continue

            print(f"\nQuerying: {user_input}\n")

            chunks = query_database(model, user_input, settings)

            reranked = rerank_candidates(
                reranker,
                user_input,
                chunks,
                settings
            )

            for i, chunk in enumerate(reranked):

                print(
                    f"[{i+1}] "
                    f"CrossScore={chunk['cross_score']:.4f} "
                    f"Distance={chunk['dist']:.4f}"
                )

                print(chunk['text'][:500])
                print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        except Exception as e:
            print(f"\nERROR: {e}\n")