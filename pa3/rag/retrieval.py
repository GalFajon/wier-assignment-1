from sqlalchemy import create_engine, text
import os


def get_database_connection(db_url=None):
    if db_url is None:
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        database = os.getenv("DB_NAME", "pa2db")
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    engine = create_engine(db_url)
    return engine


def query_similar_chunks(engine, query_vector, dimension=384, metric="cosine", top_n=5, table_name="page_segment_vec384"):
    table_map = {
        384: "page_segment_vec384",
        768: "page_segment_vec768",
        1024: "page_segment_vec1024"
    }
    
    if dimension not in table_map:
        raise ValueError(f"Dimension {dimension} not supported. Use 384, 768, or 1024.")
    
    actual_table = table_map.get(dimension, table_name)
    
    operator_map = {
        "cosine": "<=>",
        "l2": "<->",
        "l1": "<+>"
    }
    
    if metric not in operator_map:
        raise ValueError(f"Metric {metric} not supported. Use cosine, l2, or l1.")
    
    operator = operator_map[metric]
    
    vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"
    
    sql = f"""
        SELECT 
            page_segment,
            embedding {operator} '{vector_str}'::vector AS distance
        FROM public.{actual_table}
        ORDER BY embedding {operator} '{vector_str}'::vector
        LIMIT :top_n;
    """
    
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(sql),
                {"top_n": top_n}
            )
            
            chunks = []
            for row in result:
                chunks.append((row[0], float(row[1])))
            
            return chunks
    except Exception as e:
        print(f"Database query error: {e}")
        return []


def health_check(engine):
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Database health check failed: {e}")
        return False





def get_chunks_for_page_id_and_model(engine, page_id, model_id, dimension=384):
    table_map = {
        384: "page_segment_vec384",
        768: "page_segment_vec768",
        1024: "page_segment_vec1024"
    }

    if dimension not in table_map:
        raise ValueError(f"Dimension {dimension} not supported. Use 384, 768, or 1024.")

    table = table_map[dimension]

    sql = f"""
        SELECT
            id,
            page_segment
        FROM public.{table}
        WHERE page_id = :page_id
          AND model_id = :model_id
        ORDER BY id ASC;
    """

    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql), {"page_id": page_id, "model_id": model_id})
            return [(row[0], row[1]) for row in result]
    except Exception as e:
        print(f"Database query error: {e}")
        return []
    
    
def get_longest_pages_by_chunk_length(engine, model_id, dimension=384, top_n=50):
    table_map = {
        384: "page_segment_vec384",
        768: "page_segment_vec768",
        1024: "page_segment_vec1024"
    }

    if dimension not in table_map:
        raise ValueError(f"Dimension {dimension} not supported. Use 384, 768, or 1024.")

    table = table_map[dimension]

    sql = f"""
        SELECT
            page_id,
            SUM(LENGTH(page_segment)) AS total_length
        FROM public.{table}
        WHERE model_id = :model_id
        GROUP BY page_id
        ORDER BY total_length DESC
        LIMIT :top_n;
    """

    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql), {"model_id": model_id, "top_n": top_n})
            return [(row[0], row[1]) for row in result]
    except Exception as e:
        print(f"Database query error: {e}")
        return []
    
    
def get_pages_by_chunk_length_sum_interval(engine, model_id, min_length, max_length, dimension=384):
    table_map = {
        384: "page_segment_vec384",
        768: "page_segment_vec768",
        1024: "page_segment_vec1024"
    }

    if dimension not in table_map:
        raise ValueError(f"Dimension {dimension} not supported. Use 384, 768, or 1024.")

    table = table_map[dimension]

    sql = f"""
        SELECT
            page_id,
            SUM(LENGTH(page_segment)) AS total_length
        FROM public.{table}
        WHERE model_id = :model_id
        GROUP BY page_id
        HAVING SUM(LENGTH(page_segment)) BETWEEN :min_length AND :max_length
        ORDER BY total_length DESC;
    """

    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql), {"model_id": model_id, "min_length": min_length, "max_length": max_length})
            return [(row[0], row[1]) for row in result]
    except Exception as e:
        print(f"Database query error: {e}")
        return []
    
    
def get_page_by_id(engine, page_id):
    sql = """
        SELECT
            id,
            site_id,
            page_type_code,
            url,
            html_content,
            http_status_code,
            accessed_time,
            content_hash,
            priority,
            cleaned_content
        FROM public.page
        WHERE id = :page_id;
    """

    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql), {"page_id": page_id})
            row = result.mappings().fetchone()
            return dict(row) if row else None
    except Exception as e:
        print(f"Database query error: {e}")
        return None