
from __future__ import annotations

from typing import Any

from sqlalchemy import MetaData, Table, and_, create_engine, select, insert, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql.base import ischema_names
from pgvector.sqlalchemy import Vector
ischema_names["vector"] = Vector



def get_source_table(engine: Engine, schema_name: str, table_name: str) -> Table:
	metadata = MetaData()
	return Table(table_name, metadata, schema=schema_name, autoload_with=engine)




def register_model(engine: Engine, model_table: Table, model_name: str) -> int:
	with engine.connect() as connection:
		try:
			result = connection.execute(
				insert(model_table).values(model_name=model_name).returning(model_table.c.id)
			)
			connection.commit()
			return result.scalar_one()
		except IntegrityError:
			connection.rollback()
			result = connection.execute(
				select(model_table.c.id).where(model_table.c.model_name == model_name)
			)
			return result.scalar_one()



def get_model_id(engine: Engine, model_table: Table, model_name: str) -> int | None:
    with engine.connect() as connection:
        result = connection.execute(
            select(model_table.c.id)
            .where(model_table.c.model_name == model_name)
        ).scalar_one_or_none()
    return result




def drop_pgvector_indexes(engine: Engine):
    tables = [
        "page_segment_vec384",
        "page_segment_vec768",
        "page_segment_vec1024",
    ]

    with engine.connect() as connection:
        for table in tables:
            connection.exec_driver_sql(f"""
                DROP INDEX IF EXISTS public.idx_{table}_embedding_cosine;
            """)
            connection.exec_driver_sql(f"""
                DROP INDEX IF EXISTS public.idx_{table}_embedding_l2;
            """)
            connection.exec_driver_sql(f"""
                DROP INDEX IF EXISTS public.idx_{table}_model;
            """)
        connection.commit()

def create_pgvector_indexes(engine: Engine):
    tables = [
        "page_segment_vec384",
        "page_segment_vec768",
        "page_segment_vec1024",
    ]

    with engine.connect() as connection:
        for table in tables:
            # cosine index
            connection.exec_driver_sql(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_embedding_cosine
                ON public.{table}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)

            # l2 index
            connection.exec_driver_sql(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_embedding_l2
                ON public.{table}
                USING hnsw (embedding vector_l2_ops)
                WITH (m = 16, ef_construction = 64);
            """)

            # model_id index
            connection.exec_driver_sql(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_model
                ON public.{table}(model_id);
            """)

        connection.commit()




def put_segments(engine: Engine, table: Table, segments: list[dict[str, Any]]):
	with engine.connect() as connection:
		result = connection.execute(
			insert(table),
			segments
		)
		connection.commit()
	
 
 

def fetch_html_content_rows(
    engine: Engine,
    source_table: Table,
    limit: int,
) -> list[dict[str, Any]]:
    query = (
        select(source_table.c.id, source_table.c.html_content)
        .where(
            and_(
                source_table.c.html_content.is_not(None),
                source_table.c.html_content != "",
            )
        )
        .order_by(source_table.c.id)
    )

    if limit > 0:
        query = query.limit(limit)

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [
        {
            "id": row["id"],
            "html_content": row["html_content"],
        }
        for row in rows
    ]

def get_segments_by_model(
    engine: Engine,
    model_id: int,
    max_segments=2000,
    vector_size=384
):
    query = f"""
        SELECT page_segment, embedding FROM page_segment_vec{vector_size}
        WHERE model_id= :model_id
        LIMIT :max_segments
    """

    with engine.connect() as connection:
        result = connection.execute(
            text(query),
            {
                "model_id": model_id,
                "max_segments": max_segments,
            }
        ).fetchall()
    return result

def get_segments_by_id(
    engine: Engine,
    ids: list[int],
    dim=384
):

    table_map = {
        384: "page_segment_vec384",
        768: "page_segment_vec768",
        1024: "page_segment_vec1024",
    }

    if dim not in table_map:
        raise ValueError(f"Unsupported vector length: {dim}")

    table = table_map[dim]


    sql = f"""
    select id, page_segment from public.{table}
    where id in {tuple(ids)}
    """

    with engine.connect() as connection:
        result = connection.execute(
            text(sql),
            {
                
            }
        ).fetchall()

    return result

def get_page_segment_ids(
    engine: Engine,
    page_ids: list[int],
    model_id: int,
    dim: int
):

    table_map = {
        384: "page_segment_vec384",
        768: "page_segment_vec768",
        1024: "page_segment_vec1024",
    }

    if dim not in table_map:
        raise ValueError(f"Unsupported vector length: {dim}")

    table = table_map[dim]


    sql = f"""
    select page_id, array_agg(id) from public.{table}
    group by page_id, model_id
    having page_id in {tuple(page_ids)} and model_id={model_id}
    """

    with engine.connect() as connection:
        result = connection.execute(
            text(sql),
            {
                
            }
        ).fetchall()

    return result   

def get_random_page_ids(
    engine: Engine,
    n: int
):

    sql = f"""
    select page_id from public.page_segment_vec1024
    group by page_id
    having COUNT(*) > 6
    order by random()
    LIMIT {n};
    """

    with engine.connect() as connection:
        result = connection.execute(
            text(sql),
            {
                
            }
        ).fetchall()

    return result

def get_random_page_segments(
    engine,
    dimension=384,
    n=50
):
     
    dim = dimension

    table_map = {
        384: "page_segment_vec384",
        768: "page_segment_vec768",
        1024: "page_segment_vec1024",
    }

    if dim not in table_map:
        raise ValueError(f"Unsupported vector length: {dim}")

    table = table_map[dim]


    sql = f"""
    select page_id, COUNT(*), array_agg(id) from public.{table}
    group by page_id
    having COUNT(*) > 12
    order by random()
    LIMIT {n};
    """

    with engine.connect() as connection:
        result = connection.execute(
            text(sql),
            {
                
            }
        ).fetchall()

    return result

def query_page_segments(
    engine,
    model_id: int,
    metric: str,
    query_vector,
    dimension=384,
    top_n: int = 5
):
    ann_candidate_mult = 50
    dim = len(query_vector)

    table_map = {
        384: "page_segment_vec384",
        768: "page_segment_vec768",
        1024: "page_segment_vec1024",
    }

    if dim not in table_map:
        raise ValueError(f"Unsupported vector length: {dim}")

    table = table_map[dim]

    op_map = {
        "cosine": "<=>",
        "l2": "<->",
        "l1": "<+>",
    }

    if metric not in op_map:
        raise ValueError("metric must be one of: cosine, l2, l1")

    op = op_map[metric]

    ann_limit = max(
        top_n * ann_candidate_mult,
        1000
    )

    sql = f"""
    WITH ann_candidates AS (

        SELECT id
        FROM public.{table}
        ORDER BY embedding {op} (:query_vec)::vector
        LIMIT :ann_limit

    )

    SELECT
        p.page_segment,
        p.embedding {op} (:query_vec)::vector AS distance,
        p.id

    FROM public.{table} p

    INNER JOIN ann_candidates a
        ON p.id = a.id

    WHERE p.model_id = :model_id

    ORDER BY p.embedding {op} (:query_vec)::vector

    LIMIT :top_n;
    """

    with engine.connect() as connection:

        connection.execute(
            text("SET enable_seqscan = on;")
        )

        connection.execute(
            text("SET enable_indexscan = on;")
        )

        connection.execute(
            text("SET enable_bitmapscan = on;")
        )

        ef_search = max(
            ann_limit,
            200
        )

        connection.execute(
            text(f"SET hnsw.ef_search = {ef_search};")
        )

        result = connection.execute(
            text(sql),
            {
                "query_vec": query_vector,
                "model_id": model_id,
                "top_n": top_n,
                "ann_limit": ann_limit
            }
        ).fetchall()

    return result