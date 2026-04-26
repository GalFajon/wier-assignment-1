
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
    with engine.connect() as connection:
        connection.exec_driver_sql("DROP INDEX IF EXISTS public.idx_ps_embedding_cosine;")
        connection.exec_driver_sql("DROP INDEX IF EXISTS public.idx_ps_embedding_l2;")
        connection.commit()


def create_pgvector_indexes(engine: Engine):
    with engine.connect() as connection:
        # cos dist
        connection.exec_driver_sql("""
            CREATE INDEX IF NOT EXISTS idx_ps_embedding_cosine
            ON public.page_segment
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

        # l2 dist
        connection.exec_driver_sql("""
            CREATE INDEX IF NOT EXISTS idx_ps_embedding_l2
            ON public.page_segment
            USING hnsw (embedding vector_l2_ops)
            WITH (m = 16, ef_construction = 64);
        """)

        # model id
        connection.exec_driver_sql("""
            CREATE INDEX IF NOT EXISTS idx_page_segment_model
            ON public.page_segment(model_id);
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


def query_page_segments(
    engine,
    model_id: int,
    metric: str,
    query_vector,
    top_n: int = 5
):
    if metric == "cosine":
        op = "<=>"
    elif metric == "l2":
        op = "<->"
    elif metric == "l1":
        op = "<+>"
    else:
        raise ValueError("metric must be one of: cosine, l2, l1")

    sql = f"""
    SELECT page_segment, embedding {op} (:query_vec)::vector AS distance
    FROM public.page_segment
    WHERE model_id = :model_id
    ORDER BY embedding {op} (:query_vec)::vector
    LIMIT :top_n;
    """

    with engine.connect() as connection:
        result = connection.execute(
            text(sql),
            {
                "query_vec": query_vector,
                "model_id": model_id,
                "top_n": top_n
            }
        ).fetchall()

    return result