from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from tqdm import tqdm

from sqlalchemy import MetaData, Table, and_, create_engine, select, insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql.base import ischema_names
from pgvector.sqlalchemy import Vector
ischema_names["vector"] = Vector

from parsing import parse_html
from chunking import chunk_segments
from embedding import embed_chunks, load_model, embed_string




@dataclass(frozen=True)
class ParserSettings:
	database_url: str
	parse_limit: int
	table_schema: str
	table_name: str
	chunk_length: int
	model_run_device: str
	model_name: str
	embedding_dimension: int
	batch_size: int
 
	def __str__(self) -> str:
		return (
			f"ParserSettings(\n"
			f"	database_url='{self.database_url}',\n"
			f"	parse_limit={self.parse_limit},\n"
			f"	table_schema='{self.table_schema}',\n"
			f"	table_name='{self.table_name}',\n"
			f"	chunk_length={self.chunk_length},\n"
			f"	model_run_device='{self.model_run_device}',\n"
   			f"	model_name='{self.model_name}',\n"
			f"	embedding_dimension={self.embedding_dimension},\n"
			f"	batch_size={self.batch_size}\n"
			f")"
		)




def load_settings() -> ParserSettings:
	database_url = os.getenv("DATABASE_URL","postgresql+psycopg://crawler:crawler@localhost:5432/crawler")
	parse_limit = int(os.getenv("PARSE_LIMIT", "10"))
	table_schema = os.getenv("PARSER_TABLE_SCHEMA")
	table_name = os.getenv("PARSER_TABLE_NAME", "page")
	chunk_lenght = int(os.getenv("CHUNK_LENGTH", "64"))
	model_run_device = os.getenv("MODEL_RUN_DEVICE", "cpu")
	model_name = os.getenv("MODEL_NAME", "no-model-found-error-in-env")
	embedding_dimension = int(os.getenv("EMBEDING_DIMENSION", "768"))
	batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

	return ParserSettings(
		database_url=database_url,
		parse_limit=parse_limit,
		table_schema=table_schema,
		table_name=table_name,
		chunk_length=chunk_lenght,
		model_run_device=model_run_device,
		model_name=model_name,
  		embedding_dimension=embedding_dimension,
		batch_size=batch_size
	)




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





def parse_and_embed() -> None:
	settings = load_settings()
	print(settings)
 
	engine = create_engine(settings.database_url, pool_pre_ping=True)
	model = load_model(settings)
 
	try:
		source_table = get_source_table(
			engine=engine,
			schema_name=settings.table_schema,
			table_name=settings.table_name,
		)
		
		source_schema = source_table.schema or settings.table_schema
		print(f"Using table: {source_schema}.{source_table.name}")

		model_table = get_source_table(engine, source_schema, "model")
		model_id = register_model(engine, model_table, settings.model_name)
		print(f"Inserted model '{settings.model_name}' with id={model_id}")

		rows = fetch_html_content_rows(
			engine=engine,
			source_table=source_table,
			limit=settings.parse_limit,
		)
		
		print(f"Fetched rows with html_content: {len(rows)}")
		all_segments = []
		for row in tqdm(rows, desc='Parsing HTML Pages and Embedding their Chunks'):
			parsed = parse_html(row["html_content"])
			
			chunks = chunk_segments(parsed.get("title") + ". " + parsed.get("article-content", ""), max_words=settings.chunk_length)
			embeddings = embed_chunks(model, chunks, settings=settings)

			for (chunk, embedding) in zip(chunks, embeddings):
				all_segments.append({
					"page_id": row["id"],
					"page_segment": chunk,
					"embedding": embedding,
					"model_id": model_id,
				})

			# print(
			# 	"page_id={id} title={title!r} article-content={article_content!r}".format(
			# 		id=row["id"],
			# 		title=parsed["title"],
			# 		article_content=parsed["article-content"],
			# 	)
			# )

		#print(all_segments)

		put_segments(engine, get_source_table(engine, source_schema, "page_segment"), all_segments)


	finally:
		engine.dispose()


if __name__ == "__main__":
	parse_and_embed()
 
	#html_content = r.get("https://www.24ur.com/novice/slovenija-odloca/posveti-pri-predsednici-poslanci-se-bodo-prvic-uradno-prestevali.html").text
	#parsed = parse_html(html_content=html_content)
	#print(parsed)
 
