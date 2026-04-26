from __future__ import annotations

from tqdm import tqdm
from sqlalchemy import create_engine

from ParserSettings import ParserSettings, load_settings
from db_api import get_source_table, register_model, fetch_html_content_rows, put_segments, drop_pgvector_indexes, create_pgvector_indexes
from parsing import parse_html
from chunking import chunk_segments
from embedding import embed_chunks, load_embedding_model


def parse_and_embed() -> None:
	settings = load_settings()
	print(settings)
 
	engine = create_engine(settings.database_url, pool_pre_ping=True)
	model = load_embedding_model(settings)
 
	try:
		drop_pgvector_indexes(engine)

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

		put_segments(engine, get_source_table(engine, source_schema, "page_segment"), all_segments)
		create_pgvector_indexes(engine)

	finally:
		engine.dispose()


if __name__ == "__main__":
	parse_and_embed()
 
	#html_content = r.get("https://www.24ur.com/novice/slovenija-odloca/posveti-pri-predsednici-poslanci-se-bodo-prvic-uradno-prestevali.html").text
	#parsed = parse_html(html_content=html_content)
	#print(parsed)
 
