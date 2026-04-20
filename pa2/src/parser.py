from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from lxml import etree
from lxml import html as lxml_html
from sqlalchemy import MetaData, Table, and_, create_engine, select, insert
from sqlalchemy.engine import Engine
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm

from sqlalchemy.dialects.postgresql.base import ischema_names
from pgvector.sqlalchemy import Vector
ischema_names["vector"] = Vector

from chunk_embedding import embed_chunks, load_model

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

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

def parse_html(html_content: str) -> dict[str, Any]:
	#print("")
	try:
		tree = lxml_html.fromstring(html_content)
	except (etree.ParserError, ValueError):
		return {
			"title": "",
			"article-content": "",
		}

	raw_title = tree.xpath("normalize-space(string(//meta[@property='og:title']/@content))")
	if not raw_title:
		raw_title = tree.xpath("normalize-space(string(//title))")
	title = re.sub(r"\s*\|\s*24ur\.com\s*$", "", raw_title, flags=re.IGNORECASE).strip()

	article_nodes = tree.xpath("//*[@id='article-body']")
	if not article_nodes:
		return {
			"title": title,
			"article-content": "",
		}
	
		

	article_node = article_nodes[0]
	for removable in article_node.xpath(".//script|.//style|.//noscript|.//img|.//figure|.//svg|.//picture"):
		parent = removable.getparent()
		if parent is not None:
			parent.remove(removable)

	article_text = re.sub(r"\s+", " ", article_node.text_content()).strip()

	# extract sumamry by detecting the main article image, and moving along hierarchy to the summary <p> element
	summary = tree.xpath("//picture[@tabindex=0 and @class='media-object']//parent::div//parent::div//parent::div/p//text()")
	if summary:
		article_text = f'{summary[0]} {article_text}'

	return {
		"title": title,
		"article-content": article_text,
	}




def chunk_fixed_length(text, chunk_size=50):
    """Fixed length chunking."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_segments(text, max_words=256):
	"""Splits text into sentence-based chunks with a max word count limit."""
	sentences = sent_tokenize(text, language="slovene")  # Split into sentences
	chunks, current_chunk = [], []
	current_length = 0

	for sentence in sentences:
		words = sentence.split()
		#print(f"sentence: {sentence}")
		if current_length + len(words) > max_words:
			chunks.append(" ".join(current_chunk))  # Save current chunk
			current_chunk, current_length = [], 0  # Reset chunk
		current_chunk.append(sentence)
		current_length += len(words)
    
	if current_chunk:
		chunks.append(" ".join(current_chunk))  # Add last chunk

	return chunks


def main() -> None:
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
					"embedding": embedding
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
	main()
	#html_content = r.get("https://www.24ur.com/novice/slovenija-odloca/posveti-pri-predsednici-poslanci-se-bodo-prvic-uradno-prestevali.html").text
	#parsed = parse_html(html_content=html_content)
	#print(parsed)
