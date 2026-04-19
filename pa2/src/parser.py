from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from lxml import etree
from lxml import html as lxml_html
from sqlalchemy import MetaData, Table, and_, create_engine, select
from sqlalchemy.engine import Engine

@dataclass(frozen=True)
class ParserSettings:
	database_url: str
	parse_limit: int
	table_schema: str
	table_name: str


def load_settings() -> ParserSettings:
	database_url = os.getenv("DATABASE_URL","postgresql+psycopg://crawler:crawler@localhost:5432/crawler")
	parse_limit = int(os.getenv("PARSE_LIMIT", "10"))
	table_schema = os.getenv("PARSER_TABLE_SCHEMA", "crawldb")
	table_name = os.getenv("PARSER_TABLE_NAME", "page")
	return ParserSettings(
		database_url=database_url,
		parse_limit=parse_limit,
		table_schema=table_schema,
		table_name=table_name,
	)

def get_source_table(engine: Engine, schema_name: str, table_name: str) -> Table:
	metadata = MetaData()
	return Table(table_name, metadata, schema=schema_name, autoload_with=engine)


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
		.limit(limit)
	)

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


	return {
		"title": title,
		"article-content": article_text,
	}


def main() -> None:
	settings = load_settings()
	engine = create_engine(settings.database_url, pool_pre_ping=True)

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

		for row in rows:
			parsed = parse_html(row["html_content"])
			print(
				"page_id={id} title={title!r} article-content={article_content!r}".format(
					id=row["id"],
					title=parsed["title"],
					article_content=parsed["article-content"],
				)
			)
	finally:
		engine.dispose()


if __name__ == "__main__":
	main()
