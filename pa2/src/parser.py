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


def parse_html_stub(html_content: str) -> dict[str, Any]:
	try:
		tree = lxml_html.fromstring(html_content)
	except (etree.ParserError, ValueError):
		return {
			"title": "",
			"first_h1": "",
			"token_count": 0,
		}

	title = tree.xpath("normalize-space(string(//title))")
	first_h1 = tree.xpath("normalize-space(string((//h1)[1]))")

	text_nodes = tree.xpath("//body//text()")
	text_blob = " ".join(text_nodes)
	text_blob = re.sub(r"\s+", " ", text_blob).strip()

	tokens = re.findall(r"[A-Za-z0-9_]+", text_blob)

	return {
		"title": title,
		"first_h1": first_h1,
		"token_count": len(tokens),
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
			parsed = parse_html_stub(row["html_content"])
			print(
				"page_id={id} title={title!r} h1={h1!r} token_count={token_count}".format(
					id=row["id"],
					title=parsed["title"],
					h1=parsed["first_h1"],
					token_count=parsed["token_count"],
				)
			)
	finally:
		engine.dispose()


if __name__ == "__main__":
	main()
