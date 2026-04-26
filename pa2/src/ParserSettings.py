from dataclasses import dataclass
import os

@dataclass(frozen=True)
class ParserSettings:
    #DB
	database_url: str
	parse_limit: int
	table_schema: str
	table_name: str
 
	#CHUNKING
	chunk_length: int
 
	#EMBEDDING
	model_run_device: str
	model_name: str
	embedding_dimension: int
	batch_size: int
 
	#QUERYING AND RERANKING
	query_return_n: int
	distance_metric: str
	reranking_model_name: str
	rerank_return_n: int
	
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
			f"	batch_size={self.batch_size},\n"
			f"	query_return_n={self.query_return_n},\n"
			f"	distance_metric={self.distance_metric},\n"
			f"	reranking_model_name={self.reranking_model_name},\n"
			f"	rerank_return_n={self.rerank_return_n}\n"
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
	query_return_n = int(os.getenv("QUERY_RETURN_N", "100"))
	distance_metric = os.getenv("DISTANCE_METRIC", "cos_similarity")
	reranking_model_name = os.getenv("RERANKING_MODEL_NAME", "no-model-found-error-in-env")
	rerank_return_n = int(os.getenv("RERANK_RETURN_N", "10"))

	return ParserSettings(
		database_url=database_url,
		parse_limit=parse_limit,
		table_schema=table_schema,
		table_name=table_name,
		chunk_length=chunk_lenght,
		model_run_device=model_run_device,
		model_name=model_name,
  		embedding_dimension=embedding_dimension,
		batch_size=batch_size,
		query_return_n=query_return_n,
		distance_metric=distance_metric,
		reranking_model_name=reranking_model_name,
		rerank_return_n=rerank_return_n
	)

