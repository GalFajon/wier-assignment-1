# PA3
## Demo:
The demo requires the docker container for the database to be up and running. Additionally it requires that an Ollama instance is running on the host computer, either via the provided docker container or, ideally, on the machine itself. The database is imported from an sql file (DB_with_embeddings.sql), that must be placed in the pa3 folder. The link to the database is provided in the database.txt file.

```
docker compose up db
```

In case you need an Ollama instance:

```
docker compose up
```

Once the containers are initialized and the database is imported, the demo program can be run using:

```
python -m venv .venv
pip install -r requirements.txt
python app.py
```

The program can then be queried and will return results for all DSPy signatures, described in the report.