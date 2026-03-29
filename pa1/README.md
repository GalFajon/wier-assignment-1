## WIER - Assignment 1 - Web crawler

Web crawler implementation, consists of:
- crawler client (Python + Selenium) that fetches pages and parses content
- Flask API server that stores crawl results,
- PostgreSQL database for persistence.

The crawler is configured via environment variables and runs with a preferential frontier strategy and robots.txt compliance.

## Install, set up, and run (Docker Compose)

Prerequisites:
- Docker
- Docker Compose

Steps:
1. Open [pa1/crawler/docker-compose.yml](pa1/crawler/docker-compose.yml) and adjust:
   - `URL_SEED` (seed URL)
   - `MAX_PAGES` (crawl limit)
   - `WORKER_COUNT` (parallel workers)
   - `QUERY` (topic for relevance scoring)
2. From the `pa1/crawler` folder, run:
   ```bash
   docker compose up --build

## Import database

The database dump linked in `googledrive.txt` can be imported with `pg_restore` using the credentials from `docker-compose.yml`.

```bash
pg_restore -h localhost -p 5432 -U postgres --dbname=crawldb ./dump.sql