# Quick Start Guide

## Prerequisites

The PA1/PA2 database dump file must be present at: `pa3/DB_with_embeddings.sql`

This file should be a PostgreSQL custom format dump containing the PA1 crawled pages and PA2 embeddings.

## 1. One-Command Setup

```bash
cd pa3/rag
bash setup.sh
```

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run Examples
### Example 1: Default Comparison
```bash
python main.py
```

### Example 2: Custom Question
```bash
python main.py "What are the major industries in Slovenia?"
```

### Example 3: Custom Candidate Count
```bash
python main.py "What happened in 2020?" 20
```

Searches for 20 candidate chunks before reranking for context.

## 4. Stop Services

```bash
docker compose down
```

Keep data:
```bash
docker compose stop
```