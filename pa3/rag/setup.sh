#!/bin/bash
# Setup script for PA3 RAG System

set -e
mkdir -p db_data ollama_data models
docker compose up -d