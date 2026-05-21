#!/bin/bash
set -e

echo "Waiting for PostgreSQL to be ready..."
until pg_isready -U postgres; do
  sleep 1
done

echo "PostgreSQL is ready. Creating pa2db database..."
psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'pa2db'" | grep -q 1 || psql -U postgres -c "CREATE DATABASE pa2db;"

echo "Restoring database..."
pg_restore -U postgres -d pa2db --no-owner --no-privileges /tmp/DB_with_embeddings.sql

echo "Database restore completed!"