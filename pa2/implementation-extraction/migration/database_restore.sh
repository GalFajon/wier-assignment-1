#!/bin/bash
set -e

echo "RUNNING RESTORE SCRIPT"

curl -L \
'https://drive.usercontent.google.com/download?id=1ecdRDZfovBy1ImHaQOKT6i7MkgZ5rpZs&export=download&confirm=t' \
-o /tmp/db.dump

echo "RESTORING DB"

pg_restore \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    --no-owner \
    /tmp/db.dump

echo "DONE"