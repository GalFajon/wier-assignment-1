from __future__ import annotations

from pathlib import Path

import sqlparse
from sqlalchemy.exc import SQLAlchemyError

from .config import load_settings
from .db import create_db_engine


def _read_sql(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Migration file was not found: {path}")
    return path.read_text(encoding="utf-8")


def apply_migration() -> None:
    settings = load_settings()
    raw_sql = _read_sql(settings.migration_file)
    statements = [stmt.strip() for stmt in sqlparse.split(raw_sql) if stmt.strip()]

    if not statements:
        print("No SQL statements found in migration file. Nothing to run.")
        return

    engine = create_db_engine(settings.database_url)

    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
        print(f"Migration applied successfully using {settings.migration_file}")
    except SQLAlchemyError as exc:
        print(f"Migration failed: {exc}")
        raise
    finally:
        engine.dispose()


if __name__ == "__main__":
    apply_migration()
