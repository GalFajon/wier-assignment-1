from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str
    migration_file: Path


def load_settings() -> Settings:
    default_migration = Path(__file__).resolve().parents[2] / "migration.sql"

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://crawler:crawler@localhost:5432/crawler",
    )
    migration_file = Path(os.getenv("MIGRATION_FILE", str(default_migration))).resolve()

    return Settings(database_url=database_url, migration_file=migration_file)
