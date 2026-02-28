"""
PostgreSQL database connection and helper utilities.

Uses asyncpg for async connectivity and pgvector for vector search.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncpg  # type: ignore


class Database:
    """Async PostgreSQL connection pool wrapper."""

    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def connect(cls) -> None:
        """Initialise the connection pool from DATABASE_URL env var."""
        dsn = os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:password@localhost:5432/edulens",
        )
        cls._pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)

    @classmethod
    async def disconnect(cls) -> None:
        """Close the connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None

    @classmethod
    def pool(cls) -> asyncpg.Pool:
        if cls._pool is None:
            raise RuntimeError("Database pool is not initialised. Call Database.connect() first.")
        return cls._pool

    @classmethod
    @asynccontextmanager
    async def acquire(cls) -> AsyncGenerator[asyncpg.Connection, None]:
        async with cls.pool().acquire() as conn:
            yield conn

    @classmethod
    async def execute(cls, query: str, *args: Any) -> str:
        async with cls.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query: str, *args: Any) -> List[asyncpg.Record]:
        async with cls.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query: str, *args: Any) -> Optional[asyncpg.Record]:
        async with cls.acquire() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def fetchval(cls, query: str, *args: Any) -> Any:
        async with cls.acquire() as conn:
            return await conn.fetchval(query, *args)


async def apply_schema() -> None:
    """Apply the SQL schema file to the connected database (idempotent)."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema_sql = fh.read()
    async with Database.acquire() as conn:
        await conn.execute(schema_sql)
