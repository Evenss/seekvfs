"""FilesDAO — database access layer for OceanbaseFsBackend.

All SQL lives here. To adapt the Maximal recipe to a different schema,
table name, column names, or database engine, subclass ``FilesDAO`` and
override only the methods you need, then pass your DAO to the backend::

    class MyDAO(FilesDAO):
        \"\"\"Custom schema: renamed columns, extra business fields.\"\"\"

        async def upsert_init(self, path: str) -> None:
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        \"INSERT INTO my_docs (uri, summary, overview, vec)\"
                        \" VALUES (%s, NULL, NULL, NULL)\"
                        \" ON DUPLICATE KEY UPDATE\"
                        \"   summary = NULL, overview = NULL, vec = NULL\",
                        (path,),
                    )
                    await conn.commit()

        async def update_derivatives(
            self, path: str, l0: str, l1: str, emb: list[float]
        ) -> None:
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        \"UPDATE my_docs\"
                        \" SET summary = %s, overview = %s, vec = %s\"
                        \" WHERE uri = %s\",
                        (l0, l1, _vec_to_str(emb), path),
                    )
                    await conn.commit()

        # ... override other methods as needed ...


    backend = OceanbaseFsBackend(
        ob_pool=pool,
        fs_root=\"/data/agent_files\",
        summarizer=...,
        embedder=...,
        dao=MyDAO(pool),   # ← inject your custom DAO
    )

Default schema (``schema.sql``)::

    CREATE TABLE vfs_storage (
        path       VARCHAR(512)  NOT NULL PRIMARY KEY,
        l0         TEXT          DEFAULT NULL,
        l1         MEDIUMTEXT    DEFAULT NULL,
        embedding  VECTOR(1536)  DEFAULT NULL,
        updated_at TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
                                 ON UPDATE CURRENT_TIMESTAMP
    );
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _vec_to_str(vec: list[float]) -> str:
    """Encode a float list to OceanBase ``VECTOR`` literal ``'[x,y,...]'``."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


class FilesDAO:
    """Default OceanBase data access layer for the Maximal recipe.

    Every public method corresponds to exactly one logical database
    operation.  Subclass and override any method to adapt the schema or
    SQL dialect without touching the backend orchestration logic.

    Args:
        pool: An ``asyncmy`` connection pool.
        table: Table name (default ``"files"``).
        vector_dim: Dimension of the embedding vector column (default 1536).
            Must match the output dimension of your :class:`Embedder`.
            Common values: OpenAI text-embedding-3-small → 1536,
            text-embedding-v3 (Qwen) → 1024.
    """

    def __init__(self, pool: Any, table: str = "vfs_storage", vector_dim: int = 1536) -> None:
        self._pool = pool
        self._table = table
        self._vector_dim = vector_dim

    async def initialize(self) -> None:
        """Create the table and vector index if they do not already exist.

        Safe to call multiple times — uses ``CREATE TABLE IF NOT EXISTS``.
        Called automatically by :class:`OceanbaseFsBackend` on first use,
        so you rarely need to invoke this directly.

        Override in a subclass if you need a different schema or DDL dialect.
        """
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            f"  path        VARCHAR(512)  NOT NULL,"
            f"  l0          TEXT          DEFAULT NULL,"
            f"  l1          MEDIUMTEXT    DEFAULT NULL,"
            f"  embedding   VECTOR({self._vector_dim}) DEFAULT NULL,"
            f"  updated_at  TIMESTAMP     NOT NULL"
            f"              DEFAULT CURRENT_TIMESTAMP"
            f"              ON UPDATE CURRENT_TIMESTAMP,"
            f"  PRIMARY KEY (path),"
            f"  VECTOR INDEX idx_emb (embedding)"
            f"    WITH (distance = L2, type = HNSW, lib = vsag)"
            f")"
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(ddl)
            await conn.commit()
        logger.info("FilesDAO.initialize: table %r ready (dim=%d)", self._table, self._vector_dim)

    # ------------------------------------------------------------------ #
    # Write-path                                                           #
    # ------------------------------------------------------------------ #

    async def upsert_init(self, path: str) -> None:
        """Ensure a row for *path* exists; reset stale derivatives to NULL.

        Called by ``write()`` before scheduling derivative generation.

        Uses ``REPLACE INTO`` (DELETE + INSERT) instead of
        ``ON DUPLICATE KEY UPDATE`` to avoid OceanBase HNSW vector-index
        conflicts when overwriting an existing embedding column.
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"REPLACE INTO {self._table} (path, l0, l1, embedding)"
                    f" VALUES (%s, NULL, NULL, NULL)",
                    (path,),
                )
                await conn.commit()

    async def update_derivatives(
        self, path: str, l0: str, l1: str, emb: list[float]
    ) -> None:
        """Write generated L0/L1/embedding for *path*.

        Called after async (or sync) derivative generation completes.
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE {self._table}"
                    f" SET l0 = %s, l1 = %s, embedding = %s"
                    f" WHERE path = %s",
                    (l0, l1, _vec_to_str(emb), path),
                )
                await conn.commit()

    async def clear_derivatives(self, path: str) -> None:
        """Set L0/L1/embedding back to NULL after an ``edit()``.

        Marks the stored derivatives as stale; a new generation pass
        will fill them in.
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE {self._table}"
                    f" SET l0 = NULL, l1 = NULL, embedding = NULL"
                    f" WHERE path = %s",
                    (path,),
                )
                await conn.commit()

    async def delete(self, path: str) -> None:
        """Remove the row for *path* from the database."""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {self._table} WHERE path = %s", (path,)
                )
                await conn.commit()

    # ------------------------------------------------------------------ #
    # Read-path                                                            #
    # ------------------------------------------------------------------ #

    async def get_l0(self, path: str) -> tuple[bool, str | None]:
        """Fetch the L0 abstract for *path*.

        Returns:
            ``(row_exists, l0_value)``

            - ``row_exists = False`` → no DB record (path was never written).
            - ``row_exists = True, l0_value = None`` → row exists but L0 not yet generated.
            - ``row_exists = True, l0_value = "..."`` → L0 is available.
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT l0 FROM {self._table} WHERE path = %s", (path,)
                )
                row = await cur.fetchone()
        if row is None:
            return False, None
        return True, row[0]

    async def get_l1(self, path: str) -> tuple[bool, str | None]:
        """Fetch the L1 overview for *path*.  Same semantics as ``get_l0``."""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT l1 FROM {self._table} WHERE path = %s", (path,)
                )
                row = await cur.fetchone()
        if row is None:
            return False, None
        return True, row[0]

    async def get_l1_l0(
        self, path: str
    ) -> tuple[bool, str | None, str | None]:
        """Fetch L1 and L0 together (used for ``hint=None`` waterfall).

        Returns:
            ``(row_exists, l1_value, l0_value)``
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT l1, l0 FROM {self._table} WHERE path = %s", (path,)
                )
                row = await cur.fetchone()
        if row is None:
            return False, None, None
        return True, row[0], row[1]

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    async def vector_search(
        self,
        emb: list[float],
        path_like: str | None,
        score_threshold: float | None,
        limit: int,
    ) -> list[tuple[str, str | None, float]]:
        """Run vector similarity search.

        Args:
            emb: Query embedding (dense float vector).
            path_like: Optional SQL ``LIKE`` pattern to restrict paths.
            score_threshold: Minimum score (``HAVING score >= threshold``).
            limit: Maximum number of hits to return.

        Returns:
            List of ``(path, l0_snippet, score)`` ordered by score DESC.
            ``l0_snippet`` may be ``None`` if L0 is not yet generated.
        """
        emb_str = _vec_to_str(emb)
        sql = (
            f"SELECT path, l0, 1 - l2_distance(embedding, %s) AS score"
            f" FROM {self._table}"
            f" WHERE embedding IS NOT NULL"
        )
        params: list = [emb_str]
        if path_like:
            sql += " AND path LIKE %s"
            params.append(path_like)
        if score_threshold is not None:
            sql += " HAVING score >= %s"
            params.append(score_threshold)
        sql += " ORDER BY score DESC LIMIT %s"
        params.append(limit)

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()

        return [(r[0], r[1], float(r[2])) for r in rows]

    # ------------------------------------------------------------------ #
    # Bulk / utility                                                       #
    # ------------------------------------------------------------------ #

    async def batch_l0(self, paths: list[str]) -> dict[str, str | None]:
        """Fetch L0 snippets for multiple *paths* in one query.

        Returns a mapping ``{path: l0_or_None}``; paths with no DB row
        are absent from the result.
        """
        if not paths:
            return {}
        placeholders = ", ".join(["%s"] * len(paths))
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT path, l0 FROM {self._table}"
                    f" WHERE path IN ({placeholders})",
                    paths,
                )
                rows = await cur.fetchall()
        return {r[0]: r[1] for r in rows}

    async def find_incomplete(
        self, all_paths: list[str]
    ) -> tuple[set[str], set[str]]:
        """Identify paths that need derivative generation.

        Used by ``reconcile()`` to find what needs repair.

        Returns:
            ``(missing_derivatives, no_db_record)``

            - ``missing_derivatives``: paths that have a DB row but at
              least one of l0/l1/embedding is NULL.
            - ``no_db_record``: FS files that have no DB row at all.
        """
        if not all_paths:
            return set(), set()

        placeholders = ", ".join(["%s"] * len(all_paths))

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT path FROM {self._table}"
                    f" WHERE path IN ({placeholders})"
                    f" AND (l0 IS NULL OR l1 IS NULL OR embedding IS NULL)",
                    all_paths,
                )
                missing_deriv: set[str] = {r[0] for r in await cur.fetchall()}

                await cur.execute(
                    f"SELECT path FROM {self._table}"
                    f" WHERE path IN ({placeholders})",
                    all_paths,
                )
                in_db: set[str] = {r[0] for r in await cur.fetchall()}

        no_db_record = set(all_paths) - in_db
        return missing_deriv, no_db_record


__all__ = ["FilesDAO"]
