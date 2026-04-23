"""OceanbaseFsBackend — L2 on local filesystem, L0/L1/embedding in OceanBase.

This is the **Maximal recipe** for seekvfs. Storage layout:

- **L2** full content      → local filesystem under ``fs_root``
- **L0** abstract (~100 t) → OceanBase via :class:`FilesDAO`
- **L1** overview (~2k t)  → OceanBase via :class:`FilesDAO`
- **embedding**            → OceanBase via :class:`FilesDAO` (vector column)

All database interactions are delegated to a :class:`~seekvfs_recipes.maximal.dao.FilesDAO`
instance, which you can subclass to adapt the table structure, column names,
or swap the database engine entirely.

Full usage guide: ``docs/recipes/maximal.md``.
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from seekvfs.exceptions import BackendError, NotFoundError
from seekvfs.models import FileData, FileInfo, GrepMatch, SearchHit, SearchResult
from seekvfs.uri import SCHEME as _SCHEME
from seekvfs_recipes.maximal.dao import FilesDAO
from seekvfs_recipes.maximal.exceptions import TierNotAvailable
from seekvfs_recipes.maximal.protocol import Embedder, Summarizer

logger = logging.getLogger(__name__)

GenerationMode = Literal["sync", "async"]


def _to_bytes(content: bytes | str) -> bytes:
    return content if isinstance(content, bytes) else content.encode("utf-8")


class OceanbaseFsBackend:
    """Production 3-tier backend: filesystem for L2, OceanBase for L0/L1/embedding.

    Hint values accepted by ``read(path, hint=...)``:

    +----------+----------------------------------------------------------+
    | hint     | behaviour                                                |
    +==========+==========================================================+
    | ``None`` | waterfall: L1 → L0 → truncated L2                       |
    | ``"l0"`` | strict L0; raises ``TierNotAvailable`` if not generated  |
    | ``"l1"`` | strict L1; raises ``TierNotAvailable`` if not generated  |
    | ``"l2"`` | full content (equivalent to ``read_full``)               |
    | other    | ``BackendError``                                         |
    +----------+----------------------------------------------------------+

    Args:
        ob_pool: An ``asyncmy`` connection pool
            (``await asyncmy.create_pool(...)``).
        fs_root: Root directory for L2 files. Created automatically if absent.
        summarizer: Produces L0 (``abstract``) and L1 (``overview``) from
            raw content.
        embedder: Embeds the L0 text into a dense vector.
        dao: Optional custom :class:`~seekvfs_recipes.maximal.dao.FilesDAO`.
            Pass a subclass to adapt the schema, column names, or DB engine.
            If omitted, a default ``FilesDAO(ob_pool, table)`` is created.
        generation: ``"async"`` (default) — ``write`` returns immediately and
            derivatives are generated in the background.
            ``"sync"`` — ``write`` blocks until derivatives are committed.
        fallback_l2_chars: Maximum chars returned as a truncated-L2 fallback
            when no L1/L0 is available yet (default ``8000``).
        table: OceanBase table name passed to the default ``FilesDAO``
            (ignored when a custom ``dao`` is supplied).
    """

    def __init__(
        self,
        *,
        ob_pool: object,
        fs_root: str | Path,
        summarizer: Summarizer,
        embedder: Embedder,
        dao: FilesDAO | None = None,
        generation: GenerationMode = "async",
        fallback_l2_chars: int = 8000,
        table: str = "vfs_storage",
        l0_threshold: int = 300,
        l1_threshold: int = 2000,
    ) -> None:
        """
        Args:
            l0_threshold: If content (in chars) is shorter than this, L0 is set
                to the content itself — no LLM call is made.  Default 300.
            l1_threshold: If content is shorter than this, L1 is also set to
                the content itself.  Default 2000.
        """
        if generation not in ("sync", "async"):
            raise ValueError(
                f"generation must be 'sync' or 'async', got {generation!r}"
            )
        self._dao = dao if dao is not None else FilesDAO(ob_pool, table)
        self._fs_root = Path(fs_root).resolve()
        self._fs_root.mkdir(parents=True, exist_ok=True)
        self._summarizer = summarizer
        self._embedder = embedder
        self._generation: GenerationMode = generation
        self._fallback_l2_chars = fallback_l2_chars
        self._l0_threshold = l0_threshold
        self._l1_threshold = l1_threshold
        self._pending: dict[str, asyncio.Task] = {}
        self._edit_lock = asyncio.Lock()
        self._initialized = False
        self._init_lock = asyncio.Lock()

    # ---------- lazy initialization ----------

    async def _ensure_ready(self) -> None:
        """Initialize DB table and fs_root on first use (idempotent)."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._dao.initialize()
            self._initialized = True

    # ---------- path helpers ----------

    def _local(self, path: str) -> Path:
        """Map a VFS path (``seekvfs://…``) to a local ``Path``."""
        return self._fs_root / path.removeprefix(_SCHEME)

    def _reconstruct(self, fp: Path) -> str:
        """Reconstruct the VFS path from a local filesystem ``Path``."""
        return _SCHEME + str(fp.relative_to(self._fs_root))

    # ---------- pending task bookkeeping ----------

    def _register(self, path: str, task: asyncio.Task) -> None:
        self._pending[path] = task

        def _cleanup(t: asyncio.Task) -> None:
            if self._pending.get(path) is t:
                self._pending.pop(path, None)

        task.add_done_callback(_cleanup)

    def _cancel_pending(self, path: str) -> bool:
        task = self._pending.get(path)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    # ---------- derivative generation ----------

    async def _generate_derivatives(
        self, content: bytes | str
    ) -> tuple[str, str, list[float]]:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        n = len(text)

        # Short content: skip LLM calls and use the text directly.
        # This avoids the LLM "expanding" already-brief notes into long documents.
        if n <= self._l0_threshold:
            l0 = text
        else:
            l0 = await self._summarizer.abstract(content)

        if n <= self._l1_threshold:
            l1 = text
        else:
            l1 = await self._summarizer.overview(content)

        emb = await self._embedder.embed(l0)
        return l0, l1, emb

    async def _async_body(self, path: str, raw: bytes) -> None:
        try:
            l0, l1, emb = await self._generate_derivatives(raw)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("async derivative generation failed for path=%r", path)
            return

        # Race-condition guard: verify FS content hasn't changed since we started.
        fp = self._local(path)
        try:
            current = await asyncio.to_thread(fp.read_bytes)
        except OSError:
            return  # file was deleted while we were generating
        if current != raw:
            return  # overwritten with newer content; a new task will take over

        await self._dao.update_derivatives(path, l0, l1, emb)

    def _schedule_async(self, path: str, raw: bytes) -> asyncio.Task:
        self._cancel_pending(path)
        task = asyncio.create_task(self._async_body(path, raw))
        self._register(path, task)
        return task

    # ---------- BackendProtocol ----------

    async def write(self, path: str, content: bytes | str) -> None:
        await self._ensure_ready()
        raw = _to_bytes(content)

        # 1. Persist L2 to the filesystem.
        fp = self._local(path)

        def _write_fs() -> None:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(raw)

        await asyncio.to_thread(_write_fs)

        # 2. Ensure DB row exists; mark any stale derivatives as NULL.
        await self._dao.upsert_init(path)

        # 3. Generate L0/L1/embedding sync or async.
        if self._generation == "sync":
            l0, l1, emb = await self._generate_derivatives(content)
            await self._dao.update_derivatives(path, l0, l1, emb)
        else:
            self._schedule_async(path, raw)

    async def read(self, path: str, hint: str | None = None) -> FileData:
        await self._ensure_ready()
        if hint == "l2":
            return await self.read_full(path)

        if hint == "l0":
            exists, val = await self._dao.get_l0(path)
            if not exists:
                raise NotFoundError(path)
            if val is None:
                raise TierNotAvailable(path)
            return FileData(val.encode("utf-8"), "utf-8")

        if hint == "l1":
            exists, val = await self._dao.get_l1(path)
            if not exists:
                raise NotFoundError(path)
            if val is None:
                raise TierNotAvailable(path)
            return FileData(val.encode("utf-8"), "utf-8")

        if hint is not None:
            raise BackendError(
                f"unknown hint {hint!r};"
                " OceanbaseFsBackend accepts None / 'l0' / 'l1' / 'l2'"
            )

        # hint is None → waterfall: L1 → L0 → truncated L2
        exists, l1_val, l0_val = await self._dao.get_l1_l0(path)
        if exists:
            if l1_val is not None:
                return FileData(l1_val.encode("utf-8"), "utf-8")
            if l0_val is not None:
                return FileData(l0_val.encode("utf-8"), "utf-8")

        # Fallback: truncated L2 from filesystem.
        fp = self._local(path)

        def _read_truncated() -> bytes:
            if not fp.exists():
                raise NotFoundError(path)
            return fp.read_bytes()[: self._fallback_l2_chars]

        return FileData(await asyncio.to_thread(_read_truncated), "utf-8")

    async def read_full(self, path: str) -> FileData:
        fp = self._local(path)

        def _read() -> bytes:
            if not fp.exists():
                raise NotFoundError(path)
            return fp.read_bytes()

        return FileData(await asyncio.to_thread(_read), "utf-8")

    async def read_batch(self, paths: list[str]) -> dict[str, FileData]:
        out: dict[str, FileData] = {}
        for p in paths:
            out[p] = await self.read(p)
        return out

    async def search(
        self,
        query: str,
        path_pattern: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> SearchResult:
        """Vector search via OceanBase; falls back to lexical if embedder fails."""
        await self._ensure_ready()
        query_emb: list[float] | None = None
        try:
            query_emb = await self._embedder.embed(query)
        except Exception:
            logger.exception(
                "embedder failed on query %r; falling back to lexical search", query
            )

        hits: list[SearchHit] = []
        searched: list[str] = []

        if query_emb is not None:
            # Convert glob path_pattern to SQL LIKE (basic: * → %, ? → _).
            path_like = (
                path_pattern.replace("*", "%").replace("?", "_")
                if path_pattern
                else None
            )
            rows = await self._dao.vector_search(
                query_emb, path_like, score_threshold, limit
            )
            for path, snippet, score in rows:
                searched.append(path)
                hits.append(SearchHit(path=path, snippet=snippet or "", score=score))
        else:
            # Lexical fallback: scan filesystem content.
            q_low = query.lower()

            def _scan() -> list[tuple[str, bytes]]:
                pairs: list[tuple[str, bytes]] = []
                for fp in self._fs_root.rglob("*"):
                    if not fp.is_file():
                        continue
                    vfs_path = self._reconstruct(fp)
                    if path_pattern and not fnmatch.fnmatch(vfs_path, path_pattern):
                        continue
                    try:
                        data = fp.read_bytes()
                    except OSError:
                        continue
                    pairs.append((vfs_path, data))
                return pairs

            for vfs_path, data in await asyncio.to_thread(_scan):
                searched.append(vfs_path)
                text = data.decode("utf-8", errors="replace")
                score = 1.0 if q_low and q_low in text.lower() else 0.0
                if score_threshold is not None and score < score_threshold:
                    continue
                if score <= 0:
                    continue
                hits.append(SearchHit(path=vfs_path, snippet="", score=score))

        return SearchResult(query=query, hits=hits[:limit], searched_paths=searched)

    async def ls(
        self,
        path: str,
        pattern: str | None = None,
        recursive: bool = False,
    ) -> list[FileInfo]:
        await self._ensure_ready()
        prefix = path if path.endswith("/") else path + "/"
        local_rel = prefix.removeprefix(_SCHEME).rstrip("/")
        local_dir = self._fs_root / local_rel if local_rel else self._fs_root

        def _scan_fs() -> list[tuple[str, int, datetime]]:
            out: list[tuple[str, int, datetime]] = []
            if not local_dir.exists():
                return out
            candidates = local_dir.rglob("*") if recursive else local_dir.iterdir()
            for fp in candidates:
                if not fp.is_file():
                    continue
                rest = str(fp.relative_to(local_dir))
                if pattern is not None and not fnmatch.fnmatch(rest, pattern):
                    continue
                stat = fp.stat()
                out.append(
                    (
                        prefix + rest,
                        stat.st_size,
                        datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                    )
                )
            return out

        file_tuples = await asyncio.to_thread(_scan_fs)
        if not file_tuples:
            return []

        # Enrich with L0 snippets from the database.
        snippets = await self._dao.batch_l0([t[0] for t in file_tuples])

        result = [
            FileInfo(
                path=p,
                size=sz,
                mtime=mt,
                is_dir=False,
                snippet=snippets.get(p),
            )
            for p, sz, mt in file_tuples
        ]
        result.sort(key=lambda fi: fi.path)
        return result

    async def edit(self, path: str, old: str, new: str) -> int:
        fp = self._local(path)
        async with self._edit_lock:

            def _edit_fs() -> tuple[int, bytes | None]:
                if not fp.exists():
                    raise NotFoundError(path)
                text = fp.read_bytes().decode("utf-8", errors="replace")
                count = text.count(old)
                if count == 0:
                    return 0, None
                new_raw = text.replace(old, new).encode("utf-8")
                fp.write_bytes(new_raw)
                return count, new_raw

            count, new_raw = await asyncio.to_thread(_edit_fs)

        if count == 0 or new_raw is None:
            return 0

        # Mark stale derivatives in the database.
        await self._dao.clear_derivatives(path)

        # Regenerate derivatives.
        if self._generation == "sync":
            l0, l1, emb = await self._generate_derivatives(new_raw)
            await self._dao.update_derivatives(path, l0, l1, emb)
        else:
            self._schedule_async(path, new_raw)

        return count

    async def grep(
        self,
        pattern: str,
        path_pattern: str | None = None,
    ) -> list[GrepMatch]:
        await self._ensure_ready()

        def _grep() -> list[GrepMatch]:
            out: list[GrepMatch] = []
            for fp in self._fs_root.rglob("*"):
                if not fp.is_file():
                    continue
                vfs_path = self._reconstruct(fp)
                if path_pattern and not fnmatch.fnmatch(vfs_path, path_pattern):
                    continue
                try:
                    text = fp.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                for idx, line in enumerate(text.splitlines(), start=1):
                    if pattern in line:
                        out.append(GrepMatch(path=vfs_path, line_number=idx, line=line))
            return out

        return await asyncio.to_thread(_grep)

    async def delete(self, path: str) -> None:
        await self._ensure_ready()
        self._cancel_pending(path)

        fp = self._local(path)

        def _delete_fs() -> None:
            if not fp.exists():
                raise NotFoundError(path)
            fp.unlink()
            parent = fp.parent
            while parent != self._fs_root:
                try:
                    parent.rmdir()
                    parent = parent.parent
                except OSError:
                    break

        await asyncio.to_thread(_delete_fs)
        await self._dao.delete(path)

    async def aclose(self) -> None:
        """Wait for all in-flight derivative generation tasks to complete."""
        pending = list(self._pending.values())
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


__all__ = ["OceanbaseFsBackend", "GenerationMode"]
