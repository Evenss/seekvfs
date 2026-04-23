"""File-system backend: :class:`FileBackend`.

Stores one file per VFS path under a local directory tree. No summaries,
no embeddings — the simplest *persistent* backend that satisfies
:class:`seekvfs.BackendProtocol`.

Every VFS path is mapped to a local file by stripping the ``seekvfs://``
scheme prefix (if present) and appending to *root_dir*::

    seekvfs://notes/a.md  →  {root_dir}/notes/a.md
    notes/a.md            →  {root_dir}/notes/a.md

Scan results (``ls``, ``search``, ``grep``) preserve the scheme format of
the input path — callers that use ``seekvfs://`` URIs get URIs back,
callers that use plain relative paths get plain paths back.  When no path
context is available (e.g. ``search`` with no ``path_pattern``), the
``seekvfs://`` scheme is used by default, which is the correct behaviour
when called through :class:`seekvfs.VFS`.

Full usage / adaptation guide: ``docs/recipes/minimal.md``.
"""
from __future__ import annotations

import asyncio
import fnmatch
from datetime import UTC, datetime
from pathlib import Path

from seekvfs.exceptions import NotFoundError
from seekvfs.models import (
    FileData,
    FileInfo,
    GrepMatch,
    SearchHit,
    SearchResult,
)
from seekvfs.uri import SCHEME as _SCHEME


def _to_bytes(content: bytes | str) -> bytes:
    return content if isinstance(content, bytes) else content.encode("utf-8")


def _detect_scheme(*paths: str | None) -> str:
    """Return ``'seekvfs://'`` if the first non-None path uses it, else ``''``.

    Falls back to ``'seekvfs://'`` when all inputs are ``None`` — correct for
    the VFS-call scenario where no path context is available.
    """
    for p in paths:
        if p is not None:
            return _SCHEME if p.startswith(_SCHEME) else ""
    return _SCHEME


class FileBackend:
    """Minimal persistent backend: one file per VFS path on local disk.

    Args:
        root_dir: Root directory for all stored files. Created automatically
            if it does not already exist.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        # Used only for edit (read-modify-write atomicity).
        self._edit_lock = asyncio.Lock()

    # ---------- internal helpers ----------

    def _local(self, path: str) -> Path:
        """Map a VFS path (with or without scheme) to a local ``Path``."""
        rel = path.removeprefix(_SCHEME)
        return self._root / rel

    def _reconstruct(self, fp: Path, scheme: str) -> str:
        """Reconstruct the VFS path for a local file."""
        rel = str(fp.relative_to(self._root))
        return scheme + rel

    # ---------- writes ----------

    async def write(self, path: str, content: bytes | str) -> None:
        data = _to_bytes(content)
        fp = self._local(path)

        def _write() -> None:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(data)

        await asyncio.to_thread(_write)

    # ---------- reads ----------

    async def read(self, path: str, hint: str | None = None) -> FileData:
        """Read the stored content.

        ``hint`` is accepted and silently ignored — this backend stores
        only one representation per path.
        """
        fp = self._local(path)

        def _read() -> bytes:
            if not fp.exists():
                raise NotFoundError(path)
            return fp.read_bytes()

        data = await asyncio.to_thread(_read)
        return FileData(data, "utf-8")

    async def read_full(self, path: str) -> FileData:
        return await self.read(path)

    async def read_batch(self, paths: list[str]) -> dict[str, FileData]:
        out: dict[str, FileData] = {}
        for p in paths:
            out[p] = await self.read(p)
        return out

    # ---------- search ----------

    async def search(
        self,
        query: str,
        path_pattern: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> SearchResult:
        """Literal substring search across stored files.

        Scores are ``1.0`` for a match, ``0.0`` otherwise (no ranking).
        For vector / semantic search, use :mod:`seekvfs_recipes.maximal`
        with an embedder.
        """
        scheme = _detect_scheme(path_pattern)
        q_low = query.lower()

        def _scan() -> list[tuple[str, bytes]]:
            pairs: list[tuple[str, bytes]] = []
            for fp in self._root.rglob("*"):
                if not fp.is_file():
                    continue
                vfs_path = self._reconstruct(fp, scheme)
                if path_pattern is not None and not fnmatch.fnmatch(vfs_path, path_pattern):
                    continue
                try:
                    data = fp.read_bytes()
                except OSError:
                    continue
                pairs.append((vfs_path, data))
            return pairs

        files = await asyncio.to_thread(_scan)
        hits: list[SearchHit] = []
        searched: list[str] = []
        for vfs_path, data in files:
            searched.append(vfs_path)
            text = data.decode("utf-8", errors="replace")
            score = 1.0 if q_low and q_low in text.lower() else 0.0
            if score_threshold is not None and score < score_threshold:
                continue
            if score <= 0:
                continue
            hits.append(SearchHit(path=vfs_path, snippet="", score=score))
        return SearchResult(query=query, hits=hits[:limit], searched_paths=searched)

    # ---------- listing ----------

    async def ls(
        self,
        path: str,
        pattern: str | None = None,
        recursive: bool = False,
    ) -> list[FileInfo]:
        prefix = path if path.endswith("/") else path + "/"
        # Local directory: strip scheme and trailing slash
        local_rel = prefix.removeprefix(_SCHEME).rstrip("/")
        local_dir = (self._root / local_rel) if local_rel else self._root

        def _ls() -> list[FileInfo]:
            out: list[FileInfo] = []
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
                    FileInfo(
                        path=prefix + rest,
                        size=stat.st_size,
                        mtime=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                        is_dir=False,
                    )
                )
            out.sort(key=lambda fi: fi.path)
            return out

        return await asyncio.to_thread(_ls)

    # ---------- edit ----------

    async def edit(self, path: str, old: str, new: str) -> int:
        fp = self._local(path)
        async with self._edit_lock:

            def _edit() -> int:
                if not fp.exists():
                    raise NotFoundError(path)
                text = fp.read_bytes().decode("utf-8", errors="replace")
                count = text.count(old)
                if count == 0:
                    return 0
                fp.write_bytes(text.replace(old, new).encode("utf-8"))
                return count

            return await asyncio.to_thread(_edit)

    # ---------- grep ----------

    async def grep(
        self,
        pattern: str,
        path_pattern: str | None = None,
    ) -> list[GrepMatch]:
        scheme = _detect_scheme(path_pattern)

        def _grep() -> list[GrepMatch]:
            out: list[GrepMatch] = []
            for fp in self._root.rglob("*"):
                if not fp.is_file():
                    continue
                vfs_path = self._reconstruct(fp, scheme)
                if path_pattern is not None and not fnmatch.fnmatch(vfs_path, path_pattern):
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

    # ---------- delete ----------

    async def delete(self, path: str) -> None:
        fp = self._local(path)

        def _delete() -> None:
            if not fp.exists():
                raise NotFoundError(path)
            fp.unlink()
            # Remove empty parent directories up to (but not including) root.
            parent = fp.parent
            while parent != self._root:
                try:
                    parent.rmdir()
                    parent = parent.parent
                except OSError:
                    break

        await asyncio.to_thread(_delete)

    # ---------- lifecycle ----------

    async def aclose(self) -> None:
        # No background state — nothing to wait on.
        return None


__all__ = ["FileBackend"]
