from __future__ import annotations

import asyncio
import fnmatch
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from seekvfs.exceptions import NotFoundError
from seekvfs.models import FileData, FileInfo, GrepMatch, SearchHit, SearchResult


def _approx(a: Iterable[float], b: Iterable[float], tol: float = 1e-6) -> bool:
    return all(abs(x - y) < tol for x, y in zip(a, b, strict=False))


class _StubBackend:
    """Minimal dict-backed backend for unit tests only.

    NOT a recipe — not exported from seekvfs_recipes. Used as a zero-setup
    backend stub in protocol-layer unit tests where the goal is to test
    VFS routing / tool export, not the backend itself.
    """

    def __init__(self) -> None:
        self._files: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def write(self, path: str, content: bytes | str) -> None:
        data = content if isinstance(content, bytes) else content.encode("utf-8")
        async with self._lock:
            self._files[path] = {"content": data, "mtime": datetime.now(tz=UTC)}

    async def read(self, path: str, hint: str | None = None) -> FileData:
        async with self._lock:
            entry = self._files.get(path)
            if entry is None:
                raise NotFoundError(path)
            return FileData(entry["content"], "utf-8")

    async def read_full(self, path: str) -> FileData:
        return await self.read(path)

    async def read_batch(self, paths: list[str]) -> dict[str, FileData]:
        return {p: await self.read(p) for p in paths}

    async def search(
        self,
        query: str,
        path_pattern: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> SearchResult:
        hits: list[SearchHit] = []
        searched: list[str] = []
        q_low = query.lower()
        async with self._lock:
            for path, entry in self._files.items():
                if path_pattern is not None and not fnmatch.fnmatch(path, path_pattern):
                    continue
                searched.append(path)
                text = entry["content"].decode("utf-8", errors="replace")
                score = 1.0 if q_low and q_low in text.lower() else 0.0
                if score_threshold is not None and score < score_threshold:
                    continue
                if score <= 0:
                    continue
                hits.append(SearchHit(path=path, snippet="", score=score))
        return SearchResult(query=query, hits=hits[:limit], searched_paths=searched)

    async def ls(
        self,
        path: str,
        pattern: str | None = None,
        recursive: bool = False,
    ) -> list[FileInfo]:
        prefix = path if path.endswith("/") else path + "/"
        out: list[FileInfo] = []
        async with self._lock:
            for p, entry in self._files.items():
                if not p.startswith(prefix):
                    continue
                rest = p[len(prefix):]
                if not recursive and "/" in rest:
                    continue
                if pattern is not None and not fnmatch.fnmatch(rest, pattern):
                    continue
                out.append(
                    FileInfo(
                        path=p,
                        size=len(entry["content"]),
                        mtime=entry.get("mtime", datetime.now(tz=UTC)),
                        is_dir=False,
                    )
                )
        out.sort(key=lambda fi: fi.path)
        return out

    async def edit(self, path: str, old: str, new: str) -> int:
        async with self._lock:
            entry = self._files.get(path)
            if entry is None:
                raise NotFoundError(path)
            text = entry["content"].decode("utf-8", errors="replace")
            count = text.count(old)
            if count == 0:
                return 0
            entry["content"] = text.replace(old, new).encode("utf-8")
            entry["mtime"] = datetime.now(tz=UTC)
            return count

    async def grep(
        self,
        pattern: str,
        path_pattern: str | None = None,
    ) -> list[GrepMatch]:
        out: list[GrepMatch] = []
        async with self._lock:
            for p, entry in self._files.items():
                if path_pattern is not None and not fnmatch.fnmatch(p, path_pattern):
                    continue
                text = entry["content"].decode("utf-8", errors="replace")
                for idx, line in enumerate(text.splitlines(), start=1):
                    if pattern in line:
                        out.append(GrepMatch(path=p, line_number=idx, line=line))
        return out

    async def delete(self, path: str) -> None:
        async with self._lock:
            if path not in self._files:
                raise NotFoundError(path)
            del self._files[path]

    async def aclose(self) -> None:
        return None


__all__ = ["_approx", "_StubBackend"]
