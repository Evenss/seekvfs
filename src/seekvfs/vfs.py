"""Top-level VFS facade.

Protocol-layer responsibilities only:

- URI parsing + longest-prefix routing
- Cross-backend search fan-out + reranker merge
- Tool export
- Lifecycle forwarding (``aclose``)

No tier concepts, no derivative scheduling, no summarizer / embedder
injection — those all live inside backend implementations (see e.g.
:mod:`seekvfs_recipes.maximal`).
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from seekvfs.exceptions import InvalidRouteConfig, VFSError
from seekvfs.models import (
    FileData,
    FileInfo,
    GrepMatch,
    RouteConfig,
    SearchResult,
)
from seekvfs.observability import trace_span
from seekvfs.reranker import LinearReranker
from seekvfs.router import Router
from seekvfs.uri import SCHEME

if TYPE_CHECKING:
    from seekvfs.protocol import Reranker
    from seekvfs.tools import ToolSpecSet


class VFS:
    """Unified front door for SeekVFS storage protocol."""

    def __init__(
        self,
        routes: dict[str, RouteConfig],
        reranker: Reranker | None = None,
    ) -> None:
        normalized = {self._normalize(k): v for k, v in routes.items()}
        self._validate_routes(normalized)
        self._routes = normalized
        self._router = Router(normalized)
        self._reranker: Reranker = reranker or LinearReranker()

    @staticmethod
    def _normalize(path: str) -> str:
        """Ensure *path* carries the ``seekvfs://`` scheme.

        * Already starts with ``seekvfs://`` → returned as-is (idempotent).
        * Bare path with no ``://`` → ``seekvfs://`` is prepended automatically.
        * Contains ``://`` but not ``seekvfs://`` → raises :class:`VFSError`.
        """
        if path.startswith(SCHEME):
            return path
        if "://" in path:
            raise VFSError(
                f"path uses an unknown scheme; expected {SCHEME!r}, got {path!r}"
            )
        return SCHEME + path

    @staticmethod
    def _validate_routes(routes: dict[str, RouteConfig]) -> None:
        if not routes:
            raise InvalidRouteConfig("routes must be a non-empty dict")
        for key, cfg in routes.items():
            if not key.startswith(SCHEME):
                raise InvalidRouteConfig(
                    f"route key must start with {SCHEME!r}, got {key!r}"
                )
            if "backend" not in cfg:
                raise InvalidRouteConfig(f"route {key!r} missing 'backend'")

    # ---------- main API ----------

    @trace_span("vfs.write")
    async def write(self, path: str, content: bytes | str) -> None:
        path = self._normalize(path)
        _, route = self._router.resolve(path)
        await route["backend"].write(path, content)

    @trace_span("vfs.read")
    async def read(self, path: str, hint: str | None = None) -> FileData:
        path = self._normalize(path)
        _, route = self._router.resolve(path)
        return await route["backend"].read(path, hint=hint)

    @trace_span("vfs.read_full")
    async def read_full(self, path: str) -> FileData:
        path = self._normalize(path)
        _, route = self._router.resolve(path)
        return await route["backend"].read_full(path)

    @trace_span("vfs.search")
    async def search(
        self,
        query: str,
        path_pattern: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> SearchResult:
        """Fan out across every route in parallel, then rerank."""
        routes = self._router.all_routes()
        if not routes:
            return SearchResult(query=query, hits=[], searched_paths=[])

        async def _one(prefix: str, route: RouteConfig) -> SearchResult:
            out = await route["backend"].search(
                query,
                path_pattern=path_pattern,
                limit=limit,
                score_threshold=score_threshold,
            )
            if prefix not in out.searched_paths:
                out.searched_paths.append(prefix)
            return out

        per_backend = await asyncio.gather(
            *(_one(prefix, route) for prefix, route in routes)
        )
        return self._reranker.merge(list(per_backend), limit=limit)

    @trace_span("vfs.ls")
    async def ls(
        self,
        path: str,
        pattern: str | None = None,
        recursive: bool = False,
    ) -> list[FileInfo]:
        path = self._normalize(path)
        _, route = self._router.resolve(path)
        return await route["backend"].ls(path, pattern=pattern, recursive=recursive)

    @trace_span("vfs.edit")
    async def edit(self, path: str, old: str, new: str) -> int:
        path = self._normalize(path)
        _, route = self._router.resolve(path)
        return await route["backend"].edit(path, old, new)

    @trace_span("vfs.grep")
    async def grep(
        self,
        pattern: str,
        path_pattern: str | None = None,
    ) -> list[GrepMatch]:
        results: list[GrepMatch] = []
        for _, route in self._router.all_routes():
            backend = route["backend"]
            results.extend(await backend.grep(pattern, path_pattern=path_pattern))
        return results

    @trace_span("vfs.delete")
    async def delete(self, path: str) -> None:
        path = self._normalize(path)
        _, route = self._router.resolve(path)
        await route["backend"].delete(path)

    async def read_batch(self, paths: list[str]) -> dict[str, FileData]:
        # group paths by backend to minimize calls
        by_backend: dict[int, tuple[object, list[str]]] = {}
        for p in paths:
            p = self._normalize(p)
            _, route = self._router.resolve(p)
            b = route["backend"]
            key = id(b)
            by_backend.setdefault(key, (b, []))[1].append(p)

        out: dict[str, FileData] = {}
        for _, (backend, sub_paths) in by_backend.items():
            partial = await backend.read_batch(sub_paths)  # type: ignore[attr-defined]
            out.update(partial)
        return out

    # ---------- introspection ----------

    def iter_routes(self) -> list[tuple[str, RouteConfig]]:
        """Return ``(prefix, RouteConfig)`` pairs, sorted by prefix length
        descending (same order used by longest-prefix resolution).
        """
        return self._router.all_routes()

    # ---------- tools ----------

    @property
    def tools(self) -> ToolSpecSet:
        from seekvfs.tools import build_tools

        return build_tools(self)

    # ---------- lifecycle ----------

    async def aclose(self) -> None:
        """Forward to every backend's ``aclose`` (if the backend exposes one).

        Safe to call even when no backend has background state; backends
        with no resources simply return immediately.
        """
        # Dedup by backend identity — two routes may share a backend instance.
        seen: set[int] = set()
        for _, route in self._router.all_routes():
            backend = route["backend"]
            if id(backend) in seen:
                continue
            seen.add(id(backend))
            aclose = getattr(backend, "aclose", None)
            if aclose is None:
                continue
            await aclose()


__all__ = ["VFS"]
