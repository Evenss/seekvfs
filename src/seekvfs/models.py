"""Core data models for SeekVFS protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from seekvfs.protocol import BackendProtocol


EncodingT = Literal["utf-8", "base64"]


@dataclass
class FileData:
    """Data returned by ``read`` / ``read_full``.

    The protocol carries no notion of "which tier this is" — what the
    backend returns is what the backend returns. Callers who need to
    distinguish views can use ``read_full`` (guaranteed original content)
    vs ``read`` (backend's preferred representation, possibly derived).
    """

    content: bytes
    encoding: EncodingT = "utf-8"


@dataclass
class FileInfo:
    path: str
    size: int
    mtime: datetime
    is_dir: bool
    snippet: str | None = None  # optional preview; backend-defined content


@dataclass
class SearchHit:
    path: str
    snippet: str = ""  # backend-defined short preview; may be empty
    score: float = 0.0


@dataclass
class SearchResult:
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    searched_paths: list[str] = field(default_factory=list)


@dataclass
class GrepMatch:
    path: str
    line_number: int
    line: str


class RouteConfig(TypedDict, total=False):
    """Per-prefix route configuration.

    Required: ``backend``. The protocol core exposes no other knobs —
    anything tier-/derivative-related is a backend implementation detail
    (see e.g. :mod:`seekvfs_recipes.maximal`).
    """

    backend: BackendProtocol


__all__ = [
    "EncodingT",
    "FileData",
    "FileInfo",
    "SearchHit",
    "SearchResult",
    "GrepMatch",
    "RouteConfig",
]
