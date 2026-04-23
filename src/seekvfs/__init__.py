"""SeekVFS — protocol-level unified storage for AI agents.

The installed distribution version is available via
``importlib.metadata.version("seekvfs")``; this package does not expose a
runtime ``__version__`` constant.
"""
from __future__ import annotations

from seekvfs.exceptions import (
    BackendError,
    InvalidRouteConfig,
    NotFoundError,
    VFSError,
)
from seekvfs.models import (
    FileData,
    FileInfo,
    GrepMatch,
    RouteConfig,
    SearchHit,
    SearchResult,
)
from seekvfs.protocol import BackendProtocol, Reranker
from seekvfs.reranker import LinearReranker
from seekvfs.uri import SCHEME, parse_uri
from seekvfs.vfs import VFS

__all__ = [
    "VFS",
    "RouteConfig",
    "SCHEME",
    "parse_uri",
    # data
    "FileData",
    "FileInfo",
    "GrepMatch",
    "SearchHit",
    "SearchResult",
    # protocols
    "BackendProtocol",
    "Reranker",
    # rerankers
    "LinearReranker",
    # exceptions
    "VFSError",
    "BackendError",
    "InvalidRouteConfig",
    "NotFoundError",
]
