"""Protocol-level exception hierarchy for SeekVFS."""
from __future__ import annotations


class VFSError(Exception):
    """Base class for all VFS exceptions."""


class NotFoundError(VFSError):
    """Raised when a path does not exist in the target backend."""


class InvalidRouteConfig(VFSError):
    """Raised during VFS construction when a :class:`RouteConfig` is malformed."""


class BackendError(VFSError):
    """Wraps an error raised inside a concrete backend implementation."""


__all__ = [
    "VFSError",
    "NotFoundError",
    "InvalidRouteConfig",
    "BackendError",
]
