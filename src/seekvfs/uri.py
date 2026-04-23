"""URI parsing for SeekVFS.

The protocol is intentionally lenient: it only verifies the scheme,
extracts the path, and reports directory-ness via trailing slash. Case
is preserved; percent-encoding / character validation is delegated to
the concrete backend.
"""
from __future__ import annotations

from seekvfs.exceptions import VFSError

SCHEME = "seekvfs://"


def parse_uri(s: str) -> str:
    """Return the path portion of an ``seekvfs://`` URI.

    Raises :class:`VFSError` if ``s`` does not start with the scheme.
    The returned path preserves the input's case and any trailing slash.
    """
    if not isinstance(s, str):
        raise VFSError(f"uri must be a string, got {type(s).__name__}")
    if not s.startswith(SCHEME):
        raise VFSError(f"uri must start with {SCHEME!r}, got {s!r}")
    return s[len(SCHEME) :]


def is_dir_uri(s: str) -> bool:
    """Return True when the URI represents a directory (trailing slash)."""
    path = parse_uri(s)
    return path.endswith("/")


def with_scheme(path: str) -> str:
    """Attach the scheme to a bare path. Idempotent."""
    if path.startswith(SCHEME):
        return path
    return SCHEME + path


__all__ = ["SCHEME", "parse_uri", "is_dir_uri", "with_scheme"]
