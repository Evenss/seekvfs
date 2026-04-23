"""Recipe-local protocols for the tiered backend.

These were part of the protocol core in earlier versions; they are now
scoped to this recipe because the notion of "summary" is a recipe
concern, not a protocol concern.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Summarizer(Protocol):
    """Produces L0 (short abstract) and L1 (overview) from raw content."""

    def abstract(self, content: bytes | str) -> str: ...

    def overview(self, content: bytes | str) -> str: ...


@runtime_checkable
class Embedder(Protocol):
    """Turns a string into a dense vector."""

    def embed(self, text: str) -> list[float]: ...


__all__ = ["Summarizer", "Embedder"]
