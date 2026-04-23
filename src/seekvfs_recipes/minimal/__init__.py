"""Minimal recipe — the simplest persistent built-in backend.

Stores one file per VFS path under a local directory tree. No summaries,
no embeddings, no background work. Use this when you need durable storage
without the overhead of a database or vector index.

Full guide: ``docs/recipes/minimal.md``.
"""
from __future__ import annotations

from seekvfs_recipes.minimal.backend import FileBackend

__all__ = ["FileBackend"]
