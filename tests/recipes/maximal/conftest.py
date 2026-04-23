from __future__ import annotations

import math

import pytest


class DummySummarizer:
    """Deterministic summarizer for tiered-recipe tests."""

    def __init__(self) -> None:
        self.abstract_calls = 0
        self.overview_calls = 0

    async def abstract(self, content: bytes | str) -> str:
        self.abstract_calls += 1
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        return "L0:" + text[:40]

    async def overview(self, content: bytes | str) -> str:
        self.overview_calls += 1
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        return "L1:" + text[:200]


class DummyEmbedder:
    """Deterministic hash-based embedder for tiered-recipe tests."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        vec = [0.0] * self.dim
        for i, ch in enumerate(text):
            vec[i % self.dim] += (ord(ch) % 97) / 97.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@pytest.fixture
def dummy_summarizer() -> DummySummarizer:
    return DummySummarizer()


@pytest.fixture
def dummy_embedder() -> DummyEmbedder:
    return DummyEmbedder()


__all__ = ["DummySummarizer", "DummyEmbedder"]
