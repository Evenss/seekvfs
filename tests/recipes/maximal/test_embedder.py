from __future__ import annotations

import pytest

from seekvfs_recipes.maximal.embedder import LangChainEmbedder


class _FakeEmbeddings:
    """Minimal stub that mimics LangChain Embeddings.aembed_query."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.last_text: str | None = None

    async def aembed_query(self, text: str) -> list[float]:
        self.last_text = text
        return self.vector


@pytest.mark.asyncio
async def test_embed_returns_vector() -> None:
    emb = _FakeEmbeddings([0.1, 0.2, 0.3])
    e = LangChainEmbedder(embeddings=emb)
    out = await e.embed("hello")
    assert out == [0.1, 0.2, 0.3]
    assert emb.last_text == "hello"


@pytest.mark.asyncio
async def test_embed_returns_list_type() -> None:
    emb = _FakeEmbeddings([1.0, 2.0])
    e = LangChainEmbedder(embeddings=emb)
    out = await e.embed("test")
    assert isinstance(out, list)


@pytest.mark.asyncio
async def test_embed_passes_text_verbatim() -> None:
    emb = _FakeEmbeddings([])
    e = LangChainEmbedder(embeddings=emb)
    await e.embed("some input text")
    assert emb.last_text == "some input text"
