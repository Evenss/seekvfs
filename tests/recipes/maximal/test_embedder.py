from __future__ import annotations

from seekvfs_recipes.maximal.embedder import LangChainEmbedder


class _FakeEmbeddings:
    """Minimal stub that mimics LangChain Embeddings.embed_query."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.last_text: str | None = None

    def embed_query(self, text: str) -> list[float]:
        self.last_text = text
        return self.vector


def test_embed_returns_vector() -> None:
    emb = _FakeEmbeddings([0.1, 0.2, 0.3])
    e = LangChainEmbedder(embeddings=emb)
    out = e.embed("hello")
    assert out == [0.1, 0.2, 0.3]
    assert emb.last_text == "hello"


def test_embed_returns_list_type() -> None:
    emb = _FakeEmbeddings([1.0, 2.0])
    e = LangChainEmbedder(embeddings=emb)
    out = e.embed("test")
    assert isinstance(out, list)


def test_embed_passes_text_verbatim() -> None:
    emb = _FakeEmbeddings([])
    e = LangChainEmbedder(embeddings=emb)
    e.embed("some input text")
    assert emb.last_text == "some input text"
