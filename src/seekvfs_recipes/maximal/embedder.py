"""LangChain-based Embedder for the maximal recipe.

Depends on ``langchain-core``. The underlying embedding model is supplied by
the caller as any ``Embeddings`` instance — swap providers (OpenAI, Cohere,
HuggingFace, Bedrock, Ollama, …) by simply passing a different object.

Usage example::

    from langchain_openai import OpenAIEmbeddings
    from langchain_community.embeddings import OllamaEmbeddings

    # OpenAI
    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key="sk-...")
    # — or Ollama (local) —
    emb = OllamaEmbeddings(model="nomic-embed-text")

    embedder = LangChainEmbedder(embeddings=emb)
    vector = embedder.embed("some text")
"""
from __future__ import annotations


class LangChainEmbedder:
    """Embedder backed by any LangChain ``Embeddings`` implementation.

    Parameters
    ----------
    embeddings:
        A LangChain embeddings instance (e.g. ``OpenAIEmbeddings``,
        ``BedrockEmbeddings``).  The object must support ``embed_query``
        (all LangChain embeddings implementations do).
    """

    def __init__(self, *, embeddings: object) -> None:
        self._embeddings = embeddings

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*."""
        result = self._embeddings.embed_query(text)  # type: ignore[attr-defined]
        return list(result)


__all__ = ["LangChainEmbedder"]
