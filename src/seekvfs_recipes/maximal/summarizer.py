"""LangChain-based Summarizer for the maximal recipe.

Depends on ``langchain-core``. The underlying LLM is supplied by the caller
as any ``BaseChatModel`` instance — swap providers (OpenAI, Anthropic, Bedrock,
Ollama, …) by simply passing a different object; no code change required.

Usage example::

    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    # Anthropic
    llm = ChatAnthropic(model="claude-opus-4-5", api_key="sk-ant-...")
    # — or OpenAI —
    llm = ChatOpenAI(model="gpt-4o", api_key="sk-...")

    summarizer = LangChainSummarizer(
        llm=llm,
        abstract_prompt="Return a one-sentence abstract of the document.",
        overview_prompt="Summarise the document in 3-5 bullet points.",
    )

    abstract = summarizer.abstract(b"... file content ...")
    overview = summarizer.overview(b"... file content ...")
"""
from __future__ import annotations


def _as_text(content: bytes | str) -> str:
    return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content


class LangChainSummarizer:
    """Summarizer backed by any LangChain ``BaseChatModel``.

    Parameters
    ----------
    llm:
        A LangChain chat model instance (e.g. ``ChatOpenAI``, ``ChatAnthropic``).
        The object must support ``invoke`` (all LangChain chat models do).
    abstract_prompt:
        System prompt used when generating the short abstract (L0, ~100 tokens).
    overview_prompt:
        System prompt used when generating the fuller overview (L1, ~2 k tokens).
    """

    def __init__(
        self,
        *,
        llm: object,
        abstract_prompt: str,
        overview_prompt: str,
    ) -> None:
        if not abstract_prompt:
            raise ValueError("LangChainSummarizer requires a non-empty 'abstract_prompt'")
        if not overview_prompt:
            raise ValueError("LangChainSummarizer requires a non-empty 'overview_prompt'")
        self._llm = llm
        self.abstract_prompt = abstract_prompt
        self.overview_prompt = overview_prompt

    def _call(self, system_prompt: str, text: str) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LangChainSummarizer requires 'langchain-core'. "
                "Install with: pip install langchain-core"
            ) from exc
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]
        response = self._llm.invoke(messages)  # type: ignore[attr-defined]
        return str(response.content).strip()

    def abstract(self, content: bytes | str) -> str:
        """Generate a short abstract (L0)."""
        return self._call(self.abstract_prompt, _as_text(content))

    def overview(self, content: bytes | str) -> str:
        """Generate a fuller overview (L1)."""
        return self._call(self.overview_prompt, _as_text(content))


__all__ = ["LangChainSummarizer"]
