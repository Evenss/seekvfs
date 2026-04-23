from __future__ import annotations

from types import SimpleNamespace

import pytest

from seekvfs_recipes.maximal.summarizer import LangChainSummarizer


class _FakeLLM:
    """Minimal stub that mimics LangChain BaseChatModel.ainvoke."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages: list | None = None

    async def ainvoke(self, messages):
        self.last_messages = messages
        return SimpleNamespace(content=self.reply)


def _make(llm, *, ab="ABSTRACT_SYS", ov="OVERVIEW_SYS"):
    return LangChainSummarizer(llm=llm, abstract_prompt=ab, overview_prompt=ov)


@pytest.mark.asyncio
async def test_abstract_uses_abstract_prompt_and_returns_text() -> None:
    llm = _FakeLLM("SHORT SUMMARY")
    s = _make(llm, ab="CUSTOM_SYS")
    out = await s.abstract("the content")
    assert out == "SHORT SUMMARY"
    assert llm.last_messages is not None
    sys_msg, human_msg = llm.last_messages
    assert sys_msg.content == "CUSTOM_SYS"
    assert human_msg.content == "the content"


@pytest.mark.asyncio
async def test_overview_uses_overview_prompt() -> None:
    llm = _FakeLLM("LONG OVERVIEW")
    s = _make(llm, ov="OV_SYS")
    out = await s.overview(b"bytes here")
    assert out == "LONG OVERVIEW"
    sys_msg = llm.last_messages[0]
    assert sys_msg.content == "OV_SYS"


@pytest.mark.asyncio
async def test_abstract_decodes_bytes() -> None:
    llm = _FakeLLM("ok")
    s = _make(llm)
    await s.abstract(b"\xe4\xb8\xad\xe6\x96\x87")  # UTF-8 Chinese
    assert llm.last_messages[1].content == "中文"


def test_missing_prompts_raise() -> None:
    llm = _FakeLLM("x")
    with pytest.raises(ValueError, match="abstract_prompt"):
        LangChainSummarizer(llm=llm, abstract_prompt="", overview_prompt="b")
    with pytest.raises(ValueError, match="overview_prompt"):
        LangChainSummarizer(llm=llm, abstract_prompt="a", overview_prompt="")
