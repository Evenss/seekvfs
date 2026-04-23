from __future__ import annotations

from seekvfs.tools import build_tools
from seekvfs.tools.openai import to_openai
from seekvfs.vfs import VFS
from tests.conftest import _StubBackend


def test_openai_format() -> None:
    vfs = VFS(routes={"seekvfs://mem/": {"backend": _StubBackend()}})
    openai_tools = to_openai(build_tools(vfs))
    assert len(openai_tools) == 8
    for t in openai_tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        assert fn["parameters"]["type"] == "object"
