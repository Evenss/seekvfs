from __future__ import annotations

from seekvfs.tools import build_tools
from seekvfs.tools.anthropic import to_anthropic
from seekvfs.vfs import VFS
from tests.conftest import _StubBackend


def test_anthropic_format() -> None:
    vfs = VFS(routes={"seekvfs://mem/": {"backend": _StubBackend()}})
    tools = to_anthropic(build_tools(vfs))
    assert len(tools) == 8
    for t in tools:
        assert set(t.keys()) == {"name", "description", "input_schema"}
        assert t["input_schema"]["type"] == "object"
