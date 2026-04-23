from __future__ import annotations

from seekvfs.tools import build_tools
from seekvfs.vfs import VFS
from tests.conftest import _StubBackend


def _vfs() -> VFS:
    return VFS(routes={"seekvfs://mem/": {"backend": _StubBackend()}})


def test_eight_tools_present() -> None:
    specs = build_tools(_vfs())
    expected = {"search", "read", "read_full", "write", "edit", "ls", "grep", "delete"}
    assert set(specs.names()) == expected


def test_schema_structure() -> None:
    for s in build_tools(_vfs()).specs:
        assert s.parameters_schema["type"] == "object"
        assert "properties" in s.parameters_schema
        assert "required" in s.parameters_schema


def test_description_override_applied() -> None:
    vfs = _vfs()
    specs = build_tools(vfs).with_description_overrides({"search": "OVERRIDE"})
    assert specs.by_name("search").description == "OVERRIDE"
    assert build_tools(vfs).by_name("search").description != "OVERRIDE"


def test_write_read_roundtrip_via_tools() -> None:
    specs = build_tools(_vfs())
    specs.by_name("write").callable(path="seekvfs://mem/a", content="hello")
    out = specs.by_name("read").callable(path="seekvfs://mem/a")
    assert 'path="seekvfs://mem/a"' in out
    assert "hello" in out
    assert "returned_level" not in out


def test_search_tool_returns_shape() -> None:
    specs = build_tools(_vfs())
    specs.by_name("write").callable(path="seekvfs://mem/a", content="alpha beta")
    result = specs.by_name("search").callable(query="alpha", limit=3)
    assert "hits" in result
    assert "searched_paths" in result
    if result["hits"]:
        assert "snippet" in result["hits"][0]
        assert "level" not in result["hits"][0]


def test_ls_tool() -> None:
    specs = build_tools(_vfs())
    specs.by_name("write").callable(path="seekvfs://mem/a", content="x")
    rows = specs.by_name("ls").callable(path="seekvfs://mem/")
    assert any(r["path"] == "seekvfs://mem/a" for r in rows)
    assert all("snippet" in r for r in rows)
