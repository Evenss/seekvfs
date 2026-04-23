from __future__ import annotations

import pytest

from seekvfs.tools import build_tools
from seekvfs.vfs import VFS
from tests.conftest import _StubBackend


@pytest.fixture
def vfs() -> VFS:
    return VFS(routes={"seekvfs://mem/": {"backend": _StubBackend()}})


def test_eight_tools_present(vfs: VFS) -> None:
    specs = build_tools(vfs)
    expected = {"search", "read", "read_full", "write", "edit", "ls", "grep", "delete"}
    assert set(specs.names()) == expected


def test_schema_structure(vfs: VFS) -> None:
    for s in build_tools(vfs).specs:
        assert s.parameters_schema["type"] == "object"
        assert "properties" in s.parameters_schema
        assert "required" in s.parameters_schema


def test_description_override_applied(vfs: VFS) -> None:
    specs = build_tools(vfs).with_description_overrides({"search": "OVERRIDE"})
    assert specs.by_name("search").description == "OVERRIDE"
    # original unchanged
    assert build_tools(vfs).by_name("search").description != "OVERRIDE"


async def test_write_read_roundtrip_via_tools(vfs: VFS) -> None:
    specs = build_tools(vfs)
    await specs.by_name("write").callable(path="seekvfs://mem/a", content="hello")
    out = await specs.by_name("read").callable(path="seekvfs://mem/a")
    assert 'path="seekvfs://mem/a"' in out
    assert "hello" in out
    # The protocol no longer exposes tier metadata — ensure the old
    # attribute is not leaking back into the rendered tag.
    assert "returned_level" not in out


async def test_search_tool_returns_shape(vfs: VFS) -> None:
    specs = build_tools(vfs)
    await specs.by_name("write").callable(path="seekvfs://mem/a", content="alpha beta")
    result = await specs.by_name("search").callable(query="alpha", limit=3)
    assert "hits" in result
    assert "searched_paths" in result
    if result["hits"]:
        assert "snippet" in result["hits"][0]
        assert "level" not in result["hits"][0]


async def test_ls_tool(vfs: VFS) -> None:
    specs = build_tools(vfs)
    await specs.by_name("write").callable(path="seekvfs://mem/a", content="x")
    rows = await specs.by_name("ls").callable(path="seekvfs://mem/")
    assert any(r["path"] == "seekvfs://mem/a" for r in rows)
    assert all("snippet" in r for r in rows)
