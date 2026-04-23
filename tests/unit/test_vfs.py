from __future__ import annotations

import pytest

from seekvfs.exceptions import InvalidRouteConfig, NotFoundError, VFSError
from seekvfs.vfs import VFS
from tests.conftest import _StubBackend


def test_invalid_route_key_rejected() -> None:
    with pytest.raises((InvalidRouteConfig, VFSError)):
        VFS(routes={"foo://": {"backend": _StubBackend()}})


# ── bare-path auto-normalization ──────────────────────────────────────────────

def test_bare_route_key_auto_normalized() -> None:
    """Route key without seekvfs:// scheme is accepted and auto-normalized."""
    vfs = VFS(routes={"notes/": {"backend": _StubBackend()}})
    vfs.write("notes/a.md", "hello")
    fd = vfs.read("notes/a.md")
    assert fd.content == b"hello"


def test_bare_path_write_read_roundtrip() -> None:
    """Bare paths in write/read are equivalent to their seekvfs:// versions."""
    be = _StubBackend()
    vfs = VFS(routes={"seekvfs://a/": {"backend": be}})
    vfs.write("a/x", "body")
    fd = vfs.read("a/x")
    assert fd.content == b"body"


def test_bare_and_full_paths_are_equivalent() -> None:
    """Writing with seekvfs:// and reading with bare path (and vice versa) works."""
    vfs = VFS(routes={"a/": {"backend": _StubBackend()}})
    vfs.write("seekvfs://a/x", "written with full path")
    fd = vfs.read("a/x")
    assert fd.content == b"written with full path"


def test_unknown_scheme_raises_vfs_error() -> None:
    """A path like 'http://...' should raise VFSError, not be silently mangled."""
    vfs = VFS(routes={"a/": {"backend": _StubBackend()}})
    with pytest.raises(VFSError):
        vfs.read("http://a/x")


def test_bare_path_read_batch() -> None:
    """read_batch accepts bare paths and returns normalized keys."""
    be = _StubBackend()
    vfs = VFS(routes={"a/": {"backend": be}})
    vfs.write("a/x", "cx")
    vfs.write("a/y", "cy")
    got = vfs.read_batch(["a/x", "a/y"])
    assert got["seekvfs://a/x"].content == b"cx"
    assert got["seekvfs://a/y"].content == b"cy"


def test_empty_routes_rejected() -> None:
    with pytest.raises(InvalidRouteConfig):
        VFS(routes={})


def test_missing_backend_rejected() -> None:
    with pytest.raises(InvalidRouteConfig):
        VFS(routes={"seekvfs://a/": {}})  # type: ignore[typeddict-item]


def test_write_and_read_roundtrip() -> None:
    vfs = VFS(
        routes={
            "seekvfs://a/": {"backend": _StubBackend()},
        }
    )
    vfs.write("seekvfs://a/x", "body")
    fd = vfs.read("seekvfs://a/x")
    assert fd.content == b"body"


def test_read_passes_hint_through() -> None:
    """Protocol-level: the hint is a pass-through string. The VFS must forward it."""

    observed: list[str | None] = []

    class HintCapturingBackend(_StubBackend):
        def read(self, path, hint=None):
            observed.append(hint)
            return super().read(path, hint=hint)

    be = HintCapturingBackend()
    vfs = VFS(routes={"seekvfs://a/": {"backend": be}})
    vfs.write("seekvfs://a/x", "body")
    vfs.read("seekvfs://a/x")
    vfs.read("seekvfs://a/x", hint="anything")
    assert observed == [None, "anything"]


def test_read_full_bypasses_hint() -> None:
    vfs = VFS(routes={"seekvfs://a/": {"backend": _StubBackend()}})
    vfs.write("seekvfs://a/x", "full body")
    fd = vfs.read_full("seekvfs://a/x")
    assert fd.content == b"full body"


def test_route_not_found() -> None:
    vfs = VFS(routes={"seekvfs://a/": {"backend": _StubBackend()}})
    with pytest.raises(NotFoundError):
        vfs.read("seekvfs://b/foo")


def test_search_fans_out_across_backends() -> None:
    be1 = _StubBackend()
    be2 = _StubBackend()
    vfs = VFS(
        routes={
            "seekvfs://x/": {"backend": be1},
            "seekvfs://y/": {"backend": be2},
        }
    )
    vfs.write("seekvfs://x/a", "alpha content")
    vfs.write("seekvfs://y/b", "alpha mirror")
    sr = vfs.search("alpha")
    assert len(sr.hits) == 2
    assert set(sr.searched_paths).issuperset({"seekvfs://x/", "seekvfs://y/"})


def test_edit_and_grep_end_to_end() -> None:
    vfs = VFS(routes={"seekvfs://n/": {"backend": _StubBackend()}})
    vfs.write("seekvfs://n/a", "foo bar foo")
    n = vfs.edit("seekvfs://n/a", "foo", "baz")
    assert n == 2
    matches = vfs.grep("baz")
    assert len(matches) == 1
    vfs.delete("seekvfs://n/a")
    with pytest.raises(NotFoundError):
        vfs.read("seekvfs://n/a")


def test_read_batch_across_backends() -> None:
    be1 = _StubBackend()
    be2 = _StubBackend()
    vfs = VFS(
        routes={
            "seekvfs://x/": {"backend": be1},
            "seekvfs://y/": {"backend": be2},
        }
    )
    vfs.write("seekvfs://x/a", "content_x")
    vfs.write("seekvfs://y/b", "content_y")
    got = vfs.read_batch(["seekvfs://x/a", "seekvfs://y/b"])
    assert got["seekvfs://x/a"].content == b"content_x"
    assert got["seekvfs://y/b"].content == b"content_y"


def test_iter_routes_sorted_longest_first() -> None:
    be1 = _StubBackend()
    be2 = _StubBackend()
    vfs = VFS(
        routes={
            "seekvfs://a/": {"backend": be1},
            "seekvfs://a/sub/": {"backend": be2},
        }
    )
    assert [p for p, _ in vfs.iter_routes()] == [
        "seekvfs://a/sub/",
        "seekvfs://a/",
    ]


def test_close_forwards_to_backends_and_dedupes_shared() -> None:
    calls: list[str] = []

    class TrackingBackend(_StubBackend):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def close(self) -> None:
            calls.append(self.name)

    shared = TrackingBackend("shared")
    alone = TrackingBackend("alone")
    vfs = VFS(
        routes={
            "seekvfs://x/": {"backend": shared},
            "seekvfs://y/": {"backend": shared},  # deliberate sharing
            "seekvfs://z/": {"backend": alone},
        }
    )
    vfs.close()
    assert sorted(calls) == ["alone", "shared"]


def test_close_tolerates_backend_without_close() -> None:
    class MinimalBackend:
        def write(self, path, content): pass
        def read(self, path, hint=None):
            from seekvfs.models import FileData
            return FileData(b"", "utf-8")
        def read_full(self, path):
            from seekvfs.models import FileData
            return FileData(b"", "utf-8")
        def search(self, q, **kw):
            from seekvfs.models import SearchResult
            return SearchResult(query=q)
        def ls(self, *a, **k): return []
        def edit(self, *a, **k): return 0
        def grep(self, *a, **k): return []
        def delete(self, *a, **k): pass
        def read_batch(self, paths): return {}
        # no close method at all

    vfs = VFS(routes={"seekvfs://a/": {"backend": MinimalBackend()}})
    vfs.close()


def test_context_manager() -> None:
    """VFS supports 'with' statement and calls close on exit."""
    calls: list[str] = []

    class TrackingBackend(_StubBackend):
        def close(self) -> None:
            calls.append("closed")

    with VFS(routes={"seekvfs://a/": {"backend": TrackingBackend()}}) as vfs:
        vfs.write("seekvfs://a/x", "hello")

    assert calls == ["closed"]
