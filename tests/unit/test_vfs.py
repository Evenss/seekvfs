from __future__ import annotations

import pytest

from seekvfs.exceptions import InvalidRouteConfig, NotFoundError
from seekvfs.vfs import VFS
from tests.conftest import _StubBackend


async def test_invalid_route_key_rejected() -> None:
    with pytest.raises(InvalidRouteConfig):
        VFS(routes={"foo://": {"backend": _StubBackend()}})


async def test_empty_routes_rejected() -> None:
    with pytest.raises(InvalidRouteConfig):
        VFS(routes={})


async def test_missing_backend_rejected() -> None:
    with pytest.raises(InvalidRouteConfig):
        VFS(routes={"seekvfs://a/": {}})  # type: ignore[typeddict-item]


async def test_write_and_read_roundtrip() -> None:
    vfs = VFS(
        routes={
            "seekvfs://a/": {"backend": _StubBackend()},
        }
    )
    await vfs.write("seekvfs://a/x", "body")
    fd = await vfs.read("seekvfs://a/x")
    assert fd.content == b"body"


async def test_read_passes_hint_through() -> None:
    """Protocol-level: the hint is a pass-through string. The VFS must forward it."""

    observed: list[str | None] = []

    class HintCapturingBackend(_StubBackend):
        async def read(self, path, hint=None):
            observed.append(hint)
            return await super().read(path, hint=hint)

    be = HintCapturingBackend()
    vfs = VFS(routes={"seekvfs://a/": {"backend": be}})
    await vfs.write("seekvfs://a/x", "body")
    await vfs.read("seekvfs://a/x")
    await vfs.read("seekvfs://a/x", hint="anything")
    assert observed == [None, "anything"]


async def test_read_full_bypasses_hint() -> None:
    vfs = VFS(routes={"seekvfs://a/": {"backend": _StubBackend()}})
    await vfs.write("seekvfs://a/x", "full body")
    fd = await vfs.read_full("seekvfs://a/x")
    assert fd.content == b"full body"


async def test_route_not_found() -> None:
    vfs = VFS(routes={"seekvfs://a/": {"backend": _StubBackend()}})
    with pytest.raises(NotFoundError):
        await vfs.read("seekvfs://b/foo")


async def test_search_fans_out_across_backends() -> None:
    be1 = _StubBackend()
    be2 = _StubBackend()
    vfs = VFS(
        routes={
            "seekvfs://x/": {"backend": be1},
            "seekvfs://y/": {"backend": be2},
        }
    )
    await vfs.write("seekvfs://x/a", "alpha content")
    await vfs.write("seekvfs://y/b", "alpha mirror")
    sr = await vfs.search("alpha")
    assert len(sr.hits) == 2
    assert set(sr.searched_paths).issuperset({"seekvfs://x/", "seekvfs://y/"})


async def test_edit_and_grep_end_to_end() -> None:
    vfs = VFS(routes={"seekvfs://n/": {"backend": _StubBackend()}})
    await vfs.write("seekvfs://n/a", "foo bar foo")
    n = await vfs.edit("seekvfs://n/a", "foo", "baz")
    assert n == 2
    matches = await vfs.grep("baz")
    assert len(matches) == 1
    await vfs.delete("seekvfs://n/a")
    with pytest.raises(NotFoundError):
        await vfs.read("seekvfs://n/a")


async def test_read_batch_across_backends() -> None:
    be1 = _StubBackend()
    be2 = _StubBackend()
    vfs = VFS(
        routes={
            "seekvfs://x/": {"backend": be1},
            "seekvfs://y/": {"backend": be2},
        }
    )
    await vfs.write("seekvfs://x/a", "content_x")
    await vfs.write("seekvfs://y/b", "content_y")
    got = await vfs.read_batch(["seekvfs://x/a", "seekvfs://y/b"])
    assert got["seekvfs://x/a"].content == b"content_x"
    assert got["seekvfs://y/b"].content == b"content_y"


async def test_iter_routes_sorted_longest_first() -> None:
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


async def test_aclose_forwards_to_backends_and_dedupes_shared() -> None:
    calls: list[str] = []

    class TrackingBackend(_StubBackend):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        async def aclose(self) -> None:
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
    await vfs.aclose()
    assert sorted(calls) == ["alone", "shared"]


async def test_aclose_tolerates_backend_without_aclose() -> None:
    class MinimalBackend:
        async def write(self, path, content): pass
        async def read(self, path, hint=None):
            from seekvfs.models import FileData
            return FileData(b"", "utf-8")
        async def read_full(self, path):
            from seekvfs.models import FileData
            return FileData(b"", "utf-8")
        async def search(self, q, **kw):
            from seekvfs.models import SearchResult
            return SearchResult(query=q)
        async def ls(self, *a, **k): return []
        async def edit(self, *a, **k): return 0
        async def grep(self, *a, **k): return []
        async def delete(self, *a, **k): pass
        async def read_batch(self, paths): return {}
        # no aclose method at all

    vfs = VFS(routes={"seekvfs://a/": {"backend": MinimalBackend()}})
    # Must not raise
    await vfs.aclose()
