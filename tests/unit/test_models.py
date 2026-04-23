from __future__ import annotations

from datetime import datetime

from seekvfs.models import FileData, FileInfo, GrepMatch, SearchHit, SearchResult


def test_filedata_defaults() -> None:
    fd = FileData(content=b"hello")
    assert fd.content == b"hello"
    assert fd.encoding == "utf-8"


def test_fileinfo_snippet_defaults_none() -> None:
    fi = FileInfo(path="a", size=0, mtime=datetime(2026, 1, 1), is_dir=False)
    assert fi.snippet is None


def test_search_result_default_fields() -> None:
    sr = SearchResult(query="q")
    assert sr.hits == []
    assert sr.searched_paths == []


def test_search_hit_defaults() -> None:
    h = SearchHit(path="a")
    assert h.snippet == ""
    assert h.score == 0.0

    h2 = SearchHit(path="b", snippet="preview", score=0.5)
    assert h2.snippet == "preview"
    assert h2.score == 0.5


def test_grep_match_structure() -> None:
    m = GrepMatch(path="p", line_number=3, line="content")
    assert m.line_number == 3
