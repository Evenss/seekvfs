from __future__ import annotations

from seekvfs.models import SearchHit, SearchResult
from seekvfs.reranker import LinearReranker


def _hit(path: str, score: float) -> SearchHit:
    return SearchHit(path=path, snippet=path, score=score)


def test_single_backend_normalized_and_sorted() -> None:
    sr = SearchResult(
        query="q",
        hits=[_hit("a", 1.0), _hit("b", 3.0), _hit("c", 5.0)],
        searched_paths=["p1"],
    )
    merged = LinearReranker().merge([sr], limit=10)
    # min=1, max=5, span=4 -> 0, 0.5, 1.0 before sort
    assert [h.path for h in merged.hits] == ["c", "b", "a"]
    assert merged.hits[0].score == 1.0
    assert merged.hits[-1].score == 0.0
    assert merged.searched_paths == ["p1"]


def test_multi_backend_merge_by_score() -> None:
    r = LinearReranker()
    a = SearchResult(query="q", hits=[_hit("a1", 10), _hit("a2", 20)], searched_paths=["A"])
    b = SearchResult(query="q", hits=[_hit("b1", 0.1), _hit("b2", 0.2)], searched_paths=["B"])
    merged = r.merge([a, b], limit=3)
    # after normalization, a2=1.0, a1=0.0, b2=1.0, b1=0.0 -> ties at top
    assert len(merged.hits) == 3
    top_scores = [h.score for h in merged.hits[:2]]
    assert all(abs(s - 1.0) < 1e-9 for s in top_scores)
    assert set(merged.searched_paths) == {"A", "B"}


def test_empty_hits_backend_contributes_nothing() -> None:
    empty = SearchResult(query="q", hits=[], searched_paths=["E"])
    present = SearchResult(query="q", hits=[_hit("x", 1.0)], searched_paths=["P"])
    merged = LinearReranker().merge([empty, present], limit=5)
    assert [h.path for h in merged.hits] == ["x"]
    assert set(merged.searched_paths) == {"E", "P"}


def test_all_scores_equal_map_to_one() -> None:
    sr = SearchResult(query="q", hits=[_hit("a", 2.0), _hit("b", 2.0)])
    merged = LinearReranker().merge([sr], limit=5)
    assert all(abs(h.score - 1.0) < 1e-9 for h in merged.hits)


def test_limit_zero_returns_empty() -> None:
    sr = SearchResult(query="q", hits=[_hit("a", 1.0)])
    merged = LinearReranker().merge([sr], limit=0)
    assert merged.hits == []
