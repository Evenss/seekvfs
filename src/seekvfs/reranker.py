"""Default cross-backend reranker (min-max normalization + score sort)."""
from __future__ import annotations

from seekvfs.models import SearchHit, SearchResult


class LinearReranker:
    """Normalize each backend's hit scores into [0, 1] via min-max then
    merge + sort descending and take the top ``limit``.

    Edge cases:
      * A backend with zero hits contributes nothing.
      * If all scores inside a backend are equal, they are all mapped to 1.0
        (preserving order from that backend but letting them compete fairly
        against other backends' normalized scores).
    """

    def merge(self, per_backend: list[SearchResult], limit: int) -> SearchResult:
        normalized: list[SearchHit] = []
        query = ""
        searched: list[str] = []
        for sr in per_backend:
            if sr.query:
                query = sr.query
            searched.extend(sr.searched_paths)
            if not sr.hits:
                continue
            scores = [h.score for h in sr.hits]
            lo, hi = min(scores), max(scores)
            span = hi - lo
            for h in sr.hits:
                new_score = 1.0 if span == 0 else (h.score - lo) / span
                normalized.append(
                    SearchHit(
                        path=h.path,
                        snippet=h.snippet,
                        score=new_score,
                    )
                )
        normalized.sort(key=lambda h: h.score, reverse=True)
        return SearchResult(
            query=query,
            hits=normalized[: max(0, limit)],
            searched_paths=searched,
        )


__all__ = ["LinearReranker"]
