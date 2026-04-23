"""Recipe-local exception types."""
from __future__ import annotations

from seekvfs.exceptions import VFSError


class TierNotAvailable(VFSError):
    """Raised when a strict-tier read targets a tier that has not been
    produced (e.g. L0 requested but summarization has not run yet).
    """


__all__ = ["TierNotAvailable"]
