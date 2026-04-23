"""Offline reconcile job for :class:`OceanbaseFsBackend`.

Scans the filesystem and backfills missing L0 / L1 / embedding in OceanBase
for entries that were never generated (bulk-loaded files, crashed async tasks,
or files written directly to the filesystem bypassing the backend).

Usage::

    from seekvfs_recipes.maximal import OceanbaseFsBackend, reconcile

    backend = OceanbaseFsBackend(...)
    stats = await reconcile(backend)
    # {"checked": 120, "repaired": 4, "failed": 0}

``checked``  — total FS files inspected
``repaired`` — entries whose derivatives were regenerated and committed
``failed``   — entries where generation or DB write raised an exception
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seekvfs_recipes.maximal.backend import OceanbaseFsBackend

logger = logging.getLogger(__name__)


@dataclass
class ReconcileStats:
    checked: int = 0
    repaired: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "checked": self.checked,
            "repaired": self.repaired,
            "failed": self.failed,
        }


async def reconcile(backend: OceanbaseFsBackend) -> dict[str, int]:
    """Walk every file in *backend*'s filesystem and backfill missing derivatives.

    Two repair scenarios are handled:

    1. **FS file exists, no DB row** — file was written directly to disk.
       A new row is inserted and derivatives are generated.
    2. **DB row exists, l0/l1/embedding is NULL** — async generation crashed.
       The NULL columns are regenerated from the current FS content.
    """
    stats = ReconcileStats()

    # 1. Collect all VFS paths present on the filesystem.
    def _scan_fs() -> list[str]:
        return [
            backend._reconstruct(fp)  # noqa: SLF001
            for fp in backend._fs_root.rglob("*")  # noqa: SLF001
            if fp.is_file()
        ]

    all_paths = await asyncio.to_thread(_scan_fs)
    if not all_paths:
        return stats.as_dict()

    # 2. Ask the DAO which paths need repair.
    missing_deriv, no_db_record = await backend._dao.find_incomplete(all_paths)  # noqa: SLF001
    to_repair = missing_deriv | no_db_record

    # 3. Inspect every FS path; repair those that need it.
    for path in all_paths:
        stats.checked += 1
        if path not in to_repair:
            continue

        fp = backend._local(path)  # noqa: SLF001
        try:
            raw = await asyncio.to_thread(fp.read_bytes)
        except OSError:
            logger.warning("reconcile: cannot read FS file for path=%r", path)
            stats.failed += 1
            continue

        try:
            l0, l1, emb = await backend._generate_derivatives(raw)  # noqa: SLF001
        except Exception:
            logger.exception(
                "reconcile: derivative generation failed for path=%r", path
            )
            stats.failed += 1
            continue

        try:
            if path in no_db_record:
                await backend._dao.upsert_init(path)  # noqa: SLF001
            await backend._dao.update_derivatives(path, l0, l1, emb)  # noqa: SLF001
            stats.repaired += 1
        except Exception:
            logger.exception("reconcile: DB write failed for path=%r", path)
            stats.failed += 1

    return stats.as_dict()


__all__ = ["reconcile", "ReconcileStats"]
