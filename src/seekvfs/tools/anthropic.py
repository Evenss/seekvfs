"""Convert ToolSpecSet to Anthropic tool-use format."""
from __future__ import annotations

from typing import Any

from seekvfs.tools.spec import ToolSpecSet


def to_anthropic(specs: ToolSpecSet) -> list[dict[str, Any]]:
    return [
        {
            "name": s.name,
            "description": s.description,
            "input_schema": s.parameters_schema,
        }
        for s in specs.specs
    ]


__all__ = ["to_anthropic"]
