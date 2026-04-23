"""Convert ToolSpecSet to OpenAI tool-calling format."""
from __future__ import annotations

from typing import Any

from seekvfs.tools.spec import ToolSpecSet


def to_openai(specs: ToolSpecSet) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters_schema,
            },
        }
        for s in specs.specs
    ]


__all__ = ["to_openai"]
