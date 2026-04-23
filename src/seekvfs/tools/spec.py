"""Neutral ToolSpec + ToolSpecSet and the 8 agent-facing tool builders.

Each built tool is an async callable bound to a :class:`VFS` instance.
Output of read-like tools is wrapped as ``<file path=...>...</file>`` so
agents can clearly distinguish file content from chat text.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any

from seekvfs.models import FileData

if TYPE_CHECKING:
    from seekvfs.vfs import VFS


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    callable: Callable[..., Any]


@dataclass
class ToolSpecSet:
    specs: list[ToolSpec] = field(default_factory=list)

    # ---- ergonomics ----

    def __iter__(self):
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def by_name(self, name: str) -> ToolSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    def with_description_overrides(
        self, overrides: dict[str, str]
    ) -> ToolSpecSet:
        new_specs: list[ToolSpec] = []
        for s in self.specs:
            if s.name in overrides:
                new_specs.append(replace(s, description=overrides[s.name]))
            else:
                new_specs.append(s)
        return ToolSpecSet(specs=new_specs)

    # ---- framework converters ----

    def to_openai(self) -> list[dict[str, Any]]:
        from seekvfs.tools.openai import to_openai

        return to_openai(self)

    def to_anthropic(self) -> list[dict[str, Any]]:
        from seekvfs.tools.anthropic import to_anthropic

        return to_anthropic(self)

    def to_langgraph(self) -> list[Any]:
        from seekvfs.tools.langgraph import to_langgraph

        return to_langgraph(self)

    def to_mcp(self) -> Any:
        from seekvfs.tools.mcp import to_mcp

        return to_mcp(self)


# ---------- helpers ----------


def _wrap_file_output(fd: FileData, path: str) -> str:
    text = fd.content.decode("utf-8", errors="replace")
    return f'<file path="{path}">\n{text}\n</file>'


# ---------- tool callable wrappers ----------

async def _search(vfs: VFS, query: str, limit: int = 10) -> dict[str, Any]:
    sr = await vfs.search(query, limit=limit)
    return {
        "query": sr.query,
        "hits": [
            {
                "path": h.path,
                "snippet": h.snippet,
                "score": h.score,
            }
            for h in sr.hits
        ],
        "searched_paths": sr.searched_paths,
    }


async def _read(vfs: VFS, path: str) -> str:
    fd = await vfs.read(path)
    return _wrap_file_output(fd, path)


async def _read_full(vfs: VFS, path: str) -> str:
    fd = await vfs.read_full(path)
    return _wrap_file_output(fd, path)


async def _write(vfs: VFS, path: str, content: str) -> str:
    await vfs.write(path, content)
    return f"wrote {path}"


async def _edit(vfs: VFS, path: str, old: str, new: str) -> str:
    n = await vfs.edit(path, old, new)
    return f"{n} replacement(s) in {path}"


async def _ls(
    vfs: VFS,
    path: str,
    pattern: str | None = None,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    infos = await vfs.ls(path, pattern=pattern, recursive=recursive)
    return [
        {
            "path": i.path,
            "size": i.size,
            "is_dir": i.is_dir,
            "snippet": i.snippet,
        }
        for i in infos
    ]


async def _grep(
    vfs: VFS,
    pattern: str,
    path_pattern: str | None = None,
) -> list[dict[str, Any]]:
    matches = await vfs.grep(pattern, path_pattern=path_pattern)
    return [
        {"path": m.path, "line_number": m.line_number, "line": m.line}
        for m in matches
    ]


async def _delete(vfs: VFS, path: str) -> str:
    await vfs.delete(path)
    return f"deleted {path}"


# ---------- schemas ----------

def _schema(**props: dict[str, Any]) -> dict[str, Any]:
    required = [k for k, v in props.items() if v.pop("_required", True)]
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


_DESCRIPTIONS: dict[str, str] = {
    "search": (
        "Search across files. Returns matching paths with a short snippet "
        "(when the backend provides one). If a snippet is insufficient, "
        "call read_full(path) for complete content."
    ),
    "read": (
        "Read the backend's preferred representation of a file. May be a "
        "derived summary if the backend keeps one, or the full content "
        "otherwise. For the guaranteed original content, call read_full(path)."
    ),
    "read_full": "Read complete original content.",
    "write": "Write content. Indexing behavior is backend-defined.",
    "edit": "Literal string replacement. Last-write-wins on concurrent edits.",
    "ls": (
        "List files. pattern supports glob wildcards (e.g. '*.md'); "
        "recursive=True lists subtree."
    ),
    "grep": "Literal search in file contents.",
    "delete": "Delete a file by path.",
}


def build_tools(vfs: VFS) -> ToolSpecSet:
    """Produce the 8 agent-facing tools bound to ``vfs``."""
    specs = [
        ToolSpec(
            name="search",
            description=_DESCRIPTIONS["search"],
            parameters_schema=_schema(
                query={"type": "string", "description": "Search query"},
                limit={
                    "type": "integer",
                    "description": "Max hits",
                    "default": 10,
                    "_required": False,
                },
            ),
            callable=partial(_search, vfs),
        ),
        ToolSpec(
            name="read",
            description=_DESCRIPTIONS["read"],
            parameters_schema=_schema(
                path={"type": "string", "description": "Full seekvfs:// URI"},
            ),
            callable=partial(_read, vfs),
        ),
        ToolSpec(
            name="read_full",
            description=_DESCRIPTIONS["read_full"],
            parameters_schema=_schema(
                path={"type": "string", "description": "Full seekvfs:// URI"},
            ),
            callable=partial(_read_full, vfs),
        ),
        ToolSpec(
            name="write",
            description=_DESCRIPTIONS["write"],
            parameters_schema=_schema(
                path={"type": "string", "description": "Full seekvfs:// URI"},
                content={"type": "string", "description": "File content"},
            ),
            callable=partial(_write, vfs),
        ),
        ToolSpec(
            name="edit",
            description=_DESCRIPTIONS["edit"],
            parameters_schema=_schema(
                path={"type": "string", "description": "Full seekvfs:// URI"},
                old={"type": "string", "description": "Literal text to replace"},
                new={"type": "string", "description": "Replacement text"},
            ),
            callable=partial(_edit, vfs),
        ),
        ToolSpec(
            name="ls",
            description=_DESCRIPTIONS["ls"],
            parameters_schema=_schema(
                path={"type": "string", "description": "Directory URI"},
                pattern={
                    "type": "string",
                    "description": "Optional glob, e.g. *.md",
                    "_required": False,
                },
                recursive={
                    "type": "boolean",
                    "description": "Recurse into subdirs",
                    "default": False,
                    "_required": False,
                },
            ),
            callable=partial(_ls, vfs),
        ),
        ToolSpec(
            name="grep",
            description=_DESCRIPTIONS["grep"],
            parameters_schema=_schema(
                pattern={"type": "string", "description": "Literal substring"},
                path_pattern={
                    "type": "string",
                    "description": "Optional glob to filter paths",
                    "_required": False,
                },
            ),
            callable=partial(_grep, vfs),
        ),
        ToolSpec(
            name="delete",
            description=_DESCRIPTIONS["delete"],
            parameters_schema=_schema(
                path={"type": "string", "description": "Full seekvfs:// URI"},
            ),
            callable=partial(_delete, vfs),
        ),
    ]
    return ToolSpecSet(specs=specs)


__all__ = ["ToolSpec", "ToolSpecSet", "build_tools"]
