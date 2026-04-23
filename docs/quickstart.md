# Quickstart

This tutorial walks through a minimal `VFS` with the `minimal` recipe. For tiered reads + vector search, see [`recipes/maximal.md`](recipes/maximal.md).

## Install

```bash
pip install seekvfs
```

## Minimal VFS

```python
import asyncio
from seekvfs import VFS
from seekvfs_recipes.minimal import FileBackend


async def main() -> None:
    vfs = VFS(routes={
        "seekvfs://notes/": {"backend": FileBackend("/data/agent_notes")},
    })

    # Write — stored as a real file on disk
    await vfs.write("seekvfs://notes/hello.md", "hello world")

    # Read — returns the stored content
    fd = await vfs.read("seekvfs://notes/hello.md")
    print(fd.content.decode())
    # hello world

    # List
    for info in await vfs.ls("seekvfs://notes/"):
        print(info.path, info.size)

    # Grep
    for m in await vfs.grep("hello"):
        print(m.path, m.line_number, m.line)


asyncio.run(main())
```

> Prefix names (`notes/`, `memories/`, `scratch/`, etc.) are entirely up to you. The protocol does not recommend or reserve any naming convention.

## Picking a recipe

Two built-in recipes are available at different complexity levels:

| Recipe | Use when | Details |
|---|---|---|
| `seekvfs_recipes.minimal` | **Minimal** — durable single-process storage without a database | [recipes/minimal.md](recipes/minimal.md) |
| `seekvfs_recipes.maximal` | **Maximal** — best-combination: FS + OceanBase + vector search | [recipes/maximal.md](recipes/maximal.md) |

Recipes are NOT part of the protocol. They're separate packages under `seekvfs_recipes.*` so `src/seekvfs/` stays free of concrete storage implementations.

## Mixing recipes in one VFS

Different URI prefixes can use different recipes — handy for separating "flat storage" from "tiered with search":

```python
from seekvfs import VFS
from seekvfs_recipes.minimal import FileBackend
import asyncmy
from seekvfs_recipes.maximal import OceanbaseFsBackend, ClaudeSummarizer, OpenAIEmbedder

pool = await asyncmy.create_pool(host="...", user="...", password="...", db="agent_kb")

vfs = VFS(routes={
    # Minimal: flat storage — files on disk, literal search only
    "seekvfs://docs/": {"backend": FileBackend("/data/agent_docs")},
    # Maximal: L2 on disk, L0/L1/embedding in OceanBase, vector search
    "seekvfs://notes/": {
        "backend": OceanbaseFsBackend(
            ob_pool=pool,
            fs_root="/data/agent_notes",
            summarizer=ClaudeSummarizer(
                model="claude-opus-4-7",
                abstract_prompt="...",
                overview_prompt="...",
            ),
            embedder=OpenAIEmbedder(model="text-embedding-3-small"),
        ),
    },
})
```

`vfs.search(...)` fans out across every route in parallel and merges hits via the reranker.

## Exporting tools to your agent framework

```python
openai_tools    = vfs.tools.to_openai()
anthropic_tools = vfs.tools.to_anthropic()
langgraph_tools = vfs.tools.to_langgraph()   # needs [langgraph] extra
mcp_server      = vfs.tools.to_mcp()         # needs [mcp] extra
```
