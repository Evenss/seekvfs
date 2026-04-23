"""Built-in best-practice recipes for SeekVFS.

These are NOT part of the protocol. The core :mod:`seekvfs` package
only defines the contract (:class:`seekvfs.BackendProtocol` and friends);
this package ships two recommended implementations at different
complexity levels:

- :mod:`seekvfs_recipes.minimal` — **Minimal** recipe. One file per
  path under a local directory tree, no summaries, no embeddings. Use
  this for durable single-process storage without a database. Survives
  process restarts. See :doc:`docs/recipes/minimal`.

- :mod:`seekvfs_recipes.maximal` — **Maximal** recipe. The best-combination
  backend: L2 full content on local filesystem, L0/L1/embedding in
  OceanBase, supporting vector search and "read short first, read full on
  demand" agent UX. See :doc:`docs/recipes/maximal`.

Both recipes are meant to be used as-is or copied into your project and
adapted (swap the storage backend for your own choice).
"""
