-- OceanBase schema for seekvfs tiered recipe
--
-- Prerequisites:
--   OceanBase 4.x with vector support enabled:
--     SET GLOBAL observer_vector_index_enabled = ON;
--
-- IMPORTANT: The VECTOR dimension (1536 below) must match your Embedder's
-- output dimension. For common models:
--   OpenAI text-embedding-3-small → 1536
--   OpenAI text-embedding-3-large → 3072
--   text-embedding-ada-002        → 1536
-- Adjust VECTOR(N) before running this script.

CREATE TABLE IF NOT EXISTS vfs_storage (
    path        VARCHAR(512)    NOT NULL,
    l0          TEXT            DEFAULT NULL,           -- short abstract (~100 tokens)
    l1          MEDIUMTEXT      DEFAULT NULL,           -- overview (~2k tokens)
    embedding   VECTOR(1536)    DEFAULT NULL,           -- L0 embedding (adjust dimension)
    updated_at  TIMESTAMP       NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (path),
    VECTOR INDEX idx_emb (embedding) WITH (distance = L2, type = HNSW, lib = vsag)
);
