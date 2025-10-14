-- Genesis PostgreSQL Database Initialization Script
-- This script creates the necessary tables and indices for the Genesis platform

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Programs table - stores evolved programs and their metadata
CREATE TABLE IF NOT EXISTS programs (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    language TEXT NOT NULL,
    parent_id TEXT,
    archive_inspiration_ids JSONB,  -- JSON array of program IDs
    top_k_inspiration_ids JSONB,    -- JSON array of program IDs
    generation INTEGER NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL,
    code_diff TEXT,
    combined_score DOUBLE PRECISION,
    public_metrics JSONB,  -- JSON object
    private_metrics JSONB, -- JSON object
    text_feedback TEXT,
    complexity DOUBLE PRECISION,
    embedding JSONB,  -- JSON array of floats
    embedding_pca_2d JSONB,  -- JSON array
    embedding_pca_3d JSONB,  -- JSON array
    embedding_cluster_id INTEGER,
    correct BOOLEAN DEFAULT FALSE,
    children_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB,  -- JSON object
    migration_history JSONB,  -- JSON array
    island_idx INTEGER
);

-- Create indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_programs_generation ON programs(generation);
CREATE INDEX IF NOT EXISTS idx_programs_timestamp ON programs(timestamp);
CREATE INDEX IF NOT EXISTS idx_programs_complexity ON programs(complexity);
CREATE INDEX IF NOT EXISTS idx_programs_parent_id ON programs(parent_id);
CREATE INDEX IF NOT EXISTS idx_programs_children_count ON programs(children_count);
CREATE INDEX IF NOT EXISTS idx_programs_island_idx ON programs(island_idx);
CREATE INDEX IF NOT EXISTS idx_programs_correct ON programs(correct);
CREATE INDEX IF NOT EXISTS idx_programs_combined_score ON programs(combined_score);

-- Archive table - stores elite programs
CREATE TABLE IF NOT EXISTS archive (
    program_id TEXT PRIMARY KEY,
    FOREIGN KEY (program_id) REFERENCES programs(id) ON DELETE CASCADE
);

-- Metadata store - for tracking evolution state
CREATE TABLE IF NOT EXISTS metadata_store (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Insert initial metadata
INSERT INTO metadata_store (key, value)
VALUES
    ('last_iteration', '0'),
    ('best_program_id', NULL),
    ('beam_search_parent_id', NULL)
ON CONFLICT (key) DO NOTHING;

-- Grant permissions (if needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO genesis;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO genesis;
