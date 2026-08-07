-- Migration: 0001_init_schema
-- Created at: 2026-08-06
-- Description: Initialize MarketPulse AI schema (holdings, strategies, document chunks, audit logs)

SET EXTRA_FLOAT_DIGITS = 2;

------------------------------------------------------------
-- User Portfolio Holdings
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_holdings (
    holding_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    ticker VARCHAR(10) NOT NULL,
    shares DECIMAL(12,4) NOT NULL,
    cost_basis DECIMAL(12,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

------------------------------------------------------------
-- User Strategy Memory
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_strategies (
    strategy_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    strategy_text TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_strategies_embedding
ON user_strategies
USING hnsw (embedding vector_cosine_ops);

------------------------------------------------------------
-- Document Chunks
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    document_name VARCHAR(255) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    chunk_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
ON document_chunks
USING hnsw (embedding vector_cosine_ops);

------------------------------------------------------------
-- Research Audit Logs (with Row-Level TTL)
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_audit_logs (
    log_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    prompt_query TEXT NOT NULL,
    retrieved_news TEXT NOT NULL,
    vector_distance FLOAT NOT NULL,
    bull_perspective TEXT,
    bear_perspective TEXT,
    generated_summary TEXT NOT NULL,
    session_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
)
WITH (
    ttl_expire_after = '30 days'
);
