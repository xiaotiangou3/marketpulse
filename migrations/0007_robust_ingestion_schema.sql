-- Migration: 0007_robust_ingestion_schema
-- Created at: 2026-08-13
-- Description: Robust file upload tracking and generic documents with unique constraints and background jobs

SET EXTRA_FLOAT_DIGITS = 2;

------------------------------------------------------------
-- Documents Metadata
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    document_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    name VARCHAR(255) NOT NULL UNIQUE,
    file_type VARCHAR(10) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    storage_path VARCHAR(512),
    created_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT unique_user_file_hash UNIQUE (user_id, file_hash)
);

------------------------------------------------------------
-- Ingestion Jobs
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    document_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued', -- queued, extracting, chunking, embedding, persisting, completed, failed
    progress_pct INT NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

------------------------------------------------------------
-- Add new columns safely to document_chunks
------------------------------------------------------------
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS document_id UUID REFERENCES documents(document_id) ON DELETE CASCADE;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_number INT;
ALTER TABLE document_chunks ALTER COLUMN ticker DROP NOT NULL;

------------------------------------------------------------
-- Backfill Legacy Data
------------------------------------------------------------
-- Insert unmapped default document for legacy rows that don't match any file in documents
INSERT INTO documents (document_id, name, file_type, file_hash, storage_path)
VALUES ('00000000-0000-0000-0000-000000000000', 'legacy_unmapped_documents', 'pdf', 'legacy_unmapped_hash', 'legacy')
ON CONFLICT (name) DO NOTHING;

-- Backfill legacy records based on document_name in document_chunks
-- Since document_chunks used to store the name directly, we insert rows into documents
-- generating a synthetic file_hash from the name to keep uniqueness
INSERT INTO documents (name, file_type, file_hash, storage_path)
SELECT DISTINCT document_name, 'pdf', 'legacy_migration_' || md5(document_name), 'legacy_migration'
FROM document_chunks
WHERE document_name != 'legacy_unmapped_documents'
ON CONFLICT (name) DO NOTHING;

-- Map document_id back to document_chunks
UPDATE document_chunks c
SET document_id = d.document_id
FROM documents d
WHERE c.document_name = d.name AND c.document_id IS NULL;

-- Assign any remaining unmapped rows to the default unmapped document
UPDATE document_chunks
SET document_id = '00000000-0000-0000-0000-000000000000'
WHERE document_id IS NULL;

------------------------------------------------------------
-- Add constraints and clean up duplicate chunks
------------------------------------------------------------
-- Clean up duplicate document chunks if any existed before setting uniqueness
DELETE FROM document_chunks a USING document_chunks b
WHERE a.chunk_id > b.chunk_id
  AND a.document_id = b.document_id
  AND a.chunk_index = b.chunk_index;

-- Add uniqueness constraint to document_chunks
ALTER TABLE document_chunks DROP CONSTRAINT IF EXISTS unique_document_chunk;
ALTER TABLE document_chunks ADD CONSTRAINT unique_document_chunk UNIQUE (document_id, chunk_index);

-- Drop legacy obsolete column document_name
ALTER TABLE document_chunks DROP COLUMN IF EXISTS document_name;
