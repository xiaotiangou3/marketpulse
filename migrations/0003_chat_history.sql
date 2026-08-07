-- Migration: 0003_chat_history
-- Created at: 2026-08-07
-- Description: Create chat_history table for single-session persistence

CREATE TABLE IF NOT EXISTS chat_history (
    message_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
