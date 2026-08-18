-- Migration: 0008_chat_history_metadata
-- Created at: 2026-08-18
-- Description: Add JSONB columns for backtest, trade, and debate data persistence to chat_history table

ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS backtest_data JSONB;
ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS trade_data JSONB;
ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS debate_data JSONB;
