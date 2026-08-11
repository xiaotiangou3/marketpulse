-- Migration: 0006_multi_sandbox_schema
-- Created at: 2026-08-11
-- Description: Create sandboxes and sandbox_positions tables, and link paper_trades to sandboxes

CREATE TABLE IF NOT EXISTS sandboxes (
    sandbox_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    name VARCHAR(100) NOT NULL,
    description TEXT,
    strategy_id UUID REFERENCES user_strategies(strategy_id) ON DELETE SET NULL,
    strategy_type VARCHAR(50),
    initial_capital DECIMAL(12,2) NOT NULL DEFAULT 100000.00,
    cash_balance DECIMAL(12,2) NOT NULL DEFAULT 100000.00,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sandbox_positions (
    position_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    sandbox_id UUID NOT NULL REFERENCES sandboxes(sandbox_id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    qty DECIMAL(12,4) NOT NULL,
    avg_entry_price DECIMAL(12,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT unique_sandbox_symbol UNIQUE(sandbox_id, symbol)
);

ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS sandbox_id UUID REFERENCES sandboxes(sandbox_id) ON DELETE SET NULL;
