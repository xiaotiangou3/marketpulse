-- Migration: 0005_paper_trades
-- Created at: 2026-08-11
-- Description: Create paper_trades audit table to record executed paper orders with status and timestamps

CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    order_id STRING NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    qty DECIMAL(12,4) NOT NULL,
    execution_price DECIMAL(12,2),
    status VARCHAR(30) NOT NULL,
    order_type VARCHAR(20) NOT NULL DEFAULT 'market',
    time_in_force VARCHAR(10) NOT NULL DEFAULT 'gtc',
    raw_response JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
