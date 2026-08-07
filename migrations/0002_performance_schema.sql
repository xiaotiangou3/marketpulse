-- Migration: 0002_performance_schema
-- Created at: 2026-08-06
-- Description: Create stock_prices and portfolio_snapshots tables for performance monitoring

------------------------------------------------------------
-- Stock Prices Cache Table
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stock_prices (
    ticker VARCHAR(10) PRIMARY KEY,
    price DECIMAL(12,2) NOT NULL,
    daily_change_pct DECIMAL(8,4),
    updated_at TIMESTAMPTZ DEFAULT now()
);

------------------------------------------------------------
-- Portfolio Snapshots Log Table
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    total_value DECIMAL(14,2) NOT NULL,
    total_gain_loss DECIMAL(14,2) NOT NULL,
    total_gain_loss_pct DECIMAL(8,4) NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT now()
);
