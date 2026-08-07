-- Migration: 0004_market_news
-- Created at: 2026-08-07
-- Description: Create market_news table to cache news articles and store AI action suggestions with 30-day TTL

CREATE TABLE IF NOT EXISTS market_news (
    news_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id STRING NOT NULL DEFAULT 'demo_user',
    ticker VARCHAR(10) NOT NULL,
    title TEXT NOT NULL,
    source VARCHAR(100) NOT NULL,
    url TEXT,
    summary TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    action_suggestions TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT unique_ticker_title UNIQUE (ticker, title)
) WITH (
    ttl_expire_after = '30 days'
);
