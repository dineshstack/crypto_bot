-- ============================================================
-- Claude Crypto Bot — Bot Events Schema (v3 migration)
-- Run this in: Supabase Dashboard → SQL Editor → New query
-- Run AFTER supabase_schema.sql and supabase_schema_coins.sql
-- ============================================================

-- Structured event log — powers the web dashboard activity feed
CREATE TABLE IF NOT EXISTS bot_events (
    id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    level       TEXT        DEFAULT 'info',   -- 'info' | 'warning' | 'error'
    event       TEXT        NOT NULL,          -- 'cycle_start', 'trade', 'lesson', 'review', etc.
    message     TEXT,
    data        JSONB
);

CREATE INDEX IF NOT EXISTS idx_bot_events_time ON bot_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_events_level ON bot_events (level, created_at DESC);
