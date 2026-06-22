-- ============================================================
-- Claude Crypto Bot — Supabase Schema
-- Run this in: Supabase Dashboard → SQL Editor → New query
-- ============================================================

-- Trades: every analysis cycle's result
CREATE TABLE IF NOT EXISTS trades (
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    action          TEXT        NOT NULL,
    amount_usd      NUMERIC(10,2) DEFAULT 0,
    btc_qty         NUMERIC(16,8) DEFAULT 0,
    price           NUMERIC(12,2),
    decision        JSONB,          -- full Claude decision object
    market          JSONB,          -- full market snapshot
    success         BOOLEAN   DEFAULT FALSE,
    error           TEXT,
    outcome         TEXT,           -- 'correct' | 'wrong' | 'neutral' | 'missed_opportunity'
    price_after_4h  NUMERIC(12,2),  -- price at next cycle (for outcome eval)
    lesson_generated BOOLEAN  DEFAULT FALSE
);

-- Portfolio snapshots taken at the start of each cycle
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    usdt        NUMERIC(10,2),
    btc         NUMERIC(16,8),
    price       NUMERIC(12,2),
    total_usd   NUMERIC(10,2)
);

-- Lessons learned from mistakes and weekly reviews
CREATE TABLE IF NOT EXISTS lessons (
    id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    lesson      TEXT        NOT NULL,
    source      TEXT,           -- 'self_correction' | 'weekly_review'
    active      BOOLEAN     DEFAULT TRUE,
    trade_id    UUID        REFERENCES trades(id) ON DELETE SET NULL
);

-- Weekly Opus deep-review records
CREATE TABLE IF NOT EXISTS weekly_reviews (
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    period_start    TIMESTAMPTZ,
    period_end      TIMESTAMPTZ,
    total_trades    INTEGER,
    correct_trades  INTEGER,
    wrong_trades    INTEGER,
    pnl_usd         NUMERIC(10,2),
    review_text     TEXT
);

-- Pending live-trade confirmations (Telegram approve/reject)
CREATE TABLE IF NOT EXISTS pending_confirmations (
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    decision        JSONB,
    market          JSONB,
    portfolio       JSONB,
    status          TEXT        DEFAULT 'pending',   -- 'pending'|'approved'|'rejected'|'expired'
    telegram_msg_id INTEGER
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_trades_created_at      ON trades (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_outcome         ON trades (outcome) WHERE outcome IS NULL;
CREATE INDEX IF NOT EXISTS idx_lessons_active         ON lessons (active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_created_at   ON portfolio_snapshots (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reviews_created_at     ON weekly_reviews (created_at DESC);
