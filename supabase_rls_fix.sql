-- ============================================================
-- Fix RLS (Row Level Security) for all tables
-- Run this in: Supabase Dashboard → SQL Editor → New query
-- ============================================================

-- Option A: Disable RLS (simplest — fine for a private bot with anon key)
-- The bot is the only client, so public access is OK.

ALTER TABLE trades DISABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_snapshots DISABLE ROW LEVEL SECURITY;
ALTER TABLE lessons DISABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_reviews DISABLE ROW LEVEL SECURITY;
ALTER TABLE pending_confirmations DISABLE ROW LEVEL SECURITY;
ALTER TABLE coin_research DISABLE ROW LEVEL SECURITY;
ALTER TABLE coin_watchlist DISABLE ROW LEVEL SECURITY;
ALTER TABLE bot_events DISABLE ROW LEVEL SECURITY;
