-- ═══════════════════════════════════════════════════════════════════════════
-- Crypto Bot — MySQL Schema
-- Run once: mysql -u root -p crypto_bot < mysql_schema.sql
-- ═══════════════════════════════════════════════════════════════════════════

CREATE DATABASE IF NOT EXISTS crypto_bot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE crypto_bot;

-- ── Portfolio snapshots ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    usdt        DECIMAL(16,2)  NOT NULL DEFAULT 0,
    btc         DECIMAL(16,8)  NOT NULL DEFAULT 0,
    price       DECIMAL(16,2)  NOT NULL DEFAULT 0,
    total_usd   DECIMAL(16,2)  NOT NULL DEFAULT 0,
    created_at  TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_snapshots_created (created_at)
) ENGINE=InnoDB;

-- ── Trades ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    action          VARCHAR(10)    NOT NULL DEFAULT 'hold',
    amount_usd      DECIMAL(16,2)  NOT NULL DEFAULT 0,
    btc_qty         DECIMAL(16,8)  NOT NULL DEFAULT 0,
    price           DECIMAL(16,2)  NOT NULL DEFAULT 0,
    decision        JSON           NULL,
    market          JSON           NULL,
    success         TINYINT(1)     NOT NULL DEFAULT 0,
    error           TEXT           NULL,
    outcome         VARCHAR(30)    NULL,
    price_after_4h  DECIMAL(16,2)  NULL,
    lesson_generated TINYINT(1)   NOT NULL DEFAULT 0,
    created_at      TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_trades_created (created_at),
    INDEX idx_trades_action (action),
    INDEX idx_trades_outcome (outcome)
) ENGINE=InnoDB;

-- ── Lessons ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lessons (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    lesson      TEXT           NOT NULL,
    source      VARCHAR(50)    NOT NULL DEFAULT 'self_correction',
    trade_id    BIGINT         NULL,
    active      TINYINT(1)     NOT NULL DEFAULT 1,
    created_at  TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_lessons_active (active),
    INDEX idx_lessons_created (created_at)
) ENGINE=InnoDB;

-- ── Weekly reviews ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS weekly_reviews (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    period_start    DATETIME       NOT NULL,
    period_end      DATETIME       NOT NULL,
    total_trades    INT            NOT NULL DEFAULT 0,
    correct_trades  INT            NOT NULL DEFAULT 0,
    wrong_trades    INT            NOT NULL DEFAULT 0,
    pnl_usd         DECIMAL(16,2)  NOT NULL DEFAULT 0,
    review_text     TEXT           NULL,
    created_at      TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_reviews_created (created_at)
) ENGINE=InnoDB;

-- ── Pending confirmations ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pending_confirmations (
    id          VARCHAR(36)    PRIMARY KEY,
    decision    JSON           NULL,
    market      JSON           NULL,
    portfolio   JSON           NULL,
    status      VARCHAR(20)    NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ── Bot events ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bot_events (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    event       VARCHAR(50)    NOT NULL,
    message     TEXT           NOT NULL,
    level       VARCHAR(10)    NOT NULL DEFAULT 'info',
    data        JSON           NULL,
    created_at  TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_events_created (created_at),
    INDEX idx_events_level (level)
) ENGINE=InnoDB;

-- ── Coin research ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS coin_research (
    id                BIGINT AUTO_INCREMENT PRIMARY KEY,
    coin_id           VARCHAR(100)   NULL,
    symbol            VARCHAR(20)    NULL,
    name              VARCHAR(100)   NULL,
    category          VARCHAR(100)   NULL,
    investment_score  INT            NOT NULL DEFAULT 0,
    team_score        INT            NOT NULL DEFAULT 0,
    technology_score  INT            NOT NULL DEFAULT 0,
    market_score      INT            NOT NULL DEFAULT 0,
    tokenomics_score  INT            NOT NULL DEFAULT 0,
    usecase_score     INT            NOT NULL DEFAULT 0,
    verdict           VARCHAR(20)    NULL,
    suggested_usd     DECIMAL(16,2)  NOT NULL DEFAULT 0,
    hold_months       INT            NOT NULL DEFAULT 0,
    risks             JSON           NULL,
    opportunities     JSON           NULL,
    summary           TEXT           NULL,
    price_usd         DECIMAL(20,8)  NULL,
    market_cap_usd    DECIMAL(20,2)  NULL,
    volume_24h_usd    DECIMAL(20,2)  NULL,
    price_change_7d   DECIMAL(10,2)  NULL,
    github_commits_4w INT            NULL,
    twitter_followers  INT           NULL,
    raw_data          JSON           NULL,
    on_watchlist      TINYINT(1)     NOT NULL DEFAULT 0,
    created_at        TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_research_created (created_at),
    INDEX idx_research_symbol (symbol)
) ENGINE=InnoDB;

-- ── Coin watchlist ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS coin_watchlist (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    coin_id      VARCHAR(100)   NOT NULL,
    symbol       VARCHAR(20)    NOT NULL,
    name         VARCHAR(100)   NULL,
    entry_price  DECIMAL(20,8)  NOT NULL DEFAULT 0,
    target_usd   DECIMAL(16,2)  NOT NULL DEFAULT 0,
    research_id  BIGINT         NULL,
    active       TINYINT(1)     NOT NULL DEFAULT 1,
    created_at   TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_watchlist_coin (coin_id),
    INDEX idx_watchlist_active (active)
) ENGINE=InnoDB;

-- ── Backtest runs ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS backtest_runs (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    period_months   INT            NOT NULL DEFAULT 6,
    start_date      DATE           NULL,
    end_date        DATE           NULL,
    total_trades    INT            NOT NULL DEFAULT 0,
    wins            INT            NOT NULL DEFAULT 0,
    losses          INT            NOT NULL DEFAULT 0,
    win_rate        DECIMAL(5,3)   NOT NULL DEFAULT 0,
    sharpe_ratio    DECIMAL(8,3)   NOT NULL DEFAULT 0,
    sortino_ratio   DECIMAL(8,3)   NOT NULL DEFAULT 0,
    max_drawdown_pct DECIMAL(8,3)  NOT NULL DEFAULT 0,
    profit_factor   DECIMAL(8,3)   NOT NULL DEFAULT 0,
    total_return_pct DECIMAL(8,3)  NOT NULL DEFAULT 0,
    equity_curve    JSON           NULL,
    config_snapshot JSON           NULL,
    created_at      TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_backtest_created (created_at)
) ENGINE=InnoDB;

-- ── Claude API call logs ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS claude_api_logs (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    cycle_id    VARCHAR(36)    NULL,
    agent       VARCHAR(30)    NOT NULL,
    model       VARCHAR(50)    NOT NULL,
    prompt      TEXT           NOT NULL,
    response    TEXT           NULL,
    tokens_in   INT            NOT NULL DEFAULT 0,
    tokens_out  INT            NOT NULL DEFAULT 0,
    duration_ms INT            NOT NULL DEFAULT 0,
    created_at  TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_claude_logs_created (created_at),
    INDEX idx_claude_logs_cycle (cycle_id),
    INDEX idx_claude_logs_agent (agent)
) ENGINE=InnoDB;
