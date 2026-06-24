# 🤖 Claude Crypto Bot — AI-Powered BTC & ETH Trading System

An institutional-grade **AI cryptocurrency trading bot** built with **Claude AI (Anthropic)**, featuring a multi-agent analysis pipeline, machine learning ensemble predictions, real-time WebSocket monitoring, and a full Next.js dashboard. Trades BTC/USDT and ETH/USDT on **Binance** with automated risk management, self-learning capabilities, and Telegram control.

**Built by [Dinesh Lakmal](https://dineshstack.com)** — Full-stack developer & crypto trading system architect.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Claude AI](https://img.shields.io/badge/Claude-Haiku%20%2B%20Opus-orange?logo=anthropic)](https://anthropic.com)
[![Binance](https://img.shields.io/badge/Binance-Spot%20API-yellow?logo=binance)](https://binance.com)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-blue?logo=mysql)](https://mysql.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Multi-Agent Claude Pipeline](#-multi-agent-claude-pipeline)
- [ML Ensemble Model](#-ml-ensemble-model)
- [Risk Management](#-risk-management)
- [Data Sources](#-data-sources)
- [Dashboard](#-dashboard)
- [Telegram Commands](#-telegram-commands)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Going Live](#-going-live)
- [Project Structure](#-project-structure)
- [Blog & Learning](#-blog--learning)

---

## ✨ Features

### Trading Intelligence
- **3-Agent Claude AI Pipeline** — Market Analyst, Sentiment Analyst, and Decision Maker run in parallel for every analysis cycle
- **ML Stacking Ensemble** — XGBoost + LightGBM meta-learner with 86% accuracy, trained on 5000+ candles across 3 timeframes
- **HMM 5-State Regime Detection** — Hidden Markov Model classifies market as strong_trend, weak_trend, range, high_vol, or crash with 3-bar persistence filter
- **15+ Data Sources** — Technical indicators, derivatives, news, social sentiment, on-chain, options, whale monitoring, macro correlations, MVRV-Z score
- **Multi-Asset Trading** — BTC/USDT and ETH/USDT with independent analysis cycles
- **Self-Learning Loop** — Evaluates trades 4h post-execution, generates lessons from mistakes, injects into future Claude prompts

### Risk Management
- **Quarter-Kelly Position Sizing** — Conservative 0.25× Kelly with 10pp win-rate discount
- **ATR-Based Regime Stops** — Dynamic stop-loss/take-profit that adapts to volatility regime
- **5 Circuit Breakers** — Daily loss gate (3%), consecutive loss pause (5), drawdown sizing reduction (10%), full halt (20%), equity MA filter
- **RL Position Management** — Q-learning agent adjusts position sizing based on market state and PnL
- **Live Trade Approval** — Every buy/sell in live mode requires Telegram ✅ confirmation

### Real-Time Monitoring
- **Binance WebSocket Streams** — Sub-second price updates for BTC + ETH
- **Flash Crash Detection** — >2% drop in 5 minutes → Telegram alert + emergency analysis
- **Volume Spike Detection** — 5× above average triggers alert
- **Liquidation Cascade Monitoring** — Futures liquidation data from Binance

### Advisory Tools
- **Coin Screening** — Top 50 cryptocurrencies ranked by momentum score with risk tiers
- **Market Reports** — Claude Opus generates weekly/monthly market reports for client distribution
- **Investment Thesis Generator** — Deep analysis of any cryptocurrency with entry/exit levels, position sizing, risk factors
- **New Coin Research** — Scan trending coins, score 0-100 across 5 dimensions (team, tech, market, tokenomics, use case)

### Dashboard (Next.js)
- **Real-Time Dashboard** — Portfolio value, win rate, Sharpe ratio, agent reasoning, derivatives panel
- **Trade Detail Panel** — Click any trade to see full 3-agent reasoning, ML prediction, risk data
- **Performance Analytics** — Sharpe, Sortino, drawdown, profit factor, per-asset breakdown
- **AI Logs** — Every prompt sent to Claude and every response, grouped by analysis cycle
- **Auto-Refresh Toggle** — Live/Paused mode with 30s refresh
- **Health Bar** — Drawdown %, daily P&L, streak, risk level in header
- **Contextual Tooltips** — Every metric has a `?` hover explanation
- **System Guide** — Complete documentation for users and advisors

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    VPS (Ubuntu 24.04)                │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐   ┌────────┐ │
│  │  Trading Bot  │───▶│    MySQL     │◀──│  API   │ │
│  │  (main.py)    │    │  (8 tables)  │   │ Server │ │
│  └──────┬───────┘    └──────────────┘   │ :8100  │ │
│         │                                └───┬────┘ │
│  ┌──────▼───────┐                           │      │
│  │  Claude AI    │    ┌──────────────┐      │      │
│  │  (3 agents)   │    │  WebSocket   │      │      │
│  └──────────────┘    │  (Binance)   │      │      │
│                      └──────────────┘      │      │
│  ┌──────────────┐                    ┌─────▼────┐ │
│  │  ML Model     │                    │  nginx   │ │
│  │  (XGB+LGB)    │                    │  :443    │ │
│  └──────────────┘                    └─────┬────┘ │
└─────────────────────────────────────────────┼──────┘
                                              │
                              ┌────────────────▼───────────┐
                              │   Dashboard (Next.js)      │
                              │   Vercel / VPS :3004       │
                              └────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI Engine** | Claude Haiku 4.5 (analysis), Claude Opus 4.8 (reports, research) |
| **ML Model** | XGBoost + LightGBM stacking ensemble, Optuna hyperparameter tuning |
| **Regime Detection** | Gaussian HMM (hmmlearn) with persistence filter |
| **Feature Selection** | Boruta-SHAP (with scikit-learn fallback) |
| **Trading** | Binance Spot API via CCXT |
| **Real-Time Data** | Binance WebSocket (websockets library) |
| **Database** | MySQL 8.0 (pymysql) |
| **API Server** | FastAPI + Uvicorn |
| **Dashboard** | Next.js 16, React 19, Tailwind CSS 4, Recharts, TradingView Lightweight Charts |
| **Bot Control** | python-telegram-bot (Telegram Bot API) |
| **Deployment** | systemd services, nginx reverse proxy, logrotate |
| **Language** | Python 3.12 (bot), TypeScript (dashboard) |

---

## 🧠 Multi-Agent Claude Pipeline

The bot uses **3 specialized Claude agents** instead of a single prompt:

### Agent 1: Market Analyst
Analyzes 20+ technical indicators, derivatives data, and WebSocket real-time stream. Runs on **Claude Haiku** for speed.

**Inputs:** RSI, MACD, Bollinger Bands, Stochastic RSI, ATR, OBV, VWAP, Ichimoku, funding rate, open interest, long/short ratio, Fear & Greed index (with 7-day trend), multi-timeframe regime consensus.

### Agent 2: Sentiment Analyst
Analyzes external signals from 8 data sources. Runs in **parallel** with Agent 1.

**Inputs:** News headlines (RSS), Reddit sentiment, on-chain data (hash rate, mempool), cross-asset correlations (DXY, S&P500, Gold, VIX), Deribit options (put/call, DVOL, max pain, composite flow score), whale transactions (on-chain + Binance), MVRV-Z score.

### Agent 3: Decision Maker
Synthesizes both assessments + ML prediction + portfolio state + past lessons. Has hard rules:
- Won't buy if BTC allocation > 55%
- Won't sell if RSI > 45 (avoid panic-selling uptrends)
- Defaults to HOLD when signals conflict
- Rejects trades with R:R < 1.0

All Claude API calls are **logged** to MySQL with full prompt + response for audit transparency.

---

## 📈 ML Ensemble Model

**Architecture:** Stacking ensemble — XGBoost + LightGBM base learners → LogisticRegression meta-learner

| Metric | Value |
|--------|-------|
| Accuracy | 86% (4-fold purged walk-forward CV) |
| F1 Score | 0.80 |
| Features | 62 across 3 timeframes (1h, 4h, 1d) |
| Labels | Triple Barrier (buy/hold/sell) |
| Tuning | 100 Optuna trials (60 XGB + 40 LGB) |
| Retrain | Weekly + drift-triggered (accuracy < 47%) |

**Feature groups:** RSI, Stochastic, MACD histogram, Bollinger position/width, SMA distance, ATR%, volume ratio, returns (1h–48h), volatility (6h–48h), EMA crossovers, OBV change, candle patterns, cyclical time, on-chain (fees, mempool, hash rate), regime encoding.

---

## 🛡️ Risk Management

| Protection | How It Works |
|-----------|-------------|
| **Quarter-Kelly Sizing** | 0.25× Kelly fraction with 10pp win-rate discount, 20% hard cap |
| **ATR Regime Stops** | 1.5× ATR in loss streaks, 2× normal, 3× in high vol |
| **Circuit Breakers** | Daily 3% halt, 5 loss pause, 10% sizing cut, 20% full halt |
| **RL Position Manager** | Q-learning adjusts sizing (0.3×–1.2×) based on RSI/trend/vol/PnL state |
| **Live Approval** | Telegram ✅/❌ buttons for every trade in live mode |
| **Grid/DCA Strategy** | Auto-activates during sideways regime (BB < 6%, RSI 35-65) |
| **Flash Crash Detection** | >2% drop in 5min → emergency alert + immediate analysis |
| **Allocation Caps** | BTC max 60%, ETH max 25%, total crypto max 80% |

---

## 📡 Data Sources

All free, no paid API subscriptions required:

| Source | Data | API |
|--------|------|-----|
| **Binance** | Price, volume, funding rate, OI, long/short, liquidations | REST + WebSocket |
| **CoinGecko** | Market cap, rankings, coin details, trending | REST (free tier) |
| **Alternative.me** | Fear & Greed Index (30-day history) | REST |
| **Deribit** | Options put/call ratio, DVOL, max pain, IV skew | REST (public) |
| **Blockchain.info** | Hash rate, transaction count, large transactions | REST |
| **Mempool.space** | Fee rates, mempool size | REST |
| **Reddit RSS** | r/Bitcoin, r/CryptoCurrency sentiment | RSS |
| **CoinDesk/CoinTelegraph** | Crypto news headlines | RSS |
| **Reuters/Kitco** | Macro + gold news | RSS |
| **yfinance** | DXY, S&P500, Gold, US 10Y, VIX | Python library |
| **bitcoin-data.com** | MVRV-Z Score | REST |

---

## 📊 Dashboard

The Next.js dashboard provides full visibility into the trading system:

| Page | Description |
|------|-------------|
| **Dashboard** | Portfolio overview, performance banner, agent reasoning, derivatives panel |
| **Trades** | Full history with click-to-expand detail (3-agent output, ML, risk data) |
| **Analytics** | Sharpe, Sortino, drawdown, profit factor, per-asset breakdown (7d/30d/all) |
| **Screening** | Top 50 coins table: momentum score, sparklines, risk tiers |
| **Reports** | AI-generated market reports from Claude Opus |
| **Research** | Coin research with 5-dimension scoring |
| **AI Logs** | Every Claude API call — full prompt + response audit trail |
| **Backtests** | Historical backtest results with equity curves |
| **Lessons** | Self-correction loop + weekly Opus deep reviews |
| **Activity** | Raw bot event log (trades, errors, circuit breakers) |
| **Guide** | Complete system documentation for users |

Dashboard repo: [crypto_bot_dashboard](https://github.com/dineshstack/crypto_bot_dashboard)

---

## 💬 Telegram Commands

### Trading
| Command | Description |
|---------|-------------|
| `/start` | Start the trading loop |
| `/stop` | Pause the bot |
| `/status` | Portfolio snapshot + live WebSocket data |
| `/analyze` | Trigger immediate analysis |
| `/history` | Last 5 trades with outcomes |
| `/performance` | Full analytics report |

### Advisory
| Command | Description |
|---------|-------------|
| `/screen` | Scan top 50 coins by momentum |
| `/report` | Weekly market report (Claude Opus) |
| `/thesis SOL` | Investment thesis for any coin |
| `/thesis SOL 5000` | Thesis for $5K portfolio |
| `/newcoins` | Scan & score new/trending coins |
| `/research BTC` | Deep-dive any coin |

---

## 🚀 Installation

### Prerequisites
- Ubuntu 24.04 LTS VPS (8 CPU, 32GB RAM recommended)
- MySQL 8.0
- Python 3.12
- Node.js 22+ (for dashboard)

### Quick Deploy
```bash
git clone https://github.com/dineshstack/crypto_bot.git
cd crypto_bot
bash deploy/install.sh
```

The install script handles: Python venv, pip dependencies, systemd services, logrotate, directory structure.

### MySQL Setup
```bash
sudo mysql -e "CREATE DATABASE crypto_bot;"
sudo mysql -e "CREATE USER 'crypto_bot'@'localhost' IDENTIFIED BY 'YOUR_PASSWORD';"
sudo mysql -e "GRANT ALL ON crypto_bot.* TO 'crypto_bot'@'localhost';"
sudo mysql crypto_bot < mysql_schema.sql
```

### Train ML Model
```bash
source venv/bin/activate
python3 -c "import ml_signal, market_data as md; e=md.get_exchange(); ml_signal.train_model(e)"
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and set:

```env
# Claude AI
ANTHROPIC_API_KEY=sk-ant-...

# Binance (Spot API only, no withdrawals)
BINANCE_API_KEY=your_key
BINANCE_SECRET=your_secret

# Telegram
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# MySQL
MYSQL_HOST=127.0.0.1
MYSQL_PASSWORD=your_password

# API (for dashboard)
API_SECRET_KEY=generate_with_openssl_rand_hex_32

# Mode
TESTNET=true
```

---

## 🔴 Going Live

1. Create Binance **LIVE** API key — Spot Trading only, no withdrawals, IP-restricted
2. Set `TESTNET=false` in `.env`
3. Deposit USDT to Binance Spot wallet
4. Restart: `sudo systemctl restart crypto-bot`
5. Every trade now requires your Telegram ✅ approval before executing

---

## 📁 Project Structure

```
crypto_bot/
├── main.py                 # Telegram bot + trading loop + circuit breakers
├── claude_analyzer.py      # 3-agent Claude pipeline with full logging
├── ml_signal.py            # ML ensemble + HMM regime + training pipeline
├── market_data.py          # Technical indicators + derivatives + F&G trend
├── executor.py             # Trade execution with risk-managed sizing
├── risk_manager.py         # Quarter-Kelly + ATR stops + RL adjustment
├── database.py             # MySQL data layer
├── api_server.py           # FastAPI REST API (20+ endpoints)
├── ws_stream.py            # WebSocket real-time data + anomaly detection
├── multi_asset.py          # ETH/USDT support
├── analytics.py            # Performance metrics (Sharpe, Sortino, etc.)
├── coin_screener.py        # Top 50 coin momentum screening
├── report_generator.py     # AI market report generation
├── thesis_generator.py     # Investment thesis generator
├── grid_dca.py             # Grid/DCA sideways strategy
├── rl_position.py          # Q-learning position management
├── onchain_macro.py        # MVRV-Z score + exchange flow
├── options_data.py         # Deribit options + composite flow signal
├── cross_asset.py          # DXY, S&P500, Gold, VIX correlations
├── whale_monitor.py        # On-chain + exchange whale detection
├── social_sentiment.py     # Reddit RSS sentiment
├── news_fetcher.py         # Crypto + macro news headlines
├── onchain_data.py         # Blockchain.info + mempool.space
├── self_correction.py      # Trade outcome evaluation + lesson generation
├── weekly_review.py        # Claude Opus weekly deep review
├── coin_researcher.py      # New coin research + scoring
├── backtester.py           # Historical backtesting framework
├── config.py               # All configuration + thresholds
├── mysql_schema.sql        # Database schema (12 tables)
├── requirements.txt        # Python dependencies
├── deploy/
│   ├── install.sh          # Full deployment script
│   ├── health_check.sh     # System health checker
│   └── update.sh           # Update + restart script
└── .env.example            # Environment template
```

---

## 📚 Blog & Learning

Visit **[dineshstack.com](https://dineshstack.com)** for articles on:
- Building AI-powered trading systems
- Claude AI multi-agent architectures
- Machine learning for cryptocurrency prediction
- Real-time WebSocket data processing
- Full-stack development with Next.js + Python
- VPS deployment and DevOps

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. The authors are not financial advisors. Always do your own research and never trade with money you cannot afford to lose.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Keywords:** AI crypto trading bot, Claude AI trading, cryptocurrency automated trading, Bitcoin trading bot, Ethereum trading bot, Binance trading bot Python, machine learning crypto prediction, XGBoost cryptocurrency, sentiment analysis crypto, on-chain analytics, options flow trading, whale monitoring, portfolio management, risk management Kelly criterion, ATR stop-loss, reinforcement learning trading, Next.js trading dashboard, real-time WebSocket crypto, multi-agent AI system, Claude Haiku Opus, algorithmic trading, quantitative trading crypto, market regime detection HMM, fear greed index trading, MVRV-Z score, derivatives analysis, funding rate trading
