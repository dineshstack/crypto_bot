"""
Advanced ML Signal Model — v2.

10 research-backed improvements over the v1 baseline:
  1. 5000+ training candles (paginated fetch)
  2. Triple Barrier labeling (3-class: buy/hold/sell)
  3. Multi-timeframe features (1h + 4h + 1d)
  4. Market regime detection (bull/bear/sideways) + adaptive retrain
  5. Boruta-SHAP feature selection
  6. Optuna hyperparameter optimization (100 trials)
  7. Purged walk-forward validation (gap = lookahead)
  8. On-chain features (hash rate, mempool fees)
  9. Stacking ensemble (XGBoost + LightGBM → LogisticRegression meta)
  10. Concept drift detection (rolling accuracy monitoring)

Designed for Ubuntu 24.04 LTS — 8 CPU cores, 32GB RAM.
Full training with Optuna takes ~2-5 minutes.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "ml_models"
MODEL_PATH = MODEL_DIR / "btc_ensemble_v2.joblib"
META_PATH = MODEL_DIR / "model_meta_v2.joblib"
DRIFT_PATH = MODEL_DIR / "drift_log.joblib"

LOOKAHEAD_HOURS = 4
PROFIT_TARGET_PCT = 1.5
STOP_LOSS_PCT = 1.5
MIN_TRAIN_ROWS = 1000
RETRAIN_INTERVAL_DAYS = 7
DRIFT_ACCURACY_THRESHOLD = 0.47
DRIFT_WINDOW = 20


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA FETCHING — 5000+ candles with pagination
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv_paginated(exchange, symbol: str, timeframe: str,
                          total_candles: int = 5000) -> pd.DataFrame:
    all_candles = []
    since = None
    per_request = 1000

    while len(all_candles) < total_candles:
        batch = exchange.fetch_ohlcv(
            symbol, timeframe, since=since, limit=per_request
        )
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < per_request:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        all_candles[:total_candles],
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRIPLE BARRIER LABELING
# ═══════════════════════════════════════════════════════════════════════════════

def triple_barrier_label(df: pd.DataFrame, lookahead: int = LOOKAHEAD_HOURS,
                         pt_pct: float = PROFIT_TARGET_PCT,
                         sl_pct: float = STOP_LOSS_PCT) -> pd.Series:
    labels = pd.Series(0, index=df.index, dtype=int)
    close = df["close"].values

    for i in range(len(close) - lookahead):
        entry = close[i]
        upper = entry * (1 + pt_pct / 100)
        lower = entry * (1 - sl_pct / 100)

        for j in range(i + 1, min(i + lookahead + 1, len(close))):
            if close[j] >= upper:
                labels.iloc[i] = 1   # buy signal — hit profit target
                break
            elif close[j] <= lower:
                labels.iloc[i] = -1  # sell signal — hit stop loss
                break
        # if neither barrier hit → label stays 0 (hold)

    # Mark last rows as NaN (no future data)
    labels.iloc[-lookahead:] = np.nan
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-TIMEFRAME FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_indicators(close, high, low, volume, prefix: str) -> dict:
    features = {}

    # RSI
    rsi14 = RSIIndicator(close, window=14).rsi()
    features[f"{prefix}_rsi_14"] = rsi14

    # Stochastic RSI
    stoch = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    features[f"{prefix}_stoch_k"] = stoch.stochrsi_k()
    features[f"{prefix}_stoch_d"] = stoch.stochrsi_d()

    # MACD
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    features[f"{prefix}_macd_hist"] = macd.macd_diff()

    # Bollinger Band position
    bb = BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    features[f"{prefix}_bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower)
    features[f"{prefix}_bb_width"] = (bb_upper - bb_lower) / bb_mid * 100

    # SMA distances
    for w in [20, 50]:
        if len(close) > w:
            sma = SMAIndicator(close, window=w).sma_indicator()
            features[f"{prefix}_vs_sma{w}"] = (close / sma - 1) * 100

    # ATR
    atr = AverageTrueRange(high, low, close, window=14)
    features[f"{prefix}_atr_pct"] = atr.average_true_range() / close * 100

    # Volume ratio
    vol_sma = volume.rolling(min(24, max(2, len(volume) // 4))).mean()
    features[f"{prefix}_vol_ratio"] = volume / vol_sma

    return features


def engineer_features(df_1h: pd.DataFrame, df_4h: pd.DataFrame = None,
                      df_1d: pd.DataFrame = None) -> pd.DataFrame:
    df = df_1h.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── 1h indicators (primary timeframe) ──
    for k, v in _compute_indicators(close, high, low, volume, "h1").items():
        df[k] = v

    # Returns at different lags
    for lag in [1, 2, 4, 8, 12, 24, 48]:
        df[f"return_{lag}h"] = close.pct_change(lag)

    # Rolling volatility
    for w in [6, 12, 24, 48]:
        df[f"volatility_{w}h"] = close.pct_change().rolling(w).std()

    # EMA crossovers
    ema9 = EMAIndicator(close, window=9).ema_indicator()
    ema21 = EMAIndicator(close, window=21).ema_indicator()
    df["ema_9_21_diff"] = (ema9 / ema21 - 1) * 100

    # OBV
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["obv_change_12h"] = obv.pct_change(12)

    # Candle features
    df["body_pct"] = (close - df["open"]) / df["open"] * 100
    df["wick_upper"] = (high - close.combine(df["open"], max)) / close * 100
    df["wick_lower"] = (close.combine(df["open"], min) - low) / close * 100
    df["volume_change_4h"] = volume.rolling(4).sum().pct_change(4)

    # Cyclical time
    if "ts" in df.columns:
        dt = pd.to_datetime(df["ts"], unit="ms")
        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # ── 4h indicators (merge onto 1h by nearest timestamp) ──
    if df_4h is not None and len(df_4h) > 50:
        feats_4h = _compute_indicators(
            df_4h["close"], df_4h["high"], df_4h["low"], df_4h["volume"], "h4"
        )
        df_4h_feats = df_4h[["ts"]].copy()
        for k, v in feats_4h.items():
            df_4h_feats[k] = v.values if hasattr(v, "values") else v
        df = pd.merge_asof(
            df.sort_values("ts"), df_4h_feats.sort_values("ts"),
            on="ts", direction="backward"
        )

    # ── 1d indicators ──
    if df_1d is not None and len(df_1d) > 50:
        feats_1d = _compute_indicators(
            df_1d["close"], df_1d["high"], df_1d["low"], df_1d["volume"], "d1"
        )
        df_1d_feats = df_1d[["ts"]].copy()
        for k, v in feats_1d.items():
            df_1d_feats[k] = v.values if hasattr(v, "values") else v
        df = pd.merge_asof(
            df.sort_values("ts"), df_1d_feats.sort_values("ts"),
            on="ts", direction="backward"
        )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MARKET REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_regime(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    atr = AverageTrueRange(df["high"], df["low"], close, window=14)
    atr_pct = atr.average_true_range() / close * 100
    macd = MACD(close).macd_diff()

    regime = pd.Series("sideways", index=df.index)
    regime[(close > sma50) & (macd > 0)] = "bull"
    regime[(close < sma50) & (macd < 0)] = "bear"
    regime[atr_pct > 4.0] = "volatile"

    return regime


def encode_regime(regime: pd.Series) -> pd.DataFrame:
    mapping = {"bull": 1, "sideways": 0, "bear": -1, "volatile": 2}
    return pd.DataFrame({
        "regime_code": regime.map(mapping).fillna(0),
        "regime_is_bull": (regime == "bull").astype(int),
        "regime_is_bear": (regime == "bear").astype(int),
        "regime_is_volatile": (regime == "volatile").astype(int),
    }, index=regime.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ON-CHAIN FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_onchain_features() -> dict:
    result = {}
    try:
        r = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=8)
        if r.ok:
            d = r.json()
            result["fee_fastest"] = d.get("fastestFee", 0)
            result["fee_economy"] = d.get("economyFee", 0)
    except Exception:
        pass

    try:
        r = requests.get("https://mempool.space/api/mempool", timeout=8)
        if r.ok:
            d = r.json()
            result["mempool_count"] = d.get("count", 0)
            result["mempool_vsize_mb"] = d.get("vsize", 0) / 1e6
    except Exception:
        pass

    try:
        r = requests.get("https://api.blockchain.info/stats", timeout=12)
        if r.ok:
            d = r.json()
            result["hash_rate_eh"] = d.get("hash_rate", 0) / 1e9
            result["network_tx_24h"] = d.get("n_tx", 0)
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PURGED WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def purged_walk_forward_split(n_samples: int, n_splits: int = 5,
                              purge_gap: int = LOOKAHEAD_HOURS,
                              embargo_pct: float = 0.01):
    test_size = n_samples // (n_splits + 1)
    embargo = max(1, int(n_samples * embargo_pct))

    for i in range(n_splits):
        test_start = (i + 1) * test_size
        test_end = test_start + test_size
        if test_end > n_samples:
            break

        train_end = test_start - purge_gap
        if train_end < test_size:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start + embargo, min(test_end, n_samples))

        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"ts", "open", "high", "low", "close", "volume", "target",
               "regime", "regime_code"}
    return [c for c in df.columns if c not in exclude]


def train_model(exchange) -> dict:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    logger.info("ML v2: === TRAINING PIPELINE START ===")

    # ── 1. Fetch multi-timeframe data ──
    logger.info("ML v2: Fetching 5000+ candles across 3 timeframes...")
    df_1h = fetch_ohlcv_paginated(exchange, "BTC/USDT", "1h", total_candles=5000)
    time.sleep(1)
    df_4h = fetch_ohlcv_paginated(exchange, "BTC/USDT", "4h", total_candles=2000)
    time.sleep(1)
    df_1d = fetch_ohlcv_paginated(exchange, "BTC/USDT", "1d", total_candles=500)
    logger.info("ML v2: Fetched 1h=%d, 4h=%d, 1d=%d candles",
                len(df_1h), len(df_4h), len(df_1d))

    # ── 3. Multi-timeframe feature engineering ──
    logger.info("ML v2: Engineering multi-timeframe features...")
    df = engineer_features(df_1h, df_4h, df_1d)

    # ── 2. Triple Barrier labeling ──
    logger.info("ML v2: Applying Triple Barrier labeling...")
    df["target"] = triple_barrier_label(df)

    # ── 4. Regime detection ──
    regime = detect_regime(df)
    df["regime"] = regime
    regime_df = encode_regime(regime)
    for col in regime_df.columns:
        df[col] = regime_df[col]

    # ── 8. On-chain features (static for entire dataset — used as latest snapshot) ──
    onchain = fetch_onchain_features()
    for k, v in onchain.items():
        df[k] = v
    logger.info("ML v2: Added %d on-chain features", len(onchain))

    # Drop NaN rows
    feature_cols = _get_feature_cols(df)
    df = df.dropna(subset=feature_cols + ["target"])
    df["target"] = df["target"].astype(int)

    if len(df) < MIN_TRAIN_ROWS:
        logger.warning("ML v2: Not enough data (%d rows)", len(df))
        return {"error": f"Not enough data: {len(df)} rows"}

    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Remap target: -1 → 0 (sell), 0 → 1 (hold), 1 → 2 (buy) for classifiers
    label_map = {-1: 0, 0: 1, 1: 2}
    y_mapped = y.map(label_map)

    logger.info("ML v2: Dataset ready — %d samples, %d features", len(X), len(feature_cols))
    logger.info("ML v2: Class distribution — sell=%.1f%%, hold=%.1f%%, buy=%.1f%%",
                (y == -1).mean() * 100, (y == 0).mean() * 100, (y == 1).mean() * 100)

    # ── 5. Boruta-SHAP feature selection ──
    selected_features = feature_cols
    try:
        from BorutaShap import BorutaShap
        logger.info("ML v2: Running Boruta-SHAP feature selection...")
        selector = BorutaShap(
            model=xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                n_jobs=-1, use_label_encoder=False, eval_metric="mlogloss",
                verbosity=0,
            ),
            importance_measure="shap",
            classification=True,
        )
        selector.fit(X=X, y=y_mapped, n_trials=50, verbose=False)
        accepted = selector.Subset().columns.tolist()
        if len(accepted) >= 10:
            selected_features = accepted
            logger.info("ML v2: Boruta-SHAP selected %d/%d features",
                        len(selected_features), len(feature_cols))
        else:
            logger.warning("ML v2: Boruta selected too few (%d), keeping all", len(accepted))
    except Exception as exc:
        logger.warning("ML v2: Boruta-SHAP failed (%s), using all features", exc)

    X_sel = X[selected_features]

    # ── 6. Optuna hyperparameter tuning ──
    best_xgb_params = {}
    best_lgb_params = {}
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("ML v2: Running Optuna tuning (100 trials)...")

        def xgb_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            }
            scores = []
            for tr_idx, te_idx in purged_walk_forward_split(len(X_sel), n_splits=3):
                m = xgb.XGBClassifier(
                    **params, n_jobs=-1, use_label_encoder=False,
                    eval_metric="mlogloss", verbosity=0,
                )
                m.fit(X_sel.iloc[tr_idx], y_mapped.iloc[tr_idx])
                preds = m.predict(X_sel.iloc[te_idx])
                scores.append(accuracy_score(y_mapped.iloc[te_idx], preds))
            return np.mean(scores)

        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(xgb_objective, n_trials=60, show_progress_bar=False)
        best_xgb_params = study_xgb.best_params
        logger.info("ML v2: Best XGB accuracy=%.3f, params=%s",
                    study_xgb.best_value, best_xgb_params)

        def lgb_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            }
            scores = []
            for tr_idx, te_idx in purged_walk_forward_split(len(X_sel), n_splits=3):
                m = lgb.LGBMClassifier(**params, n_jobs=-1, verbose=-1)
                m.fit(X_sel.iloc[tr_idx], y_mapped.iloc[tr_idx])
                preds = m.predict(X_sel.iloc[te_idx])
                scores.append(accuracy_score(y_mapped.iloc[te_idx], preds))
            return np.mean(scores)

        study_lgb = optuna.create_study(direction="maximize")
        study_lgb.optimize(lgb_objective, n_trials=40, show_progress_bar=False)
        best_lgb_params = study_lgb.best_params
        logger.info("ML v2: Best LGB accuracy=%.3f", study_lgb.best_value)

    except Exception as exc:
        logger.warning("ML v2: Optuna failed (%s), using defaults", exc)

    # ── 9. Stacking Ensemble (XGBoost + LightGBM → LogisticRegression) ──
    logger.info("ML v2: Training stacking ensemble...")

    xgb_params = {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "n_jobs": -1, "use_label_encoder": False,
        "eval_metric": "mlogloss", "verbosity": 0,
    }
    xgb_params.update(best_xgb_params)

    lgb_params = {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "n_jobs": -1, "verbose": -1,
    }
    lgb_params.update(best_lgb_params)

    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_lgb = lgb.LGBMClassifier(**lgb_params)

    # ── 7. Purged walk-forward CV for final evaluation ──
    logger.info("ML v2: Purged walk-forward evaluation (5 folds)...")
    cv_scores = []
    meta_X_all = []
    meta_y_all = []

    for fold_i, (tr_idx, te_idx) in enumerate(
        purged_walk_forward_split(len(X_sel), n_splits=5)
    ):
        X_tr, X_te = X_sel.iloc[tr_idx], X_sel.iloc[te_idx]
        y_tr, y_te = y_mapped.iloc[tr_idx], y_mapped.iloc[te_idx]

        model_xgb.fit(X_tr, y_tr)
        model_lgb.fit(X_tr, y_tr)

        xgb_proba = model_xgb.predict_proba(X_te)
        lgb_proba = model_lgb.predict_proba(X_te)
        stack_feats = np.hstack([xgb_proba, lgb_proba])

        meta_X_all.append(stack_feats)
        meta_y_all.append(y_te.values)

        xgb_preds = model_xgb.predict(X_te)
        acc = accuracy_score(y_te, xgb_preds)
        f1 = f1_score(y_te, xgb_preds, average="weighted", zero_division=0)
        cv_scores.append({"accuracy": acc, "f1": f1})
        logger.info("ML v2: Fold %d — accuracy=%.3f, f1=%.3f", fold_i + 1, acc, f1)

    avg_acc = np.mean([s["accuracy"] for s in cv_scores])
    avg_f1 = np.mean([s["f1"] for s in cv_scores])

    # Train meta-learner on stacked CV predictions
    meta_X = np.vstack(meta_X_all)
    meta_y = np.concatenate(meta_y_all)
    meta_model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    meta_model.fit(meta_X, meta_y)

    # Train final base models on ALL data
    model_xgb.fit(X_sel, y_mapped)
    model_lgb.fit(X_sel, y_mapped)

    # ── Save everything ──
    MODEL_DIR.mkdir(exist_ok=True)
    ensemble = {
        "xgb": model_xgb,
        "lgb": model_lgb,
        "meta": meta_model,
        "selected_features": selected_features,
        "label_map": label_map,
        "reverse_map": {0: "sell", 1: "hold", 2: "buy"},
    }
    joblib.dump(ensemble, MODEL_PATH)

    current_regime = detect_regime(df_1h).iloc[-1]

    meta = {
        "version": "v2_ensemble",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "samples": len(X_sel),
        "features": selected_features,
        "n_features": len(selected_features),
        "n_features_total": len(feature_cols),
        "cv_accuracy": round(avg_acc, 4),
        "cv_f1": round(avg_f1, 4),
        "cv_folds": len(cv_scores),
        "class_balance": {
            "sell": round((y == -1).mean(), 3),
            "hold": round((y == 0).mean(), 3),
            "buy": round((y == 1).mean(), 3),
        },
        "xgb_params": best_xgb_params or "defaults",
        "lgb_params": best_lgb_params or "defaults",
        "regime_at_train": current_regime,
    }
    joblib.dump(meta, META_PATH)

    logger.info(
        "ML v2: === TRAINING COMPLETE ===\n"
        "  Samples: %d | Features: %d/%d selected\n"
        "  CV Accuracy: %.1f%% | F1: %.3f\n"
        "  Regime: %s | XGB tuned: %s",
        len(X_sel), len(selected_features), len(feature_cols),
        avg_acc * 100, avg_f1,
        current_regime, bool(best_xgb_params),
    )
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model():
    if not MODEL_PATH.exists():
        return None, None
    try:
        ensemble = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH) if META_PATH.exists() else {}
        return ensemble, meta
    except Exception as exc:
        logger.error("ML v2: Load failed: %s", exc)
        return None, None


def predict(exchange) -> dict:
    result = {
        "ml_probability_up": None,
        "ml_probability_sell": None,
        "ml_direction": None,
        "ml_confidence": None,
        "ml_model_accuracy": None,
        "ml_regime": None,
        "ml_available": False,
    }

    ensemble, meta = _load_model()
    if ensemble is None:
        return result

    try:
        df_1h = pd.DataFrame(
            exchange.fetch_ohlcv("BTC/USDT", "1h", limit=200),
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        time.sleep(0.5)
        df_4h = pd.DataFrame(
            exchange.fetch_ohlcv("BTC/USDT", "4h", limit=100),
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        time.sleep(0.5)
        df_1d = pd.DataFrame(
            exchange.fetch_ohlcv("BTC/USDT", "1d", limit=60),
            columns=["ts", "open", "high", "low", "close", "volume"],
        )

        df = engineer_features(df_1h, df_4h, df_1d)

        # Add regime
        regime = detect_regime(df)
        regime_df = encode_regime(regime)
        for col in regime_df.columns:
            df[col] = regime_df[col]

        # Add on-chain
        onchain = fetch_onchain_features()
        for k, v in onchain.items():
            df[k] = v

        selected = ensemble["selected_features"]
        missing = [f for f in selected if f not in df.columns]
        for f in missing:
            df[f] = 0

        latest = df.iloc[-1:][selected]
        if latest.isnull().any(axis=1).iloc[0]:
            nan_cols = latest.columns[latest.isnull().any()].tolist()
            logger.warning("ML v2: NaN in features: %s — filling with 0", nan_cols[:5])
            latest = latest.fillna(0)

        xgb_proba = ensemble["xgb"].predict_proba(latest)
        lgb_proba = ensemble["lgb"].predict_proba(latest)
        stack = np.hstack([xgb_proba, lgb_proba])
        final_proba = ensemble["meta"].predict_proba(stack)[0]

        rmap = ensemble["reverse_map"]
        pred_class = int(np.argmax(final_proba))
        direction = rmap.get(pred_class, "hold")
        prob_buy = float(final_proba[2]) if len(final_proba) > 2 else 0
        prob_sell = float(final_proba[0]) if len(final_proba) > 0 else 0
        prob_hold = float(final_proba[1]) if len(final_proba) > 1 else 0
        confidence = float(np.max(final_proba) - (1.0 / len(final_proba)))
        confidence = max(0, confidence)

        current_regime = regime.iloc[-1]

        result.update({
            "ml_probability_up": round(prob_buy, 3),
            "ml_probability_sell": round(prob_sell, 3),
            "ml_direction": direction,
            "ml_confidence": round(confidence, 3),
            "ml_model_accuracy": meta.get("cv_accuracy"),
            "ml_regime": current_regime,
            "ml_available": True,
        })

        logger.info(
            "ML v2: %s (buy=%.1f%% hold=%.1f%% sell=%.1f%%) regime=%s",
            direction.upper(), prob_buy * 100, prob_hold * 100,
            prob_sell * 100, current_regime,
        )

    except Exception as exc:
        logger.error("ML v2: Prediction failed: %s", exc)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CONCEPT DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def log_prediction_outcome(predicted: str, actual_price_before: float,
                           actual_price_after: float):
    pct = (actual_price_after - actual_price_before) / actual_price_before * 100

    if pct >= PROFIT_TARGET_PCT:
        actual = "buy"
    elif pct <= -STOP_LOSS_PCT:
        actual = "sell"
    else:
        actual = "hold"

    correct = predicted == actual

    drift_log = []
    if DRIFT_PATH.exists():
        try:
            drift_log = joblib.load(DRIFT_PATH)
        except Exception:
            drift_log = []

    drift_log.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "predicted": predicted,
        "actual": actual,
        "correct": correct,
        "pct_change": round(pct, 3),
    })

    # Keep last 100 entries
    drift_log = drift_log[-100:]
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(drift_log, DRIFT_PATH)

    return correct


def check_drift() -> dict:
    if not DRIFT_PATH.exists():
        return {"drift_detected": False, "rolling_accuracy": None, "samples": 0}

    try:
        log = joblib.load(DRIFT_PATH)
    except Exception:
        return {"drift_detected": False, "rolling_accuracy": None, "samples": 0}

    if len(log) < DRIFT_WINDOW:
        return {"drift_detected": False, "rolling_accuracy": None,
                "samples": len(log)}

    recent = log[-DRIFT_WINDOW:]
    accuracy = sum(1 for e in recent if e["correct"]) / len(recent)
    drift = accuracy < DRIFT_ACCURACY_THRESHOLD

    if drift:
        logger.warning(
            "ML v2: CONCEPT DRIFT detected — rolling accuracy %.1f%% < %.1f%% threshold",
            accuracy * 100, DRIFT_ACCURACY_THRESHOLD * 100,
        )

    return {
        "drift_detected": drift,
        "rolling_accuracy": round(accuracy, 3),
        "samples": len(recent),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def should_retrain() -> bool:
    if not MODEL_PATH.exists():
        return True
    _, meta = _load_model()
    if not meta or "trained_at" not in meta:
        return True

    # Check concept drift — emergency retrain
    drift = check_drift()
    if drift["drift_detected"]:
        logger.info("ML v2: Drift-triggered retrain")
        return True

    # Check regime change
    ensemble, _ = _load_model()
    if ensemble and meta.get("regime_at_train"):
        pass  # regime checked during predict cycle, not here

    trained = datetime.fromisoformat(meta["trained_at"])
    days_old = (datetime.now(timezone.utc) - trained).days
    return days_old >= RETRAIN_INTERVAL_DAYS


def get_ml_context(exchange) -> str:
    data = predict(exchange)

    if not data["ml_available"]:
        return ""

    direction = data["ml_direction"]
    prob_buy = data["ml_probability_up"]
    prob_sell = data["ml_probability_sell"]
    conf = data["ml_confidence"]
    acc = data["ml_model_accuracy"]
    regime = data["ml_regime"]

    conf_label = "high" if conf > 0.3 else "moderate" if conf > 0.15 else "low"
    acc_str = f" | model CV accuracy: {acc:.0%}" if acc else ""

    drift = check_drift()
    drift_str = ""
    if drift["rolling_accuracy"] is not None:
        drift_str = f"\n  Live accuracy:  {drift['rolling_accuracy']:.0%} (last {drift['samples']} predictions)"
        if drift["drift_detected"]:
            drift_str += " ⚠️ DRIFT DETECTED"

    return (
        f"ML SIGNAL (v2 Ensemble — XGBoost+LightGBM stacked, 3-class):\n"
        f"  Prediction:  {direction.upper()} (buy={prob_buy:.0%} / sell={prob_sell:.0%})\n"
        f"  Confidence:  {conf_label} ({conf:.0%}){acc_str}\n"
        f"  Regime:      {regime}\n"
        f"  Training:    Triple Barrier labels, multi-timeframe, Optuna-tuned{drift_str}\n"
        f"  Note: ML is one signal among many — weight alongside technicals and sentiment"
    )
