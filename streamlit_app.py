"""
MT5 Scalper Pro — Real-Time Trading Assistant
=============================================
Multi-strategy scalping dashboard with 6 strategies including Micro Scalp & Quick Scalp for high-frequency M1 trading.
Features:
  - Centralized PAIR_CONFIG (edit one place to change supported pairs globally)
  - TradingView-style chart (no zoom, pan only, double-click/annotation editing disabled)
  - 6 strategies: Pullback EMA, EMA Crossover, RSI Range, ATR Breakout, Micro Scalp, Quick Scalp
  - MT5 signal file export (JSON for Expert Advisor import)
  - Scanner with sound alerts + browser notifications
  - Detailed single-pair backtest with session/duration/session breakdown, best/worst trade, equity/drawdown/P&L charts
  - Session filter (London + New York)
  - Yahoo Finance v8 REST API + SDK fallback
  - Real trading risk & position size calculator
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import json
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings('ignore')

import ta

# ============================================================================
# CONSTANTS
# ============================================================================

# ============================================================================
# PAIR CONFIGURATION — Edit this section to change supported pairs globally
# ============================================================================
PAIR_CONFIG = {
    "EURUSD": {"yf_ticker": "EURUSD=X", "type": "forex",  "pip": 0.0001, "digits": 5},
    "GBPUSD": {"yf_ticker": "GBPUSD=X", "type": "forex",  "pip": 0.0001, "digits": 5},
    "USDJPY": {"yf_ticker": "USDJPY=X", "type": "forex",  "pip": 0.01,   "digits": 3},
    "XAUUSD": {"yf_ticker": "GC=F",     "type": "metal",  "pip": 0.01,   "digits": 2},
    "AUDUSD": {"yf_ticker": "AUDUSD=X", "type": "forex",  "pip": 0.0001, "digits": 5},
    "USDCAD": {"yf_ticker": "USDCAD=X", "type": "forex",  "pip": 0.0001, "digits": 5},
    "NZDUSD": {"yf_ticker": "NZDUSD=X", "type": "forex",  "pip": 0.0001, "digits": 5},
    "GBPAUD": {"yf_ticker": "GBPAUD=X", "type": "forex",  "pip": 0.0001, "digits": 5},
}

def get_pairs():
    """Returns list of enabled pair names from PAIR_CONFIG."""
    return list(PAIR_CONFIG.keys())

def get_yf_ticker(pair: str) -> str:
    return PAIR_CONFIG.get(pair, {}).get("yf_ticker", pair)

def get_pip_value(pair: str) -> float:
    if pair is None:
        return 0.0001
    return PAIR_CONFIG.get(pair, {}).get("pip", 0.0001)

def is_jpy_pair(pair: str) -> bool:
    return "JPY" in pair.upper()

# Legacy aliases for backward compatibility
ALL_PAIRS = list(PAIR_CONFIG.keys())
YF_MAP = {pair: cfg["yf_ticker"] for pair, cfg in PAIR_CONFIG.items()}
TF_INTERVAL = {"M1": "1m", "M5": "5m", "H1": "60m"}
TF_PERIOD = {"M1": "5d", "M5": "5d", "H1": "60d"}
TF_LABEL = {"M1": "1 Min", "M5": "5 Min", "H1": "1 Hr"}

# Chart layout defaults — TradingView style (pan only, no zoom, no edit)
CHART_LAYOUT = dict(
    template='plotly_dark',
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    margin=dict(l=40, r=20, t=40, b=40),
)

# ============================================================================
# STRATEGIES
# ============================================================================

STRATEGIES = {
    "Pullback EMA": {
        "description": "EMA9/21 pullback within EMA20/50 trend. RSI confirms. Best for ranging + trending markets.",
        "params": {
            "ema_fast": 9, "ema_slow": 21,
            "ema_trend_fast": 20, "ema_trend_slow": 50,
            "rsi_period": 14, "atr_period": 14,
            "sl_atr_mult": 1.2, "sl_min_atr": 0.5,
            "min_trend_dist_pct": 0.00006, "min_atr_pct": 0.00003,
            "candle_body_min": 0.50, "min_score": 60,
            "cooldown": 3, "swing_lookback": 5,
        }
    },
    "EMA Crossover": {
        "description": "Fast EMA crosses slow EMA. Price must be above/below EMA50. Pure momentum.",
        "params": {
            "ema_fast": 9, "ema_slow": 21,
            "ema_trend_fast": 20, "ema_trend_slow": 50,
            "rsi_period": 14, "atr_period": 14,
            "sl_atr_mult": 1.5, "sl_min_atr": 0.8,
            "min_trend_dist_pct": 0.00004, "min_atr_pct": 0.00002,
            "candle_body_min": 0.40, "min_score": 50,
            "cooldown": 2, "swing_lookback": 5,
        }
    },
    "RSI Range": {
        "description": "RSI crosses above oversold / below overbought threshold. EMA20/50 trend filter.",
        "params": {
            "ema_fast": 9, "ema_slow": 21,
            "ema_trend_fast": 20, "ema_trend_slow": 50,
            "rsi_period": 14, "atr_period": 14,
            "sl_atr_mult": 1.0, "sl_min_atr": 0.5,
            "min_trend_dist_pct": 0.00003, "min_atr_pct": 0.00002,
            "candle_body_min": 0.40, "min_score": 40,
            "cooldown": 2, "swing_lookback": 5,
            "rsi_buy_max": 35, "rsi_sell_min": 65,
        }
    },
    "ATR Breakout": {
        "description": "Price breaks EMA9 with ATR expansion. No RSI filter. Pure momentum + volatility.",
        "params": {
            "ema_fast": 9, "ema_slow": 21,
            "ema_trend_fast": 20, "ema_trend_slow": 50,
            "rsi_period": 14, "atr_period": 14,
            "sl_atr_mult": 2.0, "sl_min_atr": 1.0,
            "min_trend_dist_pct": 0.00010, "min_atr_pct": 0.00005,
            "candle_body_min": 0.70, "min_score": 70,
            "cooldown": 1, "swing_lookback": 3,
        }
    },
    "Micro Scalp": {
        "description": "⚡ High-frequency EMA3/7 cross within EMA20 trend. Tight SL (0.5×ATR). Targets 5-10 pips per trade. Designed for M1 scalping with multiple trades per hour.",
        "params": {
            "ema_fast": 3, "ema_slow": 7,
            "ema_trend_fast": 20, "ema_trend_slow": 50,
            "rsi_period": 6, "atr_period": 5,
            "sl_atr_mult": 0.5, "sl_min_atr": 0.3,
            "min_trend_dist_pct": 0.00002, "min_atr_pct": 0.00001,
            "candle_body_min": 0.30, "min_score": 30,
            "cooldown": 0, "swing_lookback": 2,
            "use_tp_fixed": True, "tp_pips": 8.0,
        }
    },
    "Quick Scalp": {
        "description": "⚡⚡ Ultra-fast EMA2/5 cross within EMA8 trend. Minimal SL (0.3×ATR). Fixed 4-6 pip TP. No session filter. Designed for 3-10+ trades per hour on M1.",
        "params": {
            "ema_fast": 2, "ema_slow": 5,
            "ema_trend_fast": 8, "ema_trend_slow": 20,
            "rsi_period": 4, "atr_period": 3,
            "sl_atr_mult": 0.3, "sl_min_atr": 0.15,
            "min_trend_dist_pct": 0.00001, "min_atr_pct": 0.000005,
            "candle_body_min": 0.20, "min_score": 20,
            "cooldown": 0, "swing_lookback": 1,
            "use_tp_fixed": True, "tp_pips": 5.0,
        }
    },
}
DEFAULT_STRATEGY = "Pullback EMA"

# ============================================================================
# SESSION STATE
# ============================================================================

def init_session():
    defaults = {
        "signal_history": [],
        "selected_pairs": ALL_PAIRS[:4],
        "last_signal_hashes": [],
        "refresh_count": 0,
        "pair_cooldown": {},
        "bt_result": None,
        "pinned_symbol": ALL_PAIRS[0],
        "alert_config": {"min_score": 50, "min_rr": 0.8, "sound_enabled": True, "pairs": ALL_PAIRS[:4]},
        "mt5_signal": None,
        "scanner_symbol": ALL_PAIRS[0],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ============================================================================
# DATA SOURCE
# ============================================================================

def fetch_yf_v8(symbol: str, interval: str = "1m", range_str: str = "5d") -> pd.DataFrame:
    yf_sym = YF_MAP.get(symbol.upper(), symbol)
    iv = {"1m": "1m", "5m": "5m", "60m": "60m"}.get(interval, "1m")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}"
    params = {"interval": iv, "range": range_str}
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame()
        r = result[0]
        ts = r["timestamp"]
        ohlcv = r["indicators"]["quote"][0]
        df = pd.DataFrame({
            "time": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
            "open": ohlcv["open"], "high": ohlcv["high"],
            "low": ohlcv["low"], "close": ohlcv["close"],
            "volume": ohlcv.get("volume", [0] * len(ts)),
        })
        df = df.dropna(subset=["close"])
        df["volume"] = df["volume"].fillna(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def get_candles(symbol: str, interval: str = "1m", range_str: str = "5d", count: int = 500) -> pd.DataFrame:
    df = fetch_yf_v8(symbol, interval, range_str)
    if df.empty:
        try:
            import yfinance
            yf_sym = YF_MAP.get(symbol.upper(), symbol)
            t = yfinance.Ticker(yf_sym)
            raw = t.history(period="5d", interval=interval, prepost=False)
            if not raw.empty:
                df = raw.reset_index()
                df.columns = [c.capitalize() if c != 'Datetime' else 'time' for c in df.columns]
                df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
                df = df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                df['volume'] = df['volume'].fillna(0)
        except Exception:
            return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.drop_duplicates(subset='time', keep='last')
    return df.tail(count).reset_index(drop=True)

# ============================================================================
# INDICATORS
# ============================================================================

def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=params["ema_fast"]).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=params["ema_slow"]).ema_indicator()
    df['ema_trend_fast'] = ta.trend.EMAIndicator(df['close'], window=params["ema_trend_fast"]).ema_indicator()
    df['ema_trend_slow'] = ta.trend.EMAIndicator(df['close'], window=params["ema_trend_slow"]).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=params["rsi_period"]).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=params["atr_period"]).average_true_range()
    df['atr_pct'] = df['atr'] / df['close']
    df['trend_dist'] = abs(df['ema_trend_fast'] - df['ema_trend_slow'])
    df['candle_range'] = df['high'] - df['low']
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_body_pct'] = (df['candle_body'] / df['candle_range'].replace(0, np.nan)).fillna(0).clip(0, 1)
    df['rsi_change'] = df['rsi'].diff()
    return df

def swing_high(df, idx, lookback):
    return df['high'].iloc[max(0, idx-lookback):idx+1].max()

def swing_low(df, idx, lookback):
    return df['low'].iloc[max(0, idx-lookback):idx+1].min()

# ============================================================================
# SESSION HELPERS
# ============================================================================

def session_active(dt) -> bool:
    h = dt.hour
    return (8 <= h < 12) or (13 <= h <= 17)


# ─── patched session check that backtest can override ────────────────────────
def _check_session(dt, params) -> bool:
    """Returns True if session filter passes, or if _no_session_filter is set."""
    if params.get("_no_session_filter"):
        return True
    return session_active(dt)

def get_sessions(dt):
    h = dt.hour
    return {"London": 8 <= h < 12, "NewYork": 13 <= h <= 17, "Asia": 0 <= h < 7}

def session_boxes(dt) -> str:
    sess = get_sessions(dt)
    boxes = []
    for name, active in sess.items():
        c = '#4caf50' if active else '#424242'
        txt = 'white' if active else '#888'
        boxes.append(f"<div style='background:{c};border-radius:8px;padding:7px 12px;color:{txt};font-weight:bold;font-size:12px;'>● {name}</div>")
    return f"<div style='display:flex;gap:8px;flex-wrap:wrap;'>{''.join(boxes)}</div>"

def rsi_display(v):
    if v > 70: return f"{v:.1f} 🔴", "#ef5350"
    elif v < 30: return f"{v:.1f} 🟢", "#4caf50"
    elif v > 60 or v < 40: return f"{v:.1f} 🟡", "#ff9800"
    return f"{v:.1f} ⚪", "#9e9e9e"

# ============================================================================
# SIGNAL SCORE
# ============================================================================

def score_signal(df, idx, direction, params):
    row = df.iloc[idx]
    dist = row['trend_dist']
    max_dist = params['min_trend_dist_pct'] * 5
    trend = min(30.0, (dist / max_dist) * 30.0) if dist > 0 else 0.0
    rsi = row['rsi']
    rsi_dist = max(0, rsi - 50) if direction == 'BUY' else max(0, 50 - rsi)
    rsi_score = min(25.0, (rsi_dist / 35) * 25.0)
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    candle_score = min(25.0, (body / 0.80) * 25.0)
    atr_ratio = row['atr_pct'] / (params['min_atr_pct'] + 1e-10)
    atr_score = min(20.0, max(0.0, (atr_ratio - 0.5) * 10.0))
    return round(trend + rsi_score + candle_score + atr_score, 1)

# ============================================================================
# STOP LOSS / TAKE PROFIT
# ============================================================================

def calc_sl(signal, entry, df, idx, params):
    lb = params['swing_lookback']
    atr = df.iloc[idx]['atr']
    if signal == 'BUY':
        sl = swing_low(df, idx, lb) - (atr * params['sl_atr_mult'])
        min_sl = entry - (atr * params['sl_min_atr'])
        if entry - sl < atr * params['sl_min_atr']:
            sl = min_sl
    else:
        sl = swing_high(df, idx, lb) + (atr * params['sl_atr_mult'])
        min_sl = entry + (atr * params['sl_min_atr'])
        if sl - entry < atr * params['sl_min_atr']:
            sl = min_sl
    return sl

def calc_tp(signal, entry, sl, df, idx, params, symbol: str = None):
    if params.get('use_tp_fixed') and params.get('tp_pips'):
        # Micro Scalp / Quick Scalp: fixed TP in pips
        pip_val = get_pip_value(symbol)
        tp_pips = params['tp_pips']
        return entry + (tp_pips * pip_val) if signal == 'BUY' else entry - (tp_pips * pip_val)
    # Dynamic TP
    atr_ratio = df.iloc[idx]['atr'] / (params['min_atr_pct'] + 1e-10)
    rr = 1.2 if atr_ratio < 1.0 else 1.35 if atr_ratio < 2.0 else 1.5
    risk = abs(entry - sl)
    return entry + (risk * rr) if signal == 'BUY' else entry - (risk * rr)

# ============================================================================
# STRATEGY SIGNAL DETECTION
# ============================================================================

def find_signal(df, idx, strategy, params):
    if idx < 2:
        return None
    row, prev = df.iloc[idx], df.iloc[idx - 1]

    if strategy == "Pullback EMA":
        return _sig_pullback_ema(df, idx, params)
    elif strategy == "EMA Crossover":
        return _sig_ema_crossover(df, idx, params)
    elif strategy == "RSI Range":
        return _sig_rsi_range(df, idx, params)
    elif strategy == "ATR Breakout":
        return _sig_atr_breakout(df, idx, params)
    elif strategy == "Micro Scalp":
        return _sig_micro_scalp(df, idx, params)
    elif strategy == "Quick Scalp":
        return _sig_quick_scalp(df, idx, params)
    return None


def _sig_pullback_ema(df, idx, params):
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    cross_up = prev['ema_fast'] <= prev['ema_slow'] and row['ema_fast'] > row['ema_slow']
    cross_down = prev['ema_fast'] >= prev['ema_slow'] and row['ema_fast'] < row['ema_slow']
    if not (cross_up or cross_down): return None
    direction = 'BUY' if cross_up else 'SELL'
    if direction == 'BUY' and not (row['ema_trend_fast'] > row['ema_trend_slow']): return None
    if direction == 'SELL' and not (row['ema_trend_fast'] < row['ema_trend_slow']): return None
    if direction == 'BUY' and not (row['rsi'] > 50 and row['rsi_change'] > 0): return None
    if direction == 'SELL' and not (row['rsi'] < 50 and row['rsi_change'] < 0): return None
    if row['trend_dist'] < params['min_trend_dist_pct']: return None
    if row['atr_pct'] < params['min_atr_pct']: return None
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    if body < params['candle_body_min']: return None
    if not _check_session(row['time'], params): return None
    return direction


def _sig_ema_crossover(df, idx, params):
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    cross_up = prev['ema_fast'] <= prev['ema_slow'] and row['ema_fast'] > row['ema_slow']
    cross_down = prev['ema_fast'] >= prev['ema_slow'] and row['ema_fast'] < row['ema_slow']
    if not (cross_up or cross_down): return None
    direction = 'BUY' if cross_up else 'SELL'
    if direction == 'BUY' and row['close'] <= row['ema_trend_slow']: return None
    if direction == 'SELL' and row['close'] >= row['ema_trend_slow']: return None
    if row['atr_pct'] < params['min_atr_pct']: return None
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    if body < params['candle_body_min']: return None
    if not _check_session(row['time'], params): return None
    return direction


def _sig_rsi_range(df, idx, params):
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    rsi_cross_up = prev['rsi'] < params.get('rsi_buy_max', 35) and row['rsi'] >= params.get('rsi_buy_max', 35)
    rsi_cross_down = prev['rsi'] > params.get('rsi_sell_min', 65) and row['rsi'] <= params.get('rsi_sell_min', 65)
    if not (rsi_cross_up or rsi_cross_down): return None
    direction = 'BUY' if rsi_cross_up else 'SELL'
    if direction == 'BUY' and not (row['ema_trend_fast'] > row['ema_trend_slow']): return None
    if direction == 'SELL' and not (row['ema_trend_fast'] < row['ema_trend_slow']): return None
    if row['atr_pct'] < params['min_atr_pct']: return None
    if not _check_session(row['time'], params): return None
    return direction


def _sig_atr_breakout(df, idx, params):
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    buy_cond = row['close'] > row['ema_fast'] and row['atr_pct'] > params['min_atr_pct']
    sell_cond = row['close'] < row['ema_fast'] and row['atr_pct'] > params['min_atr_pct']
    if not (buy_cond or sell_cond): return None
    direction = 'BUY' if buy_cond else 'SELL'
    if direction == 'BUY' and not (row['ema_trend_fast'] > row['ema_trend_slow']): return None
    if direction == 'SELL' and not (row['ema_trend_fast'] < row['ema_trend_slow']): return None
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    if body < params['candle_body_min']: return None
    if not _check_session(row['time'], params): return None
    return direction


def _sig_micro_scalp(df, idx, params):
    """
    ⚡ Micro Scalp: EMA3/7 ultra-fast cross within EMA20/50 trend.
    Targets multiple trades per hour with tight 0.5×ATR SL and fixed TP.
    RSI momentum confirms. No session filter for max signals.
    """
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    cross_up = prev['ema_fast'] <= prev['ema_slow'] and row['ema_fast'] > row['ema_slow']
    cross_down = prev['ema_fast'] >= prev['ema_slow'] and row['ema_fast'] < row['ema_slow']
    if not (cross_up or cross_down): return None
    direction = 'BUY' if cross_up else 'SELL'
    if direction == 'BUY' and not (row['ema_trend_fast'] > row['ema_trend_slow']): return None
    if direction == 'SELL' and not (row['ema_trend_fast'] < row['ema_trend_slow']): return None
    if direction == 'BUY' and not (row['rsi'] > 50): return None
    if direction == 'SELL' and not (row['rsi'] < 50): return None
    if row['atr_pct'] < params['min_atr_pct']: return None
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    if body < params['candle_body_min']: return None
    # No session filter — trade all day for max signals
    return direction


def _sig_quick_scalp(df, idx, params):
    """
    ⚡⚡ Quick Scalp: Ultra-fast EMA2/5 cross within EMA8/20 trend.
    Maximum frequency M1 scalping with minimal SL (0.3×ATR) and fixed 5-pip TP.
    No session filter. Designed for 3-10+ trades per hour.
    """
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    cross_up = prev['ema_fast'] <= prev['ema_slow'] and row['ema_fast'] > row['ema_slow']
    cross_down = prev['ema_fast'] >= prev['ema_slow'] and row['ema_fast'] < row['ema_slow']
    if not (cross_up or cross_down): return None
    direction = 'BUY' if cross_up else 'SELL'
    if direction == 'BUY' and not (row['ema_trend_fast'] > row['ema_trend_slow']): return None
    if direction == 'SELL' and not (row['ema_trend_fast'] < row['ema_trend_slow']): return None
    if direction == 'BUY' and not (row['rsi'] > 50): return None
    if direction == 'SELL' and not (row['rsi'] < 50): return None
    if row['atr_pct'] < params['min_atr_pct']: return None
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    if body < params['candle_body_min']: return None
    # No session filter — trade all day
    return direction

# ============================================================================
# LIVE SCAN
# ============================================================================

def scan_pair(symbol: str, interval: str, strategy: str, params: dict) -> dict | None:
    tf_int = TF_INTERVAL.get(interval, "1m")
    tf_rng = TF_PERIOD.get(interval, "5d")
    df = get_candles(symbol, tf_int, tf_rng, count=150)
    if df.empty or len(df) < 30:
        return None
    df = add_indicators(df, params)
    cooldown = st.session_state.pair_cooldown.get(symbol, 0)
    if cooldown > 0:
        return None
    for ci in range(-12, -1):
        direction = find_signal(df, ci, strategy, params)
        if direction is None:
            continue
        score = score_signal(df, ci, direction, params)
        if score < params['min_score']:
            continue
        entry = df.iloc[ci]['close']
        sl = calc_sl(direction, entry, df, ci, params)
        tp = calc_tp(direction, entry, sl, df, ci, params, symbol)
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0
        return {
            'symbol': symbol, 'signal': direction,
            'entry': round(entry, 5), 'sl': round(sl, 5), 'tp': round(tp, 5),
            'rr': rr, 'score': score,
            'time': df.iloc[ci]['time'],
            'rsi': round(df.iloc[ci]['rsi'], 1),
            'atr': round(df.iloc[ci]['atr'], 5),
            'trend_dist': round(df.iloc[ci]['trend_dist'], 6),
            'candle_pct': round(df.iloc[ci]['candle_body_pct'] * 100, 0),
            'timeframe': interval, 'strategy': strategy,
        }
    return None

def scan_pair_historical(symbol: str, interval: str, strategy: str, params: dict, max_trades: int = 10) -> list:
    """Scan recent candles and return the last N completed signals (non-overlapping)."""
    tf_int = TF_INTERVAL.get(interval, "1m")
    tf_rng = TF_PERIOD.get(interval, "5d")
    # Use 500 candles to maximize historical lookback; more candles = more completed trades
    df = get_candles(symbol, tf_int, tf_rng, count=500)
    if df.empty or len(df) < 50:
        return []
    df = add_indicators(df, params)
    df = df.reset_index(drop=True)

    # Disable session filter for historical scan so we get trades from all hours
    scan_params = {**params, "_no_session_filter": True}

    signals_found = []
    position = None
    lookback = max(50, params.get('ema_trend_slow', 50))

    # Scan from lookback to near the end (don't use last 2 bars — they may be incomplete)
    for i in range(lookback, len(df) - 2):
        direction = find_signal(df, i, strategy, scan_params)
        if direction and position is None:
            score = score_signal(df, i, direction, params)
            if score >= params['min_score']:
                entry = df.iloc[i]['close']
                sl = calc_sl(direction, entry, df, i, params)
                tp = calc_tp(direction, entry, sl, df, i, params, symbol)
                position = direction
                pos_entry, pos_sl, pos_tp = entry, sl, tp
                pos_score, pos_time = score, df.iloc[i]['time']
                pos_idx = i
            continue

        if position:
            close = df.iloc[i]['close']
            exit_p, result = None, None
            if position == 'BUY':
                if close <= pos_sl:
                    exit_p, result = pos_sl, 'LOSS'
                elif close >= pos_tp:
                    exit_p, result = pos_tp, 'WIN'
            else:
                if close >= pos_sl:
                    exit_p, result = pos_sl, 'LOSS'
                elif close <= pos_tp:
                    exit_p, result = pos_tp, 'WIN'
            if result:
                risk = abs(pos_entry - pos_sl)
                reward = abs(exit_p - pos_entry)
                rr = round(reward / risk, 2) if risk > 0 else 0
                pip_val = get_pip_value(symbol)
                signals_found.append({
                    'symbol': symbol,
                    'signal': position,
                    'direction': position,
                    'entry': round(pos_entry, 5),
                    'exit': round(exit_p, 5),
                    'sl': round(pos_sl, 5),
                    'tp': round(pos_tp, 5),
                    'rr': rr,
                    'score': pos_score,
                    'result': result,
                    'pnl_pips': round((reward / pip_val) if result == 'WIN' else -(risk / pip_val), 1),
                    'entry_time': pos_time,
                    'exit_time': df.iloc[i]['time'],
                    'entry_idx': pos_idx,
                    'exit_idx': i,
                    'timeframe': interval,
                    'strategy': strategy,
                    'rsi': round(df.iloc[pos_idx]['rsi'], 1),
                    'atr': round(df.iloc[pos_idx]['atr'], 5),
                    'atr_pct': round(df.iloc[pos_idx]['atr_pct'], 6),
                    'candle_pct': round(df.iloc[pos_idx]['candle_body_pct'] * 100, 0),
                    'trend_dist': round(df.iloc[pos_idx]['trend_dist'], 6),
                    'hold_candles': i - pos_idx,
                    'duration_min': round((df.iloc[i]['time'] - pos_time).total_seconds() / 60, 1),
                })
                position = None
                if len(signals_found) >= max_trades:
                    break

    # Return in chronological order (oldest first)
    return list(reversed(signals_found))


def scan_all_pairs(strategy, params) -> list:
    pairs = st.session_state.selected_pairs
    interval = st.session_state.get("active_tf", "M1")
    if not pairs:
        return []
    ac = st.session_state.alert_config
    signals = []
    with ThreadPoolExecutor(max_workers=min(len(pairs), 8)) as ex:
        futures = {ex.submit(scan_pair, p, interval, strategy, params): p for p in pairs}
        for f in as_completed(futures):
            try:
                s = f.result()
                if s and s['score'] >= ac['min_score'] and s['rr'] >= ac['min_rr']:
                    signals.append(s)
            except Exception:
                pass
    return signals

# ============================================================================
# MT5 SIGNAL FILE GENERATOR
# ============================================================================

def generate_mt5_signal(sig: dict, params: dict) -> dict:
    """
    Generate MT5-compatible signal dict.
    Creates a JSON signal that can be written to a file for MT5 EA to read.
    Format compatible with common MT5 file-based signal systems.
    """
    pip_val = get_pip_value(sig['symbol'])
    sl_pips = abs(sig['entry'] - sig['sl']) / pip_val
    tp_pips = abs(sig['tp'] - sig['entry']) / pip_val
    return {
        "action": "OPEN",
        "symbol": sig['symbol'],
        "type": "BUY" if sig['signal'] == 'BUY' else "SELL",
        "entry": sig['entry'],
        "stop_loss": sig['sl'],
        "take_profit": sig['tp'],
        "sl_pips": round(sl_pips, 1),
        "tp_pips": round(tp_pips, 1),
        "risk": "MEDIUM",
        "score": sig['score'],
        "strategy": sig['strategy'],
        "timeframe": sig['timeframe'],
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "magic": 2024,
    }


def mt5_signal_file(symbol: str, signals: list) -> bytes:
    """Generate JSON file with current signals for MT5 EA."""
    data = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "symbol": symbol,
        "total_signals": len(signals),
        "signals": [generate_mt5_signal(s, {}) for s in signals],
    }
    return json.dumps(data, indent=2).encode('utf-8')

def _serialize_trade(row: pd.Series) -> dict:
    """Convert a trade row to a JSON-safe dict with all types coerced."""
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, pd.Timestamp):
            out[k] = str(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out

# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(symbol: str, candles: int, interval: str, strategy: str, params: dict) -> tuple:
    tf_int = TF_INTERVAL.get(interval, "1m")
    tf_rng = TF_PERIOD.get(interval, "5d")
    df = get_candles(symbol, tf_int, tf_rng, count=candles)
    if df.empty or len(df) < 100:
        return None, None, None
    df = add_indicators(df, params)
    df = df.reset_index(drop=True)
    trades = []
    position = None
    cooldown = 0
    start = max(50, params['ema_trend_slow'])
    for i in range(start, len(df) - 1):
        if cooldown > 0:
            cooldown -= 1
        direction = find_signal(df, i, strategy, params)
        if direction and cooldown == 0 and position is None:
            score = score_signal(df, i, direction, params)
            if score >= params['min_score']:
                entry = df.iloc[i]['close']
                sl = calc_sl(direction, entry, df, i, params)
                tp = calc_tp(direction, entry, sl, df, i, params, symbol)
                position = direction
                pos_entry, pos_sl, pos_tp = entry, sl, tp
                pos_score, pos_time = score, df.iloc[i]['time']
                cooldown = params['cooldown']
            continue
        if position:
            close = df.iloc[i]['close']
            exit_p, result = None, None
            if position == 'BUY':
                if close <= pos_sl: exit_p, result = pos_sl, 'LOSS'
                elif close >= pos_tp: exit_p, result = pos_tp, 'WIN'
            else:
                if close >= pos_sl: exit_p, result = pos_sl, 'LOSS'
                elif close <= pos_tp: exit_p, result = pos_tp, 'WIN'
            if result:
                risk = abs(pos_entry - pos_sl)
                reward = abs(exit_p - pos_entry)
                rr = round(reward / risk, 2) if risk > 0 else 0
                pip_val = get_pip_value(symbol)
                pnl = (reward / pip_val) if result == 'WIN' else -(risk / pip_val)
                hold_candles = i - df[df['time'] == pos_time].index[0]
                trades.append({
                    'signal': position, 'entry': pos_entry, 'exit': exit_p,
                    'sl': pos_sl, 'tp': pos_tp, 'rr': rr,
                    'pnl_pips': round(pnl, 1), 'result': result,
                    'score': pos_score, 'entry_time': pos_time, 'exit_time': df.iloc[i]['time'],
                    'hold_candles': hold_candles,
                    'sl_pips': round(abs(pos_entry - pos_sl) / pip_val, 1),
                    'tp_pips': round(abs(pos_tp - pos_entry) / pip_val, 1),
                })
                position = None
    if not trades:
        return None, df, None
    td = pd.DataFrame(trades)
    total = len(td)
    wins = len(td[td['result'] == 'WIN'])
    losses = total - wins
    wr = wins / total * 100 if total > 0 else 0
    total_pnl = td['pnl_pips'].sum()
    avg_rr = td['rr'].mean()
    avg_win = td[td['result'] == 'WIN']['pnl_pips'].mean() if wins > 0 else 0
    avg_loss = abs(td[td['result'] == 'LOSS']['pnl_pips'].mean()) if losses > 0 else 0
    pf = abs(td[td['result'] == 'WIN']['pnl_pips'].sum() / td[td['result'] == 'LOSS']['pnl_pips'].sum()) if losses > 0 and td[td['result'] == 'LOSS']['pnl_pips'].sum() != 0 else float('inf')
    cum = td['pnl_pips'].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()
    expectancy = (td['pnl_pips'] / td['rr'].replace(0, np.nan)).mean() if not td.empty else 0
    td['win_streak'] = (td['result'] == 'WIN').astype(int) * (td.groupby((td['result'] != td['result'].shift()).cumsum()).cumcount() + 1) * (td['result'] == 'WIN').astype(int)
    td['loss_streak'] = (td['result'] == 'LOSS').astype(int) * (td.groupby((td['result'] != td['result'].shift()).cumsum()).cumcount() + 1) * (td['result'] == 'LOSS').astype(int)
    max_win_streak = int(td[td['result']=='WIN']['win_streak'].max()) if wins > 0 else 0
    max_loss_streak = int(td[td['result']=='LOSS']['loss_streak'].max()) if losses > 0 else 0
    td['hour'] = pd.to_datetime(td['entry_time']).dt.hour
    hourly = td.groupby('hour')['pnl_pips'].agg(['sum', 'count']).reset_index()
    hourly.columns = ['hour', 'pnl', 'count']
    td['month'] = pd.to_datetime(td['entry_time']).dt.to_period('M')
    monthly = td.groupby('month')['pnl_pips'].agg(['sum', 'count']).reset_index()
    monthly.columns = ['month', 'pnl', 'count']
    # Session breakdown: London 8-12, NY 13-17
    def get_session(h):
        if 8 <= h < 12: return 'London'
        elif 13 <= h <= 17: return 'NewYork'
        else: return 'Other'
    td['session'] = td['hour'].apply(get_session)
    session_grp = td.groupby('session')['pnl_pips'].agg(['sum', 'count', 'mean']).reset_index()
    session_grp.columns = ['session', 'pnl', 'count', 'avg_pnl']
    # Duration in minutes
    td['duration_min'] = (pd.to_datetime(td['exit_time']) - pd.to_datetime(td['entry_time'])).dt.total_seconds() / 60
    dur_bins = [0, 1, 3, 5, 15, 60, 999]
    dur_labels = ['<1m', '1-3m', '3-5m', '5-15m', '15-60m', '>60m']
    td['dur_bucket'] = pd.cut(td['duration_min'], bins=dur_bins, labels=dur_labels, right=True)
    dur_grp = td.groupby('dur_bucket', observed=True)['pnl_pips'].agg(['sum', 'count', 'mean']).reset_index()
    dur_grp.columns = ['duration', 'pnl', 'count', 'avg_pnl']
    buy_td = td[td['signal'] == 'BUY']
    sell_td = td[td['signal'] == 'SELL']
    return td, df, {
        'total': total, 'wins': wins, 'losses': losses,
        'win_rate': round(wr, 1), 'total_pnl': round(total_pnl, 1),
        'avg_rr': round(avg_rr, 2), 'profit_factor': round(pf, 2) if pf != float('inf') else '∞',
        'max_drawdown': round(max_dd, 1),
        'avg_win': round(avg_win, 1), 'avg_loss': round(avg_loss, 1),
        'best': td['pnl_pips'].max(), 'worst': td['pnl_pips'].min(),
        'expectancy': round(expectancy, 2),
        'max_win_streak': max_win_streak, 'max_loss_streak': max_loss_streak,
        'avg_hold': round(td['hold_candles'].mean(), 1),
        'buy_wr': round(len(buy_td[buy_td['result']=='WIN'])/len(buy_td)*100, 1) if len(buy_td) > 0 else 0,
        'sell_wr': round(len(sell_td[sell_td['result']=='WIN'])/len(sell_td)*100, 1) if len(sell_td) > 0 else 0,
        'buy_pnl': round(buy_td['pnl_pips'].sum(), 1),
        'sell_pnl': round(sell_td['pnl_pips'].sum(), 1),
        'candles_used': len(df),
        'hourly': hourly.to_dict('records'),
        'monthly': monthly.to_dict('records'),
        'trades_per_hour': round(total / (len(df) / 60), 2) if len(df) > 0 else 0,
        # New detailed metrics
        'session_breakdown': session_grp.to_dict('records'),
        'duration_breakdown': dur_grp.to_dict('records'),
        'avg_duration_min': round(td['duration_min'].mean(), 1),
        'median_duration_min': round(td['duration_min'].median(), 1),
        'best_trade': _serialize_trade(td.loc[td['pnl_pips'].idxmax()]) if len(td) > 0 else {},
        'worst_trade': _serialize_trade(td.loc[td['pnl_pips'].idxmin()]) if len(td) > 0 else {},
        'total_hours': round(len(df) / 60, 1),
        'recovery_factor': round(abs(total_pnl / max_dd), 2) if max_dd != 0 else '∞',
        'avg_trade_pnl': round(total_pnl / total, 2) if total > 0 else 0,
        'sharp_ratio_like': round((td['pnl_pips'].mean() / td['pnl_pips'].std()), 2) if td['pnl_pips'].std() > 0 else 0,
    }

# ============================================================================
# PLOT: CANDLESTICK + INDICATORS (TradingView-like, pan only)
# ============================================================================

def plot_chart(df, trades_df=None, sig=None, params=None):
    if df is None or df.empty:
        return None
    if params is None:
        params = {}
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.20, 0.20, 0.20],
        vertical_spacing=0.06,
        subplot_titles=('', '', '', ''),
    )
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        name='Price', hoverinfo='x+y'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_fast'], name=f"EMA{params.get('ema_fast', 9)}",
        line=dict(color='#2196F3', width=1.5), hoverinfo='y'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_slow'], name=f"EMA{params.get('ema_slow', 21)}",
        line=dict(color='#FF9800', width=1.5), hoverinfo='y'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_trend_fast'], name=f"EMA{params.get('ema_trend_fast', 20)}",
        line=dict(color='#00BCD4', width=1.2, dash='dash'), opacity=0.7, hoverinfo='y'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_trend_slow'], name=f"EMA{params.get('ema_trend_slow', 50)}",
        line=dict(color='#E91E63', width=1.2, dash='dash'), opacity=0.7, hoverinfo='y'), row=1, col=1)

    if 'volume' in df.columns and df['volume'].sum() > 0:
        colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name='Volume',
            marker_color=colors, opacity=0.6, hoverinfo='y'), row=2, col=1)

    if sig is not None and len(df) > 1:
        c = '#26a69a' if sig['signal'] == 'BUY' else '#ef5350'
        fig.add_trace(go.Scatter(
            x=[sig['time']], y=[sig['entry']],
            mode='markers+text',
            marker=dict(size=22, color=c, symbol='arrow-up' if sig['signal'] == 'BUY' else 'arrow-down'),
            text=[f"▶ {sig['signal']} {sig['score']}"],
            textposition='top center', textfont=dict(color=c, size=11),
            name=f"SIGNAL {sig['signal']}", hoverinfo='text'
        ), row=1, col=1)
        fig.add_hline(y=sig['sl'], line_dash='dash', line_color='#f44336', line_width=1.2,
            annotation_text=f"SL {sig['sl']}", annotation_position="bottom-right",
            annotation_font_color='#f44336', row=1, col=1)
        fig.add_hline(y=sig['tp'], line_dash='dash', line_color='#4caf50', line_width=1.2,
            annotation_text=f"TP {sig['tp']}", annotation_position="top right",
            annotation_font_color='#4caf50', row=1, col=1)

    if trades_df is not None and not trades_df.empty:
        for _, t in trades_df.iterrows():
            c = '#26a69a' if t['signal'] == 'BUY' else '#ef5350'
            fig.add_trace(go.Scatter(x=[t['entry_time']], y=[t['entry']],
                mode='markers', marker=dict(size=9, color=c, symbol='circle'),
                showlegend=False, hoverinfo='y'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[t['exit_time']], y=[t['exit']],
                mode='markers', marker=dict(size=8, color='white', symbol='x'),
                showlegend=False, hoverinfo='y'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], name='RSI',
        line=dict(color='#9C27B0', width=1.5), hoverinfo='y'), row=3, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='gray', row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', line_width=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', line_width=0.5, row=3, col=1)

    fig.add_trace(go.Scatter(x=df['time'], y=df['atr'], name='ATR',
        fill='tozeroy', line=dict(color='#FF5722', width=1),
        fillcolor='rgba(255,87,34,0.12)', hoverinfo='y'), row=4, col=1)

    fig.update_layout(
        template='plotly_dark', height=720, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=50, r=20, t=30, b=40),
    )
    return fig

# ============================================================================
# PLOT: EQUITY CURVE
# ============================================================================

def equity_chart(td):
    if td is None or td.empty:
        return None
    cum = td['pnl_pips'].cumsum()
    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in cum.values]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cum)+1)), y=cum.values,
        mode='lines+markers', line=dict(color='#26a69a', width=2),
        marker=dict(color=colors, size=4),
        name='Equity (pips)', fill='tozeroy', fillcolor='rgba(38,166,154,0.1)',
        hoverinfo='x+y'
    ))
    fig.update_layout(
        template='plotly_dark', height=220,
        xaxis_title="Trade #", yaxis_title="P&L (pips)",
        hovermode='x unified', margin=dict(l=50, r=20, t=10, b=40),
    )
    fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=0.5)
    return fig

# ============================================================================
# PLOT: DRAWDOWN
# ============================================================================

def drawdown_chart(td):
    if td is None or td.empty:
        return None
    cum = td['pnl_pips'].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(dd)+1)), y=dd.values,
        mode='lines+markers', line=dict(color='#ef5350', width=2),
        marker=dict(color='#ef5350', size=3),
        fill='tozeroy', fillcolor='rgba(239,83,80,0.15)',
        name='Drawdown (pips)', hoverinfo='x+y'
    ))
    fig.update_layout(
        template='plotly_dark', height=170,
        xaxis_title="Trade #", yaxis_title="Drawdown (pips)",
        hovermode='x unified', margin=dict(l=50, r=20, t=10, b=40),
    )
    fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=0.5)
    return fig

# ============================================================================
# PLOT: P&L DISTRIBUTION
# ============================================================================

def pnl_dist_chart(td):
    if td is None or td.empty:
        return None
    fig = go.Figure()
    win_pnls = td[td['result']=='WIN']['pnl_pips']
    loss_pnls = td[td['result']=='LOSS']['pnl_pips']
    fig.add_trace(go.Histogram(x=win_pnls, name='Wins', marker_color='#26a69a', opacity=0.75, nbinsx=25))
    fig.add_trace(go.Histogram(x=loss_pnls, name='Losses', marker_color='#ef5350', opacity=0.75, nbinsx=25))
    fig.update_layout(
        template='plotly_dark', height=200,
        xaxis_title="P&L (pips)", yaxis_title="Count",
        barmode='overlay', margin=dict(l=50, r=20, t=10, b=40),
        hovermode='x unified',
    )
    return fig

# ============================================================================
# PLOT: HOURLY RETURNS
# ============================================================================

def hourly_chart(metrics):
    records = metrics.get('hourly', [])
    if not records:
        return None
    hr = pd.DataFrame(records)
    hr['hour'] = hr['hour'].astype(int)
    fig = go.Figure(data=go.Bar(
        x=hr['hour'], y=hr['pnl'],
        marker_color=hr['pnl'],
        marker=dict(coloraxis='coloraxis'),
        name='P&L by Hour', hoverinfo='x+y'
    ))
    fig.update_layout(
        template='plotly_dark', height=200,
        xaxis_title="Hour (UTC)", yaxis_title="Net P&L (pips)",
        coloraxis=dict(colorscale='RdYlGn'), margin=dict(l=50, r=20, t=10, b=40),
        hovermode='x unified',
    )
    return fig

# ============================================================================
# PLOT: MONTHLY RETURNS HEATMAP
# ============================================================================

def monthly_heatmap(metrics):
    records = metrics.get('monthly', [])
    if not records:
        return None
    mr = pd.DataFrame(records)
    mr['month'] = mr['month'].astype(str)
    fig = go.Figure(data=go.Heatmap(
        x=mr['month'], y=['P&L'],
        z=mr['pnl'], text=mr[['pnl', 'count']].apply(lambda r: f"{r['pnl']:.0f}p / {r['count']}t", axis=1),
        texttemplate="%{text}", textfont={"color": "white"},
        colorscale='RdYlGn', zmid=0,
        hoverinfo='x+y+text',
    ))
    fig.update_layout(
        template='plotly_dark', height=120,
        xaxis_title="Month", yaxis_title="",
        margin=dict(l=50, r=20, t=10, b=40),
    )
    return fig

# ============================================================================
# TRADINGVIEW EMBED
# ============================================================================

def tv_widget_html(symbol: str, interval: str = "M1") -> str:
    tv_int = {"M1": "1", "M5": "5", "H1": "60"}.get(interval, "1")
    tv_sym = YF_MAP.get(symbol.upper(), symbol)
    url = (
        f"https://www.tradingview.com/widgetembed/?"
        f"symbol={tv_sym}&interval={tv_int}&"
        f"hide_sidebar=false&hide_top_toolbar=false&save_image=false&"
        f"studies=RSI@tv-basicthemes%3B!EMA@tv-basicthemes%3B!&"
        f"theme=dark&style=1&locale=en&toolbar_bg=%23363636&"
        f"enable_publishing=false&allow_symbol_change=true&"
        f"support_host=https://www.tradingview.com"
    )
    return f'<iframe src="{url}" width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no" style="border-radius:10px;border:1px solid #2a2a2a;" title="TV {symbol}"></iframe>'


def fetch_yf_history(symbol: str, interval: str = "1m", days: int = 5) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance with configurable days."""
    import yfinance
    yf_sym = YF_MAP.get(symbol.upper(), symbol)
    tf_map = {"1m": "1m", "5m": "5m", "60m": "60m"}
    try:
        t = yfinance.Ticker(yf_sym)
        raw = t.history(period=f"{days}d", interval=tf_map.get(interval, "1m"), prepost=False)
        if raw.empty:
            return pd.DataFrame()
        df = raw.reset_index()
        # yfinance: after reset_index(), Datetime becomes 'Datetime' column
        df.columns = ['time' if c.lower() == 'datetime' else c for c in df.columns]
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df['volume'] = df['Volume'].fillna(0) if 'Volume' in df.columns else 0
        return df[['time', 'Open', 'High', 'Low', 'Close', 'volume']].rename(
            columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    except Exception:
        return pd.DataFrame()




# ============================================================================
# SIGNAL CARD
# ============================================================================

def signal_card(sig, show_mt5=False) -> str:
    is_buy = sig['signal'] == 'BUY'
    border = '#4caf50' if is_buy else '#f44336'
    bg = '#0d2e12' if is_buy else '#2e0d0d'
    arrow = '🟢 BUY' if is_buy else '🔴 SELL'
    pip_val = get_pip_value(sig['symbol'])
    sl_pips = round(abs(sig['entry'] - sig['sl']) / pip_val, 1)
    tp_pips = round(abs(sig['tp'] - sig['entry']) / pip_val, 1)
    return f"""
    <div style="padding:16px;border-radius:12px;border:2px solid {border};background:{bg};color:white;margin:6px 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <h3 style="margin:0;">{arrow} — {sig['symbol']} <span style="font-size:12px;opacity:0.7;">[{sig['timeframe']}]</span> <span style="font-size:11px;color:#aaa;">{sig.get('strategy','')}</span></h3>
            <div style="background:{border};border-radius:50%;width:60px;height:60px;display:flex;flex-direction:column;align-items:center;justify-content:center;">
                <span style="font-size:18px;font-weight:bold;">{sig['score']}</span>
                <span style="font-size:7px;">SCORE</span>
            </div>
        </div>
        <hr style="opacity:0.2;margin:8px 0;border-color:#333;">
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:8px;">
            <div><small>ENTRY</small><br><b>{sig['entry']}</b></div>
            <div><small>SL</small><br><b style='color:#f44336;'>{sig['sl']}</b> <small style='color:#888;'>{sl_pips}p</small></div>
            <div><small>TP</small><br><b style='color:#4caf50;'>{sig['tp']}</b> <small style='color:#888;'>{tp_pips}p</small></div>
            <div><small>R:R</small><br><b>1:{sig['rr']}</b></div>
            <div><small>RSI</small><br><b>{sig['rsi']}</b></div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-top:6px;">
            <div><small>ATR</small><br><b>{sig['atr']}</b></div>
            <div><small>Body</small><br><b>{sig['candle_pct']:.0f}%</b></div>
            <div><small>TrendDist</small><br><b>{sig['trend_dist']:.6f}</b></div>
            <div><small>UTC</small><br><b>{sig['time']}</b></div>
        </div>
    </div>"""

# ============================================================================
# SOUND ALERT (browser beep via JS)
# ============================================================================

SOUND_JS = """
<script>
function playBeep() {
    try {
        var ctx = new (window.AudioContext || window.webkitAudioContext)();
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = 880;
        osc.type = 'sine';
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.4);
    } catch(e) {}
}
playBeep();
</script>
"""

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="MT5 Scalper Pro", page_icon="📈", layout="wide")
st.title("📈 MT5 Scalper Pro")
init_session()

# ─────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    strat_options = list(STRATEGIES.keys())
    active_strat = st.selectbox(
        "🎯 Strategy", strat_options,
        index=strat_options.index(st.session_state.get("active_strategy", DEFAULT_STRATEGY))
    )
    st.session_state["active_strategy"] = active_strat
    st.caption(f"_{STRATEGIES[active_strat]['description']}_")

    sym_sidebar = st.selectbox("📌 Pinned Symbol", ALL_PAIRS,
        index=ALL_PAIRS.index(st.session_state.get("pinned_symbol", ALL_PAIRS[0])) if st.session_state.get("pinned_symbol", ALL_PAIRS[0]) in ALL_PAIRS else 0)
    st.session_state["pinned_symbol"] = sym_sidebar

    tf = st.selectbox("⏱️ Timeframe", ["M1", "M5", "H1"], index=0)
    st.session_state["active_tf"] = tf

    st.markdown("---")
    st.subheader(f"📐 {active_strat}")
    p = STRATEGIES[active_strat]["params"].copy()

    p["ema_fast"] = st.slider("EMA Fast (pullback)", 3, 21, p["ema_fast"])
    p["ema_slow"] = st.slider("EMA Slow (signal)", 5, 55, p["ema_slow"])
    p["ema_trend_fast"] = st.slider("EMA Trend Fast", 5, 30, p["ema_trend_fast"])
    p["ema_trend_slow"] = st.slider("EMA Trend Slow", 20, 100, p["ema_trend_slow"])
    p["rsi_period"] = st.slider("RSI Period", 3, 21, p["rsi_period"])
    p["atr_period"] = st.slider("ATR Period", 3, 21, p["atr_period"])

    st.subheader("🎯 Filters")
    p["min_trend_dist_pct"] = st.slider("Trend Dist (×10000)", 1, 30, int(p["min_trend_dist_pct"] * 10000)) / 10000
    p["min_atr_pct"] = st.slider("Min ATR % (×10000)", 1, 20, int(p["min_atr_pct"] * 10000)) / 10000
    p["candle_body_min"] = st.slider("Body Min %", 20, 80, int(p["candle_body_min"] * 100)) / 100

    st.subheader("🛡️ Risk")
    p["sl_atr_mult"] = st.slider("SL ATR ×", 0.3, 4.0, p["sl_atr_mult"], 0.1)
    p["sl_min_atr"] = st.slider("Min SL ATR ×", 0.1, 2.0, p["sl_min_atr"], 0.1)

    st.subheader("⏱️ Cooldown & Score")
    p["cooldown"] = st.slider("Cooldown (candles)", 0, 15, p["cooldown"])
    p["min_score"] = st.slider("Min Score", 20, 95, p["min_score"])

    if active_strat == "RSI Range":
        st.subheader("📊 RSI Thresholds")
        p["rsi_buy_max"] = st.slider("RSI BUY (oversold)", 20, 50, p.get("rsi_buy_max", 35))
        p["rsi_sell_min"] = st.slider("RSI SELL (overbought)", 50, 80, p.get("rsi_sell_min", 65))

    if active_strat == "Micro Scalp":
        st.subheader("⚡ Micro Scalp")
        p["use_tp_fixed"] = True
        p["tp_pips"] = st.slider("Fixed TP (pips)", 3, 20, int(p.get("tp_pips", 8)), 1)

    if active_strat == "Quick Scalp":
        st.subheader("⚡⚡ Quick Scalp")
        p["use_tp_fixed"] = True
        p["tp_pips"] = st.slider("Fixed TP (pips)", 2, 15, int(p.get("tp_pips", 5)), 1)

    st.session_state["active_params"] = p

    st.markdown("---")
    st.subheader("🔍 Scanner Pairs")
    selected = st.multiselect("Pairs", ALL_PAIRS, default=st.session_state.selected_pairs)
    st.session_state.selected_pairs = selected

    st.markdown("**🕐 Sessions:** London 08-12 UTC | NY 13-17 UTC")

    if st.session_state.signal_history:
        st.subheader("📜 Recent Signals")
        for h in reversed(st.session_state.signal_history[-8:]):
            c = '#4caf50' if h['signal'] == 'BUY' else '#f44336'
            st.markdown(
                f"<span style='color:{c};'>● {h['signal']}</span> {h['symbol']} "
                f"{h['time'].strftime('%H:%M')} <small>score:{h['score']}</small>",
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Backtest", "🔴 Scanner"])

# ═══════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════
with tab1:
    now = datetime.now()
    st.markdown(session_boxes(now), unsafe_allow_html=True)

    pinned = st.session_state.get("pinned_symbol", ALL_PAIRS[0])
    active = st.session_state.get("active_strategy", DEFAULT_STRATEGY)
    params = st.session_state.get("active_params", STRATEGIES[active]["params"].copy())

    # Top bar: symbol TF strategy selectors
    col_sym, col_tf, col_strat, col_pin = st.columns([2, 1, 2, 1])
    with col_sym:
        dash_sym = st.selectbox("Symbol", ALL_PAIRS,
            index=ALL_PAIRS.index(pinned) if pinned in ALL_PAIRS else 0, key="dash_sym")
        st.session_state["scanner_symbol"] = dash_sym
    with col_tf:
        dash_tf = st.selectbox("TF", ["M1", "M5", "H1"], index=["M1", "M5", "H1"].index(st.session_state.get("active_tf", "M1")), key="dash_tf")
    with col_strat:
        dash_strat = st.selectbox("Strategy", list(STRATEGIES.keys()),
            index=list(STRATEGIES.keys()).index(active), key="dash_strat_v2")
    with col_pin:
        st.markdown("")
        if st.button("📌 Pin Symbol", use_container_width=True):
            st.session_state["pinned_symbol"] = dash_sym
            st.session_state["active_strategy"] = dash_strat
            st.rerun()

    # Override params with selected strategy
    dash_params = STRATEGIES[dash_strat]["params"].copy()
    dash_params.update({k: v for k, v in params.items() if k in dash_params})

    # TradingView chart
    st.markdown("### 📊 Live TradingView Chart")
    st.markdown(tv_widget_html(dash_sym, dash_tf), unsafe_allow_html=True)

    # Strategy data
    tf_int = TF_INTERVAL.get(dash_tf, "1m")
    tf_rng = TF_PERIOD.get(dash_tf, "5d")
    df_dash = get_candles(dash_sym, tf_int, tf_rng, count=200)

    if df_dash.empty:
        st.warning(f"❌ No data for {dash_sym} [{dash_tf}]. Market closed or Yahoo rate-limiting.")
    else:
        df_dash = add_indicators(df_dash, dash_params)
        last = df_dash.iloc[-1]
        age = (datetime.now() - last['time']).total_seconds() / 60
        fc = "🟢" if age < 15 else "🟡" if age < 60 else "🔴"
        trend_dir = "🟢 UP" if last['ema_trend_fast'] > last['ema_trend_slow'] else "🔴 DOWN"
        rsi_str, rsi_col = rsi_display(float(last['rsi']))

        st.caption(f"{fc} {len(df_dash)} candles | {last['time'].strftime('%Y-%m-%d %H:%M')} UTC | Yahoo v8 | 📌 {dash_strat} on {dash_sym}")

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        c1.metric("Price", f"{last['close']:.5f}")
        c2.metric(f"EMA{dash_params['ema_fast']}", f"{last['ema_fast']:.5f}")
        c3.metric(f"EMA{dash_params['ema_slow']}", f"{last['ema_slow']:.5f}")
        c4.metric(f"EMA{dash_params['ema_trend_fast']}", f"{last['ema_trend_fast']:.5f}")
        c5.metric(f"EMA{dash_params['ema_trend_slow']}", f"{last['ema_trend_slow']:.5f}")
        c6.markdown(f"<div style='text-align:center;padding:4px 0;'><small>RSI</small><br><b style='color:{rsi_col};'>{rsi_str}</b></div>", unsafe_allow_html=True)
        c7.metric("ATR", f"{last['atr']:.5f}")
        c8.metric("Trend", trend_dir)

        st.markdown("---")
        fig = plot_chart(df_dash, params=dash_params)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'hoverCompareCartesian', 'toggleSpikelines'], 'doubleClick': False, 'editable': False})

        sig = scan_pair(dash_sym, dash_tf, dash_strat, dash_params)
        if sig:
            st.markdown(signal_card(sig), unsafe_allow_html=True)
            st.toast(f"📢 {sig['signal']} {sig['symbol']} @ {sig['entry']} (score:{sig['score']})", icon="📈")
            st.session_state["mt5_signal"] = generate_mt5_signal(sig, dash_params)
            # Browser beep for new signal
            st.markdown(SOUND_JS, unsafe_allow_html=True)

            # ── Real Trading: Risk & Position Size Calculator ────────────────────
            with st.expander("🧮 Risk & Position Size Calculator", expanded=False):
                pip_val = get_pip_value(sig['symbol'])
                sl_pips_calc = round(abs(sig['entry'] - sig['sl']) / pip_val, 1)
                tp_pips_calc = round(abs(sig['tp'] - sig['entry']) / pip_val, 1)
                risk_account = st.number_input("Account Balance ($)", min_value=100.0, value=1000.0, step=50.0, key="risk_account")
                risk_pct = st.slider("Risk per trade (%)", 0.25, 5.0, 1.0, 0.25, key="risk_pct")
                risk_amount = risk_account * risk_pct / 100
                position_size = round(risk_amount / (sl_pips_calc * pip_val), 2) if sl_pips_calc > 0 else 0
                potential_profit = round(position_size * tp_pips_calc * pip_val, 2)
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;padding:10px;background:#1a1a2e;border-radius:10px;">
                    <div><small>Account</small><br><b>${risk_account:,.0f}</b></div>
                    <div><small>Risk Amount</small><br><b style="color:#f44336;">${risk_amount:.2f} ({risk_pct}%)</b></div>
                    <div><small>SL Distance</small><br><b>{sl_pips_calc} pips</b></div>
                    <div><small>TP Distance</small><br><b>{tp_pips_calc} pips</b></div>
                    <div><small>Position Size</small><br><b style="color:#2196F3;">{position_size:,} units</b></div>
                    <div><small>Potential Profit</small><br><b style="color:#4caf50;">${potential_profit:.2f}</b></div>
                    <div><small>Risk:Reward</small><br><b>1:{sig['rr']:.1f}</b></div>
                    <div><small>Leverage (forex)</small><br><b>~1:{int(position_size * sig['entry'] / risk_account) if position_size > 0 else 0}x</b></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"⚪ No signal ({dash_strat}) | score≥{dash_params['min_score']} | ATR%≥{dash_params['min_atr_pct']:.5f} | body≥{dash_params['candle_body_min']*100:.0f}% | RSI={last['rsi']:.1f} | {trend_dir}")

# ═══════════════════════════════════════════════════
# TAB 2: BACKTEST
# ═══════════════════════════════════════════════════
with tab2:
    st.header("📈 Backtest Engine")
    st.caption(f"🔧 {active} | SL: ATR×{params.get('sl_atr_mult','?')} | TP: {'fixed ' + str(params.get('tp_pips','?')) + 'pips' if params.get('use_tp_fixed') else 'dynamic'} | Session: London+NY | Cooldown: {params.get('cooldown',0)}")

    # ── Controls Row ─────────────────────────────────────────────────────────
    bt_ctrl1, bt_ctrl2, bt_ctrl3, bt_ctrl4 = st.columns([2, 1, 2, 2])
    with bt_ctrl1:
        bt_sym = st.selectbox("📊 Symbol", ALL_PAIRS, index=0, key="bt_sym")
    with bt_ctrl2:
        bt_tf = st.selectbox("⏱️ TF", ["M1", "M5", "H1"],
            index=["M1", "M5", "H1"].index(st.session_state.get("active_tf", "M1")), key="bt_tf")
    with bt_ctrl3:
        bt_strat = st.selectbox("🎯 Strategy", list(STRATEGIES.keys()),
            index=list(STRATEGIES.keys()).index(active), key="bt_strat")
    with bt_ctrl4:
        bt_candles = st.selectbox("📐 Candles", [500, 1000, 1500, 2000, 3000], index=2, key="bt_candles")

    bt_ctrl5, bt_ctrl6, bt_ctrl7 = st.columns([2, 1, 1])
    with bt_ctrl5:
        bt_no_session = st.checkbox("🔓 No Session Filter", value=False, key="bt_no_session")
    with bt_ctrl6:
        bt_compare = st.checkbox("⚡ Compare All Strategies", value=False, key="bt_compare")
    with bt_ctrl7:
        auto_refresh = st.checkbox("🔄 Auto-Refresh (30s)", value=False, key="bt_autorefresh")

    bt_params = STRATEGIES[bt_strat]["params"].copy()
    bt_params.update({k: v for k, v in params.items() if k in bt_params})
    if bt_no_session:
        bt_params["_no_session_filter"] = True

    # Auto-refresh
    if auto_refresh:
        refresh_key = f"bt_refresh_{bt_sym}_{bt_tf}_{bt_strat}_{bt_candles}"
        st_autorefresh(interval=30000, key=refresh_key)

    # Run button
    run_cols = st.columns([1, 2, 1])
    with run_cols[1]:
        if st.button("▶️ ▶️  RUN BACKTEST", use_container_width=True, type="primary", key="run_bt_v4"):
            with st.spinner(f"🔄 {bt_strat} on {bt_sym} [{bt_tf}] — {bt_candles} candles | Fetching data..."):
                tick = datetime.now()
                td, df_bt, m = run_backtest(bt_sym, bt_candles, bt_tf, bt_strat, bt_params)
                fetch_ms = (datetime.now() - tick).total_seconds() * 1000
                st.session_state.bt_result = (td, df_bt, m, bt_sym, bt_tf, bt_strat, bt_params)
                st.session_state.bt_fetch_ms = fetch_ms

    bt_res = st.session_state.get("bt_result")
    if bt_res:
        td, df_bt, m, sym_b, tf_b, strat_b, par_b = bt_res[:7]
        fetch_ms = st.session_state.get("bt_fetch_ms", 0)

        if m is None:
            st.error(f"❌ No trades for {sym_b} [{tf_b}] using {strat_b}. Try: more candles, enable 'No Session Filter', lower min_score, or switch TF.")
        else:
            # ── LIVE STATUS BAR ─────────────────────────────────────────────────
            now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            st.success(f"✅ Backtest complete | {m['total']} trades | {sym_b} [{tf_b}] | {bt_candles} candles | Data fetched in {fetch_ms:.0f}ms | {now_utc}")

            # ── PERFORMANCE SUMMARY CARDS ─────────────────────────────────────
            st.markdown("#### 📊 Performance Summary")
            # Row 1: Key metrics in 4 clean cards
            s1, s2, s3, s4 = st.columns(4)
            pnl_val = m['total_pnl']
            pnl_color = "normal" if pnl_val >= 0 else "inverse"
            s1.metric("Total Trades", m['total'], delta=f"{m['wins']}W / {m['losses']}L")
            s2.metric("Win Rate", f"{m['win_rate']}%", delta=f"{m['profit_factor']} PF")
            s3.metric("Net P&L", f"{pnl_val:.1f} pips", delta="▲" if pnl_val > 0 else "▼", delta_color=pnl_color)
            s4.metric("Max Drawdown", f"{m['max_drawdown']} pips", delta="⚠️ risk" if m['max_drawdown'] > 20 else "ok")

            # Row 2: Detailed metrics
            d1, d2, d3, d4, d5, d6 = st.columns(6)
            d1.metric("Avg R:R", f"1:{m['avg_rr']}")
            d2.metric("Avg Win", f"{m['avg_win']} pips")
            d3.metric("Avg Loss", f"{m['avg_loss']} pips")
            d4.metric("Expectancy", f"{m.get('expectancy', 0):.2f} pips")
            d5.metric("Best Trade", f"{m.get('best', 0):.1f} pips")
            d6.metric("Worst Trade", f"{m.get('worst', 0):.1f} pips")

            # Row 3: Direction & streaks
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Win Streak", f"×{m['max_win_streak']}", delta="🟢")
            g2.metric("Loss Streak", f"×{m['max_loss_streak']}", delta="🔴")
            g3.metric("BUY Win Rate", f"{m['buy_wr']}%", delta=f"{m['buy_pnl']:.1f}p")
            g4.metric("SELL Win Rate", f"{m['sell_wr']}%", delta=f"{m['sell_pnl']:.1f}p")
            g5.metric("Sharpe-like", m.get('sharp_ratio_like', 0), delta="risk-adj" if m.get('sharp_ratio_like', 0) > 0.5 else "low")

            # ── BEST / WORST TRADE CARDS ──────────────────────────────────────
            bc, wc = st.columns(2)
            best = m.get('best_trade', {})
            worst = m.get('worst_trade', {})
            with bc:
                if best:
                    bcd = "🟢 WIN" if best.get('result') == 'WIN' else "🔴 LOSS"
                    bbg = "#0d2e12" if best.get('result') == 'WIN' else "#2e0d0d"
                    btc = "#4caf50" if best.get('result') == 'WIN' else "#f44336"
                    st.markdown(f"""
                    <div style="background:{bbg};border-left:4px solid {btc};border-radius:8px;padding:16px;margin:4px 0">
                        <h4 style="color:{btc};margin:0">🏆 Best Trade — {bcd} {best.get('pnl_pips', 0):.1f} pips</h4>
                        <p style="color:#aaa;margin:4px 0">R:R 1:{best.get('rr', 0)} | Score {best.get('score', 0)} | Hold: {best.get('hold_candles', 0)} cndls ({best.get('duration_min', 0):.1f}m)</p>
                        <p style="color:#888;margin:2px 0">⏰ {str(best.get('entry_time', ''))[:19]} → {str(best.get('exit_time', ''))[:19]}</p>
                        <p style="color:#888;margin:2px 0">{best.get('signal', '')} @ {best.get('entry', 0):.5f} → {best.get('exit', 0):.5f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            with wc:
                if worst:
                    wcd = "🟢 WIN" if worst.get('result') == 'WIN' else "🔴 LOSS"
                    wbg = "#0d2e12" if worst.get('result') == 'WIN' else "#2e0d0d"
                    wtc = "#4caf50" if worst.get('result') == 'WIN' else "#f44336"
                    st.markdown(f"""
                    <div style="background:{wbg};border-left:4px solid {wtc};border-radius:8px;padding:16px;margin:4px 0">
                        <h4 style="color:{wtc};margin:0">💔 Worst Trade — {wcd} {worst.get('pnl_pips', 0):.1f} pips</h4>
                        <p style="color:#aaa;margin:4px 0">R:R 1:{worst.get('rr', 0)} | Score {worst.get('score', 0)} | Hold: {worst.get('hold_candles', 0)} cndls ({worst.get('duration_min', 0):.1f}m)</p>
                        <p style="color:#888;margin:2px 0">⏰ {str(worst.get('entry_time', ''))[:19]} → {str(worst.get('exit_time', ''))[:19]}</p>
                        <p style="color:#888;margin:2px 0">{worst.get('signal', '')} @ {worst.get('entry', 0):.5f} → {worst.get('exit', 0):.5f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ── SESSION + DURATION BREAKDOWN ───────────────────────────────────
            sess_data = m.get('session_breakdown', [])
            dur_data = m.get('duration_breakdown', [])
            if sess_data or dur_data:
                s_col, d_col = st.columns(2)
                with s_col:
                    if sess_data:
                        st.markdown("##### 🌐 Session Breakdown (London 08-12 UTC | New York 13-17 UTC)")
                        sess_rows = []
                        for s in sess_data:
                            icon = "🟢" if s['pnl'] > 0 else "🔴" if s['pnl'] < 0 else "⚪"
                            sess_rows.append({"Session": f"{icon} {s['session']}", "Trades": s['count'], "P&L (pips)": round(s['pnl'], 1), "Avg (pips)": round(s['avg_pnl'], 1)})
                        st.dataframe(pd.DataFrame(sess_rows), hide_index=True, use_container_width=True)
                with d_col:
                    if dur_data:
                        st.markdown("##### ⏱️ Duration Breakdown")
                        dur_rows = []
                        for d in dur_data:
                            icon = "🟢" if d['pnl'] > 0 else "🔴" if d['pnl'] < 0 else "⚪"
                            dur_rows.append({"Duration": f"{icon} {d['duration']}", "Trades": d['count'], "P&L (pips)": round(d['pnl'], 1), "Avg (pips)": round(d['avg_pnl'], 1)})
                        st.dataframe(pd.DataFrame(dur_rows), hide_index=True, use_container_width=True)

            # ── EQUITY CURVE + DRAWDOWN ────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📈 Equity Curve & Drawdown")
            eq_col, dd_col = st.columns(2)
            with eq_col:
                eq = equity_chart(td)
                if eq:
                    st.plotly_chart(eq, use_container_width=True,
                        config={'scrollZoom': False, 'displayModeBar': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'hoverCompareCartesian'],
                            'doubleClick': False, 'editable': False})
            with dd_col:
                dd = drawdown_chart(td)
                if dd:
                    st.plotly_chart(dd, use_container_width=True,
                        config={'scrollZoom': False, 'displayModeBar': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'hoverCompareCartesian'],
                            'doubleClick': False, 'editable': False})

            # ── P&L DISTRIBUTION + HOURLY RETURNS ──────────────────────────────
            st.markdown("#### 📊 P&L Distribution & Hourly Returns")
            dist_col, hr_col = st.columns(2)
            with dist_col:
                dist = pnl_dist_chart(td)
                if dist:
                    st.plotly_chart(dist, use_container_width=True,
                        config={'scrollZoom': False, 'displayModeBar': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'hoverCompareCartesian'],
                            'doubleClick': False, 'editable': False})
            with hr_col:
                hc = hourly_chart(m)
                if hc:
                    st.plotly_chart(hc, use_container_width=True,
                        config={'scrollZoom': False, 'displayModeBar': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'hoverCompareCartesian'],
                            'doubleClick': False, 'editable': False})

            # ── PRICE CHART WITH TRADE MARKERS ─────────────────────────────────
            st.markdown("#### 📊 Price Chart with Trade Markers")
            if df_bt is not None and not df_bt.empty:
                fig_bt = plot_chart(df_bt, td, params=par_b)
                if fig_bt:
                    st.plotly_chart(fig_bt, use_container_width=True,
                        config={'scrollZoom': False, 'displayModeBar': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'hoverCompareCartesian'],
                            'doubleClick': False, 'editable': False})

            # ── DETAILED TRADE LOG TABLE ─────────────────────────────────────────
            st.markdown(f"#### 📋 Complete Trade Log ({m['total']} trades)")

            # Build full display dataframe
            disp = td.copy()
            disp['#'] = range(1, len(disp) + 1)
            disp['P&L'] = disp['pnl_pips'].apply(lambda x: f"{x:.1f}p")
            disp['R:R'] = disp['rr'].apply(lambda x: f"1:{x:.2f}")
            disp['Score'] = disp['score'].apply(lambda x: f"{x:.0f}")
            disp['Entry'] = disp['entry'].apply(lambda x: f"{x:.5f}")
            disp['Exit'] = disp['exit'].apply(lambda x: f"{x:.5f}")
            disp['SL'] = disp['sl'].apply(lambda x: f"{x:.5f}")
            disp['TP'] = disp['tp'].apply(lambda x: f"{x:.5f}")
            disp['Dur'] = disp['duration_min'].apply(lambda x: f"{x:.1f}m")
            disp['Hold'] = disp['hold_candles'].apply(lambda x: f"{x}")
            disp['SL_pips'] = disp['sl_pips'].apply(lambda x: f"{x:.1f}p")
            disp['TP_pips'] = disp['tp_pips'].apply(lambda x: f"{x:.1f}p")
            disp['Time'] = pd.to_datetime(disp['entry_time']).dt.strftime('%m-%d %H:%M')
            disp['ExitTime'] = pd.to_datetime(disp['exit_time']).dt.strftime('%m-%d %H:%M')
            disp['Dir'] = disp['signal'].apply(lambda x: "🟢 BUY" if x == 'BUY' else "🔴 SELL")
            disp['Result'] = disp['result'].apply(lambda x: "🟢 WIN" if x == 'WIN' else "🔴 LOSS")

            # Sortable columns
            show_cols = ['#', 'Dir', 'Time', 'Entry', 'Exit', 'SL', 'TP', 'R:R', 'Score', 'Result', 'P&L', 'SL_pips', 'TP_pips', 'Hold', 'Dur', 'ExitTime']
            disp_show = disp[show_cols].rename(columns={
                'Time': 'Entry Time', 'ExitTime': 'Exit Time',
                'Hold': 'Hold (c)', 'Dur': 'Duration',
                'SL_pips': 'SL (pips)', 'TP_pips': 'TP (pips)',
            })

            st.dataframe(
                disp_show.style.map(
                    lambda _: "background-color:#0d2e12;color:#4caf50;font-weight:bold",
                    subset=pd.IndexSlice[disp_show[disp_show['Result'] == '🟢 WIN'].index, ['P&L', 'Result']]
                ).map(
                    lambda _: "background-color:#2e0d0d;color:#f44336;font-weight:bold",
                    subset=pd.IndexSlice[disp_show[disp_show['Result'] == '🔴 LOSS'].index, ['P&L', 'Result']]
                ).map(
                    lambda _: "color:#4caf50;font-weight:bold",
                    subset=pd.IndexSlice[disp_show[disp_show['Dir'] == '🟢 BUY'].index, ['Dir']]
                ).map(
                    lambda _: "color:#f44336;font-weight:bold",
                    subset=pd.IndexSlice[disp_show[disp_show['Dir'] == '🔴 SELL'].index, ['Dir']]
                ),
                column_config={
                    "#": st.column_config.NumberColumn("#", width="small"),
                    "Dir": st.column_config.TextColumn("Direction", width="small"),
                    "Entry Time": st.column_config.TextColumn("Entry Time", width="medium"),
                    "Exit Time": st.column_config.TextColumn("Exit Time", width="medium"),
                    "Entry": st.column_config.TextColumn("Entry Price", width="medium"),
                    "Exit": st.column_config.TextColumn("Exit Price", width="medium"),
                    "SL": st.column_config.TextColumn("Stop Loss", width="medium"),
                    "TP": st.column_config.TextColumn("Take Profit", width="medium"),
                    "R:R": st.column_config.TextColumn("Risk:Reward", width="small"),
                    "Score": st.column_config.TextColumn("Score", width="small"),
                    "Result": st.column_config.TextColumn("Result", width="small"),
                    "P&L": st.column_config.TextColumn("P&L (pips)", width="small"),
                    "SL (pips)": st.column_config.TextColumn("SL (pips)", width="small"),
                    "TP (pips)": st.column_config.TextColumn("TP (pips)", width="small"),
                    "Hold (c)": st.column_config.TextColumn("Hold (c)", width="small"),
                    "Duration": st.column_config.TextColumn("Duration", width="small"),
                },
                hide_index=True, use_container_width=True, height=450
            )

            # ── DOWNLOAD BUTTONS ───────────────────────────────────────────────
            dl1, dl2 = st.columns(2)
            with dl1:
                csv = td.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Trade CSV", csv,
                    f"bt_trades_{sym_b}_{tf_b}_{strat_b}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv",
                    use_container_width=True)
            with dl2:
                if not td.empty:
                    mt5_data = {
                        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "symbol": sym_b, "timeframe": tf_b, "strategy": strat_b,
                        "total_trades": m['total'], "net_pnl": m['total_pnl'],
                        "win_rate": m['win_rate'], "profit_factor": str(m['profit_factor']),
                        "parameters": {k: v for k, v in par_b.items() if not callable(v)},
                        "trades": [
                            {
                                "trade_num": i+1,
                                "entry_time": str(r['entry_time']), "exit_time": str(r['exit_time']),
                                "direction": r['signal'],
                                "entry": round(r['entry'], 5), "exit": round(r['exit'], 5),
                                "sl": round(r['sl'], 5), "tp": round(r['tp'], 5),
                                "rr": round(r['rr'], 2), "score": r['score'],
                                "result": r['result'],
                                "pnl_pips": round(r['pnl_pips'], 1),
                                "sl_pips": round(r['sl_pips'], 1), "tp_pips": round(r['tp_pips'], 1),
                                "hold_candles": r['hold_candles'],
                                "duration_min": round(r['duration_min'], 1),
                                "session": r.get('session', 'Other'),
                            }
                            for i, (_, r) in enumerate(td.iterrows())
                        ]
                    }
                    mt5_bytes = json.dumps(mt5_data, indent=2).encode('utf-8')
                    st.download_button("🤖 Download MT5 JSON", mt5_bytes,
                        f"mt5_bt_{sym_b}_{tf_b}_{strat_b}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        "application/json", use_container_width=True)

    else:
        # ── EMPTY STATE ──────────────────────────────────────────────────────
        st.info("👆 **Configure your backtest above and click `RUN BACKTEST`**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
            **📊 Data Source**\nYahoo Finance + TradingView enriched data\nHistorical candles with real OHLCV\n\n**🎯 Strategies**\n6 strategies available:\n- Pullback EMA\n- EMA Crossover\n- RSI Range\n- ATR Breakout\n- Micro Scalp\n- Quick Scalp
            """)
        with col_b:
            st.markdown("""
            **📋 Metrics Tracked**\n- Win rate, Profit Factor\n- Max Drawdown, Recovery Factor\n- Sharpe-like ratio\n- Expectancy per trade\n- Session breakdown\n- Duration analysis\n            """)
        with col_c:
            st.markdown("""
            **📅 Timeframes**\nM1 (1 minute) — 500-3000 candles\nM5 (5 minute) — 500-3000 candles\nH1 (1 hour) — 500-3000 candles\n\n**🌐 Sessions**\nLondon: 08-12 UTC\nNew York: 13-17 UTC
            """)

@st.cache_data(ttl=15, show_spinner=False)
def get_live_prices(symbols_timeframes: list[tuple]) -> dict:
    """Fetch latest close for each unique (symbol, tf) pair. Returns {symbol: price}."""
    result = {}
    for sym, tf in dict.fromkeys(symbols_timeframes).keys():
        tf_int = TF_INTERVAL.get(tf, "1m")
        df = get_candles(sym, tf_int, "5d", count=3)
        if not df.empty:
            result[sym] = df.iloc[-1]['close']
    return result


# ═══════════════════════════════════════════════════
# TAB 3: SCANNER
# ═══════════════════════════════════════════════════
with tab3:
    st.header("🔴 Multi-Pair Scanner")

    # ── STRATEGY PARAMS: read from sidebar (same as tab1 uses) ─────────────
    active = st.session_state.get("active_strategy", DEFAULT_STRATEGY)
    params = st.session_state.get("active_params", STRATEGIES[active]["params"].copy())
    scan_tf = st.session_state.get("active_tf", "M1")

    # ── SESSION STATE INIT ────────────────────────────────────────────────
    if "signal_log" not in st.session_state:
        st.session_state.signal_log = []
    if "active_signals" not in st.session_state:
        st.session_state.active_signals = {}
    if "last_signal_hashes" not in st.session_state:
        st.session_state.last_signal_hashes = []
    if "pair_cooldown" not in st.session_state:
        st.session_state.pair_cooldown = {}
    if "refresh_count" not in st.session_state:
        st.session_state.refresh_count = 0
    if "scan_paused" not in st.session_state:
        st.session_state.scan_paused = False
    if "first_scan_done" not in st.session_state:
        st.session_state.first_scan_done = False

    # Remove dead first-scan block (was referencing undefined build_scan_params, scan_mode_local, max_log, strat_name)

    def build_scan_params(strategy_name: str) -> dict:
        """Build params dict for a strategy, merging global param overrides from sidebar."""
        p = STRATEGIES[strategy_name]["params"].copy()
        p.update({k: v for k, v in params.items() if k in p})
        return p

    def sig_key(s):
        return f"{s['symbol']}|{s['signal']}|{s.get('strategy','')}|{s.get('timeframe','')}"

    def fmt_time(t):
        try:
            return pd.to_datetime(t).strftime('%H:%M:%S')
        except Exception:
            return str(t)[:8]

    # ── SETTINGS (always top, no rerun dependency) ────────────────────────
    ac = st.session_state.alert_config

    with st.expander("⚙️ Scanner Settings", expanded=True):
        row1 = st.columns([1, 1, 1, 1])
        with row1[0]:
            ac['min_score'] = st.slider("🔔 Min Score", 20, 95, ac['min_score'], key="alert_sc")
        with row1[1]:
            ac['min_rr'] = st.slider("📐 Min R:R", 0.5, 3.0, float(ac['min_rr']), 0.1, key="alert_rr")
        with row1[2]:
            ac['sound_enabled'] = st.checkbox("🔔 Sound", value=ac.get('sound_enabled', True), key="alert_snd")
        with row1[3]:
            scan_tf = st.selectbox("⏱️ TF", ["M1", "M5", "H1"],
                index=["M1", "M5", "H1"].index(st.session_state.get("active_tf", "M1")), key="scan_tf")
        row2 = st.columns([2, 1, 1])
        with row2[0]:
            scan_pairs = ac.get('pairs', ALL_PAIRS[:4])
            ac['pairs'] = st.multiselect("Pairs", ALL_PAIRS, default=scan_pairs, key="alert_prs")
        with row2[1]:
            max_log = st.number_input("📝 Max Log", 5, 50, 20, key="max_log_size")
        with row2[2]:
            scan_mode_local = st.selectbox("📡 Mode", ["Active Strategy", "All Strategies (×6)"],
                key="scan_mode_v2")
        st.session_state.alert_config = ac

    # ── CONTROLS ──────────────────────────────────────────────────────────
    ctrl = st.columns([1, 1, 5])
    with ctrl[0]:
        btn_label = "▶️ Resume" if st.session_state.scan_paused else "⏸️ Pause"
        if st.button(btn_label, use_container_width=True, key="pause_btn"):
            st.session_state.scan_paused = not st.session_state.scan_paused
            st.rerun()
    with ctrl[1]:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_log"):
            st.session_state.signal_log = []
            st.session_state.active_signals = {}
            st.session_state.last_signal_hashes = []
            st.rerun()

    # ── STATUS BAR ────────────────────────────────────────────────────────
    mode_label = active if scan_mode_local == "Active Strategy" else f"All ({len(STRATEGIES)})"
    st.info(
        f"{'⏸️ PAUSED' if st.session_state.scan_paused else '▶️ LIVE'} | "
        f"📡 {mode_label} | {scan_tf} | "
        f"Score≥{ac['min_score']} | R:R≥{ac['min_rr']} | "
        f"Pairs: {', '.join(st.session_state.selected_pairs[:4])}{'...' if len(st.session_state.selected_pairs) > 4 else ''}"
    )

    # ── BACKGROUND SCAN: runs BEFORE display, updates signal_log ─────────
    # This is the scan heartbeat — all logic here executes top-to-bottom each rerun
    if not st.session_state.scan_paused:
        fresh_signals = []
        if scan_mode_local == "Active Strategy":
            raw = scan_all_pairs(active, params)
            fresh_signals = raw if raw else []
        else:
            pairs_to_scan = st.session_state.selected_pairs or ALL_PAIRS
            with ThreadPoolExecutor(max_workers=min(len(pairs_to_scan) * len(STRATEGIES), 32)) as ex:
                futures = {
                    ex.submit(scan_pair, pair, scan_tf, strat, build_scan_params(strat)): (pair, strat)
                    for pair in pairs_to_scan
                    for strat in STRATEGIES.keys()
                }
                for f in as_completed(futures):
                    try:
                        s = f.result()
                        if s and s['score'] >= ac['min_score'] and s['rr'] >= ac['min_rr']:
                            fresh_signals.append(s)
                    except Exception:
                        pass

        # Set cooldowns BEFORE decrementing — cooldown counts down over subsequent scans
        for s in fresh_signals:
            st.session_state.pair_cooldown[s['symbol']] = params.get('cooldown', 0)

        # Decrement cooldowns
        for k in list(st.session_state.pair_cooldown.keys()):
            st.session_state.pair_cooldown[k] = max(0, st.session_state.pair_cooldown[k] - 1)

        if fresh_signals:
            fresh_signals.sort(key=lambda x: str(x.get('time', '')), reverse=True)
            sig_keys = [sig_key(s) for s in fresh_signals]

            # New signals only
            new_keys = set(sig_keys) - set(st.session_state.last_signal_hashes)
            new_signals = [s for s in fresh_signals if sig_key(s) in new_keys]

            if new_signals and ac.get('sound_enabled'):
                st.markdown(SOUND_JS, unsafe_allow_html=True)
            for s in new_signals:
                st.toast(f"📢 {s['signal']} {s['symbol']} @ {s['entry']:.5f} | {s['strategy']} | S:{s['score']} R:R:1:{s['rr']}", icon="📈")

            st.session_state.last_signal_hashes = sig_keys[:50]

            # Log new signals
            for s in fresh_signals:
                key = sig_key(s)
                if key in st.session_state.active_signals:
                    continue
                entry = {
                    'id': key,
                    'symbol': s['symbol'],
                    'strategy': s.get('strategy', active),
                    'timeframe': s.get('timeframe', scan_tf),
                    'direction': s['signal'],
                    'score': int(s['score']),
                    'rr': float(s['rr']),
                    'entry': float(s['entry']),
                    'sl': float(s['sl']),
                    'tp': float(s['tp']),
                    'rsi': float(s.get('rsi', 0)),
                    'atr': float(s.get('atr', 0)),
                    'atr_pct': float(s.get('atr_pct', 0)),
                    'candle_pct': float(s.get('candle_pct', 0)),
                    'trend_dist': float(s.get('trend_dist', 0)),
                    'timestamp': fmt_time(s.get('time', datetime.now())),
                    'logged_at': datetime.now().strftime('%H:%M:%S'),
                    'status': 'ACTIVE',
                    'result': None,
                    'pnl_pips': 0.0,
                    'exit_price': None,
                    'exit_time': None,
                    'hold_candles': 0,
                    'duration_min': 0.0,
                }
                st.session_state.signal_log.append(entry)
                st.session_state.active_signals[key] = entry

        # ── OUTCOME UPDATE: match ACTIVE signals to completed trades ─────────
        # Process up to 5 active signals per heartbeat to stay fast
        active_live = [s for s in st.session_state.signal_log if s['status'] == 'ACTIVE']
        for sig in active_live[:5]:
            if sig['status'] != 'ACTIVE':
                continue
            try:
                hist = scan_pair_historical(sig['symbol'], sig['timeframe'],
                    sig['strategy'], build_scan_params(sig['strategy']), max_trades=5)
                for h in hist:
                    h_time = str(h.get('entry_time', ''))[:19]
                    s_time = str(sig.get('timestamp', ''))[:19]
                    if (h.get('symbol') == sig['symbol']
                            and h.get('signal', h.get('direction', '')) == sig['direction']
                            and abs((pd.to_datetime(h_time) - pd.to_datetime(s_time)).total_seconds()) < 120):
                        sig['status'] = 'COMPLETED'
                        sig['result'] = h.get('result', 'LOSS')
                        sig['pnl_pips'] = float(h.get('pnl_pips', 0))
                        sig['exit_price'] = float(h.get('exit', 0))
                        sig['exit_time'] = str(h.get('exit_time', ''))[:16]
                        sig['hold_candles'] = int(h.get('hold_candles', 0))
                        sig['duration_min'] = float(h.get('duration_min', 0))
                        sig['rr'] = float(h.get('rr', sig['rr']))
                        key = sig['id']
                        if key in st.session_state.active_signals:
                            del st.session_state.active_signals[key]
                        break
            except Exception:
                pass

        # Trim log
        st.session_state.signal_log = st.session_state.signal_log[-max_log:]

    # Increment refresh counter (for display only)
    st.session_state.refresh_count += 1

    # ── DISPLAY: ALWAYS renders after state is fully updated ─────────────
    log = st.session_state.signal_log

    # Stats
    completed = [s for s in log if s['status'] == 'COMPLETED']
    active_sig = [s for s in log if s['status'] == 'ACTIVE']
    wins = [s for s in completed if s['result'] == 'WIN']
    losses = [s for s in completed if s['result'] == 'LOSS']
    win_rate = len(wins) / len(completed) * 100 if completed else 0
    total_pnl = sum(s['pnl_pips'] for s in completed)
    last_5 = list(reversed(log))[:5]  # newest first

    stat1, stat2, stat3, stat4, stat5 = st.columns(5)
    stat1.metric("📋 Total", len(log), delta=f"{len(active_sig)} active")
    stat2.metric("✅ Done", len(completed), delta=f"{len(wins)}W/{len(losses)}L")
    stat3.metric("📈 Win%", f"{win_rate:.0f}%", delta=f"{total_pnl:+.1f}p")
    stat4.metric("🟢 BUY", sum(1 for s in log if s['direction'] == 'BUY'))
    stat5.metric("🔴 SELL", sum(1 for s in log if s['direction'] == 'SELL'))

    # ── LAST 5 SIGNALS FEED ───────────────────────────────────────────────
    st.markdown("#### 📜 Last 5 Signals")

    live_prices = {}
    if active_sig:
        sym_tfs = [(s['symbol'], s['timeframe']) for s in active_sig]
        live_prices = get_live_prices(sym_tfs)

    if last_5:
        for sig in last_5:
            is_buy = sig['direction'] == 'BUY'
            is_win = sig['result'] == 'WIN'
            is_active = sig['status'] == 'ACTIVE'

            if is_active:
                border, bg = '#ffca28', '#1a1a0d'
                badge = '<span style="background:#ffca28;color:#000;font-size:10px;padding:2px 6px;border-radius:4px">● ACTIVE</span>'
            elif is_win:
                border, bg = '#4caf50', '#0d2e12'
                badge = f'<span style="background:#4caf50;color:#000;font-size:10px;padding:2px 6px;border-radius:4px">✓ WIN {sig["pnl_pips"]:+.1f}p</span>'
            else:
                border, bg = '#f44336', '#2e0d0d'
                badge = f'<span style="background:#f44336;color:#fff;font-size:10px;padding:2px 6px;border-radius:4px">✗ LOSS {sig["pnl_pips"]:+.1f}p</span>'

            dir_col = '#4caf50' if is_buy else '#f44336'

            # Build extra info block
            extra = ""
            if is_active:
                cur_price = live_prices.get(sig['symbol'])
                if cur_price:
                    pip_val = get_pip_value(sig['symbol'])
                    if is_buy:
                        pnl_pips = (cur_price - sig['entry']) / pip_val
                        dist_sl = (sig['entry'] - sig['sl']) / pip_val
                        dist_tp = (sig['tp'] - sig['entry']) / pip_val
                        pnl_col = '#4caf50' if pnl_pips >= 0 else '#f44336'
                    else:
                        pnl_pips = (sig['entry'] - cur_price) / pip_val
                        dist_sl = (sig['sl'] - sig['entry']) / pip_val
                        dist_tp = (sig['entry'] - sig['tp']) / pip_val
                        pnl_col = '#4caf50' if pnl_pips >= 0 else '#f44336'
                    extra = f"""<div style="color:#bbb;font-size:11px;margin-top:4px">
                        <span style="color:#fff;font-weight:bold">↗ {cur_price:.5f}</span>
                        <span style="color:{pnl_col};margin-left:6px;font-weight:bold">{pnl_pips:+.1f}p</span>
                        <span style="color:#555;margin-left:8px">SL <span style="color:#f44336">{dist_sl:.1f}p</span> | TP <span style="color:#4caf50">{dist_tp:.1f}p</span></span>
                    </div>"""
                else:
                    extra = f"""<div style="color:#555;font-size:11px;margin-top:4px">Live price unavailable — check connection</div>"""
            else:
                extra = f"""<div style="color:#aaa;font-size:11px;margin-top:4px">
                    ↗ {sig['exit_price']:.5f} | {sig['hold_candles']}c ({sig['duration_min']:.1f}m) | {sig.get('exit_time','—')}
                </div>"""

            st.markdown(f"""
            <div style="background:{bg};border-left:4px solid {border};border-radius:8px;padding:11px;margin:5px 0">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <span style="font-size:17px;font-weight:bold;color:{border}">{sig['symbol']}</span>
                        <span style="color:#888;margin-left:8px;font-size:12px">{sig['strategy']}</span>
                        <span style="color:#555;margin-left:6px;font-size:12px">{sig['timeframe']}</span>
                    </div>
                    <div style="text-align:right">
                        <span style="font-size:14px;font-weight:bold;color:{dir_col}">{'🟢 BUY' if is_buy else '🔴 SELL'}</span>
                        <span style="margin-left:6px">{badge}</span>
                    </div>
                </div>
                <div style="color:#bbb;font-size:12px;margin-top:4px">
                    Entry <span style="color:#fff;font-weight:bold">{sig['entry']:.5f}</span>
                    | SL <span style="color:#f44336">{sig['sl']:.5f}</span>
                    | TP <span style="color:#4caf50">{sig['tp']:.5f}</span>
                    <span style="color:#555;margin-left:8px">S:{sig['score']} | R:R 1:{sig['rr']:.1f} | {sig['logged_at']}</span>
                </div>{extra}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📡 No signals yet. Scanner is live — signals appear here automatically on closed candles.")

    # ── SIGNAL LOG TABLE ─────────────────────────────────────────────────
    if log:
        st.markdown("#### 📋 Signal Log")

        # Build live price map for active signals
        live_tbl_prices = {}
        for s in log:
            if s['status'] == 'ACTIVE' and s['symbol'] in live_prices:
                live_tbl_prices[s['symbol']] = live_prices[s['symbol']]

        def _live_pnl(sig, cur):
            if cur is None:
                return "—"
            pip_val = get_pip_value(sig['symbol'])
            if sig['direction'] == 'BUY':
                return f"{(cur - sig['entry']) / pip_val:+.1f}p"
            else:
                return f"{(sig['entry'] - cur) / pip_val:+.1f}p"

        rows = [{
            'St': '🟡' if s['status']=='ACTIVE' else ('🟢' if s['result']=='WIN' else '🔴'),
            'Pair': s['symbol'],
            'Strategy': s['strategy'],
            'TF': s['timeframe'],
            'Dir': s['direction'],
            'Entry': f"{s['entry']:.5f}",
            'Now': f"{live_prices.get(s['symbol'], 0):.5f}" if s['status']=='ACTIVE' and s['symbol'] in live_prices else "—",
            'SL': f"{s['sl']:.5f}",
            'TP': f"{s['tp']:.5f}",
            'R:R': f"1:{s['rr']:.1f}",
            'Sc': s['score'],
            'P&L': f"{s['pnl_pips']:+.1f}p" if s['status']!='ACTIVE' else (
                _live_pnl(s, live_prices.get(s['symbol'])) if s['symbol'] in live_prices else "—"),
            'Hold': f"{s['hold_candles']}c" if s['hold_candles'] else "—",
            'Dur': f"{s['duration_min']:.1f}m" if s['duration_min'] else "—",
            'Time': s['logged_at'],
        } for s in reversed(log)]
        tbl = pd.DataFrame(rows)

        def row_style(r):
            if r['St'] == '🟡':
                return ['background-color:#1a1a0d;color:#ffca28'] * len(r)
            elif r['St'] == '🟢':
                return ['background-color:#0d2e12;color:#4caf50'] * len(r)
            return ['background-color:#2e0d0d;color:#f44336'] * len(r)

        st.dataframe(
            tbl.style.apply(row_style, axis=1),
            column_config={
                "St": st.column_config.TextColumn("St", width="tiny"),
                "Pair": st.column_config.TextColumn("Pair", width="small"),
                "Strategy": st.column_config.TextColumn("Strategy", width="medium"),
                "TF": st.column_config.TextColumn("TF", width="tiny"),
                "Dir": st.column_config.TextColumn("Dir", width="small"),
                "Entry": st.column_config.TextColumn("Entry", width="medium"),
                "Now": st.column_config.TextColumn("Now", width="medium"),
                "SL": st.column_config.TextColumn("SL", width="medium"),
                "TP": st.column_config.TextColumn("TP", width="medium"),
                "R:R": st.column_config.TextColumn("R:R", width="tiny"),
                "Sc": st.column_config.NumberColumn("Sc", format="%d", width="tiny"),
                "P&L": st.column_config.TextColumn("P&L", width="small"),
                "Hold": st.column_config.TextColumn("Hold", width="tiny"),
                "Dur": st.column_config.TextColumn("Dur", width="tiny"),
                "Time": st.column_config.TextColumn("Time", width="small"),
            },
            hide_index=True, use_container_width=True, height=280
        )

        # Downloads
        dl1, dl2 = st.columns(2)
        with dl1:
            csv_bytes = pd.DataFrame(log).to_csv(index=False).encode('utf-8')
            st.download_button("📥 CSV", csv_bytes,
                f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
        with dl2:
            if completed:
                mt5 = [{
                    "action": "OPEN", "symbol": s['symbol'], "type": s['direction'],
                    "entry": s['entry'], "stop_loss": s['sl'], "take_profit": s['tp'],
                    "sl_pips": round(abs(s['entry']-s['sl'])/get_pip_value(s['symbol']), 1),
                    "tp_pips": round(abs(s['tp']-s['entry'])/get_pip_value(s['symbol']), 1),
                    "risk": "MEDIUM", "score": s['score'], "strategy": s['strategy'],
                    "timeframe": s['timeframe'], "timestamp": s['logged_at'],
                    "result": s.get('result',''), "pnl_pips": round(s['pnl_pips'], 1), "magic": 2024,
                } for s in reversed(completed[-10:])]
                st.download_button("🤖 MT5 JSON", json.dumps(mt5, indent=2).encode('utf-8'),
                    f"mt5_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "application/json", use_container_width=True)

    st.caption(
        f"🕐 {datetime.now().strftime('%H:%M:%S')} UTC | #{st.session_state.refresh_count} | "
        f"{'⏸️ PAUSED' if st.session_state.scan_paused else '▶️ LIVE'} | "
        f"{mode_label} | {scan_tf} | Log: {len(log)}/{max_log} | Active: {len(active_sig)}"
    )

    # ── AUTO-REFRESH: only active when not paused ──────────────────────────
    if not st.session_state.scan_paused:
        st_autorefresh(interval=10000, key="scanner_v9")
