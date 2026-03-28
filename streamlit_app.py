"""
MT5 Scalper Pro — Real-Time Trading Assistant
=============================================
Multi-strategy scalping dashboard with EMA/RSI pullback, crossover, RSI range, and breakout strategies.
Features:
  - TradingView embedded live chart (main feature)
  - 4 switchable strategies: Pullback EMA, EMA Crossover, RSI Range, Breakout
  - Detailed single-pair backtest: equity curve, drawdown, streaks, expectancy, hourly/monthly stats
  - Multi-pair live scanner with customizable alert thresholds
  - Auto-refresh every 10 seconds
  - Session filter (London + New York)
  - Yahoo Finance v8 REST API + SDK fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings('ignore')

import ta

# ============================================================================
# CONSTANTS
# ============================================================================

ALL_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD", "USDCAD", "NZDUSD", "GBPAUD"]
YF_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X", "GBPAUD": "GBPAUD=X",
}
TF_INTERVAL = {"M1": "1m", "M5": "5m", "H1": "60m"}
TF_PERIOD = {"M1": "5d", "M5": "5d", "H1": "60d"}
TF_LABEL = {"M1": "1 Min", "M5": "5 Min", "H1": "1 Hr"}

# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

STRATEGIES = {
    "Pullback EMA": {
        "description": "EMA9 crosses above/below EMA21 within EMA20/50 trend. RSI confirms direction. Best for ranging + trending markets.",
        "params": {
            "ema_fast": 9, "ema_slow": 21,
            "ema_trend_fast": 20, "ema_trend_slow": 50,
            "rsi_period": 14, "atr_period": 14,
            "sl_atr_mult": 1.2, "sl_min_atr": 0.5,
            "min_trend_dist_pct": 0.00006, "min_atr_pct": 0.00003,
            "candle_body_min": 0.50, "min_score": 60,
            "cooldown": 3, "swing_lookback": 5,
            "min_rsi_buy_max": 50, "min_rsi_sell_min": 50,
        }
    },
    "EMA Crossover": {
        "description": "Fast EMA crosses slow EMA. Simple momentum strategy. Works best in strong trending markets.",
        "params": {
            "ema_fast": 9, "ema_slow": 21,
            "ema_trend_fast": 20, "ema_trend_slow": 50,
            "rsi_period": 14, "atr_period": 14,
            "sl_atr_mult": 1.5, "sl_min_atr": 0.8,
            "min_trend_dist_pct": 0.00004, "min_atr_pct": 0.00002,
            "candle_body_min": 0.40, "min_score": 50,
            "cooldown": 2, "swing_lookback": 5,
            "min_rsi_buy_max": 50, "min_rsi_sell_min": 50,
        }
    },
    "RSI Range": {
        "description": "BUY when RSI crosses above oversold threshold (30), SELL when RSI crosses below overbought (70). Uses EMA trend filter.",
        "params": {
            "ema_fast": 9, "ema_slow": 21,
            "ema_trend_fast": 20, "ema_trend_slow": 50,
            "rsi_period": 14, "atr_period": 14,
            "sl_atr_mult": 1.0, "sl_min_atr": 0.5,
            "min_trend_dist_pct": 0.00003, "min_atr_pct": 0.00002,
            "candle_body_min": 0.40, "min_score": 40,
            "cooldown": 2, "swing_lookback": 5,
            "rsi_buy_max": 35, "rsi_sell_min": 65,
            "min_rsi_buy_max": 30, "min_rsi_sell_min": 70,
        }
    },
    "ATR Breakout": {
        "description": "Price breaks above/below EMA9 with strong ATR expansion. No RSI filter. Pure momentum + volatility strategy.",
        "params": {
            "ema_fast": 9, "ema_slow": 21,
            "ema_trend_fast": 20, "ema_trend_slow": 50,
            "rsi_period": 14, "atr_period": 14,
            "sl_atr_mult": 2.0, "sl_min_atr": 1.0,
            "min_trend_dist_pct": 0.00010, "min_atr_pct": 0.00005,
            "candle_body_min": 0.70, "min_score": 70,
            "cooldown": 1, "swing_lookback": 3,
            "min_rsi_buy_max": 100, "min_rsi_sell_min": 0,
        }
    },
}

# Default strategy
DEFAULT_STRATEGY = "Pullback EMA"

# ============================================================================
# SESSION STATE INIT
# ============================================================================

def init_session():
    defaults = {
        "signal_history": [],
        "selected_pairs": ALL_PAIRS[:4],
        "last_signal_hashes": set(),
        "refresh_count": 0,
        "pair_cooldown": {},
        "bt_result": None,          # single pair backtest result
        "bt_strategy": DEFAULT_STRATEGY,
        "alert_config": {
            "min_score": 60,
            "min_rr": 1.0,
            "pairs": ALL_PAIRS[:4],
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ============================================================================
# CONFIG HELPERS — read from currently selected strategy
# ============================================================================

def get_strategy_params():
    strat = st.session_state.get("active_strategy", DEFAULT_STRATEGY)
    return STRATEGIES.get(strat, STRATEGIES[DEFAULT_STRATEGY])["params"].copy()

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
    df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=params["ema_fast"]).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=params["ema_slow"]).ema_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=params["ema_trend_fast"]).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=params["ema_trend_slow"]).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=params["rsi_period"]).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=params["atr_period"]).average_true_range()
    df['atr_pct'] = df['atr'] / df['close']
    df['trend_dist'] = abs(df['ema_20'] - df['ema_50'])
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
# SESSION / TIME HELPERS
# ============================================================================

def session_active(dt) -> bool:
    h = dt.hour
    return (8 <= h < 12) or (13 <= h <= 17)

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
# SIGNAL QUALITY SCORE
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
    atr_ratio = row['atr_pct'] / params['min_atr_pct']
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

def calc_tp(signal, entry, sl, df, idx, params):
    atr_ratio = df.iloc[idx]['atr'] / (params['min_atr_pct'] + 1e-10)
    rr = 1.2 if atr_ratio < 1.0 else 1.35 if atr_ratio < 2.0 else 1.5
    risk = abs(entry - sl)
    return entry + (risk * rr) if signal == 'BUY' else entry - (risk * rr)

# ============================================================================
# STRATEGY SIGNAL DETECTION — unified interface
# ============================================================================

def find_signal(df: pd.DataFrame, idx: int, strategy: str, params: dict) -> str | None:
    """Returns 'BUY', 'SELL', or None based on the selected strategy."""
    if idx < 2:
        return None
    row = df.iloc[idx]
    prev = df.iloc[idx - 1]

    if strategy == "Pullback EMA":
        return _sig_pullback_ema(df, idx, params)
    elif strategy == "EMA Crossover":
        return _sig_ema_crossover(df, idx, params)
    elif strategy == "RSI Range":
        return _sig_rsi_range(df, idx, params)
    elif strategy == "ATR Breakout":
        return _sig_atr_breakout(df, idx, params)
    return None


def _sig_pullback_ema(df, idx, params) -> str | None:
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    cross_up = prev['ema_9'] <= prev['ema_21'] and row['ema_9'] > row['ema_21']
    cross_down = prev['ema_9'] >= prev['ema_21'] and row['ema_9'] < row['ema_21']
    if not (cross_up or cross_down):
        return None
    direction = 'BUY' if cross_up else 'SELL'
    # Trend filter
    if direction == 'BUY' and not (row['ema_20'] > row['ema_50']): return None
    if direction == 'SELL' and not (row['ema_20'] < row['ema_50']): return None
    # RSI filter
    if direction == 'BUY' and not (row['rsi'] > 50 and row['rsi_change'] > 0): return None
    if direction == 'SELL' and not (row['rsi'] < 50 and row['rsi_change'] < 0): return None
    # ATR + trend dist + body
    if row['trend_dist'] < params['min_trend_dist_pct']: return None
    if row['atr_pct'] < params['min_atr_pct']: return None
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    if body < params['candle_body_min']: return None
    if not session_active(row['time']): return None
    return direction


def _sig_ema_crossover(df, idx, params) -> str | None:
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    cross_up = prev['ema_9'] <= prev['ema_21'] and row['ema_9'] > row['ema_21']
    cross_down = prev['ema_9'] >= prev['ema_21'] and row['ema_9'] < row['ema_21']
    if not (cross_up or cross_down):
        return None
    direction = 'BUY' if cross_up else 'SELL'
    # Trend: price above EMA50 for BUY
    if direction == 'BUY' and row['close'] <= row['ema_50']: return None
    if direction == 'SELL' and row['close'] >= row['ema_50']: return None
    # ATR + body
    if row['atr_pct'] < params['min_atr_pct']: return None
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    if body < params['candle_body_min']: return None
    if not session_active(row['time']): return None
    return direction


def _sig_rsi_range(df, idx, params) -> str | None:
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    # BUY: RSI crosses above oversold threshold
    rsi_cross_up = prev['rsi'] < params.get('rsi_buy_max', 35) and row['rsi'] >= params.get('rsi_buy_max', 35)
    # SELL: RSI crosses below overbought threshold
    rsi_cross_down = prev['rsi'] > params.get('rsi_sell_min', 65) and row['rsi'] <= params.get('rsi_sell_min', 65)
    if not (rsi_cross_up or rsi_cross_down):
        return None
    direction = 'BUY' if rsi_cross_up else 'SELL'
    # Trend filter: EMA20 > EMA50 for BUY
    if direction == 'BUY' and not (row['ema_20'] > row['ema_50']): return None
    if direction == 'SELL' and not (row['ema_20'] < row['ema_50']): return None
    if row['atr_pct'] < params['min_atr_pct']: return None
    if not session_active(row['time']): return None
    return direction


def _sig_atr_breakout(df, idx, params) -> str | None:
    row, prev = df.iloc[idx], df.iloc[idx - 1]
    # BUY: close above EMA9 with ATR expansion
    buy_cond = row['close'] > row['ema_9'] and row['atr_pct'] > params['min_atr_pct']
    sell_cond = row['close'] < row['ema_9'] and row['atr_pct'] > params['min_atr_pct']
    if not (buy_cond or sell_cond):
        return None
    direction = 'BUY' if buy_cond else 'SELL'
    # Trend: EMA20 > EMA50 for BUY
    if direction == 'BUY' and not (row['ema_20'] > row['ema_50']): return None
    if direction == 'SELL' and not (row['ema_20'] < row['ema_50']): return None
    # Body filter (must be strong candle)
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    if body < params['candle_body_min']: return None
    if not session_active(row['time']): return None
    return direction

# ============================================================================
# LIVE SCAN — single pair
# ============================================================================

def scan_pair(symbol: str, interval: str, strategy: str, params: dict) -> dict | None:
    tf_int = TF_INTERVAL.get(interval, "1m")
    tf_rng = TF_PERIOD.get(interval, "5d")
    df = get_candles(symbol, tf_int, tf_rng, count=150)
    if df.empty or len(df) < 60:
        return None
    df = add_indicators(df, params)
    cooldown = st.session_state.pair_cooldown.get(symbol, 0)
    if cooldown > 0:
        return None
    for ci in range(-8, -1):
        direction = find_signal(df, ci, strategy, params)
        if direction is None:
            continue
        score = score_signal(df, ci, direction, params)
        if score < params['min_score']:
            continue
        entry = df.iloc[ci]['close']
        sl = calc_sl(direction, entry, df, ci, params)
        tp = calc_tp(direction, entry, sl, df, ci, params)
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
            'candle_pct': round(df.iloc[ci]['candle_body_pct'] * 100, 0) if not np.isnan(df.iloc[ci]['candle_body_pct']) else 0,
            'timeframe': interval, 'strategy': strategy,
        }
    return None

def scan_all_pairs(strategy, params) -> list:
    pairs = st.session_state.selected_pairs
    interval = st.session_state.get("active_tf", "M1")
    if not pairs:
        return []
    signals = []
    with ThreadPoolExecutor(max_workers=min(len(pairs), 8)) as ex:
        futures = {ex.submit(scan_pair, p, interval, strategy, params): p for p in pairs}
        for f in as_completed(futures):
            try:
                s = f.result()
                if s:
                    # Apply alert threshold filters
                    ac = st.session_state.alert_config
                    if s['score'] < ac['min_score']:
                        continue
                    if s['rr'] < ac['min_rr']:
                        continue
                    signals.append(s)
            except Exception:
                pass
    return signals

# ============================================================================
# BACKTEST — detailed, single pair
# ============================================================================

def run_backtest(symbol: str, candles: int, interval: str, strategy: str, params: dict) -> tuple:
    tf_int = TF_INTERVAL.get(interval, "1m")
    tf_rng = TF_PERIOD.get(interval, "5d")
    df = get_candles(symbol, tf_int, tf_rng, candles)
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
                tp = calc_tp(direction, entry, sl, df, i, params)
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
                pnl = (reward - risk) * 10000 if result == 'WIN' else -(risk * 10000)
                # Compute hold time
                hold_candles = i - df[df['time'] == pos_time].index[0]
                trades.append({
                    'signal': position, 'entry': pos_entry, 'exit': exit_p,
                    'sl': pos_sl, 'tp': pos_tp, 'rr': rr,
                    'pnl_pips': round(pnl, 1), 'result': result,
                    'score': pos_score, 'entry_time': pos_time, 'exit_time': df.iloc[i]['time'],
                    'hold_candles': hold_candles,
                })
                position = None
    if not trades:
        return None, df, None
    td = pd.DataFrame(trades)
    # Extended metrics
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
    # Expectancy
    expectancy = (td['pnl_pips'] / abs(td['rr'])).mean() if not td.empty else 0
    # Streaks
    td['win_streak'] = (td['result'] == 'WIN').astype(int) * (td.groupby((td['result'] != td['result'].shift()).cumsum()).cumcount() + 1) * (td['result'] == 'WIN').astype(int)
    td['loss_streak'] = (td['result'] == 'LOSS').astype(int) * (td.groupby((td['result'] != td['result'].shift()).cumsum()).cumcount() + 1) * (td['result'] == 'LOSS').astype(int)
    max_win_streak = int(td[td['result']=='WIN']['win_streak'].max()) if wins > 0 else 0
    max_loss_streak = int(td[td['result']=='LOSS']['loss_streak'].max()) if losses > 0 else 0
    # Hourly distribution
    td['hour'] = pd.to_datetime(td['entry_time']).dt.hour
    hourly = td.groupby('hour')['pnl_pips'].agg(['sum', 'count', 'mean']).reset_index()
    hourly.columns = ['hour', 'pnl', 'count', 'avg_pnl']
    # Monthly returns
    td['month'] = pd.to_datetime(td['entry_time']).dt.to_period('M')
    monthly = td.groupby('month')['pnl_pips'].agg(['sum', 'count']).reset_index()
    monthly.columns = ['month', 'pnl', 'count']
    # Trade direction breakdown
    buy_trades = td[td['signal'] == 'BUY']
    sell_trades = td[td['signal'] == 'SELL']
    metrics = {
        'total': total, 'wins': wins, 'losses': losses,
        'win_rate': round(wr, 1), 'total_pnl': round(total_pnl, 1),
        'avg_rr': round(avg_rr, 2), 'profit_factor': round(pf, 2) if pf != float('inf') else '∞',
        'max_drawdown': round(max_dd, 1),
        'avg_win': round(avg_win, 1), 'avg_loss': round(avg_loss, 1),
        'best': td['pnl_pips'].max(), 'worst': td['pnl_pips'].min(),
        'expectancy': round(expectancy, 2),
        'max_win_streak': max_win_streak, 'max_loss_streak': max_loss_streak,
        'avg_hold': round(td['hold_candles'].mean(), 1),
        'buy_wr': round(len(buy_trades[buy_trades['result']=='WIN'])/len(buy_trades)*100, 1) if len(buy_trades) > 0 else 0,
        'sell_wr': round(len(sell_trades[sell_trades['result']=='WIN'])/len(sell_trades)*100, 1) if len(sell_trades) > 0 else 0,
        'buy_pnl': round(buy_trades['pnl_pips'].sum(), 1) if len(buy_trades) > 0 else 0,
        'sell_pnl': round(sell_trades['pnl_pips'].sum(), 1) if len(sell_trades) > 0 else 0,
        'candles_used': len(df),
        'hourly': hourly.to_dict('records'),
        'monthly': monthly.to_dict('records'),
    }
    return td, df, metrics

# ============================================================================
# PLOT: CANDLESTICK + INDICATORS
# ============================================================================

def plot_chart(df, trades_df=None, sig=None):
    if df is None or df.empty:
        return None
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.20, 0.20, 0.20],
        vertical_spacing=0.06,
        subplot_titles=('Price + EMAs', 'Volume', 'RSI', 'ATR')
    )
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        name='Price'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_9'], name='EMA9',
        line=dict(color='#2196F3', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_21'], name='EMA21',
        line=dict(color='#FF9800', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_20'], name='EMA20',
        line=dict(color='#00BCD4', width=1.2, dash='dash'), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_50'], name='EMA50',
        line=dict(color='#E91E63', width=1.2, dash='dash'), opacity=0.7), row=1, col=1)
    if 'volume' in df.columns and df['volume'].sum() > 0:
        colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name='Volume',
            marker_color=colors, opacity=0.6), row=2, col=1)
    if sig is not None and len(df) > 1:
        c = '#26a69a' if sig['signal'] == 'BUY' else '#ef5350'
        fig.add_trace(go.Scatter(
            x=[sig['time']], y=[sig['entry']],
            mode='markers+text',
            marker=dict(size=24, color=c, symbol='arrow-up' if sig['signal'] == 'BUY' else 'arrow-down'),
            text=[f"▶ {sig['signal']} {sig['score']}"],
            textposition='top center', textfont=dict(color=c, size=11),
            name=f"SIGNAL {sig['signal']}"
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
                showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=[t['exit_time']], y=[t['exit']],
                mode='markers', marker=dict(size=8, color='white', symbol='x'),
                showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], name='RSI',
        line=dict(color='#9C27B0', width=1.5)), row=3, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='gray', row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', line_width=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', line_width=0.5, row=3, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['atr'], name='ATR',
        fill='tozeroy', line=dict(color='#FF5722', width=1),
        fillcolor='rgba(255,87,34,0.12)'), row=4, col=1)
    fig.update_layout(template='plotly_dark', height=750, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_rangeslider_visible=False)
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
        name='Cumulative P&L (pips)',
        fill='tozeroy', fillcolor='rgba(38,166,154,0.1)',
    ))
    fig.update_layout(template='plotly_dark', height=220,
        xaxis_title="Trade #", yaxis_title="P&L (pips)",
        hovermode='x unified', margin=dict(l=0, r=0, t=10, b=0))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=0.5)
    return fig

# ============================================================================
# PLOT: DRAWDOWN CHART
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
        name='Drawdown (pips)',
    ))
    fig.update_layout(template='plotly_dark', height=180,
        xaxis_title="Trade #", yaxis_title="Drawdown (pips)",
        hovermode='x unified', margin=dict(l=0, r=0, t=10, b=0))
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
    fig.add_trace(go.Histogram(
        x=win_pnls, name='Wins', marker_color='#26a69a',
        opacity=0.7, nbinsx=20,
    ))
    fig.add_trace(go.Histogram(
        x=loss_pnls, name='Losses', marker_color='#ef5350',
        opacity=0.7, nbinsx=20,
    ))
    fig.update_layout(template='plotly_dark', height=200,
        xaxis_title="P&L (pips)", yaxis_title="Frequency",
        barmode='overlay', margin=dict(l=0, r=0, t=10, b=0))
    return fig

# ============================================================================
# PLOT: HOURLY RETURNS HEATMAP
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
        name='P&L by Hour',
    ))
    fig.update_layout(template='plotly_dark', height=200,
        xaxis_title="Hour (UTC)", yaxis_title="Net P&L (pips)",
        coloraxis=dict(colorscale='RdYlGn'), margin=dict(l=0, r=0, t=10, b=0))
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
        f"enable_publishing=false&allow_symbol_change=true"
    )
    return f'<iframe src="{url}" width="100%" height="520" frameborder="0" allowtransparency="true" scrolling="yes" style="border-radius:10px;border:1px solid #333;" title="TV {symbol}"></iframe>'

# ============================================================================
# SIGNAL CARD
# ============================================================================

def signal_card(sig) -> str:
    is_buy = sig['signal'] == 'BUY'
    border = '#4caf50' if is_buy else '#f44336'
    bg = '#1b5e20' if is_buy else '#7f0000'
    arrow = '🟢 BUY' if is_buy else '🔴 SELL'
    return f"""
    <div style="padding:16px;border-radius:12px;border:3px solid {border};background:{bg};color:white;margin:6px 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <h3 style="margin:0;">{arrow} — {sig['symbol']} <span style="font-size:12px;opacity:0.7;">[{sig['timeframe']}]</span></h3>
            <div style="background:{border};border-radius:50%;width:60px;height:60px;display:flex;flex-direction:column;align-items:center;justify-content:center;">
                <span style="font-size:18px;font-weight:bold;">{sig['score']}</span>
                <span style="font-size:7px;">SCORE</span>
            </div>
        </div>
        <hr style="opacity:0.25;margin:8px 0;">
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;">
            <div><small>ENTRY</small><br><b>{sig['entry']}</b></div>
            <div><small>SL</small><br><b>{sig['sl']}</b></div>
            <div><small>TP</small><br><b>{sig['tp']}</b></div>
            <div><small>R:R</small><br><b>1:{sig['rr']}</b></div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-top:6px;">
            <div><small>RSI</small><br><b>{sig['rsi']}</b></div>
            <div><small>ATR</small><br><b>{sig['atr']}</b></div>
            <div><small>Body</small><br><b>{sig['candle_pct']:.0f}%</b></div>
            <div><small>UTC</small><br><b>{sig['time']}</b></div>
        </div>
    </div>"""

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="MT5 Scalper Pro", page_icon="📈", layout="wide")
st.title("📈 MT5 Scalper Pro — Real-Time Trading Assistant")
init_session()

# ─────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Strategy selector
    strat_options = list(STRATEGIES.keys())
    active_strat = st.selectbox(
        "🎯 Strategy",
        strat_options,
        index=strat_options.index(st.session_state.get("active_strategy", DEFAULT_STRATEGY)),
        help="Select the trading strategy"
    )
    st.session_state["active_strategy"] = active_strat
    params = STRATEGIES[active_strat]["params"].copy()

    st.markdown(f"<small style='color:#888'>{STRATEGIES[active_strat]['description']}</small>", unsafe_allow_html=True)

    # Symbol
    sym = st.text_input("📌 Symbol", value="EURUSD").upper().strip()

    # Timeframe
    tf_options = ["M1", "M5", "H1"]
    tf = st.selectbox("⏱️ Timeframe", tf_options, index=0)
    st.session_state["active_tf"] = tf

    st.markdown("---")
    st.subheader(f"📐 {active_strat} Parameters")

    # Dynamic params based on strategy
    params["ema_fast"] = st.slider("EMA Pullback (Fast)", 3, 21, params["ema_fast"])
    params["ema_slow"] = st.slider("EMA Signal (Slow)", 5, 55, params["ema_slow"])
    params["ema_trend_fast"] = st.slider("EMA Trend Fast", 5, 30, params["ema_trend_fast"])
    params["ema_trend_slow"] = st.slider("EMA Trend Slow", 20, 100, params["ema_trend_slow"])
    params["rsi_period"] = st.slider("RSI Period", 5, 21, params["rsi_period"])
    params["atr_period"] = st.slider("ATR Period", 5, 21, params["atr_period"])

    st.subheader("🎯 Filters")
    params["min_trend_dist_pct"] = st.slider("Min Trend Dist (×10000)", 1, 20, int(params["min_trend_dist_pct"] * 10000)) / 10000
    params["min_atr_pct"] = st.slider("Min ATR % (×10000)", 1, 15, int(params["min_atr_pct"] * 10000)) / 10000
    params["candle_body_min"] = st.slider("Candle Body Min %", 20, 80, int(params["candle_body_min"] * 100)) / 100

    st.subheader("🛡️ Risk")
    params["sl_atr_mult"] = st.slider("SL ATR Multiplier", 0.5, 4.0, params["sl_atr_mult"], 0.1)
    params["sl_min_atr"] = st.slider("Min SL ATR Factor", 0.3, 3.0, params["sl_min_atr"], 0.1)

    st.subheader("⏱️ Cooldown & Score")
    params["cooldown"] = st.slider("Cooldown Candles", 1, 15, params["cooldown"])
    params["min_score"] = st.slider("Min Signal Score", 30, 95, params["min_score"])

    # RSI Range specific
    if active_strat == "RSI Range":
        st.subheader("📊 RSI Thresholds")
        params["rsi_buy_max"] = st.slider("RSI BUY threshold (cross above)", 20, 50, params.get("rsi_buy_max", 35))
        params["rsi_sell_min"] = st.slider("RSI SELL threshold (cross below)", 50, 80, params.get("rsi_sell_min", 65))

    st.markdown("---")
    st.subheader("📈 Backtest Settings")
    bt_candles = st.slider("Candles", 300, 3000, 1500, 100, key="bt_candles")

    st.subheader("🔍 Scanner Pairs")
    selected = st.multiselect("Pairs", ALL_PAIRS, default=st.session_state.selected_pairs)
    st.session_state.selected_pairs = selected

    st.markdown("---")
    st.markdown("**🕐 Sessions:** London 08-12 UTC | NY 13-17 UTC")

    if st.session_state.signal_history:
        st.subheader("📜 Recent Signals")
        for h in reversed(st.session_state.signal_history[-10:]):
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

    col_sym, col_tf, col_strat = st.columns([2, 1, 2])
    with col_sym:
        dash_sym = st.selectbox("Symbol", ALL_PAIRS,
            index=ALL_PAIRS.index(sym) if sym in ALL_PAIRS else 0, key="dash_sym")
    with col_tf:
        dash_tf = st.selectbox("TF", ["M1", "M5", "H1"],
            index=["M1", "M5", "H1"].index(tf), key="dash_tf")
    with col_strat:
        dash_strat = st.selectbox("Strategy", list(STRATEGIES.keys()),
            index=list(STRATEGIES.keys()).index(active_strat), key="dash_strat")
    dash_params = STRATEGIES[dash_strat]["params"].copy()
    # Override with sidebar sliders
    dash_params.update({k: v for k, v in params.items() if k in dash_params})

    st.markdown("### 📊 Live TradingView Chart")
    st.markdown(tv_widget_html(dash_sym, dash_tf), unsafe_allow_html=True)

    # Strategy data
    tf_int = TF_INTERVAL.get(dash_tf, "1m")
    tf_rng = TF_PERIOD.get(dash_tf, "5d")
    df_dash = get_candles(dash_sym, tf_int, tf_rng, count=200)

    if df_dash.empty:
        st.warning(f"❌ No data for {dash_sym} [{dash_tf}]. Market may be closed or Yahoo rate-limiting.")
    else:
        df_dash = add_indicators(df_dash, dash_params)
        last = df_dash.iloc[-1]
        age = (datetime.now() - last['time']).total_seconds() / 60
        age_str = f"{int(age)} min ago" if age > 0 else "just now"
        fc = "🟢" if age < 15 else "🟡" if age < 60 else "🔴"
        trend_dir = "🟢 UP" if last['ema_20'] > last['ema_50'] else "🔴 DOWN"
        rsi_str, rsi_col = rsi_display(float(last['rsi']))

        st.caption(f"{fc} {len(df_dash)} candles | Last: {last['time'].strftime('%Y-%m-%d %H:%M')} UTC ({age_str}) | Yahoo Finance v8")

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        c1.metric("Price", f"{last['close']:.5f}")
        c2.metric("EMA9", f"{last['ema_9']:.5f}")
        c3.metric("EMA21", f"{last['ema_21']:.5f}")
        c4.metric("EMA20", f"{last['ema_20']:.5f}")
        c5.metric("EMA50", f"{last['ema_50']:.5f}")
        c6.markdown(f"<div style='text-align:center;padding:4px 0;'><small>RSI</small><br><b style='color:{rsi_col};'>{rsi_str}</b></div>", unsafe_allow_html=True)
        c7.metric("ATR", f"{last['atr']:.5f}")
        c8.metric("Trend", trend_dir)

        st.markdown("---")
        fig = plot_chart(df_dash)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        sig = scan_pair(dash_sym, dash_tf, dash_strat, dash_params)
        if sig:
            st.markdown(signal_card(sig), unsafe_allow_html=True)
            st.toast(f"📢 {sig['signal']} on {sig['symbol']} @ {sig['entry']} (score:{sig['score']})", icon="📈")
        else:
            st.info(
                f"⚪ No signal ({dash_strat}) | score≥{dash_params['min_score']}, "
                f"ATR%≥{dash_params['min_atr_pct']:.5f}, body≥{dash_params['candle_body_min']*100:.0f}%, "
                f"session=London/NY | RSI={last['rsi']:.1f} | Trend={trend_dir}"
            )

# ═══════════════════════════════════════════════════
# TAB 2: BACKTEST — single pair, detailed
# ═══════════════════════════════════════════════════
with tab2:
    st.header("📈 Backtest — Single Pair Detailed Analysis")
    st.caption(f"Strategy: {active_strat} | SL: ATR×{params['sl_atr_mult']} | TP: dynamic 1.2–1.5× | Session: London+NY | Cooldown: {params['cooldown']} candles")

    # Controls row
    bk_col1, bk_col2, bk_col3, bk_col4 = st.columns([2, 1, 1, 1])
    with bk_col1:
        bt_sym = st.selectbox("Symbol", ALL_PAIRS,
            index=ALL_PAIRS.index(sym) if sym in ALL_PAIRS else 0, key="bt_sym_select")
    with bk_col2:
        bt_tf = st.selectbox("TF", ["M1", "M5", "H1"],
            index=["M1", "M5", "H1"].index(tf), key="bt_tf_select")
    with bk_col3:
        bt_strat = st.selectbox("Strategy", list(STRATEGIES.keys()),
            index=list(STRATEGIES.keys()).index(active_strat), key="bt_strat_select")
    with bk_col4:
        bt_candles_val = st.selectbox("Candles", [300, 500, 1000, 1500, 2000, 3000],
            index=3, key="bt_candles_select")

    bt_params = STRATEGIES[bt_strat]["params"].copy()
    # Use same slider values from sidebar for quick backtest
    bt_params.update({k: v for k, v in params.items() if k in bt_params})

    run_key = "bt_run_v2"
    if st.button("▶️ Run Backtest", use_container_width=True, type="primary", key=run_key):
        with st.spinner(f"Running {bt_strat} backtest on {bt_sym} ({bt_tf}, {bt_candles_val} candles)..."):
            td, df_bt, metrics = run_backtest(bt_sym, bt_candles_val, bt_tf, bt_strat, bt_params)
            st.session_state.bt_result = (td, df_bt, metrics, bt_sym, bt_tf, bt_strat)

    # ── Show results ──
    bt_res = st.session_state.get("bt_result")
    if bt_res:
        td, df_bt, m, bt_sym_saved, bt_tf_saved, bt_strat_saved = bt_res
        if m is None:
            st.error(f"❌ No trades generated for {bt_sym_saved} [{bt_tf_saved}] using {bt_strat_saved}. Try: lowering min_score, lowering min_ATR%, increasing candles, or switching timeframe to H1.")
        else:
            st.success(f"✅ {bt_strat_saved} on {bt_sym_saved} [{bt_tf_saved}] — {m['total']} trades | {m['win_rate']}% WR | {m['total_pnl']} pips net | PF {m['profit_factor']}")

            # ── Summary Metrics ──
            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Total Trades", m['total'])
            sm2.metric("Win Rate", f"{m['win_rate']}%")
            sm3.metric("Profit Factor", m['profit_factor'])
            sm4.metric("Max Drawdown", f"{m['max_drawdown']} pips")
            sm5.metric("Net P&L", f"{m['total_pnl']} pips {'🟢' if m['total_pnl'] > 0 else '🔴'}")

            sm6, sm7, sm8, sm9, sm10 = st.columns(5)
            sm6.metric("W / L", f"{m['wins']} / {m['losses']}")
            sm7.metric("Avg R:R", f"1:{m['avg_rr']}")
            sm8.metric("Expectancy", f"{m['expectancy']} pips/trade")
            sm9.metric("Avg Hold", f"{m['avg_hold']} candles")
            sm10.metric("Best / Worst", f"{m['best']} / {m['worst']} pips")

            sm11, sm12, sm13, sm14 = st.columns(4)
            sm11.metric("Max Win Streak", f"{m['max_win_streak']} 🟢")
            sm12.metric("Max Loss Streak", f"{m['max_loss_streak']} 🔴")
            sm13.metric("Buy WR", f"{m['buy_wr']}% (PnL: {m['buy_pnl']})")
            sm14.metric("Sell WR", f"{m['sell_wr']}% (PnL: {m['sell_pnl']})")

            # ── Charts row ──
            st.markdown("### 📉 Analysis Charts")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                eq = equity_chart(td)
                if eq:
                    st.plotly_chart(eq, use_container_width=True)
            with chart_col2:
                dd = drawdown_chart(td)
                if dd:
                    st.plotly_chart(dd, use_container_width=True)

            dist_col1, dist_col2 = st.columns(2)
            with dist_col1:
                dist = pnl_dist_chart(td)
                if dist:
                    st.plotly_chart(dist, use_container_width=True)
            with dist_col2:
                hr_chart = hourly_chart(m)
                if hr_chart:
                    st.plotly_chart(hr_chart, use_container_width=True)

            # ── Price chart with trades ──
            st.markdown("### 📊 Price Chart with Trade Markers")
            if df_bt is not None and not df_bt.empty:
                fig_bt = plot_chart(df_bt, td)
                if fig_bt:
                    st.plotly_chart(fig_bt, use_container_width=True)

            # ── Trade Log ──
            st.markdown(f"### 📋 Trade Log ({m['total']} trades)")
            disp = td[['entry_time', 'signal', 'entry', 'sl', 'tp', 'rr', 'score', 'result', 'pnl_pips', 'hold_candles']].tail(50).copy()
            disp.columns = ['Time (UTC)', 'Signal', 'Entry', 'SL', 'TP', 'R:R', 'Score', 'Result', 'P&L', 'Hold(c)']
            disp['Time (UTC)'] = disp['Time (UTC)'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(
                disp.style.apply(
                    lambda r: ['background-color:#1b3d2e;color:#4caf50']*len(r) if r['Result']=='WIN'
                    else ['background-color:#3d1a1a;color:#f44336']*len(r)
                    for _ in r
                ), axis=1,
                use_container_width=True, height=400
            )
            csv = td.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full CSV",
                csv,
                f"backtest_{bt_sym_saved}_{bt_tf_saved}_{bt_strat_saved}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    else:
        st.info("👆 Select symbol, timeframe, and strategy above, then click **Run Backtest**.")

# ═══════════════════════════════════════════════════
# TAB 3: SCANNER
# ═══════════════════════════════════════════════════
with tab3:
    st.header("🔴 Multi-Pair Live Scanner")

    # Alert settings
    with st.expander("⚙️ Alert Settings", expanded=False):
        ac = st.session_state.alert_config
        ac['min_score'] = st.slider("🔔 Min Score Alert", 30, 95, ac['min_score'], key="alert_score")
        ac['min_rr'] = st.slider("📐 Min R:R Alert", 0.5, 3.0, float(ac['min_rr']), 0.1, key="alert_rr")
        ac['pairs'] = st.multiselect("Pairs to scan", ALL_PAIRS, default=ac.get('pairs', ALL_PAIRS[:4]), key="alert_pairs")

    st.info(
        f"⏱️ Auto-refresh 10s | TF: {TF_LABEL.get(tf, tf)} | "
        f"Strategy: {active_strat} | Score≥{st.session_state.alert_config['min_score']} | R:R≥{st.session_state.alert_config['min_rr']} | "
        f"{', '.join(st.session_state.alert_config.get('pairs', []))}"
    )

    refresh_count = st_autorefresh(interval=10000, key="scanner_v4")
    st.session_state.refresh_count = refresh_count

    # Decrement cooldown
    for k in list(st.session_state.pair_cooldown.keys()):
        st.session_state.pair_cooldown[k] = max(0, st.session_state.pair_cooldown[k] - 1)

    signals = scan_all_pairs(active_strat, params)

    if signals:
        signals.sort(key=lambda x: x['score'], reverse=True)
        sig_ids = {id(s) for s in signals}
        new_sigs = [s for s in signals if id(s) not in st.session_state.last_signal_hashes]
        for s in new_sigs:
            st.toast(f"📢 {s['signal']} {s['symbol']} @ {s['entry']} (score:{s['score']}, R:R:1:{s['rr']})", icon="📈")
        st.session_state.last_signal_hashes = sig_ids

        for sig in signals:
            st.markdown(signal_card(sig), unsafe_allow_html=True)

        rows = []
        for s in signals:
            rows.append({
                'Pair': s['symbol'], 'Signal': s['signal'],
                'Entry': s['entry'], 'SL': s['sl'], 'TP': s['tp'],
                'R:R': f"1:{s['rr']}", 'Score': s['score'],
                'RSI': s['rsi'], 'ATR': s['atr'],
                'Time': str(s['time'])[:19], 'TF': s.get('timeframe', tf),
            })
        tbl = pd.DataFrame(rows)
        st.dataframe(
            tbl.style.apply(
                lambda r: ['background-color:#1b3d2e;color:#4caf50']*len(r) if r['Signal']=='BUY'
                else ['background-color:#3d1a1a;color:#f44336']*len(r)
                for _ in r
            ), use_container_width=True
        )

        strong = [s for s in signals if s['score'] >= 75]
        ms1, ms2, ms3 = st.columns(3)
        ms1.metric("🟢 Strong (≥75)", len(strong))
        ms2.metric("📊 Total Signals", len(signals))
        avg_rr = sum(s['rr'] for s in signals) / len(signals) if signals else 0
        ms3.metric("📐 Avg R:R", f"1:{avg_rr:.2f}")

        for s in signals:
            st.session_state.pair_cooldown[s['symbol']] = params['cooldown']

        st.session_state.signal_history.extend(signals)
        st.session_state.signal_history = st.session_state.signal_history[-50:]
    else:
        st.warning(
            "⚠️ **No signals.**\n\n"
            "Try: lower Min Score, lower Min ATR%, H1 TF, or "
            "relax Alert Settings above."
        )

    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} UTC | Refresh #{st.session_state.refresh_count} | {active_strat} | Yahoo Finance v8")
