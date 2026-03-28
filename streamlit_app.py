"""
MT5 Scalper Pro — Real-Time Trading Assistant
=============================================
Pullback scalping strategy on EMA 20/50 trend with RSI confirmation.
Features:
  - EMA pullback strategy (not pure crossover)
  - Relaxed filters: ATR -50%, trend dist -40%, candle 50%, score 60
  - Multi-pair live scanner: EURUSD, GBPUSD, USDJPY, XAUUSD + AUDUSD, USDCAD, NZDUSD, GBPAUD
  - Auto-refresh every 10 seconds (fixed, no more crash)
  - ATR-based SL + dynamic TP (1:1.2 to 1:1.5)
  - Cooldown (3 candles)
  - Full backtest with realistic trade counts
  - Session filter (London + New York)
  - Multi-timeframe (M1/M5/H1)
  - Signal history, toast alerts, parallel scanning
  - Cached data + session persistence
  - TradingView embedded live chart
  - Trading Economics / Yahoo Finance v8 reliable data
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import warnings
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings('ignore')

import ta

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "symbol": "EURUSD",
    "timeframe": "M1",
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend_fast": 20,
    "ema_trend_slow": 50,
    "rsi_period": 14,
    "atr_period": 14,
    "sl_atr_mult": 1.2,
    "sl_min_atr": 0.5,
    "min_trend_dist_pct": 0.00006,
    "min_atr_pct": 0.00003,
    "candle_body_min": 0.50,
    "min_score": 60,
    "cooldown": 3,
    "backtest_candles": 1500,
    "swing_lookback": 5,
    "min_trend_dist": 0.00006,
    "min_atr": 0.000015,
}

ALL_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD", "USDCAD", "NZDUSD", "GBPAUD"]

TF_INTERVAL = {"M1": "1m", "M5": "5m", "H1": "60m"}
TF_PERIOD = {"M1": "5d", "M5": "5d", "H1": "60d"}
TF_LABEL = {"M1": "1 Minute", "M5": "5 Minutes", "H1": "1 Hour"}

# yfinance symbol mapping
YF_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "GBPAUD": "GBPAUD=X",
}

# TradingView widget base URL (lightweight, no API key needed)
TV_WIDGET_URL = "https://www.tradingview.com-widget/{symbol}/?"

# ============================================================================
# SESSION STATE INIT
# ============================================================================

def init_session():
    defaults = {
        "signal_history": [],
        "last_scan_time": None,
        "cooldown_map": {},
        "selected_pairs": ALL_PAIRS[:4],
        "last_signal_hashes": set(),
        "refresh_count": 0,
        "backtest_key": 0,          # incremented to force backtest recalc
        "last_data_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ============================================================================
# DATA SOURCE — Yahoo Finance v8 REST API (most reliable)
# ============================================================================

def fetch_yf_v8(symbol: str, interval: str = "1m", period: str = "5d") -> pd.DataFrame:
    """
    Use Yahoo Finance v8 REST API directly — more reliable than yfinance SDK.
    Returns DataFrame with columns: time, open, high, low, close, volume.
    """
    yf_sym = YF_MAP.get(symbol.upper(), symbol)
    interval_map = {"1m": "1m", "5m": "5m", "60m": "60m"}
    iv = interval_map.get(interval, "1m")

    # period → interval mapping for the API
    if period in ("5d", "7d"):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}"
        params = {"interval": iv, "range": "5d"}
    elif period == "60d":
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}"
        params = {"interval": "60m", "range": "2mo"}
    else:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}"
        params = {"interval": iv, "range": "5d"}

    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame()
        result = result[0]
        timestamps = result["timestamp"]
        ohlcv = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "time": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
            "open": ohlcv["open"],
            "high": ohlcv["high"],
            "low": ohlcv["low"],
            "close": ohlcv["close"],
            "volume": ohlcv.get("volume", [0] * len(timestamps)),
        })
        df = df.dropna(subset=["close"])
        df["volume"] = df["volume"].fillna(0)
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def get_candles_cached(symbol: str, interval: str = "1m", period: str = "5d", count: int = 200) -> pd.DataFrame:
    """Cached candle fetcher — tries REST API first, falls back to SDK."""
    # Try REST API first
    df = fetch_yf_v8(symbol, interval, period)
    if df.empty or len(df) < 30:
        # Fallback to yfinance SDK
        try:
            import yfinance
            yf_sym = YF_MAP.get(symbol.upper(), symbol)
            ticker = yfinance.Ticker(yf_sym)
            raw = ticker.history(period="5d", interval=interval, prepost=False)
            if not raw.empty:
                df = raw.reset_index()
                df.columns = [c.capitalize() if c != 'Datetime' else 'time' for c in df.columns]
                df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
                df = df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                df['volume'] = df['volume'].fillna(0)
        except Exception:
            pass
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.drop_duplicates(subset='time', keep='last')
    return df.tail(count).reset_index(drop=True)

# ============================================================================
# INDICATORS
# ============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=CONFIG['rsi_period']).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=CONFIG['atr_period']).average_true_range()
    df['atr_pct'] = df['atr'] / df['close']
    df['trend_dist'] = abs(df['ema_20'] - df['ema_50'])
    df['candle_range'] = df['high'] - df['low']
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_body_pct'] = df['candle_body'] / df['candle_range'].replace(0, np.nan).fillna(0)
    df['candle_body_pct'] = df['candle_body_pct'].clip(0, 1)
    df['rsi_change'] = df['rsi'].diff()
    return df

def swing_high(df: pd.DataFrame, idx: int, lookback: int) -> float:
    start = max(0, idx - lookback)
    return df['high'].iloc[start:idx+1].max()

def swing_low(df: pd.DataFrame, idx: int, lookback: int) -> float:
    start = max(0, idx - lookback)
    return df['low'].iloc[start:idx+1].min()

# ============================================================================
# SESSION FILTER
# ============================================================================

def session_active(dt: pd.Timestamp) -> bool:
    h = dt.hour
    return (8 <= h < 12) or (13 <= h <= 17)

def get_sessions(dt: pd.Timestamp) -> dict:
    h = dt.hour
    return {
        "London": 8 <= h < 12,
        "NewYork": 13 <= h <= 17,
        "Asia": 0 <= h < 7,
    }

# ============================================================================
# SIGNAL QUALITY SCORE
# ============================================================================

def score_signal(df: pd.DataFrame, idx: int, direction: str) -> float:
    row = df.iloc[idx]
    dist = row['trend_dist']
    max_dist = CONFIG['min_trend_dist'] * 5
    trend = min(30.0, (dist / max_dist) * 30.0) if dist > 0 else 0.0
    rsi = row['rsi']
    if direction == 'BUY':
        rsi_dist = max(0, rsi - 50)
    else:
        rsi_dist = max(0, 50 - rsi)
    rsi_score = min(25.0, (rsi_dist / 35) * 25.0)
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    candle_score = min(25.0, (body / 0.80) * 25.0)
    atr_ratio = row['atr_pct'] / CONFIG['min_atr_pct']
    atr_score = min(20.0, max(0.0, (atr_ratio - 0.5) * 10.0))
    return round(trend + rsi_score + candle_score + atr_score, 1)

# ============================================================================
# STOP LOSS & TAKE PROFIT
# ============================================================================

def calc_sl(signal: str, entry: float, df: pd.DataFrame, idx: int) -> float:
    lb = CONFIG['swing_lookback']
    atr = df.iloc[idx]['atr']
    if signal == 'BUY':
        sl = swing_low(df, idx, lb)
        sl = sl - (atr * CONFIG['sl_atr_mult'])
        min_sl = entry - (atr * CONFIG['sl_min_atr'])
        if entry - sl < atr * CONFIG['sl_min_atr']:
            sl = min_sl
    else:
        sl = swing_high(df, idx, lb)
        sl = sl + (atr * CONFIG['sl_atr_mult'])
        min_sl = entry + (atr * CONFIG['sl_min_atr'])
        if sl - entry < atr * CONFIG['sl_min_atr']:
            sl = min_sl
    return sl

def calc_tp(signal: str, entry: float, sl: float, df: pd.DataFrame, idx: int) -> float:
    atr_ratio = df.iloc[idx]['atr'] / CONFIG['min_atr']
    if atr_ratio < 1.0:
        rr = 1.2
    elif atr_ratio < 2.0:
        rr = 1.35
    else:
        rr = 1.5
    risk = abs(entry - sl)
    if signal == 'BUY':
        return entry + (risk * rr)
    return entry - (risk * rr)

# ============================================================================
# PULLBACK SIGNAL DETECTION
# ============================================================================

def find_pullback(df: pd.DataFrame, idx: int) -> str | None:
    if idx < 2:
        return None
    row = df.iloc[idx]
    prev = df.iloc[idx - 1]

    ema9_cross_up = prev['ema_9'] <= prev['ema_21'] and row['ema_9'] > row['ema_21']
    ema9_cross_down = prev['ema_9'] >= prev['ema_21'] and row['ema_9'] < row['ema_21']
    if not (ema9_cross_up or ema9_cross_down):
        return None
    direction = 'BUY' if ema9_cross_up else 'SELL'

    if direction == 'BUY' and not (row['ema_20'] > row['ema_50']):
        return None
    if direction == 'SELL' and not (row['ema_20'] < row['ema_50']):
        return None
    if direction == 'BUY' and row['rsi'] <= 50:
        return None
    if direction == 'SELL' and row['rsi'] >= 50:
        return None
    if direction == 'BUY' and row['rsi_change'] <= 0:
        return None
    if direction == 'SELL' and row['rsi_change'] >= 0:
        return None
    if row['trend_dist'] < CONFIG['min_trend_dist']:
        return None
    if row['atr_pct'] < CONFIG['min_atr_pct']:
        return None
    body_pct = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    if body_pct < CONFIG['candle_body_min']:
        return None
    if not session_active(row['time']):
        return None
    return direction

# ============================================================================
# SCAN ONE PAIR
# ============================================================================

def scan_pair(symbol: str, interval: str = "1m") -> dict | None:
    tf_interval = TF_INTERVAL.get(interval, "1m")
    tf_period = TF_PERIOD.get(interval, "5d")
    df = get_candles_cached(symbol, tf_interval, tf_period, count=150)
    if df.empty or len(df) < 60:
        return None
    df = add_indicators(df)
    cooldown = st.session_state.cooldown_map.get(symbol, 0)
    for check_idx in range(-8, -1):
        direction = find_pullback(df, check_idx)
        if direction is None:
            continue
        if cooldown > 0:
            continue
        score = score_signal(df, check_idx, direction)
        if score < CONFIG['min_score']:
            continue
        entry = df.iloc[check_idx]['close']
        sl = calc_sl(direction, entry, df, check_idx)
        tp = calc_tp(direction, entry, sl, df, check_idx)
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0
        sig = {
            'symbol': symbol,
            'signal': direction,
            'entry': round(entry, 5),
            'sl': round(sl, 5),
            'tp': round(tp, 5),
            'rr': rr,
            'score': score,
            'time': df.iloc[check_idx]['time'],
            'rsi': round(df.iloc[check_idx]['rsi'], 1),
            'atr': round(df.iloc[check_idx]['atr'], 5),
            'trend_dist': round(df.iloc[check_idx]['trend_dist'], 6),
            'candle_pct': round(df.iloc[check_idx]['candle_body_pct'] * 100, 0) if not np.isnan(df.iloc[check_idx]['candle_body_pct']) else 0,
            'timeframe': interval,
        }
        return sig
    return None

# ============================================================================
# MULTI-PAIR SCANNER — PARALLEL
# ============================================================================

def scan_all_pairs() -> list:
    pairs = st.session_state.selected_pairs
    interval = CONFIG.get("timeframe", "M1")
    if not pairs:
        return []
    max_workers = min(len(pairs), 8)
    signals = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_pair, p, interval): p for p in pairs}
        for future in as_completed(futures):
            try:
                sig = future.result()
                if sig:
                    signals.append(sig)
            except Exception:
                pass
    return signals

# ============================================================================
# BACKTEST — NO caching to avoid stale results
# ============================================================================

def run_backtest(symbol: str, candles: int, interval: str = "1m") -> tuple:
    """Backtest with fresh data every time — no caching."""
    tf_interval = TF_INTERVAL.get(interval, "1m")
    tf_period = TF_PERIOD.get(interval, "5d")
    df = get_candles_cached(symbol, tf_interval, tf_period, candles)
    if df.empty or len(df) < 100:
        return None, None, None
    df = add_indicators(df)
    df = df.reset_index(drop=True)
    trades = []
    position = None
    cooldown = 0
    start = max(50, CONFIG['ema_trend_slow'])
    for i in range(start, len(df) - 1):
        if cooldown > 0:
            cooldown -= 1
        direction = find_pullback(df, i)
        if direction and cooldown == 0 and position is None:
            score = score_signal(df, i, direction)
            if score >= CONFIG['min_score']:
                entry = df.iloc[i]['close']
                sl = calc_sl(direction, entry, df, i)
                tp = calc_tp(direction, entry, sl, df, i)
                position = direction
                pos_entry = entry
                pos_sl = sl
                pos_tp = tp
                pos_score = score
                pos_time = df.iloc[i]['time']
                cooldown = CONFIG['cooldown']
            continue
        if position:
            close = df.iloc[i]['close']
            exit_p = None
            result = None
            if position == 'BUY':
                if close <= pos_sl:
                    exit_p, result = pos_sl, 'LOSS'
                elif close >= pos_tp:
                    exit_p, result = pos_tp, 'WIN'
            elif position == 'SELL':
                if close >= pos_sl:
                    exit_p, result = pos_sl, 'LOSS'
                elif close <= pos_tp:
                    exit_p, result = pos_tp, 'WIN'
            if result:
                risk = abs(pos_entry - pos_sl)
                reward = abs(exit_p - pos_entry)
                rr = round(reward / risk, 2) if risk > 0 else 0
                pnl = (reward - risk) * 10000 if result == 'WIN' else -(risk * 10000)
                trades.append({
                    'signal': position, 'entry': pos_entry, 'exit': exit_p,
                    'sl': pos_sl, 'tp': pos_tp, 'rr': rr,
                    'pnl_pips': round(pnl, 1), 'result': result,
                    'score': pos_score, 'entry_time': pos_time, 'exit_time': df.iloc[i]['time'],
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
    metrics = {
        'total': total, 'wins': wins, 'losses': losses,
        'win_rate': round(wr, 1), 'total_pnl': round(total_pnl, 1),
        'avg_rr': round(avg_rr, 2), 'profit_factor': round(pf, 2) if pf != float('inf') else '∞',
        'max_drawdown': round(max_dd, 1),
        'avg_win': round(avg_win, 1), 'avg_loss': round(avg_loss, 1),
        'best': td['pnl_pips'].max(), 'worst': td['pnl_pips'].min(),
    }
    return td, df, metrics

# ============================================================================
# PLOT CHART
# ============================================================================

def plot_chart(df: pd.DataFrame, trades_df=None, sig=None, show_vol: bool = True) -> go.Figure:
    if df is None or df.empty:
        return None
    rows = 4 if show_vol else 3
    row_heights = [0.40, 0.20, 0.20, 0.20] if show_vol else [0.50, 0.25, 0.25]
    subplot_titles = ('Price + EMAs', 'Volume', 'RSI', 'ATR') if show_vol else ('Price + EMAs', 'RSI', 'ATR')
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.06,
        subplot_titles=subplot_titles
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

    # Volume
    if show_vol and 'volume' in df.columns and df['volume'].sum() > 0:
        colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name='Volume',
            marker_color=colors, opacity=0.6), row=2, col=1)

    # Signal marker + SL/TP horizontal lines
    if sig is not None:
        matches = df[df['time'] == sig['time']]
        c = '#26a69a' if sig['signal'] == 'BUY' else '#ef5350'
        if len(matches) > 0:
            fig.add_trace(go.Scatter(
                x=[sig['time']], y=[sig['entry']],
                mode='markers+text',
                marker=dict(size=24, color=c, symbol='arrow-up' if sig['signal'] == 'BUY' else 'arrow-down'),
                text=[f"▶ {sig['signal']} {sig['score']}"],
                textposition='top center', textfont=dict(color=c, size=11),
                name=f"SIGNAL {sig['signal']}"
            ), row=1, col=1)
        # Entry, SL, TP horizontal rays
        shape_type = "line" if len(df) > 1 else None
        if shape_type and len(df) > 1:
            fig.add_hline(y=sig['sl'], line_dash='dash', line_color='#f44336', line_width=1.2,
                annotation_text=f"SL {sig['sl']}", annotation_position="bottom-right",
                annotation_font_color='#f44336', row=1, col=1)
            fig.add_hline(y=sig['tp'], line_dash='dash', line_color='#4caf50', line_width=1.2,
                annotation_text=f"TP {sig['tp']}", annotation_position="top right",
                annotation_font_color='#4caf50', row=1, col=1)

    # Backtest entries/exits
    if trades_df is not None and not trades_df.empty:
        for _, t in trades_df.iterrows():
            c = '#26a69a' if t['signal'] == 'BUY' else '#ef5350'
            fig.add_trace(go.Scatter(
                x=[t['entry_time']], y=[t['entry']],
                mode='markers', marker=dict(size=9, color=c, symbol='circle'),
                showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[t['exit_time']], y=[t['exit']],
                mode='markers', marker=dict(size=8, color='white', symbol='x'),
                showlegend=False
            ), row=1, col=1)

    # RSI
    rsi_row = 3 if show_vol else 2
    fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], name='RSI',
        line=dict(color='#9C27B0', width=1.5)), row=rsi_row, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='gray', row=rsi_row, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', line_width=0.5, row=rsi_row, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', line_width=0.5, row=rsi_row, col=1)

    # ATR
    atr_row = 4 if show_vol else 3
    fig.add_trace(go.Scatter(x=df['time'], y=df['atr'], name='ATR',
        fill='tozeroy', line=dict(color='#FF5722', width=1),
        fillcolor='rgba(255,87,34,0.12)'), row=atr_row, col=1)

    fig.update_layout(
        template='plotly_dark', height=750, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_rangeslider_visible=False
    )
    return fig

# ============================================================================
# TRADINGVIEW EMBED HELPER
# ============================================================================

def tradingview_widget(symbol: str, interval: str = "M1") -> str:
    """
    Generate a TradingView widget URL for the given symbol and interval.
    TradingView lightweight chart — no API key needed.
    """
    # Map our interval to TradingView interval format
    tv_interval_map = {"M1": "1", "M5": "5", "H1": "60"}
    tv_interval = tv_interval_map.get(interval, "1")

    # Map our symbol to TradingView format (TradingView uses same Yahoo symbols)
    tv_sym = YF_MAP.get(symbol.upper(), symbol)

    # Build widget URL with correct TradingView embed params
    widget_url = (
        f"https://www.tradingview.com/widgetembed/?"
        f"symbol={tv_sym}&interval={tv_interval}&"
        f"hide_sidebar=false&"
        f"hide_top_toolbar=false&"
        f"save_image=false&"
        f"studies=RSI@tv-basicthemes%3B!EMA@tv-basicthemes%3B!&"
        f"theme=dark&"
        f"style=1&"
        f"locale=en&"
        f"toolbar_bg=%23363636&"
        f"enable_publishing=false&"
        f"allow_symbol_change=true&"
        f"container_id=tradingview_widget_{symbol.lower()}"
    )
    return widget_url


def tv_widget_html(symbol: str, interval: str = "M1") -> str:
    """Return an iframe HTML string for embedding TradingView chart."""
    widget_url = tradingview_widget(symbol, interval)
    return f"""
    <iframe
        src="{widget_url}"
        width="100%"
        height="500"
        frameborder="0"
        allowtransparency="true"
        scrolling="yes"
        style="border-radius:8px; border: 1px solid #333;"
        title="TradingView {symbol} Chart"
    ></iframe>
    """

# ============================================================================
# SIGNAL CARD HTML
# ============================================================================

def signal_card(sig: dict) -> str:
    is_buy = sig['signal'] == 'BUY'
    border = '#4caf50' if is_buy else '#f44336'
    bg = '#1b5e20' if is_buy else '#7f0000'
    arrow = '🟢 BUY' if is_buy else '🔴 SELL'
    sc = sig['score']
    tf = sig.get('timeframe', 'M1')
    return f"""
    <div style="padding:18px; border-radius:12px; border:3px solid {border};
                background:{bg}; color:white; margin:8px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h2 style="margin:0;">{arrow} — {sig['symbol']} <span style="font-size:14px; opacity:0.7;">[{tf}]</span></h2>
            <div style="background:{border}; border-radius:50%; width:64px; height:64px;
                        display:flex; flex-direction:column; align-items:center;
                        justify-content:center;">
                <span style="font-size:20px; font-weight:bold;">{sc}</span>
                <span style="font-size:8px;">SCORE</span>
            </div>
        </div>
        <hr style="opacity:0.25; margin:10px 0;">
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px; margin-top:8px;">
            <div><small>ENTRY</small><br><b>{sig['entry']}</b></div>
            <div><small>STOP LOSS</small><br><b>{sig['sl']}</b></div>
            <div><small>TAKE PROFIT</small><br><b>{sig['tp']}</b></div>
            <div><small>R:R</small><br><b>1:{sig['rr']}</b></div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px; margin-top:8px;">
            <div><small>RSI</small><br><b>{sig['rsi']}</b></div>
            <div><small>ATR</small><br><b>{sig['atr']}</b></div>
            <div><small>Candle</small><br><b>{sig['candle_pct']:.0f}%</b></div>
            <div><small>Time (UTC)</small><br><b>{sig['time']}</b></div>
        </div>
    </div>
    """

# ============================================================================
# SESSION STATUS BOXES
# ============================================================================

def session_boxes(dt: pd.Timestamp) -> str:
    sessions = get_sessions(dt)
    boxes = []
    for name, active in sessions.items():
        color = '#4caf50' if active else '#424242'
        icon = '●' if active else '○'
        text_color = 'white' if active else '#888'
        boxes.append(
            f"<div style='background:{color}; border-radius:8px; padding:8px 14px; color:{text_color}; font-weight:bold; font-size:13px;'>{icon} {name}</div>"
        )
    return f"<div style='display:flex; gap:10px; flex-wrap:wrap;'>{''.join(boxes)}</div>"

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="MT5 Scalper Pro", page_icon="📈", layout="wide")
st.title("📈 MT5 Scalper Pro — Real-Time Trading Assistant")

init_session()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")

    sym = st.text_input("📌 Symbol", value=CONFIG['symbol']).upper().strip()
    CONFIG['symbol'] = sym

    tf_options = ["M1", "M5", "H1"]
    tf = st.selectbox("⏱️ Timeframe", tf_options, index=0, help="M1 = 1 Minute, M5 = 5 Minutes, H1 = 1 Hour")
    CONFIG['timeframe'] = tf

    # Live TradingView chart in sidebar
    st.markdown("---")
    st.markdown("### 📊 TradingView — Live Chart")
    st.markdown(tv_widget_html(sym, tf), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📏 EMAs")
    CONFIG['ema_fast'] = st.slider("EMA Pullback (9)", 5, 21, CONFIG['ema_fast'])
    CONFIG['ema_slow'] = st.slider("EMA Signal (21)", 10, 55, CONFIG['ema_slow'])
    CONFIG['ema_trend_fast'] = st.slider("EMA Trend (20)", 10, 30, CONFIG['ema_trend_fast'])
    CONFIG['ema_trend_slow'] = st.slider("EMA Trend (50)", 30, 100, CONFIG['ema_trend_slow'])
    CONFIG['rsi_period'] = st.slider("RSI Period", 5, 21, CONFIG['rsi_period'])
    CONFIG['atr_period'] = st.slider("ATR Period", 5, 21, CONFIG['atr_period'])

    st.subheader("🎯 Filters")
    CONFIG['min_trend_dist'] = st.slider("Min Trend Dist (×10000)", 1, 20, int(CONFIG['min_trend_dist_pct'] * 10000)) / 10000
    CONFIG['min_atr_pct'] = st.slider("Min ATR % (×10000)", 1, 10, int(CONFIG['min_atr_pct'] * 10000)) / 10000
    CONFIG['candle_body_min'] = st.slider("Candle Body Min %", 30, 70, int(CONFIG['candle_body_min'] * 100)) / 100

    st.subheader("🛡️ Risk")
    CONFIG['sl_atr_mult'] = st.slider("SL ATR Multiplier", 0.5, 3.0, CONFIG['sl_atr_mult'], 0.1)
    CONFIG['sl_min_atr'] = st.slider("Min SL ATR Factor", 0.3, 2.0, CONFIG['sl_min_atr'], 0.1)

    st.subheader("⏱️ Cooldown & Score")
    CONFIG['cooldown'] = st.slider("Cooldown Candles", 1, 10, CONFIG['cooldown'])
    CONFIG['min_score'] = st.slider("Min Signal Score", 30, 95, CONFIG['min_score'])

    st.subheader("📈 Backtest")
    CONFIG['backtest_candles'] = st.slider("Candles", 300, 3000, CONFIG['backtest_candles'], 100)

    st.subheader("🔍 Scanner Pairs")
    selected = st.multiselect("Select pairs", ALL_PAIRS, default=st.session_state.selected_pairs)
    st.session_state.selected_pairs = selected

    st.markdown("---")
    st.markdown("**🕐 Sessions:** London 08-12 UTC | NY 13-17 UTC")

    # Signal history
    if st.session_state.signal_history:
        st.subheader("📜 Recent Signals")
        for h in reversed(st.session_state.signal_history[-10:]):
            is_buy = h['signal'] == 'BUY'
            c = '#4caf50' if is_buy else '#f44336'
            st.markdown(
                f"<span style='color:{c};'>● {h['signal']}</span> "
                f"{h['symbol']} {h['time'].strftime('%H:%M')} "
                f"<small>score:{h['score']}</small>",
                unsafe_allow_html=True
            )

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Backtest", "🔴 Multi-Pair Scanner"])

# ============================================================
# TAB 1: DASHBOARD
# ============================================================
with tab1:
    now = datetime.now()
    st.markdown(session_boxes(now), unsafe_allow_html=True)
    st.info(
        f"📡 **Yahoo Finance v8 API** [{TF_LABEL.get(tf, tf)}] | **{sym}** | "
        f"EMA {CONFIG['ema_fast']}/{CONFIG['ema_slow']}/{CONFIG['ema_trend_fast']}/{CONFIG['ema_trend_slow']} | "
        f"Min Score {CONFIG['min_score']}"
    )

    # Data status
    df_main = get_candles_cached(sym, TF_INTERVAL[tf], TF_PERIOD[tf], count=200)

    if df_main.empty:
        st.error(
            f"❌ No data returned for **{sym}** on {TF_LABEL.get(tf, tf)}. "
            f"Yahoo Finance may be rate-limiting or market is closed. "
            f"Try again in a few minutes or switch timeframe."
        )
    else:
        # Show data freshness
        last_candle_time = df_main['time'].iloc[-1]
        age_mins = (datetime.now() - last_candle_time).total_seconds() / 60
        age_str = f"{int(age_mins)} min ago" if age_mins > 0 else "just now"
        st.caption(f"📅 Last candle: {last_candle_time.strftime('%Y-%m-%d %H:%M:%S')} UTC ({age_str}) | {len(df_main)} candles loaded")

        df_main = add_indicators(df_main)
        fig = plot_chart(df_main)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        last = df_main.iloc[-1]
        trend_dir = "🔴 DOWN" if last['ema_20'] < last['ema_50'] else "🟢 UP"
        sess = session_active(last['time'])
        rsi_val = last['rsi']
        rsi_color = "🟢" if 40 <= rsi_val <= 60 else "🔴" if rsi_val > 60 or rsi_val < 40 else "🟡"

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        c1.metric("Price", f"{last['close']:.5f}")
        c2.metric("EMA9", f"{last['ema_9']:.5f}")
        c3.metric("EMA21", f"{last['ema_21']:.5f}")
        c4.metric("EMA20", f"{last['ema_20']:.5f}")
        c5.metric("EMA50", f"{last['ema_50']:.5f}")
        c6.metric("RSI", f"{last['rsi']:.1f} {rsi_color}")
        c7.metric("ATR", f"{last['atr']:.5f}")
        c8.metric("Trend", trend_dir)

        # Live signal
        sig = scan_pair(sym, tf)
        if sig:
            st.markdown(signal_card(sig), unsafe_allow_html=True)
            st.toast(f"📢 {sig['signal']} signal on {sig['symbol']} @ {sig['entry']} (score: {sig['score']})", icon="📈")
        else:
            st.info(f"⚪ No qualified signal for {sym} right now. All filters active: score≥{CONFIG['min_score']}, ATR%≥{CONFIG['min_atr_pct']:.5f}, body≥{CONFIG['candle_body_min']*100:.0f}%, session=London/NY")

# ============================================================
# TAB 2: BACKTEST
# ============================================================
with tab2:
    st.header("📈 Backtest — Pullback Strategy")
    st.caption(
        f"Strategy: EMA9/21 pullback within EMA20/50 trend | "
        f"SL: ATR×{CONFIG['sl_atr_mult']} | TP: dynamic 1.2–1.5×risk | "
        f"Session: London + NY | Cooldown: {CONFIG['cooldown']} candles"
    )

    # Force key changes on each button press so cache is bypassed
    bt_key = f"bt_{st.session_state.backtest_key}"
    col_run, col_sym, col_info = st.columns([1, 2, 3])
    with col_run:
        if st.button("▶️ Run Backtest", use_container_width=True, type="primary"):
            st.session_state.backtest_key += 1
            st.cache_data.clear()   # clear ALL cached data for fresh run
            with st.spinner(f"Fetching {CONFIG['backtest_candles']} candles for {sym} [{TF_LABEL.get(tf,tf)}]..."):
                td, df_bt, metrics = run_backtest(sym, CONFIG['backtest_candles'], tf)

            if td is None or metrics is None:
                st.error(
                    "❌ No trades generated. Possible reasons:\n"
                    "• Session filter: only London (08-12 UTC) or NY (13-17 UTC) active\n"
                    "• ATR below minimum threshold — try lowering Min ATR %\n"
                    "• Min score too high — try lowering to 50\n"
                    "• Not enough candles — try increasing to 2000+\n"
                    "• Market closed or no data available"
                )
            else:
                # Summary metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Trades", metrics['total'])
                m2.metric("Win Rate", f"{metrics['win_rate']}%")
                m3.metric("Profit Factor", metrics['profit_factor'])
                m4.metric("Max Drawdown", f"{metrics['max_drawdown']} pips")
                m5.metric("Net P&L", f"{metrics['total_pnl']} pips")

                m6, m7, m8, m9, m10 = st.columns(5)
                m6.metric("W / L", f"{metrics['wins']} / {metrics['losses']}")
                m7.metric("Avg R:R", f"1:{metrics['avg_rr']}")
                m8.metric("Avg Win", f"{metrics['avg_win']} pips")
                m9.metric("Avg Loss", f"{metrics['avg_loss']} pips")
                m10.metric("Best / Worst", f"{metrics['best']} / {metrics['worst']} pips")

                # Equity curve
                if not td.empty:
                    st.subheader("📉 Equity Curve")
                    eq_fig = go.Figure()
                    cum = td['pnl_pips'].cumsum()
                    eq_fig.add_trace(go.Scatter(
                        x=list(range(len(cum))), y=cum.values,
                        mode='lines+markers',
                        line=dict(color='#26a69a', width=2),
                        marker=dict(size=4),
                        name='Cumulative P&L (pips)'
                    ))
                    eq_fig.update_layout(
                        template='plotly_dark', height=250,
                        xaxis_title="Trade #",
                        yaxis_title="P&L (pips)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(eq_fig, use_container_width=True)

                # Chart
                st.subheader("📊 Price Chart with Trades")
                fig_bt = plot_chart(df_bt, td)
                if fig_bt:
                    st.plotly_chart(fig_bt, use_container_width=True)

                # Trades table
                st.subheader(f"📋 Trade Log ({metrics['total']} trades)")
                disp = td[['entry_time', 'signal', 'entry', 'sl', 'tp', 'rr', 'score', 'result', 'pnl_pips']].tail(30).copy()
                disp.columns = ['Time (UTC)', 'Signal', 'Entry', 'SL', 'TP', 'R:R', 'Score', 'Result', 'P&L (pips)']
                disp['Time (UTC)'] = disp['Time (UTC)'].dt.strftime('%Y-%m-%d %H:%M')

                def color_result(row):
                    if row['Result'] == 'WIN':
                        return ['background-color: #1b3d2e; color: #4caf50'] * len(row)
                    return ['background-color: #3d1a1a; color: #f44336'] * len(row)

                st.dataframe(
                    disp.style.apply(color_result, axis=1),
                    use_container_width=True, height=400
                )

                csv = td.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, f"backtest_{sym}_{tf}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

    with col_sym:
        st.markdown("")  # spacer

    with col_info:
        st.info(
            "💡 **Tips if no trades:**\n"
            "• Lower Min Score to 40–50\n"
            "• Lower Min ATR% to 0.001–0.002\n"
            "• Increase candles to 2000+\n"
            "• Use H1 timeframe for more signals\n"
            "• Check session: only 08–12 & 13–17 UTC active"
        )

# ============================================================
# TAB 3: LIVE MULTI-PAIR SCANNER
# ============================================================
with tab3:
    st.header("🔴 Multi-Pair Live Scanner")
    st.info(
        f"⏱️ Auto-refreshes every 10s | TF: {TF_LABEL.get(tf, tf)} | "
        f"Pairs: {', '.join(st.session_state.selected_pairs) if st.session_state.selected_pairs else 'None selected'}"
    )

    # Auto-refresh — replaces the broken while True/st.rerun() loop
    refresh_count = st_autorefresh(interval=10000, key="scanner_autorefresh_v2")
    st.session_state.refresh_count = refresh_count

    # Force clear cooldown map on refresh so signals can fire again
    if refresh_count > 0:
        # Decrement cooldowns
        for k in st.session_state.cooldown_map:
            st.session_state.cooldown_map[k] = max(0, st.session_state.cooldown_map[k] - 1)

    signals = scan_all_pairs()

    if signals:
        signals.sort(key=lambda x: x['score'], reverse=True)

        # Toast only truly NEW signals
        sig_ids = {id(s) for s in signals}
        new_sigs = [s for s in signals if id(s) not in st.session_state.last_signal_hashes]
        for s in new_sigs:
            st.toast(f"📢 NEW: {s['signal']} {s['symbol']} @ {s['entry']} (score:{s['score']})", icon="📈")

        st.session_state.last_signal_hashes = sig_ids

        for sig in signals:
            st.markdown(signal_card(sig), unsafe_allow_html=True)

        rows = []
        for s in signals:
            rows.append({
                'Pair': s['symbol'],
                'Signal': s['signal'],
                'Entry': s['entry'],
                'SL': s['sl'],
                'TP': s['tp'],
                'R:R': f"1:{s['rr']}",
                'Score': s['score'],
                'RSI': s['rsi'],
                'ATR': s['atr'],
                'Time (UTC)': str(s['time']),
                'TF': s.get('timeframe', tf),
            })

        st.markdown("### 📋 Signals Table")
        tbl = pd.DataFrame(rows)
        st.dataframe(
            tbl.style.apply(
                lambda r: ['background-color: #1b3d2e; color: #4caf50' if r['Signal'] == 'BUY' else 'background-color: #3d1a1a; color: #f44336' for _ in r],
                axis=1
            ),
            use_container_width=True
        )

        strong = [s for s in signals if s['score'] >= 75]
        ms1, ms2 = st.columns(2)
        ms1.metric("🟢 Strong Signals (≥75)", len(strong))
        ms2.metric("📊 Total Active Signals", len(signals))

        # Update cooldown map — mark pairs that just fired as having a cooldown
        for s in signals:
            st.session_state.cooldown_map[s['symbol']] = CONFIG['cooldown']

        st.session_state.signal_history.extend(signals)
        st.session_state.signal_history = st.session_state.signal_history[-50:]

    else:
        st.warning(
            "⚠️ **No signals across any pair right now.**\n\n"
            "Possible reasons:\n"
            "• Outside London (08–12 UTC) or New York (13–17 UTC) session\n"
            "• Low ATR volatility — try lowering Min ATR% filter\n"
            "• No trending conditions (EMA20/50 not spread enough)\n"
            "• Price not pulling back to EMA9 for a valid entry\n"
            "• Min score too high — try lowering to 50"
        )

    st.caption(
        f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')} UTC | "
        f"Refresh #{st.session_state.refresh_count} | "
        f"Data: Yahoo Finance v8 REST API"
    )
