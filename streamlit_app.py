"""
MT5 Scalper Pro — Real-Time Trading Assistant
=============================================
Pullback scalping strategy on EMA 20/50 trend with RSI confirmation.
Features:
  - EMA pullback strategy (not pure crossover)
  - Relaxed filters: ATR -50%, trend dist -40%, candle 50%, score 60
  - Multi-pair live scanner: EURUSD, GBPUSD, USDJPY, XAUUSD
  - Auto-refresh every 10 seconds
  - ATR-based SL + dynamic TP (1:1.2 to 1:1.5)
  - Cooldown (3 candles)
  - Full backtest with realistic trade counts
  - Session filter (London + New York)
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
import warnings
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

def install_deps():
    deps = ['ta', 'yfinance', 'plotly']
    for dep in deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])

install_deps()

import ta
import yfinance

# ============================================================================
# CONFIGURATION — relaxed defaults for realistic signal frequency
# ============================================================================

CONFIG = {
    "symbol": "EURUSD",
    # Pullback EMAs
    "ema_fast": 9,        # pullback detection
    "ema_slow": 21,       # signal line
    "ema_trend_fast": 20,  # trend direction
    "ema_trend_slow": 50,  # trend direction
    # RSI
    "rsi_period": 14,
    # ATR
    "atr_period": 14,
    # Stop loss / TP
    "sl_atr_mult": 1.2,    # SL buffer = ATR * this
    "sl_min_atr": 0.5,     # minimum SL distance = ATR * this
    # FILTERS (relaxed from previous version)
    "min_trend_dist": 0.00006,   # EMA20/50 distance (was 0.00010, now -40%)
    "min_atr": 0.000015,        # minimum ATR (was 0.00003, now -50%)
    "candle_body_min": 0.50,    # body > 50% of range (was 60%)
    "min_score": 60,             # minimum signal score (was 70)
    # Cooldown
    "cooldown": 3,              # ignore 3 candles after signal
    # Backtest
    "backtest_candles": 1500,
    "swing_lookback": 5,
}

SCAN_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

# ============================================================================
# DATA SOURCE
# ============================================================================

class DataSource:
    name = "yfinance"

    def _yf_symbol(self, sym: str) -> str:
        sym = sym.upper().strip()
        if "=" in sym:
            return sym
        if len(sym) == 6:
            return f"{sym[:3]}{sym[3:]}CF=X"
        return f"{sym}=X"

    def get_candles(self, symbol: str, count: int = 200) -> pd.DataFrame:
        yf_sym = self._yf_symbol(symbol)
        try:
            ticker = yfinance.Ticker(yf_sym)
            df = ticker.history(period="5d", interval="1m", prepost=False)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume', 'Datetime': 'time'
            }, inplace=True)
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
            df['volume'] = df['volume'].fillna(0)
            # Drop duplicate timestamps, keep last
            df = df.drop_duplicates(subset='time', keep='last')
            return df.tail(count)[['time', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            return pd.DataFrame()

ds = DataSource()

# ============================================================================
# INDICATORS
# ============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=CONFIG['rsi_period']).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=CONFIG['atr_period']).average_true_range()
    df['trend_dist'] = abs(df['ema_20'] - df['ema_50'])
    df['candle_range'] = df['high'] - df['low']
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_body_pct'] = df['candle_body'] / df['candle_range'].replace(0, np.nan)
    # RSI change for momentum
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
    return (8 <= h < 12) or (13 <= h < 17)  # London or New York

# ============================================================================
# SIGNAL QUALITY SCORE
# ============================================================================

def score_signal(df: pd.DataFrame, idx: int, direction: str) -> float:
    """
    Score 0-100:
      - Trend strength (EMA20/50 distance): 30 pts
      - RSI confirmation (distance from 50): 25 pts
      - Candle body strength: 25 pts
      - ATR volatility level: 20 pts
    """
    row = df.iloc[idx]

    # Trend (30 pts)
    dist = row['trend_dist']
    max_dist = CONFIG['min_trend_dist'] * 5
    trend = min(30.0, (dist / max_dist) * 30.0) if dist > 0 else 0.0

    # RSI (25 pts) — how far is RSI from 50 in trade direction?
    rsi = row['rsi']
    if direction == 'BUY':
        rsi_dist = max(0, rsi - 50)
    else:
        rsi_dist = max(0, 50 - rsi)
    rsi_score = min(25.0, (rsi_dist / 35) * 25.0)

    # Candle (25 pts)
    body = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    candle_score = min(25.0, (body / 0.80) * 25.0)

    # ATR volatility (20 pts)
    atr_ratio = row['atr'] / CONFIG['min_atr']
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
    # Dynamic RR: 1.2 (low vol) → 1.5 (high vol)
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
    """
    Pullback strategy:
    BUY: EMA9 crosses above EMA21 while in uptrend (EMA20 > EMA50)
         and price is pulling back near EMA9
    SELL: EMA9 crosses below EMA21 while in downtrend (EMA20 < EMA50)
          and price is pulling back near EMA9

    Also requires:
      - RSI confirming direction and showing momentum
      - ATR above minimum
      - Trend distance above minimum
      - Candle body > threshold
      - Session active
    """
    if idx < 2:
        return None

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]

    ema9_cross_up = prev['ema_9'] <= prev['ema_21'] and row['ema_9'] > row['ema_21']
    ema9_cross_down = prev['ema_9'] >= prev['ema_21'] and row['ema_9'] < row['ema_21']

    if not (ema9_cross_up or ema9_cross_down):
        return None

    direction = 'BUY' if ema9_cross_up else 'SELL'

    # TREND: EMA20 > EMA50 for BUY, EMA20 < EMA50 for SELL
    if direction == 'BUY' and not (row['ema_20'] > row['ema_50']):
        return None
    if direction == 'SELL' and not (row['ema_20'] < row['ema_50']):
        return None

    # RSI: must be on correct side of 50
    if direction == 'BUY' and row['rsi'] <= 50:
        return None
    if direction == 'SELL' and row['rsi'] >= 50:
        return None

    # RSI momentum: rising for BUY, falling for SELL
    if direction == 'BUY' and row['rsi_change'] <= 0:
        return None
    if direction == 'SELL' and row['rsi_change'] >= 0:
        return None

    # TREND DISTANCE filter
    if row['trend_dist'] < CONFIG['min_trend_dist']:
        return None

    # VOLATILITY filter
    if row['atr'] < CONFIG['min_atr']:
        return None

    # CANDLE BODY filter
    body_pct = row['candle_body_pct'] if not np.isnan(row['candle_body_pct']) else 0
    if body_pct < CONFIG['candle_body_min']:
        return None

    # SESSION filter
    if not session_active(row['time']):
        return None

    return direction

# ============================================================================
# SCAN ONE PAIR
# ============================================================================

def scan_pair(symbol: str, cooldown_counter: int = 0) -> dict | None:
    """
    Scan a single pair for pullback signal.
    Returns signal dict or None.
    """
    df = ds.get_candles(symbol, count=150)
    if df.empty or len(df) < 60:
        return None

    df = add_indicators(df)

    # Look at last 8 candles for a valid signal
    for check_idx in range(-8, -1):
        direction = find_pullback(df, check_idx)
        if direction is None:
            continue

        # Cooldown: skip if this candle is within cooldown window
        if cooldown_counter > 0:
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

        return {
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
        }

    return None

# ============================================================================
# MULTI-PAIR SCANNER
# ============================================================================

def scan_all_pairs() -> list:
    """Scan all pairs and return list of signals."""
    signals = []
    for pair in SCAN_PAIRS:
        sig = scan_pair(pair)
        if sig:
            signals.append(sig)
    return signals

# ============================================================================
# BACKTEST
# ============================================================================

def backtest(symbol: str, candles: int | None = None) -> tuple:
    """
    Backtest with pullback strategy.
    Uses only yfinance historical data.
    """
    if candles is None:
        candles = CONFIG['backtest_candles']

    df = ds.get_candles(symbol, count=candles)
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
            if score < CONFIG['min_score']:
                position = None
            else:
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

def plot_chart(df: pd.DataFrame, trades_df=None, sig=None) -> go.Figure:
    if df is None or df.empty:
        return None

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.06,
        subplot_titles=('Price + EMAs', 'RSI', 'ATR')
    )

    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        name='Price'
    ), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_9'], name='EMA9',
        line=dict(color='#2196F3', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_21'], name='EMA21',
        line=dict(color='#FF9800', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_20'], name='EMA20',
        line=dict(color='#00BCD4', width=1.2, dash='dash'), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_50'], name='EMA50',
        line=dict(color='#E91E63', width=1.2, dash='dash'), opacity=0.7), row=1, col=1)

    # Signal marker
    if sig is not None:
        matches = df[df['time'] == sig['time']]
        if len(matches) > 0:
            c = '#26a69a' if sig['signal'] == 'BUY' else '#ef5350'
            fig.add_trace(go.Scatter(
                x=[sig['time']], y=[sig['entry']],
                mode='markers+text',
                marker=dict(size=24, color=c, symbol='arrow-up' if sig['signal'] == 'BUY' else 'arrow-down'),
                text=[f"▶ {sig['signal']} {sig['score']}"],
                textposition='top center', textfont=dict(color=c, size=11),
                name=f"SIGNAL {sig['signal']}"
            ), row=1, col=1)

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
    fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], name='RSI',
        line=dict(color='#9C27B0', width=1.5)), row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='gray', row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', line_width=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', line_width=0.5, row=2, col=1)

    # ATR
    fig.add_trace(go.Scatter(x=df['time'], y=df['atr'], name='ATR',
        fill='tozeroy', line=dict(color='#FF5722', width=1),
        fillcolor='rgba(255,87,34,0.12)'), row=3, col=1)
    fig.add_hline(y=CONFIG['min_atr'], line_dash='dash', line_color='yellow',
        line_width=0.7, row=3, col=1)

    fig.update_layout(
        template='plotly_dark', height=750, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_rangeslider_visible=False
    )
    return fig

# ============================================================================
# SIGNAL CARD HTML
# ============================================================================

def signal_card(sig: dict) -> str:
    is_buy = sig['signal'] == 'BUY'
    border = '#4caf50' if is_buy else '#f44336'
    bg = '#1b5e20' if is_buy else '#7f0000'
    arrow = '🟢 BUY' if is_buy else '🔴 SELL'
    sc = sig['score']
    sc_color = '#4caf50' if sc >= 80 else '#ff9800' if sc >= 70 else '#f44336'

    return f"""
    <div style="padding:18px; border-radius:12px; border:3px solid {border};
                background:{bg}; color:white; margin:8px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h2 style="margin:0;">{arrow} — {sig['symbol']}</h2>
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
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="MT5 Scalper Pro", page_icon="📈", layout="wide")
st.title("📈 MT5 Scalper Pro — Real-Time Trading Assistant")

# Sidebar
st.sidebar.header("⚙️ Settings")

sym = st.sidebar.text_input("Symbol (Backtest)", value=CONFIG['symbol']).upper().strip()
CONFIG['symbol'] = sym

st.sidebar.subheader("📏 EMAs")
CONFIG['ema_fast'] = st.sidebar.slider("EMA Pullback (9)", 5, 21, CONFIG['ema_fast'])
CONFIG['ema_slow'] = st.sidebar.slider("EMA Signal (21)", 10, 55, CONFIG['ema_slow'])
CONFIG['ema_trend_fast'] = st.sidebar.slider("EMA Trend (20)", 10, 30, CONFIG['ema_trend_fast'])
CONFIG['ema_trend_slow'] = st.sidebar.slider("EMA Trend (50)", 30, 100, CONFIG['ema_trend_slow'])
CONFIG['rsi_period'] = st.sidebar.slider("RSI Period", 5, 21, CONFIG['rsi_period'])
CONFIG['atr_period'] = st.sidebar.slider("ATR Period", 5, 21, CONFIG['atr_period'])

st.sidebar.subheader("🎯 Filters (Relaxed)")
CONFIG['min_trend_dist'] = st.sidebar.slider("Min Trend Distance (×10000)", 1, 20, int(CONFIG['min_trend_dist'] * 10000)) / 10000
CONFIG['min_atr'] = st.sidebar.slider("Min ATR (×10000)", 1, 15, int(CONFIG['min_atr'] * 10000)) / 10000
CONFIG['candle_body_min'] = st.sidebar.slider("Candle Body Min %", 30, 70, int(CONFIG['candle_body_min'] * 100)) / 100

st.sidebar.subheader("🛡️ Risk")
CONFIG['sl_atr_mult'] = st.sidebar.slider("SL ATR Multiplier", 0.5, 3.0, CONFIG['sl_atr_mult'], 0.1)
CONFIG['sl_min_atr'] = st.sidebar.slider("Min SL ATR Factor", 0.3, 2.0, CONFIG['sl_min_atr'], 0.1)

st.sidebar.subheader("⏱️ Cooldown & Score")
CONFIG['cooldown'] = st.sidebar.slider("Cooldown Candles", 1, 10, CONFIG['cooldown'])
CONFIG['min_score'] = st.sidebar.slider("Min Signal Score", 30, 95, CONFIG['min_score'])

st.sidebar.subheader("📈 Backtest")
CONFIG['backtest_candles'] = st.sidebar.slider("Candles", 300, 3000, CONFIG['backtest_candles'], 100)

st.sidebar.markdown("---")
st.sidebar.markdown("**🕐 Sessions:** London 08-12 UTC | NY 13-17 UTC")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Backtest", "🔴 Multi-Pair Live Scanner"])

# ============================================================
# TAB 1: DASHBOARD
# ============================================================
with tab1:
    col_a, col_b = st.columns([2, 1])
    col_a.info(f"📡 YFinance | **{sym}** | Pullback EMA {CONFIG['ema_fast']}/{CONFIG['ema_slow']}/{CONFIG['ema_trend_fast']}/{CONFIG['ema_trend_slow']}")
    if col_b.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    df_main = ds.get_candles(sym, count=200)
    if df_main.empty:
        st.error(f"No data for {sym}. Try EURUSD, GBPUSD, USDJPY, XAUUSD.")
    else:
        df_main = add_indicators(df_main)
        fig = plot_chart(df_main)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        last = df_main.iloc[-1]
        trend_dir = "🔴 DOWN" if last['ema_20'] < last['ema_50'] else "🟢 UP"
        session = "✅" if session_active(last['time']) else "❌"

        s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
        s1.metric("Price", f"{last['close']:.5f}")
        s2.metric("EMA9", f"{last['ema_9']:.5f}")
        s3.metric("EMA21", f"{last['ema_21']:.5f}")
        s4.metric("RSI", f"{last['rsi']:.1f}")
        s5.metric("ATR", f"{last['atr']:.5f}")
        s6.metric("Trend", trend_dir)
        s7.metric("Session", session)

        # Live signal for main symbol
        sig = scan_pair(sym)
        if sig:
            st.markdown(signal_card(sig), unsafe_allow_html=True)
        else:
            st.info("⚪ No qualified signal for " + sym + " right now. Check scanner tab for all pairs.")

# ============================================================
# TAB 2: BACKTEST
# ============================================================
with tab2:
    st.header("📈 Backtest — Pullback Strategy")
    if st.button("▶️ Run Backtest", use_container_width=True):
        with st.spinner(f"Loading {CONFIG['backtest_candles']} candles for {sym}..."):
            td, df_bt, metrics = backtest(sym)

        if td is None or metrics is None:
            st.error("❌ Not enough data or no trades generated. Try increasing candles or adjusting filters.")
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

            # Chart
            fig_bt = plot_chart(df_bt, td)
            if fig_bt:
                st.plotly_chart(fig_bt, use_container_width=True)

            # Trades table
            disp = td[['entry_time', 'signal', 'entry', 'sl', 'tp', 'rr', 'score', 'result', 'pnl_pips']].tail(30).copy()
            disp.columns = ['Time', 'Signal', 'Entry', 'SL', 'TP', 'R:R', 'Score', 'Result', 'P&L (pips)']
            st.dataframe(disp, use_container_width=True)

            csv = td.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "backtest_results.csv", "text/csv")

# ============================================================
# TAB 3: LIVE MULTI-PAIR SCANNER
# ============================================================
with tab3:
    st.header("🔴 Multi-Pair Live Scanner")
    st.info("⏱️ Auto-refreshes every 10 seconds | Scans: EURUSD, GBPUSD, USDJPY, XAUUSD")

    # Placeholder for scanner content — refreshed every 10s
    scanner_placeholder = st.empty()

    # Auto-refresh loop
    while True:
        with scanner_placeholder.container():
            signals = scan_all_pairs()

            if signals:
                # Sort: strongest score first
                signals.sort(key=lambda x: x['score'], reverse=True)

                # Signal cards
                for sig in signals:
                    st.markdown(signal_card(sig), unsafe_allow_html=True)

                # Table view
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
                    })

                st.markdown("### 📋 Signals Table")
                tbl = pd.DataFrame(rows)

                def color_signal(row):
                    if row['Signal'] == 'BUY':
                        return ['background-color: #1b3d2e; color: #4caf50'] * len(row)
                    return ['background-color: #3d1a1a; color: #f44336'] * len(row)

                st.dataframe(
                    tbl.style.apply(lambda r: ['background-color: #1b3d2e; color: #4caf50' if r['Signal'] == 'BUY' else 'background-color: #3d1a1a; color: #f44336' for _ in r], axis=1),
                    use_container_width=True
                )

                strong = [s for s in signals if s['score'] >= 75]
                weak = [s for s in signals if s['score'] < 75]

                ms1, ms2 = st.columns(2)
                ms1.metric("Strong Signals (≥75)", len(strong))
                ms2.metric("All Signals", len(signals))

            else:
                st.warning("⚠️ No signals across any pair right now. Possible reasons:\n"
                           "- Outside London/NY session\n"
                           "- Low ATR volatility\n"
                           "- No trending conditions\n"
                           "- Price not pulling back to EMA9")

            # Current time
            st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

        time.sleep(10)
        st.rerun()
