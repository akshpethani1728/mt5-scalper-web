"""
MetaTrader 5 Scalper - Streamlit Web Interface (Enhanced)
==========================================================
EMA + RSI pullback scalping with:
  - Trend strength filter (EMA 20/50 distance)
  - Volatility filter (ATR threshold)
  - Candle confirmation (body > 60% of range)
  - Cooldown system (skip 4 candles after signal)
  - Session filter (London + New York)
  - ATR-based smart stop loss
  - Dynamic take profit (volatility-adaptive RR)
  - Signal quality score (0-100)
  - Enhanced UI with clear signal box
  - Full backtest metrics (win rate, avg RR, max DD, profit factor)
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
from datetime import datetime, timezone
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
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Core EMAs (crossover)
    "symbol": "EURUSD",
    "ema_fast": 9,
    "ema_slow": 21,
    # Trend filter EMAs
    "ema_trend_fast": 20,
    "ema_trend_slow": 50,
    # RSI
    "rsi_period": 14,
    # ATR
    "atr_period": 14,
    # Stop loss / Take profit
    "risk_reward": 1.5,
    "sl_atr_multiplier": 1.5,     # SL = swing + (ATR * multiplier)
    "sl_min_atr_factor": 0.5,     # minimum SL distance = ATR * this
    # Filters
    "min_trend_distance": 0.00010, # minimum EMA20/50 distance for trend strength
    "min_atr": 0.00003,            # minimum ATR to trade
    "candle_body_threshold": 0.60, # body must be >60% of total range
    # Cooldown
    "cooldown_candles": 4,         # ignore signals for N candles after a trade
    # Score threshold
    "min_score": 70,               # only show signals with score >= this
    # Swing lookback
    "swinglookback": 5,
    # Backtest
    "backtest_candles": 1000,
}

# ============================================================================
# DATA SOURCE: YFINANCE
# ============================================================================

class YFinanceDataSource:
    name = "yfinance"

    def _convert_symbol(self, symbol: str) -> str:
        clean = symbol.upper().strip()
        if "=" in clean:
            return clean
        return f"{clean[:3]}{clean[3:]}=X"

    def get_candles(self, symbol: str, count: int, timeframe: str = "1m") -> pd.DataFrame:
        yf_symbol = self._convert_symbol(symbol)
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "30m": "30m", "1h": "60m", "4h": "1h", "1d": "1d",
        }
        interval = interval_map.get(timeframe, "1m")

        try:
            ticker = yfinance.Ticker(yf_symbol)
            df = ticker.history(period="5d", interval=interval, prepost=False)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume', 'Datetime': 'time'
            })
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
            df['volume'] = df['volume'].fillna(0)
            return df.tail(count)[['time', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            return pd.DataFrame()

data_source = YFinanceDataSource()

# ============================================================================
# INDICATORS
# ============================================================================

def calculate_indicators(df):
    df = df.copy()
    # Crossover EMAs
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=CONFIG['ema_fast']).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=CONFIG['ema_slow']).ema_indicator()
    # Trend filter EMAs
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=CONFIG['ema_trend_fast']).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=CONFIG['ema_trend_slow']).ema_indicator()
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=CONFIG['rsi_period']).rsi()
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=CONFIG['atr_period']).average_true_range()
    # EMA distance for trend strength
    df['ema_distance'] = abs(df['ema_20'] - df['ema_50'])
    # Candle properties
    df['candle_range'] = df['high'] - df['low']
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_body_ratio'] = df['candle_body'] / df['candle_range'].replace(0, np.nan)
    return df

def get_swing_high(df, index, lookback):
    start = max(0, index - lookback)
    return df['high'].iloc[start:index+1].max()

def get_swing_low(df, index, lookback):
    start = max(0, index - lookback)
    return df['low'].iloc[start:index+1].min()

# ============================================================================
# SESSION FILTER
# ============================================================================

def is_high_activity_session(dt: pd.Timestamp) -> bool:
    """
    Allow trading only during London (08:00-12:00 UTC) and
    New York (13:00-17:00 UTC) sessions.
    """
    hour = dt.hour
    # London: 8-12 UTC
    if 8 <= hour < 12:
        return True
    # New York: 13-17 UTC
    if 13 <= hour < 17:
        return True
    return False

# ============================================================================
# CANDLE CONFIRMATION
# ============================================================================

def get_candle_body_ratio(df, index) -> float:
    """Return body-to-range ratio. 1.0 = full bullish body, 0.0 = doji."""
    row = df.iloc[index]
    body = abs(row['close'] - row['open'])
    candle_range = row['high'] - row['low']
    if candle_range == 0 or np.isnan(candle_range):
        return 0.0
    return body / candle_range

# ============================================================================
# SIGNAL QUALITY SCORE (0-100)
# ============================================================================

def calculate_signal_score(df, index, direction) -> float:
    """
    Score based on:
      - Trend strength (EMA20/50 distance)     : up to 25 pts
      - RSI confirmation (far from 50)        : up to 25 pts
      - Candle body strength                  : up to 25 pts
      - ATR volatility level                  : up to 25 pts
    """
    row = df.iloc[index]

    # 1. Trend strength (EMA 20/50 distance vs threshold)
    ema_dist = abs(row['ema_20'] - row['ema_50'])
    max_expected = CONFIG['min_trend_distance'] * 4
    trend_score = min(25.0, (ema_dist / max_expected) * 25.0) if ema_dist > 0 else 0.0

    # 2. RSI confirmation — how far from 50?
    rsi = row['rsi']
    if direction == 'BUY':
        rsi_dist = max(0, rsi - 50)
    else:
        rsi_dist = max(0, 50 - rsi)
    rsi_score = min(25.0, (rsi_dist / 40) * 25.0)  # 40 = max distance expected

    # 3. Candle body strength
    body_ratio = row['candle_body_ratio'] if not np.isnan(row['candle_body_ratio']) else 0
    candle_score = min(25.0, body_ratio * 25.0 / 0.60)  # 60% body = full 25 pts

    # 4. ATR volatility
    atr_ratio = row['atr'] / CONFIG['min_atr']
    atr_score = min(25.0, (atr_ratio - 1) * 25.0 / 3.0) if atr_ratio > 1 else 0.0
    atr_score = max(0.0, atr_score)

    return round(trend_score + rsi_score + candle_score + atr_score, 1)

# ============================================================================
# SMART STOP LOSS (ATR-based with swing buffer)
# ============================================================================

def calculate_smart_sl(signal, entry_price, df, index) -> float:
    """
    SL = swing_low/high + buffer of (ATR * sl_atr_multiplier)
    Ensures SL is not tighter than (ATR * sl_min_atr_factor)
    """
    lookback = CONFIG['swinglookback']
    atr = df.iloc[index]['atr']
    atr_mult = CONFIG['sl_atr_multiplier']

    if signal == 'BUY':
        sl = get_swing_low(df, index, lookback)
        buffer = atr * atr_mult
        sl = sl - buffer
        # Ensure minimum distance
        min_sl = entry_price - (atr * CONFIG['sl_min_atr_factor'])
        if entry_price - sl < atr * CONFIG['sl_min_atr_factor']:
            sl = min_sl
    else:
        sl = get_swing_high(df, index, lookback)
        buffer = atr * atr_mult
        sl = sl + buffer
        min_sl = entry_price + (atr * CONFIG['sl_min_atr_factor'])
        if sl - entry_price < atr * CONFIG['sl_min_atr_factor']:
            sl = min_sl
    return sl

# ============================================================================
# DYNAMIC TAKE PROFIT (volatility-adaptive RR)
# ============================================================================

def calculate_dynamic_tp(signal, entry_price, sl, df, index) -> float:
    """
    TP uses adaptive RR based on ATR:
      - High ATR  (> 2x min)  → 1:1.5 RR
      - Normal ATR(1-2x min)   → 1:1.2 RR
      - Low ATR   (< 1x min)   → 1:0.8 RR (tighter TP in slow market)
    Clamped between 1.0 and 2.0.
    """
    atr = df.iloc[index]['atr']
    atr_ratio = atr / CONFIG['min_atr']

    # Map atr_ratio to RR: 1x min → 1.0, 2x → 1.5, 3x+ → 1.8
    if atr_ratio < 1.0:
        rr = 0.8
    elif atr_ratio < 1.5:
        rr = 1.0
    elif atr_ratio < 2.0:
        rr = 1.2
    elif atr_ratio < 3.0:
        rr = 1.5
    else:
        rr = 1.8

    risk = abs(entry_price - sl)
    tp = entry_price + (risk * rr) if signal == 'BUY' else entry_price - (risk * rr)
    return tp

# ============================================================================
# CORE SIGNAL DETECTION (with all filters)
# ============================================================================

def detect_crossover(df, index):
    """
    EMA crossover + RSI filter + trend strength + candle confirmation + session.
    Returns dict with signal type or None.
    """
    if index < 1:
        return None

    row = df.iloc[index]
    prev = df.iloc[index - 1]

    # --- EMA crossover ---
    ema_fast_cross_up = prev['ema_fast'] <= prev['ema_slow'] and row['ema_fast'] > row['ema_slow']
    ema_fast_cross_down = prev['ema_fast'] >= prev['ema_slow'] and row['ema_fast'] < row['ema_slow']

    if not (ema_fast_cross_up or ema_fast_cross_down):
        return None

    direction = 'BUY' if ema_fast_cross_up else 'SELL'

    # --- RSI direction confirmation ---
    if direction == 'BUY' and row['rsi'] <= 50:
        return None
    if direction == 'SELL' and row['rsi'] >= 50:
        return None

    # --- Price must be on correct side of both ema sets ---
    if direction == 'BUY':
        if not (row['close'] > row['ema_fast'] and row['close'] > row['ema_slow']):
            return None
    else:
        if not (row['close'] < row['ema_fast'] and row['close'] < row['ema_slow']):
            return None

    # --- Trend strength filter: EMA20/EMA50 distance ---
    ema_dist = abs(row['ema_20'] - row['ema_50'])
    if ema_dist < CONFIG['min_trend_distance']:
        return None

    # --- Volatility filter: ATR minimum ---
    if row['atr'] < CONFIG['min_atr']:
        return None

    # --- Candle body confirmation ---
    body_ratio = row['candle_body_ratio'] if not np.isnan(row['candle_body_ratio']) else 0
    if body_ratio < CONFIG['candle_body_threshold']:
        return None

    # --- Session filter ---
    if not is_high_activity_session(row['time']):
        return None

    return direction

# ============================================================================
# SCAN FOR SIGNAL (with score + cooldown)
# ============================================================================

_last_signal_idx = None  # Module-level cooldown tracker

def scan_for_signal(symbol, cooldown_candles=4):
    """
    Scan for signal with all filters + quality score.
    Cooldown: if last signal was within 'cooldown_candles', skip.
    Returns signal dict with score, or None.
    """
    global _last_signal_idx

    df = data_source.get_candles(symbol, count=120)
    if df is None or df.empty or len(df) < max(CONFIG['ema_trend_slow'], CONFIG['ema_slow']) + 2:
        return None

    df = calculate_indicators(df)

    # Check last 10 candles for any crossover
    for check_idx in range(-10, -1):
        direction = detect_crossover(df, check_idx)
        if direction is None:
            continue

        # Cooldown check
        if _last_signal_idx is not None:
            candles_since_last = abs(check_idx - _last_signal_idx)
            if candles_since_last < cooldown_candles:
                continue

        # Calculate score
        score = calculate_signal_score(df, check_idx, direction)
        if score < CONFIG['min_score']:
            return None

        entry_price = df.iloc[check_idx]['close']
        sl = calculate_smart_sl(direction, entry_price, df, check_idx)
        tp = calculate_dynamic_tp(direction, entry_price, sl, df, check_idx)
        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        rr_achieved = (reward / risk) if risk > 0 else 0

        _last_signal_idx = check_idx

        return {
            'signal': direction,
            'entry': entry_price,
            'sl': sl,
            'tp': tp,
            'rr': round(rr_achieved, 2),
            'score': score,
            'time': df.iloc[check_idx]['time'],
            'candle_close': entry_price,
            'symbol': symbol,
            'ema_fast': df.iloc[check_idx]['ema_fast'],
            'ema_slow': df.iloc[check_idx]['ema_slow'],
            'ema_20': df.iloc[check_idx]['ema_20'],
            'ema_50': df.iloc[check_idx]['ema_50'],
            'rsi': df.iloc[check_idx]['rsi'],
            'atr': df.iloc[check_idx]['atr'],
            'candle_body_ratio': round(df.iloc[check_idx]['candle_body_ratio'], 3) if not np.isnan(df.iloc[check_idx]['candle_body_ratio']) else 0,
            'ema_distance': round(df.iloc[check_idx]['ema_distance'], 6),
        }

    return None

# ============================================================================
# BACKTEST (full strategy + metrics)
# ============================================================================

def run_backtest(symbol, candles=None):
    """
    Run backtest with all filters + cooldown + score threshold.
    Returns (trades_df, df_indicators) or (None, None).
    """
    if candles is None:
        candles = CONFIG['backtest_candles']

    df = data_source.get_candles(symbol, candles)
    if df is None or df.empty or len(df) < 100:
        return None, None

    df = calculate_indicators(df)
    df = df.reset_index(drop=True)

    trades = []
    position = None
    cooldown_counter = 0
    last_signal_idx = None

    start_iloc = max(CONFIG['ema_trend_slow'], CONFIG['ema_slow']) + 2

    for i in range(start_iloc, len(df) - 1):
        # Cooldown countdown
        if cooldown_counter > 0:
            cooldown_counter -= 1

        direction = detect_crossover(df, i)

        # Open position if: valid signal AND not in cooldown AND no open position
        if direction is not None and cooldown_counter == 0 and position is None:
            score = calculate_signal_score(df, i, direction)
            if score < CONFIG['min_score']:
                position = None
            else:
                entry_price = df.iloc[i]['close']
                sl = calculate_smart_sl(direction, entry_price, df, i)
                tp = calculate_dynamic_tp(direction, entry_price, sl, df, i)

                position = direction
                position_entry = entry_price
                position_sl = sl
                position_tp = tp
                position_score = score
                position_time = df.iloc[i]['time']
                cooldown_counter = CONFIG['cooldown_candles']
                last_signal_idx = i
            continue

        # Check exit
        if position is not None:
            curr_close = df.iloc[i]['close']
            exit_price = None
            result = None

            if position == 'BUY':
                if curr_close <= position_sl:
                    exit_price, result = position_sl, 'LOSS'
                elif curr_close >= position_tp:
                    exit_price, result = position_tp, 'WIN'
            elif position == 'SELL':
                if curr_close >= position_sl:
                    exit_price, result = position_sl, 'LOSS'
                elif curr_close <= position_tp:
                    exit_price, result = position_tp, 'WIN'

            if result is not None:
                risk = abs(position_entry - position_sl)
                reward = abs(exit_price - position_entry)
                rr = (reward / risk) if risk > 0 else 0

                pnl_pips = (reward - risk) * 10000 if result == 'WIN' else -(risk * 10000)

                trades.append({
                    'signal': position,
                    'entry': position_entry,
                    'exit': exit_price,
                    'sl': position_sl,
                    'tp': position_tp,
                    'rr': round(rr, 2),
                    'pnl_pips': round(pnl_pips, 1),
                    'pnl_raw': round(exit_price - position_entry, 6) if result == 'WIN' else round -(position_entry - exit_price, 6),
                    'result': result,
                    'score': position_score,
                    'entry_time': position_time,
                    'exit_time': df.iloc[i]['time'],
                })
                position = None

    if not trades:
        return None, df

    trades_df = pd.DataFrame(trades)

    # Detailed metrics
    total = len(trades_df)
    wins = len(trades_df[trades_df['result'] == 'WIN'])
    losses = total - wins
    win_rate = (wins / total * 100) if total > 0 else 0
    total_pnl_pips = trades_df['pnl_pips'].sum()
    avg_rr = trades_df['rr'].mean() if total > 0 else 0
    profit_factor = abs(trades_df[trades_df['result'] == 'WIN']['pnl_pips'].sum() / trades_df[trades_df['result'] == 'LOSS']['pnl_pips'].sum()) if losses > 0 and trades_df[trades_df['result'] == 'LOSS']['pnl_pips'].sum() != 0 else float('inf')

    # Max drawdown
    cumulative = trades_df['pnl_pips'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    trades_df['_cum_pnl'] = cumulative
    trades_df['_drawdown'] = drawdown

    metrics = {
        'total': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl_pips': total_pnl_pips,
        'avg_rr': avg_rr,
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else '∞',
        'max_drawdown': max_drawdown,
        'best_trade': trades_df['pnl_pips'].max(),
        'worst_trade': trades_df['pnl_pips'].min(),
        'avg_win': trades_df[trades_df['result'] == 'WIN']['pnl_pips'].mean() if wins > 0 else 0,
        'avg_loss': trades_df[trades_df['result'] == 'LOSS']['pnl_pips'].mean() if losses > 0 else 0,
    }

    return trades_df, df, metrics

# ============================================================================
# PLOTLY CHART
# ============================================================================

def plot_chart(df, trades_df=None, last_signal=None):
    if df is None or df.empty:
        return None

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.20, 0.20, 0.15],
        vertical_spacing=0.05,
        subplot_titles=('Price + EMAs', 'RSI', 'ATR + Volume', 'Signal Score')
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # EMAs: fast/slow crossover
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_fast'], name=f"EMA{CONFIG['ema_fast']}",
        line=dict(color='#2196F3', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_slow'], name=f"EMA{CONFIG['ema_slow']}",
        line=dict(color='#FF9800', width=1.5)), row=1, col=1)
    # Trend EMAs
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_20'], name=f"EMA{CONFIG['ema_trend_fast']}",
        line=dict(color='#00BCD4', width=1, dash='dash'), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_50'], name=f"EMA{CONFIG['ema_trend_slow']}",
        line=dict(color='#E91E63', width=1, dash='dash'), opacity=0.7), row=1, col=1)

    # Signal markers
    if last_signal is not None:
        idx_list = df[df['time'] == last_signal['time']].index
        if len(idx_list) > 0:
            color = '#26a69a' if last_signal['signal'] == 'BUY' else '#ef5350'
            fig.add_trace(go.Scatter(
                x=[last_signal['time']], y=[last_signal['entry']],
                mode='markers+text',
                marker=dict(size=22, color=color, symbol='arrow-up' if last_signal['signal'] == 'BUY' else 'arrow-down'),
                text=[f"Score: {last_signal['score']}"],
                textposition='top center',
                textfont=dict(color=color, size=10),
                name=f"SIGNAL {last_signal['signal']}"
            ), row=1, col=1)

    # Entry/Exit from backtest
    if trades_df is not None and not trades_df.empty:
        for _, t in trades_df.iterrows():
            color = '#26a69a' if t['signal'] == 'BUY' else '#ef5350'
            fig.add_trace(go.Scatter(
                x=[t['entry_time']], y=[t['entry']],
                mode='markers', marker=dict(size=10, color=color, symbol='circle'),
                showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[t['exit_time']], y=[t['exit']],
                mode='markers', marker=dict(size=8, color='white', symbol='x-thin'),
                showlegend=False
            ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], name="RSI",
        line=dict(color='#9C27B0', width=1.5)), row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=0.5, row=2, col=1)

    # ATR
    atr_color = np.where(df['atr'] >= CONFIG['min_atr'], '#FF5722', '#666666')
    fig.add_trace(go.Scatter(x=df['time'], y=df['atr'], name="ATR",
        fill='tozeroy', line=dict(color='#FF5722', width=1),
        fillcolor='rgba(255,87,34,0.1)'), row=3, col=1)
    fig.add_hline(y=CONFIG['min_atr'], line_dash="dash", line_color="yellow",
        line_width=0.8, annotation_text=f"Min ATR {CONFIG['min_atr']:.5f}", row=3, col=1)

    # Volume
    colors = ['#26a69a' if df['close'].iloc[j] >= df['open'].iloc[j] else '#ef5350' for j in range(len(df))]
    fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name='Volume',
        marker=dict(color=colors, opacity=0.4), showlegend=False), row=3, col=1)

    # Signal score on backtest (if available)
    if trades_df is not None and not trades_df.empty:
        score_colors = ['#26a69a' if r == 'WIN' else '#ef5350' for r in trades_df['result']]
        fig.add_trace(go.Bar(
            x=trades_df['entry_time'], y=trades_df['score'],
            marker_color=score_colors, name='Score', opacity=0.7
        ), row=4, col=1)
        fig.add_hline(y=CONFIG['min_score'], line_dash="dash", line_color="yellow",
            line_width=0.8, row=4, col=1)

    fig.update_layout(
        template="plotly_dark", height=800, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(row=4, col=1, title_text="Time")
    return fig

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="MT5 Scalper Pro", page_icon="📈", layout="wide")
st.title("📈 MT5 Scalper Pro — Enhanced EMA + RSI Strategy")

# Sidebar
st.sidebar.header("⚙️ Settings")
symbol = st.sidebar.text_input("Symbol", value=CONFIG['symbol']).upper().strip()
CONFIG['symbol'] = symbol

st.sidebar.subheader("🔀 Crossover EMAs")
CONFIG['ema_fast'] = st.sidebar.slider("EMA Fast", 3, 21, CONFIG['ema_fast'])
CONFIG['ema_slow'] = st.sidebar.slider("EMA Slow", 5, 55, CONFIG['ema_slow'])

st.sidebar.subheader("📏 Trend Filter EMAs")
CONFIG['ema_trend_fast'] = st.sidebar.slider("EMA 20", 10, 30, CONFIG['ema_trend_fast'])
CONFIG['ema_trend_slow'] = st.sidebar.slider("EMA 50", 30, 100, CONFIG['ema_trend_slow'])

st.sidebar.subheader("📊 RSI / ATR")
CONFIG['rsi_period'] = st.sidebar.slider("RSI Period", 5, 21, CONFIG['rsi_period'])
CONFIG['atr_period'] = st.sidebar.slider("ATR Period", 5, 21, CONFIG['atr_period'])

st.sidebar.subheader("🎯 Filters")
CONFIG['min_trend_distance'] = st.sidebar.slider("Min Trend Distance (×10000)", 1, 20, int(CONFIG['min_trend_distance'] * 10000)) / 10000
CONFIG['min_atr'] = st.sidebar.slider("Min ATR (×10000)", 1, 20, int(CONFIG['min_atr'] * 10000)) / 10000
CONFIG['candle_body_threshold'] = st.sidebar.slider("Candle Body %", 30, 80, int(CONFIG['candle_body_threshold'] * 100)) / 100

st.sidebar.subheader("🛡️ Stop Loss")
CONFIG['sl_atr_multiplier'] = st.sidebar.slider("SL ATR Multiplier", 0.5, 3.0, CONFIG['sl_atr_multiplier'], 0.1)
CONFIG['sl_min_atr_factor'] = st.sidebar.slider("Min SL Factor", 0.3, 2.0, CONFIG['sl_min_atr_factor'], 0.1)

st.sidebar.subheader("⏱️ Cooldown & Score")
CONFIG['cooldown_candles'] = st.sidebar.slider("Cooldown Candles", 1, 10, CONFIG['cooldown_candles'])
CONFIG['min_score'] = st.sidebar.slider("Min Signal Score", 30, 95, CONFIG['min_score'])

st.sidebar.subheader("📈 Backtest")
CONFIG['backtest_candles'] = st.sidebar.slider("Candles", 200, 3000, CONFIG['backtest_candles'], 50)

# ---- Session filter info ----
st.sidebar.markdown("---")
st.sidebar.markdown("**🕐 Session Filter:** London (08-12 UTC) + New York (13-17 UTC)")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Backtest", "🔴 Live Scan"])

# ============================================================
# TAB 1: DASHBOARD
# ============================================================
with tab1:
    col_header = st.columns([2, 1])
    col_header[0].info(f"📡 YFinance | **{symbol}** | EMA {CONFIG['ema_fast']}/{CONFIG['ema_slow']} | Trend EMA {CONFIG['ema_trend_fast']}/{CONFIG['ema_trend_slow']}")
    if col_header[1].button("🔄 Refresh", use_container_width=True):
        st.rerun()

    df = data_source.get_candles(symbol, count=200)
    if df.empty:
        st.error(f"No data for {symbol}. Try EURUSD, GBPUSD, USDJPY.")
    else:
        df = calculate_indicators(df)
        fig = plot_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Market stats row
        last = df.iloc[-1]
        ema_dist = abs(last['ema_20'] - last['ema_50'])
        session_active = "✅" if is_high_activity_session(last['time']) else "❌"
        body_ok = "✅" if last['candle_body_ratio'] >= CONFIG['candle_body_threshold'] else "❌"
        trend_ok = "✅" if ema_dist >= CONFIG['min_trend_distance'] else "❌"
        atr_ok = "✅" if last['atr'] >= CONFIG['min_atr'] else "❌"

        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Price", f"{last['close']:.5f}")
        m2.metric(f"EMA{CONFIG['ema_fast']}", f"{last['ema_fast']:.5f}")
        m3.metric(f"EMA{CONFIG['ema_slow']}", f"{last['ema_slow']:.5f}")
        m4.metric("RSI", f"{last['rsi']:.1f}")
        m5.metric("ATR", f"{last['atr']:.5f}")
        m6.metric("Trend Dist", f"{ema_dist:.6f}")
        m7.metric("Session", session_active)

        # Filter status
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Trend ✅" if trend_ok == "✅" else "Trend ❌", f"Dist: {ema_dist:.5f}" if trend_ok == "✅" else f"< {CONFIG['min_trend_distance']:.5f}")
        f2.metric("ATR ✅" if atr_ok == "✅" else "ATR ❌", f"{last['atr']:.5f}" if atr_ok == "✅" else f"< {CONFIG['min_atr']:.5f}")
        f3.metric("Candle ✅" if body_ok == "✅" else "Candle ❌", f"{last['candle_body_ratio']*100:.0f}%" if body_ok == "✅" else f"< {CONFIG['candle_body_threshold']*100:.0f}%")
        f4.metric("Session ✅" if session_active == "✅" else "Session ❌", "Active" if session_active == "✅" else "Closed")

        # ---- SIGNAL BOX ----
        signal = scan_for_signal(symbol)
        if signal:
            is_buy = signal['signal'] == 'BUY'
            color = "#1b5e20" if is_buy else "#7f0000"
            border_color = "#4caf50" if is_buy else "#f44336"
            arrow = "🟢 BUY" if is_buy else "🔴 SELL"

            score_gauge = signal['score']
            score_color = "#4caf50" if score_gauge >= 80 else "#ff9800" if score_gauge >= 70 else "#f44336"

            st.markdown(f"""
            <div style="
                padding:20px; border-radius:12px; border:3px solid {border_color};
                background:{color}; color:white; margin:10px 0;
            ">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2 style="margin:0; color:white;">{arrow} on {signal['symbol']}</h2>
                    <div style="text-align:center; background:{border_color}; border-radius:50%;
                                width:70px; height:70px; display:flex; align-items:center;
                                justify-content:center; flex-direction:column;">
                        <span style="font-size:18px; font-weight:bold;">{score_gauge}</span>
                        <span style="font-size:9px;">SCORE</span>
                    </div>
                </div>
                <hr style="opacity:0.3; margin:10px 0;">
                <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin-top:10px;">
                    <div><small>ENTRY</small><br><b>{signal['entry']:.5f}</b></div>
                    <div><small>STOP LOSS</small><br><b>{signal['sl']:.5f}</b></div>
                    <div><small>TAKE PROFIT</small><br><b>{signal['tp']:.5f}</b></div>
                    <div><small>R:R RATIO</small><br><b>1:{signal['rr']}</b></div>
                </div>
                <div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; margin-top:10px;">
                    <div><small>RSI</small><br><b>{signal['rsi']:.1f}</b></div>
                    <div><small>ATR</small><br><b>{signal['atr']:.5f}</b></div>
                    <div><small>EMA Dist</small><br><b>{signal['ema_distance']:.5f}</b></div>
                    <div><small>Candle Body</small><br><b>{signal['candle_body_ratio']*100:.0f}%</b></div>
                    <div><small>Time (UTC)</small><br><b>{signal['time']}</b></div>
                </div>
                <div style="margin-top:10px;">
                    <small>Signal Quality Breakdown</small><br>
                    <div style="background:rgba(255,255,255,0.1); border-radius:6px; height:8px; margin-top:4px;">
                        <div style="background:{score_color}; width:{score_gauge}%; height:8px; border-radius:6px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⚪ No qualified signal at this time. All filters must pass and score ≥ " + str(CONFIG['min_score']))

# ============================================================
# TAB 2: BACKTEST
# ============================================================
with tab2:
    st.header("📈 Backtest Results")
    if st.button("▶️ Run Backtest", use_container_width=True):
        with st.spinner("Running backtest with all filters..."):
            result = run_backtest(symbol)
            if result[0] is None:
                st.error("❌ Not enough data or no trades generated.")
            else:
                trades_df, df, metrics = result

                # Metrics grid
                mk1, mk2, mk3, mk4, mk5, mk6 = st.columns(6)
                mk1.metric("Total Trades", metrics['total'])
                mk2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                mk3.metric("Avg R:R", f"1:{metrics['avg_rr']:.2f}")
                mk4.metric("Profit Factor", metrics['profit_factor'])
                mk5.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f} pips")
                mk6.metric("Total P&L", f"{metrics['total_pnl_pips']:.1f} pips")

                mz1, mz2, mz3, mz4, mz5 = st.columns(5)
                mz1.metric("Wins / Losses", f"{metrics['wins']} / {metrics['losses']}")
                mz2.metric("Avg Win (pips)", f"{metrics['avg_win']:.1f}")
                mz3.metric("Avg Loss (pips)", f"{metrics['avg_loss']:.1f}")
                mz4.metric("Best Trade", f"{metrics['best_trade']:.1f} pips")
                mz5.metric("Worst Trade", f"{metrics['worst_trade']:.1f} pips")

                # Chart
                fig = plot_chart(df, trades_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Trades table
                display_cols = ['entry_time', 'signal', 'entry', 'sl', 'tp', 'rr', 'score', 'result', 'pnl_pips']
                disp = trades_df[display_cols].copy()
                disp.columns = ['Time', 'Signal', 'Entry', 'SL', 'TP', 'R:R', 'Score', 'Result', 'P&L (pips)']
                st.dataframe(disp.tail(30), use_container_width=True)

                csv = trades_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Full Results", csv, "enhanced_backtest_results.csv", "text/csv")

# ============================================================
# TAB 3: LIVE SCAN
# ============================================================
with tab3:
    st.header("🔴 Live Scanner")
    st.info("Checks for qualified signals. Refresh page or use the button below.")
    if st.button("🔍 Scan Now", use_container_width=True):
        st.rerun()

    signal = scan_for_signal(symbol)
    if signal:
        is_buy = signal['signal'] == 'BUY'
        border = "#4caf50" if is_buy else "#f44336"
        bg = "#1b5e20" if is_buy else "#7f0000"
        arrow = "🟢 BUY" if is_buy else "🔴 SELL"
        score_c = "#4caf50" if signal['score'] >= 80 else "#ff9800" if signal['score'] >= 70 else "#f44336"

        st.markdown(f"""
        <div style="padding:20px; border-radius:12px; border:3px solid {border};
                    background:{bg}; color:white; margin:10px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2 style="margin:0;">{arrow} — {signal['symbol']}</h2>
                <div style="background:{border}; border-radius:50%; width:60px; height:60px;
                            display:flex; flex-direction:column; align-items:center;
                            justify-content:center;">
                    <span style="font-size:18px; font-weight:bold;">{signal['score']}</span>
                    <span style="font-size:8px;">SCORE</span>
                </div>
            </div>
            <hr style="opacity:0.3;">
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:15px; margin-top:10px;">
                <div><small>ENTRY</small><br><b>{signal['entry']:.5f}</b></div>
                <div><small>STOP LOSS</small><br><b>{signal['sl']:.5f}</b></div>
                <div><small>TAKE PROFIT</small><br><b>{signal['tp']:.5f}</b></div>
                <div><small>R:R</small><br><b>1:{signal['rr']}</b></div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:15px; margin-top:10px;">
                <div><small>RSI</small><br><b>{signal['rsi']:.1f}</b></div>
                <div><small>ATR</small><br><b>{signal['atr']:.5f}</b></div>
                <div><small>Candle Body</small><br><b>{signal['candle_body_ratio']*100:.0f}%</b></div>
                <div><small>Trend Dist</small><br><b>{signal['ema_distance']:.5f}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No signal found. Make sure:\n- London (08-12 UTC) or New York (13-17 UTC) is active\n- ATR is above minimum threshold\n- EMA crossover has occurred\n- Candle body is strong (>60%)\n- Signal score ≥ " + str(CONFIG['min_score']))
