"""
MetaTrader 5 Scalper - Streamlit Web Interface
===============================================
EMA crossover + RSI scalping strategy dashboard.
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
    "symbol": "EURUSD",
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_period": 14,
    "atr_period": 14,
    "risk_reward": 1.5,
    "min_ema_distance": 0.00005,
    "min_atr": 0.00003,
    "swinglookback": 5,
    "backtest_candles": 500,
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
        except Exception as e:
            return pd.DataFrame()

data_source = YFinanceDataSource()

# ============================================================================
# INDICATORS
# ============================================================================

def calculate_indicators(df):
    df = df.copy()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=CONFIG['ema_fast']).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=CONFIG['ema_slow']).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=CONFIG['rsi_period']).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=CONFIG['atr_period']).average_true_range()
    return df

def get_swing_high(df, index, lookback):
    start = max(0, index - lookback)
    return df['high'].iloc[start:index+1].max()

def get_swing_low(df, index, lookback):
    start = max(0, index - lookback)
    return df['low'].iloc[start:index+1].min()

def detect_crossover(df, index):
    if index < 1:
        return None
    curr = df.iloc[index]
    prev = df.iloc[index - 1]

    ema_fast_cross_up = prev['ema_fast'] <= prev['ema_slow'] and curr['ema_fast'] > curr['ema_slow']
    ema_fast_cross_down = prev['ema_fast'] >= prev['ema_slow'] and curr['ema_fast'] < curr['ema_slow']

    rsi_above_50 = curr['rsi'] > 50
    rsi_below_50 = curr['rsi'] < 50

    price_above_emas = curr['close'] > curr['ema_fast'] and curr['close'] > curr['ema_slow']
    price_below_emas = curr['close'] < curr['ema_fast'] and curr['close'] < curr['ema_slow']

    ema_distance = abs(curr['ema_fast'] - curr['ema_slow'])
    ema_spread_ok = ema_distance >= CONFIG['min_ema_distance']
    atr_ok = curr['atr'] >= CONFIG['min_atr']

    if ema_fast_cross_up and rsi_above_50 and price_above_emas and ema_spread_ok and atr_ok:
        return 'BUY'
    if ema_fast_cross_down and rsi_below_50 and price_below_emas and ema_spread_ok and atr_ok:
        return 'SELL'
    return None

def calculate_sl_tp(signal, entry_price, df, index):
    lookback = CONFIG['swinglookback']
    if signal == 'BUY':
        sl = get_swing_low(df, index, lookback)
        if entry_price - sl < CONFIG['min_atr'] * 0.5:
            sl = entry_price - CONFIG['min_atr']
        risk = entry_price - sl
        tp = entry_price + (risk * CONFIG['risk_reward'])
    else:
        sl = get_swing_high(df, index, lookback)
        if sl - entry_price < CONFIG['min_atr'] * 0.5:
            sl = entry_price + CONFIG['min_atr']
        risk = sl - entry_price
        tp = entry_price - (risk * CONFIG['risk_reward'])
    return sl, tp

# ============================================================================
# SCAN & BACKTEST
# ============================================================================

def scan_for_signal(symbol):
    df = data_source.get_candles(symbol, count=50)
    if df is None or df.empty or len(df) < CONFIG['ema_slow'] + 2:
        return None
    df = calculate_indicators(df)
    signal_idx = -2
    signal = detect_crossover(df, signal_idx)
    if signal is None:
        return None
    entry_price = df.iloc[signal_idx]['close']
    candle_time = df.iloc[signal_idx]['time']
    sl, tp = calculate_sl_tp(signal, entry_price, df, signal_idx)
    return {
        'signal': signal, 'entry': entry_price, 'sl': sl, 'tp': tp,
        'time': candle_time, 'candle_close': entry_price, 'symbol': symbol,
        'ema_fast': df.iloc[signal_idx]['ema_fast'], 'ema_slow': df.iloc[signal_idx]['ema_slow'],
        'rsi': df.iloc[signal_idx]['rsi'], 'atr': df.iloc[signal_idx]['atr'],
    }

def run_backtest(symbol, candles=None):
    if candles is None:
        candles = CONFIG['backtest_candles']
    df = data_source.get_candles(symbol, candles)
    if df is None or df.empty or len(df) < 100:
        return None, None
    df = calculate_indicators(df)
    df = df.reset_index(drop=True)

    trades = []
    position = None

    for i in range(CONFIG['ema_slow'] + 2, len(df) - 1):
        signal = detect_crossover(df, i)
        if signal is not None and position is None:
            entry_price = df.iloc[i]['close']
            sl, tp = calculate_sl_tp(signal, entry_price, df, i)
            position = signal
            position_entry = entry_price
            position_sl = sl
            position_tp = tp
            position_time = df.iloc[i]['time']
            continue
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
                pnl = abs(exit_price - position_entry) if result == 'WIN' else -abs(position_entry - exit_price)
                trades.append({
                    'signal': position, 'entry': position_entry, 'exit': exit_price,
                    'sl': position_sl, 'tp': position_tp, 'pnl': pnl,
                    'result': result, 'entry_time': position_time, 'exit_time': df.iloc[i]['time'],
                })
                position = None

    if not trades:
        return None, df
    return pd.DataFrame(trades), df

# ============================================================================
# PLOTLY CHART
# ============================================================================

def plot_chart(df, trades_df=None, last_signal=None):
    if df is None or df.empty:
        return None

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05,
        subplot_titles=('Price + EMAs + Signals', 'RSI', 'ATR')
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_fast'], name=f"EMA {CONFIG['ema_fast']}",
        line=dict(color='#2196F3', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_slow'], name=f"EMA {CONFIG['ema_slow']}",
        line=dict(color='#FF9800', width=1.5)), row=1, col=1)

    # Signal markers
    if last_signal is not None:
        idx = df[df['time'] == last_signal['time']].index
        if len(idx) > 0:
            color = '#26a69a' if last_signal['signal'] == 'BUY' else '#ef5350'
            fig.add_trace(go.Scatter(
                x=[last_signal['time']], y=[last_signal['entry']],
                mode='markers', marker=dict(size=20, color=color, symbol='arrow-up' if last_signal['signal'] == 'BUY' else 'arrow-down'),
                name=f"SIGNAL {last_signal['signal']}"
            ), row=1, col=1)

    # Entry/Exit markers from backtest
    if trades_df is not None and not trades_df.empty:
        for _, t in trades_df.iterrows():
            color = '#26a69a' if t['signal'] == 'BUY' else '#ef5350'
            fig.add_trace(go.Scatter(
                x=[t['entry_time']], y=[t['entry']],
                mode='markers', marker=dict(size=12, color=color, symbol='circle'),
                showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[t['exit_time']], y=[t['exit']],
                mode='markers', marker=dict(size=12, color='white', symbol='x'),
                showlegend=False
            ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], name="RSI",
        line=dict(color='#9C27B0', width=1.5)), row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=0.5, row=2, col=1)

    # ATR
    fig.add_trace(go.Scatter(x=df['time'], y=df['atr'], name="ATR",
        fill='tozeroy', line=dict(color='#FF5722', width=1)), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=700, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(row=3, col=1, title_text="Time")
    return fig

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="MT5 Scalper", page_icon="📈", layout="wide")

st.title("📈 MT5 Scalper — EMA + RSI Dashboard")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
symbol = st.sidebar.text_input("Symbol", value=CONFIG['symbol']).upper().strip()
CONFIG['symbol'] = symbol
CONFIG['ema_fast'] = st.sidebar.slider("EMA Fast", 3, 21, CONFIG['ema_fast'])
CONFIG['ema_slow'] = st.sidebar.slider("EMA Slow", 5, 55, CONFIG['ema_slow'])
CONFIG['rsi_period'] = st.sidebar.slider("RSI Period", 5, 21, CONFIG['rsi_period'])
CONFIG['risk_reward'] = st.sidebar.slider("Risk:Reward", 0.5, 3.0, CONFIG['risk_reward'], 0.1)
CONFIG['backtest_candles'] = st.sidebar.slider("Backtest Candles", 100, 2000, CONFIG['backtest_candles'], 50)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Backtest", "🔴 Live Scan"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"📡 Data Source: **YFinance** | Symbol: **{symbol}**")
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    df = data_source.get_candles(symbol, count=200)
    if df.empty:
        st.error(f"❌ No data found for {symbol}. Try EURUSD, GBPUSD, USDJPY.")
    else:
        df = calculate_indicators(df)
        fig = plot_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Stats row
        last = df.iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Price", f"{last['close']:.5f}")
        col2.metric(f"EMA {CONFIG['ema_fast']}", f"{last['ema_fast']:.5f}")
        col3.metric(f"EMA {CONFIG['ema_slow']}", f"{last['ema_slow']:.5f}")
        col4.metric("RSI", f"{last['rsi']:.1f}")
        col5.metric("ATR", f"{last['atr']:.5f}")

        # Latest signal
        signal = scan_for_signal(symbol)
        if signal:
            color = "green" if signal['signal'] == 'BUY' else "red"
            st.markdown(f"""
            <div style="padding:15px; border-radius:10px; border:2px solid {color}; background:{'#1a3d2e' if color=='green' else '#3d1a1a'}">
                <h3 style="color:{color};margin:0">🎯 {signal['signal']} SIGNAL on {signal['symbol']}</h3>
                <p><b>Entry:</b> {signal['entry']:.5f} | <b>SL:</b> {signal['sl']:.5f} | <b>TP:</b> {signal['tp']:.5f}</p>
                <p><b>RSI:</b> {signal['rsi']:.1f} | <b>EMA Fast:</b> {signal['ema_fast']:.5f} | <b>EMA Slow:</b> {signal['ema_slow']:.5f}</p>
                <p><b>Time:</b> {signal['time']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⚪ No active signal on current timeframe.")

with tab2:
    st.header("📈 Backtest Results")
    if st.button("▶️ Run Backtest", use_container_width=True):
        with st.spinner("Running backtest..."):
            trades_df, df = run_backtest(symbol)
        if trades_df is None:
            st.error("❌ Not enough data for backtest.")
        else:
            fig = plot_chart(df, trades_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            total = len(trades_df)
            wins = len(trades_df[trades_df['result'] == 'WIN'])
            losses = total - wins
            win_rate = (wins / total * 100) if total > 0 else 0
            total_pnl = trades_df['pnl'].sum()
            max_win = trades_df[trades_df['result'] == 'WIN']['pnl'].max() if wins > 0 else 0
            max_loss = trades_df[trades_df['result'] == 'LOSS']['pnl'].min() if losses > 0 else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Trades", total)
            c2.metric("Win Rate", f"{win_rate:.1f}%")
            c3.metric("Wins / Losses", f"{wins} / {losses}")
            c4.metric("Total P&L (pips)", f"{total_pnl * 10000:.1f}")
            c5.metric("Best Win (pips)", f"{max_win * 10000:.1f}")

            st.dataframe(trades_df.tail(20), use_container_width=True)

            csv = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "backtest_results.csv", "text/csv")

with tab3:
    st.header("🔴 Live Scanner")
    st.warning("Live mode refreshes automatically every 30 seconds.")
    placeholder = st.empty()
    while True:
        signal = scan_for_signal(symbol)
        if signal:
            color = "green" if signal['signal'] == 'BUY' else "red"
            placeholder.success(f"🎯 **{signal['signal']}** on **{signal['symbol']}** @ `{signal['entry']:.5f}` | SL: `{signal['sl']:.5f}` | TP: `{signal['tp']:.5f}` | RSI: `{signal['rsi']:.1f}`")
        else:
            placeholder.info("⚪ No signal at this time.")
        time.sleep(30)
        st.rerun()
