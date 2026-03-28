import warnings
warnings.filterwarnings('ignore')
import requests, json
import pandas as pd
import numpy as np
import ta
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

YF_MAP = {'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X', 'XAUUSD': 'GC=F', 'AUDUSD': 'AUDUSD=X', 'USDCAD': 'USDCAD=X', 'NZDUSD': 'NZDUSD=X', 'GBPAUD': 'GBPAUD=X'}
TF_INTERVAL = {'M1': '1m', 'M5': '5m', 'H1': '60m'}
TF_PERIOD = {'M1': '5d', 'M5': '5d', 'H1': '60d'}

def fetch_yf_v8(symbol, interval='1m', range_str='5d'):
    yf_sym = YF_MAP.get(symbol.upper(), symbol)
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}'
    params = {'interval': interval, 'range': range_str}
    try:
        resp = requests.get(url, params=params, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        data = resp.json()
        result = data.get('chart', {}).get('result', [])
        if not result: return pd.DataFrame()
        r = result[0]; ts = r['timestamp']; ohlcv = r['indicators']['quote'][0]
        df = pd.DataFrame({'time': pd.to_datetime(ts, unit='s', utc=True).tz_convert(None), 'open': ohlcv['open'], 'high': ohlcv['high'], 'low': ohlcv['low'], 'close': ohlcv['close'], 'volume': ohlcv.get('volume', [0]*len(ts))})
        df = df.dropna(subset=['close']); df['volume'] = df['volume'].fillna(0)
        return df
    except: return pd.DataFrame()

def get_candles(symbol, interval='1m', range_str='5d', count=500):
    df = fetch_yf_v8(symbol, interval, range_str)
    if df is None or df.empty: return pd.DataFrame()
    df = df.drop_duplicates(subset='time', keep='last')
    return df.tail(count).reset_index(drop=True)

STRATEGIES = {
    'Pullback EMA': {'params': {'ema_fast': 9, 'ema_slow': 21, 'ema_trend_fast': 20, 'ema_trend_slow': 50, 'rsi_period': 14, 'atr_period': 14, 'sl_atr_mult': 1.2, 'sl_min_atr': 0.5, 'min_trend_dist_pct': 0.00006, 'min_atr_pct': 0.00003, 'candle_body_min': 0.50, 'min_score': 60, 'cooldown': 3, 'swing_lookback': 5}},
    'EMA Crossover': {'params': {'ema_fast': 9, 'ema_slow': 21, 'ema_trend_fast': 20, 'ema_trend_slow': 50, 'rsi_period': 14, 'atr_period': 14, 'sl_atr_mult': 1.5, 'sl_min_atr': 0.8, 'min_trend_dist_pct': 0.00004, 'min_atr_pct': 0.00002, 'candle_body_min': 0.40, 'min_score': 50, 'cooldown': 2, 'swing_lookback': 5}},
    'RSI Range': {'params': {'ema_fast': 9, 'ema_slow': 21, 'ema_trend_fast': 20, 'ema_trend_slow': 50, 'rsi_period': 14, 'atr_period': 14, 'sl_atr_mult': 1.0, 'sl_min_atr': 0.5, 'min_trend_dist_pct': 0.00003, 'min_atr_pct': 0.00002, 'candle_body_min': 0.40, 'min_score': 40, 'cooldown': 2, 'swing_lookback': 5, 'rsi_buy_max': 35, 'rsi_sell_min': 65}},
    'ATR Breakout': {'params': {'ema_fast': 9, 'ema_slow': 21, 'ema_trend_fast': 20, 'ema_trend_slow': 50, 'rsi_period': 14, 'atr_period': 14, 'sl_atr_mult': 2.0, 'sl_min_atr': 1.0, 'min_trend_dist_pct': 0.00010, 'min_atr_pct': 0.00005, 'candle_body_min': 0.70, 'min_score': 70, 'cooldown': 1, 'swing_lookback': 3}},
    'Micro Scalp': {'params': {'ema_fast': 3, 'ema_slow': 7, 'ema_trend_fast': 20, 'ema_trend_slow': 50, 'rsi_period': 6, 'atr_period': 5, 'sl_atr_mult': 0.5, 'sl_min_atr': 0.3, 'min_trend_dist_pct': 0.00002, 'min_atr_pct': 0.00001, 'candle_body_min': 0.30, 'min_score': 30, 'cooldown': 0, 'swing_lookback': 2, 'use_tp_fixed': True, 'tp_pips': 8.0}},
    'Quick Scalp': {'params': {'ema_fast': 2, 'ema_slow': 5, 'ema_trend_fast': 8, 'ema_trend_slow': 20, 'rsi_period': 4, 'atr_period': 3, 'sl_atr_mult': 0.3, 'sl_min_atr': 0.15, 'min_trend_dist_pct': 0.00001, 'min_atr_pct': 0.000005, 'candle_body_min': 0.20, 'min_score': 20, 'cooldown': 0, 'swing_lookback': 1, 'use_tp_fixed': True, 'tp_pips': 5.0}},
}

def add_indicators(df, params):
    if df is None or df.empty: return df
    df = df.copy()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=params['ema_fast']).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=params['ema_slow']).ema_indicator()
    df['ema_trend_fast'] = ta.trend.EMAIndicator(df['close'], window=params['ema_trend_fast']).ema_indicator()
    df['ema_trend_slow'] = ta.trend.EMAIndicator(df['close'], window=params['ema_trend_slow']).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=params['rsi_period']).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=params['atr_period']).average_true_range()
    df['atr_pct'] = df['atr'] / df['close']
    df['trend_dist'] = abs(df['ema_trend_fast'] - df['ema_trend_slow'])
    df['candle_range'] = df['high'] - df['low']
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_body_pct'] = (df['candle_body'] / df['candle_range'].replace(0, np.nan)).fillna(0).clip(0, 1)
    df['rsi_change'] = df['rsi'].diff()
    return df

def _check_session(dt, params) -> bool:
    if params.get("_no_session_filter"):
        return True
    h = dt.hour
    return (8 <= h < 12) or (13 <= h <= 17)

def swing_high(df, idx, lookback):
    return df['high'].iloc[max(0, idx-lookback):idx+1].max()
def swing_low(df, idx, lookback):
    return df['low'].iloc[max(0, idx-lookback):idx+1].min()

def score_signal(df, idx, direction, params):
    row = df.iloc[idx]; dist = row['trend_dist']
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

def calc_sl(signal, entry, df, idx, params):
    lb = params['swing_lookback']; atr = df.iloc[idx]['atr']
    if signal == 'BUY':
        sl = swing_low(df, idx, lb) - (atr * params['sl_atr_mult'])
        min_sl = entry - (atr * params['sl_min_atr'])
        if entry - sl < atr * params['sl_min_atr']: sl = min_sl
    else:
        sl = swing_high(df, idx, lb) + (atr * params['sl_atr_mult'])
        min_sl = entry + (atr * params['sl_min_atr'])
        if sl - entry < atr * params['sl_min_atr']: sl = min_sl
    return sl

def calc_tp(signal, entry, sl, df, idx, params, symbol=None):
    if params.get('use_tp_fixed') and params.get('tp_pips'):
        pip_val = 0.01 if (symbol and 'JPY' in symbol) else 0.0001
        tp_pips = params['tp_pips']
        return entry + (tp_pips * pip_val) if signal == 'BUY' else entry - (tp_pips * pip_val)
    atr_ratio = df.iloc[idx]['atr'] / (params['min_atr_pct'] + 1e-10)
    rr = 1.2 if atr_ratio < 1.0 else 1.35 if atr_ratio < 2.0 else 1.5
    risk = abs(entry - sl)
    return entry + (risk * rr) if signal == 'BUY' else entry - (risk * rr)

def _sig_pullback_ema(df, idx, params):
    row, prev = df.iloc[idx], df.iloc[idx-1]
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
    row, prev = df.iloc[idx], df.iloc[idx-1]
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
    row, prev = df.iloc[idx], df.iloc[idx-1]
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
    row, prev = df.iloc[idx], df.iloc[idx-1]
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
    row, prev = df.iloc[idx], df.iloc[idx-1]
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
    return direction

def _sig_quick_scalp(df, idx, params):
    row, prev = df.iloc[idx], df.iloc[idx-1]
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
    return direction

def find_signal(df, idx, strategy, params):
    if idx < 2: return None
    row, prev = df.iloc[idx], df.iloc[idx-1]
    if strategy == 'Pullback EMA': return _sig_pullback_ema(df, idx, params)
    elif strategy == 'EMA Crossover': return _sig_ema_crossover(df, idx, params)
    elif strategy == 'RSI Range': return _sig_rsi_range(df, idx, params)
    elif strategy == 'ATR Breakout': return _sig_atr_breakout(df, idx, params)
    elif strategy == 'Micro Scalp': return _sig_micro_scalp(df, idx, params)
    elif strategy == 'Quick Scalp': return _sig_quick_scalp(df, idx, params)
    return None

def run_backtest(symbol, candles, interval, strategy, params):
    tf_int = TF_INTERVAL.get(interval, '1m')
    tf_rng = TF_PERIOD.get(interval, '5d')
    df = get_candles(symbol, tf_int, tf_rng, candles)
    if df.empty or len(df) < 100: return None, None, None
    df = add_indicators(df, params)
    df = df.reset_index(drop=True)
    trades = []; position = None; cooldown = 0
    start = max(50, params['ema_trend_slow'])
    for i in range(start, len(df) - 1):
        if cooldown > 0: cooldown -= 1
        direction = find_signal(df, i, strategy, params)
        if direction and cooldown == 0 and position is None:
            score = score_signal(df, i, direction, params)
            if score >= params['min_score']:
                entry = df.iloc[i]['close']; sl = calc_sl(direction, entry, df, i, params)
                tp = calc_tp(direction, entry, sl, df, i, params)
                position = direction; pos_entry, pos_sl, pos_tp = entry, sl, tp
                pos_score, pos_time = score, df.iloc[i]['time']
                cooldown = params['cooldown']
            continue
        if position:
            close = df.iloc[i]['close']; exit_p, result = None, None
            if position == 'BUY':
                if close <= pos_sl: exit_p, result = pos_sl, 'LOSS'
                elif close >= pos_tp: exit_p, result = pos_tp, 'WIN'
            else:
                if close >= pos_sl: exit_p, result = pos_sl, 'LOSS'
                elif close <= pos_tp: exit_p, result = pos_tp, 'WIN'
            if result:
                risk = abs(pos_entry - pos_sl); reward = abs(exit_p - pos_entry)
                rr = round(reward / risk, 2) if risk > 0 else 0
                pnl = (reward - risk) * 10000 if result == 'WIN' else -(risk * 10000)
                pip_val = 0.0001 if 'JPY' not in symbol else 0.01
                hold_candles = i - df[df['time'] == pos_time].index[0]
                trades.append({'signal': position, 'entry': pos_entry, 'exit': exit_p, 'sl': pos_sl, 'tp': pos_tp, 'rr': rr, 'pnl_pips': round(pnl, 1), 'result': result, 'score': pos_score, 'entry_time': pos_time, 'exit_time': df.iloc[i]['time'], 'hold_candles': hold_candles, 'sl_pips': round(abs(pos_entry - pos_sl) / pip_val, 1), 'tp_pips': round(abs(pos_tp - pos_entry) / pip_val, 1)})
                position = None
    if not trades: return None, df, None
    td = pd.DataFrame(trades)
    total = len(td); wins = len(td[td['result'] == 'WIN']); losses = total - wins
    wr = wins / total * 100 if total > 0 else 0
    total_pnl = td['pnl_pips'].sum(); avg_rr = td['rr'].mean()
    avg_win = td[td['result'] == 'WIN']['pnl_pips'].mean() if wins > 0 else 0
    avg_loss = abs(td[td['result'] == 'LOSS']['pnl_pips'].mean()) if losses > 0 else 0
    pf = abs(td[td['result'] == 'WIN']['pnl_pips'].sum() / td[td['result'] == 'LOSS']['pnl_pips'].sum()) if losses > 0 and td[td['result'] == 'LOSS']['pnl_pips'].sum() != 0 else float('inf')
    cum = td['pnl_pips'].cumsum(); running_max = cum.cummax(); dd = cum - running_max
    max_dd = dd.min()
    expectancy = (td['pnl_pips'] / td['rr'].replace(0, np.nan)).mean() if not td.empty else 0
    td['win_streak'] = (td['result'] == 'WIN').astype(int) * (td.groupby((td['result'] != td['result'].shift()).cumsum()).cumcount() + 1) * (td['result'] == 'WIN').astype(int)
    td['loss_streak'] = (td['result'] == 'LOSS').astype(int) * (td.groupby((td['result'] != td['result'].shift()).cumsum()).cumcount() + 1) * (td['result'] == 'LOSS').astype(int)
    max_win_streak = int(td[td['result']=='WIN']['win_streak'].max()) if wins > 0 else 0
    max_loss_streak = int(td[td['result']=='LOSS']['loss_streak'].max()) if losses > 0 else 0
    td['hour'] = pd.to_datetime(td['entry_time']).dt.hour
    hourly = td.groupby('hour')['pnl_pips'].agg(['sum', 'count']).reset_index(); hourly.columns = ['hour', 'pnl', 'count']
    td['month'] = pd.to_datetime(td['entry_time']).dt.to_period('M')
    monthly = td.groupby('month')['pnl_pips'].agg(['sum', 'count']).reset_index(); monthly.columns = ['month', 'pnl', 'count']
    def get_session(h):
        if 8 <= h < 12: return 'London'
        elif 13 <= h <= 17: return 'NewYork'
        else: return 'Other'
    td['session'] = td['hour'].apply(get_session)
    session_grp = td.groupby('session')['pnl_pips'].agg(['sum', 'count', 'mean']).reset_index()
    session_grp.columns = ['session', 'pnl', 'count', 'avg_pnl']
    td['duration_min'] = (pd.to_datetime(td['exit_time']) - pd.to_datetime(td['entry_time'])).dt.total_seconds() / 60
    dur_bins = [0, 1, 3, 5, 15, 60, 999]; dur_labels = ['<1m', '1-3m', '3-5m', '5-15m', '15-60m', '>60m']
    td['dur_bucket'] = pd.cut(td['duration_min'], bins=dur_bins, labels=dur_labels, right=True)
    dur_grp = td.groupby('dur_bucket', observed=True)['pnl_pips'].agg(['sum', 'count', 'mean']).reset_index()
    dur_grp.columns = ['duration', 'pnl', 'count', 'avg_pnl']
    buy_td = td[td['signal'] == 'BUY']; sell_td = td[td['signal'] == 'SELL']
    return td, df, {
        'total': total, 'wins': wins, 'losses': losses, 'win_rate': round(wr, 1), 'total_pnl': round(total_pnl, 1),
        'avg_rr': round(avg_rr, 2), 'profit_factor': round(pf, 2) if pf != float('inf') else 'inf',
        'max_drawdown': round(max_dd, 1), 'avg_win': round(avg_win, 1), 'avg_loss': round(avg_loss, 1),
        'best': td['pnl_pips'].max(), 'worst': td['pnl_pips'].min(),
        'expectancy': round(expectancy, 2),
        'max_win_streak': max_win_streak, 'max_loss_streak': max_loss_streak,
        'avg_hold': round(td['hold_candles'].mean(), 1),
        'buy_wr': round(len(buy_td[buy_td['result']=='WIN'])/len(buy_td)*100, 1) if len(buy_td) > 0 else 0,
        'sell_wr': round(len(sell_td[sell_td['result']=='WIN'])/len(sell_td)*100, 1) if len(sell_td) > 0 else 0,
        'buy_pnl': round(buy_td['pnl_pips'].sum(), 1), 'sell_pnl': round(sell_td['pnl_pips'].sum(), 1),
        'candles_used': len(df), 'hourly': hourly.to_dict('records'), 'monthly': monthly.to_dict('records'),
        'trades_per_hour': round(total / (len(df) / 60), 2) if len(df) > 0 else 0,
        'session_breakdown': session_grp.to_dict('records'), 'duration_breakdown': dur_grp.to_dict('records'),
        'avg_duration_min': round(td['duration_min'].mean(), 1),
        'median_duration_min': round(td['duration_min'].median(), 1),
        'best_trade': td.loc[td['pnl_pips'].idxmax()].to_dict() if len(td) > 0 else {},
        'worst_trade': td.loc[td['pnl_pips'].idxmin()].to_dict() if len(td) > 0 else {},
        'total_hours': round(len(df) / 60, 1),
        'recovery_factor': round(abs(total_pnl / max_dd), 2) if max_dd != 0 else 'inf',
        'avg_trade_pnl': round(total_pnl / total, 2) if total > 0 else 0,
        'sharp_ratio_like': round((td['pnl_pips'].mean() / td['pnl_pips'].std()), 2) if td['pnl_pips'].std() > 0 else 0,
    }

print("Fetching data for EURUSD M1 (500 candles)...")
df_test = get_candles('EURUSD', '1m', '5d', 500)
print(f"Data fetched: {len(df_test)} rows")
print(f"Time range: {df_test['time'].min()} to {df_test['time'].max()}")
print()

print("Testing all strategies with 500 candles:")
for strat in STRATEGIES:
    p = STRATEGIES[strat]['params'].copy()
    td, df_bt, m = run_backtest('EURUSD', 500, 'M1', strat, p)
    if m:
        print(f"  {strat}: {m['total']} trades, PnL={m['total_pnl']}, WR={m['win_rate']}%")
    else:
        print(f"  {strat}: No trades generated (data ok, params too strict)")
