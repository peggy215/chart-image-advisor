# streamlit_app.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st


@dataclass
class Metrics:
    close: Optional[float] = None
    volume: Optional[float] = None
    MA5: Optional[float] = None
    MA10: Optional[float] = None
    MA20: Optional[float] = None
    MA60: Optional[float] = None
    MA120: Optional[float] = None
    MA240: Optional[float] = None
    MV5: Optional[float] = None
    MV20: Optional[float] = None
    K: Optional[float] = None
    D: Optional[float] = None
    MACD: Optional[float] = None
    DIF: Optional[float] = None
    OSC: Optional[float] = None
    RSI14: Optional[float] = None
    BB_UP: Optional[float] = None
    BB_MID: Optional[float] = None
    BB_LOW: Optional[float] = None


# ------------------------
# 技術指標計算
# ------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def calc_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns=str.title)  # Open/High/Low/Close/Adj Close/Volume

    # 均線、均量
    for n in [5, 10, 20, 60, 120, 240]:
        out[f"MA{n}"] = out["Close"].rolling(n).mean()
    for n in [5, 20]:
        out[f"MV{n}"] = out["Volume"].rolling(n).mean()

    # KD (9,3,3)
    low9 = out["Low"].rolling(9).min()
    high9 = out["High"].rolling(9).max()
    rsv = (out["Close"] - low9) / (high9 - low9) * 100
    out["K"] = rsv.rolling(3).mean()
    out["D"] = out["K"].rolling(3).mean()

    # MACD (12,26,9)
    dif = ema(out["Close"], 12) - ema(out["Close"], 26)
    macd = ema(dif, 9)
    out["DIF"], out["MACD"], out["OSC"] = dif, macd, dif - macd

    # RSI(14)
    out["RSI14"] = rsi(out["Close"], 14)

    # 布林通道 (20, 2σ)
    bb_mid = out["Close"].rolling(20).mean()
    bb_std = out["Close"].rolling(20).std(ddof=0)
    out["BB_MID"] = bb_mid
    out["BB_UP"] = bb_mid + 2 * bb_std
    out["BB_LOW"] = bb_mid - 2 * bb_std

    return out


def latest_metrics(df: pd.DataFrame) -> Metrics:
    last = df.dropna().iloc[-1]
    return Metrics(
        close=float(last["Close"]),
        volume=float(last["Volume"]),
        MA5=float(last["MA5"]), MA10=float(last["MA10"]),
        MA20=float(last["MA20"]), MA60=float(last["MA60"]),
        MA120=float(last["MA120"]), MA240=float(last["MA240"]),
        MV5=float(last["MV5"]), MV20=float(last["MV20"]),
        K=float(last["K"]), D=float(last["D"]),
        MACD=float(last["MACD"]), DIF=float(last["DIF"]), OSC=float(last["OSC"]),
        RSI14=float(last["RSI14"]),
        BB_UP=float(last["BB_UP"]), BB_MID=float(last["BB_MID"]), BB_LOW=float(last["BB_LOW"])
    )


# ------------------------
# 技術面分析（評分）
# ------------------------
def analyze(m: Metrics) -> Dict:
    notes: List[str] = []
    def gt(a, b): return (a is not None and b is not None and a > b)
    def lt(a, b): return (a is not None and b is not None and a < b)

    short_score, swing_score = 50, 50
    # 短線評分
    if gt(m.close, m.MA5): short_score += 8; notes.append("收盤>MA5 (+8)")
    if gt(m.close, m.MA10): short_score += 8; notes.append("收盤>MA10 (+8)")
    if gt(m.MA5, m.MA10): short_score += 6; notes.append("MA5>MA10 (+6)")
    if gt(m.volume, m.MV5): short_score += 6; notes.append("量>MV5 (+6)")
    if m.K is not None and m.D is not None and m.K > m.D: short_score += 8; notes.append("K>D (+8)")
    if m.K is not None and m.K < 30: short_score += 4; notes.append("K<30 (+4)")
    if m.DIF is not None and m.MACD is not None and m.DIF > m.MACD: short_score += 6; notes.append("DIF>MACD (+6)")
    if lt(m.close, m.MA20): short_score -= 6; notes.append("收盤<MA20 (-6)")
    if lt(m.volume, m.MV20): short_score -= 4; notes.append("量<MV20 (-4)")

    # 波段評分
    if gt(m.close, m.MA20): swing_score += 10; notes.append("收盤>MA20 (+10)")
    if gt(m.close, m.MA60): swing_score += 10; notes.append("收盤>MA60 (+10)")
    if gt(m.MA20, m.MA60): swing_score += 10; notes.append("MA20>MA60 (+10)")
    if gt(m.close, m.MA120): swing_score += 8; notes.append("收盤>MA120 (+8)")
    if m.DIF is not None and m.MACD is not None and m.DIF > m.MACD: swing_score += 6; notes.append("DIF>MACD (+6)")
    if m.DIF is not None and m.DIF > 0: swing_score += 4; notes.append("DIF>0 (+4)")
    if lt(m.close, m.MA60): swing_score -= 8; notes.append("收盤<MA60 (-8)")
    if lt(m.MA20, m.MA60): swing_score -= 8; notes.append("MA20<MA60 (-8)")
    if m.DIF is not None and m.MACD is not None and m.DIF < m.MACD: swing_score -= 6; notes.append("DIF<MACD (-6)")

    def verdict(score: int):
        if score >= 65: return "BUY / 加碼", "偏多，可分批買進或續抱"
        elif score >= 50: return "HOLD / 觀望", "中性，等突破或訊號"
        else: return "SELL / 減碼", "偏空，逢反彈減碼或停損"

    return {
        "short": {"score": short_score, "decision": verdict(short_score)},
        "swing": {"score": swing_score, "decision": verdict(swing_score)},
        "notes": notes,
        "inputs": asdict(m)
    }


# ------------------------
# 支撐 / 壓力估算
# ------------------------
def recent_levels(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    d = df.dropna().tail(lookback)
    return {
        "recent_high": float(d["High"].max()) if not d.empty else None,
        "recent_low": float(d["Low"].min()) if not d.empty else None,
    }

def pick_levels



