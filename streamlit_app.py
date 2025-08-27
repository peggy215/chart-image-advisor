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
    out = out.rename(columns=str.title)

    # 均線、均量
    for n in [5, 10, 20, 60, 120, 240]:
        out[f"MA{n}"] = out["Close"].rolling(n).mean()
    for n in [5, 20]:
        out[f"MV{n}"] = out["Volume"].rolling(n).mean()

    # KD
    low9 = out["Low"].rolling(9).min()
    high9 = out["High"].rolling(9).max()
    rsv = (out["Close"] - low9) / (high9 - low9) * 100
    out["K"] = rsv.rolling(3).mean()
    out["D"] = out["K"].rolling(3).mean()

    # MACD
    dif = ema(out["Close"], 12) - ema(out["Close"], 26)
    macd = ema(dif, 9)
    out["DIF"], out["MACD"], out["OSC"] = dif, macd, dif - macd

    # RSI
    out["RSI14"] = rsi(out["Close"], 14)

    # 布林通道
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
# 技術面分析
# ------------------------
def analyze(m: Metrics) -> Dict:
    notes: List[str] = []
    def gt(a, b): return (a is not None and b is not None and a > b)
    def lt(a, b): return (a is not None and b is not None and a < b)

    short_score, swing_score = 50, 50
    # 短線評分
    if gt(m.close, m.MA5): short_score += 8; notes.append("收盤>MA5")
    if gt(m.close, m.MA10): short_score += 8
    if gt(m.MA5, m.MA10): short_score += 6
    if gt(m.volume, m.MV5): short_score += 6
    if m.K and m.D and m.K > m.D: short_score += 8
    if m.K and m.K < 30: short_score += 4
    if m.DIF and m.MACD and m.DIF > m.MACD: short_score += 6
    if lt(m.close, m.MA20): short_score -= 6
    if lt(m.volume, m.MV20): short_score -= 4
    # 波段評分
    if gt(m.close, m.MA20): swing_score += 10
    if gt(m.close, m.MA60): swing_score += 10
    if gt(m.MA20, m.MA60): swing_score += 10
    if gt(m.close, m.MA120): swing_score += 8
    if m.DIF and m.MACD and m.DIF > m.MACD: swing_score += 6
    if m.DIF and m.DIF > 0: swing_score += 4
    if lt(m.close, m.MA60): swing_score -= 8
    if lt(m.MA20, m.MA60): swing_score -= 8
    if m.DIF and m.MACD and m.DIF < m.MACD: swing_score -= 6

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
# RSI / 布林訊號
# ------------------------
def rsi_status(rsi_value: Optional[float]) -> str:
    if rsi_value is None: return "—"
    if rsi_value >= 70: return f"{rsi_value:.2f}（超買）"
    if rsi_value <= 30: return f"{rsi_value:.2f}（超賣）"
    return f"{rsi_value:.2f}（中性）"

def bollinger_signal(m: Metrics) -> str:
    if not all([m.close, m.BB_UP, m.BB_LOW, m.BB_MID]): return "—"
    if m.close > m.BB_UP: return "收盤在上軌外（強勢突破）"
    if m.close < m.BB_LOW: return "收盤在下軌外（超跌/恐慌）"
    return "軌道內整理"


# ------------------------
# 個人倉位
# ------------------------
def position_analysis(m: Metrics, avg_cost: Optional[float], lots: Optional[float]) -> Dict[str, float]:
    if not avg_cost or avg_cost <= 0 or not lots or lots <= 0: return {}
    diff = m.close - avg_cost
    ret = diff / avg_cost * 100
    shares = lots * 1000
    return {"ret_pct": ret, "unrealized": diff * shares, "shares": shares}


# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Chart Advisor", layout="centered")
st.title("📈 Chart Advisor — 台股代碼直抓版（含持倉分析）")

symbol = st.text_input("輸入台股代碼", value="2330")
avg_cost = st.number_input("平均成本價", min_value=0.0, value=0.0, step=0.1)
lots = st.number_input("庫存張數", min_value=0.0, value=0.0, step=1.0)

if st.button("🔎 抓取 & 分析"):
    code = symbol.strip().upper()
    if code.isdigit(): code += ".TW"
    hist = yf.download(code, period="1y", interval="1d", progress=False)
    if hist.empty:
        st.error("抓不到資料，請檢查代碼")
    else:
        tech = calc_technicals(hist)
        m = latest_metrics(tech)
        result = analyze(m)

        st.metric("短線分數", result["short"]["score"])
        st.metric("波段分數", result["swing"]["score"])
        st.write(result["short"]["decision"], result["swing"]["decision"])
        st.write("RSI(14)：", rsi_status(m.RSI14))
        st.write("布林帶：", bollinger_signal(m))

        pa = position_analysis(m, avg_cost, lots)
        if pa:
            st.subheader("👤 個人持倉")
            st.write(f"平均成本 {avg_cost}, 現價 {m.close:.2f}, 報酬率 {pa['ret_pct']:.2f}%")
            st.write(f"未實現損益 {pa['unrealized']:.0f} 元，持有 {pa['shares']:.0f} 股")



