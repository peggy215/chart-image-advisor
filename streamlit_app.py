# streamlit_app.py
# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict

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
    MACD: Optional[float] = None  # signal
    DIF: Optional[float] = None   # macd main
    OSC: Optional[float] = None   # histogram


# ------------------------
# Technicals
# ------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calc_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # MAs
    for n in [5, 10, 20, 60, 120, 240]:
        out[f"MA{n}"] = out["Close"].rolling(n).mean()
    for n in [5, 20]:
        out[f"MV{n}"] = out["Volume"].rolling(n).mean()

    # Stochastic (%K, %D) 9,3 (Yahoo/TradingView常見變體)
    low9 = out["Low"].rolling(9).min()
    high9 = out["High"].rolling(9).max()
    rsv = (out["Close"] - low9) / (high9 - low9) * 100
    k = rsv.rolling(3).mean()
    d = k.rolling(3).mean()
    out["K"] = k
    out["D"] = d

    # MACD (12,26,9)
    dif = ema(out["Close"], 12) - ema(out["Close"], 26)
    macd = ema(dif, 9)
    osc = dif - macd
    out["DIF"] = dif
    out["MACD"] = macd
    out["OSC"] = osc

    return out


def latest_metrics(df: pd.DataFrame) -> Metrics:
    last = df.dropna().iloc[-1]
    m = Metrics(
        close=float(last["Close"]),
        volume=float(last["Volume"]),
        MA5=float(last["MA5"]),
        MA10=float(last["MA10"]),
        MA20=float(last["MA20"]),
        MA60=float(last["MA60"]),
        MA120=float(last["MA120"]),
        MA240=float(last["MA240"]),
        MV5=float(last["MV5"]),
        MV20=float(last["MV20"]),
        K=float(last["K"]),
        D=float(last["D"]),
        MACD=float(last["MACD"]),
        DIF=float(last["DIF"]),
        OSC=float(last["OSC"]),
    )
    return m


def analyze(m: Metrics) -> Dict:
    notes = []

    def gt(a, b):
        return (a is not None and b is not None and a > b)

    def lt(a, b):
        return (a is not None and b is not None and a < b)

    short_score = 50
    if gt(m.close, m.MA5): short_score += 8; notes.append("收盤>MA5 (+8)")
    if gt(m.close, m.MA10): short_score += 8; notes.append("收盤>MA10 (+8)")
    if gt(m.MA5, m.MA10): short_score += 6; notes.append("MA5>MA10 (+6)")
    if gt(m.volume, m.MV5): short_score += 6; notes.append("量>MV5 (+6)")
    if m.K is not None and m.D is not None and m.K > m.D: short_score += 8; notes.append("K>D (+8)")
    if m.K is not None and m.K < 30: short_score += 4; notes.append("K<30 (+4)")
    if m.DIF is not None and m.MACD is not None and m.DIF > m.MACD: short_score += 6; notes.append("DIF>MACD (+6)")
    if lt(m.close, m.MA20): short_score -= 6; notes.append("收盤<MA20 (-6)")
    if lt(m.volume, m.MV20): short_score -= 4; notes.append("量<MV20 (-4)")

    swing_score = 50
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
        if score >= 65:
            return "BUY / 加碼", "條件偏多，可分批買進或續抱。"
        elif score >= 50:
            return "HOLD / 觀望", "條件中性，等放量突破或更清晰訊號。"
        else:
            return "SELL / 減碼", "條件偏空，逢反彈減碼或等待回檔。"

    return {
        "short": {"score": short_score, "decision": verdict(short_score)},
        "swing": {"score": swing_score, "decision": verdict(swing_score)},
        "notes": notes,
        "inputs": asdict(m)
    }


# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Chart Advisor — 台股代碼直抓版", layout="centered")
st.title("📈 Chart Advisor — 台股代碼直抓版")
st.caption("輸入台股代碼（例如 2330、2317、3231），自動抓 Yahoo 數據；亦可於右側覆寫手動輸入。")

symbol = st.text_input("台股代碼 / Yahoo 代碼", value="2330", help="台股四位數代碼，例如 2330；或輸入完整 Yahoo 代碼，如 2330.TW")
period = st.selectbox("抓取區間", ["6mo", "1y", "2y"], index=0, help="用來計算均線/指標的歷史天數")

colA, colB = st.columns(2)
with colA:
    if st.button("🔎 抓取資料", use_container_width=True):
        st.session_state["fetch"] = True
with colB:
    if st.button("🧹 清空/重置", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]

metrics = Metrics()

if st.session_state.get("fetch"):
    code = symbol.strip().upper()
    # 補 .TW
    if code.isdigit():
        code = code + ".TW"
    try:
        # 多抓一些以計算 MA240
        hist = yf.download(code, period="2y" if period=="6mo" else period, interval="1d", progress=False)
        if hist is None or hist.empty:
            st.error("抓不到此代碼的資料，請確認代碼是否正確（例如 2330 或 2330.TW）。")
        else:
            hist = hist.rename(columns=str.title)  # make 'Close', 'Open', etc.
            tech = calc_technicals(hist)
            m = latest_metrics(tech)
            st.session_state["metrics"] = asdict(m)
            st.success("已自動擷取最新技術數據 ✅")
            st.dataframe(tech.tail(5))
    except Exception as e:
        st.error(f"擷取失敗：{e}")

# 手動輸入/覆寫
st.markdown("---")
st.markdown("### ⌨️ 手動輸入 / 覆寫（可留空）")
def num_input(label, init):
    return st.text_input(label, value=(("" if init is None else str(init))))

current = st.session_state.get("metrics", {})
for field in Metrics().__dataclass_fields__.keys():
    cur = current.get(field)
    val = num_input(field, cur)
    if val.strip():
        try:
            current[field] = float(val.replace(",", ""))
        except:
            pass
st.session_state["metrics"] = current

st.markdown("---")
if st.button("🚀 產生建議", type="primary", use_container_width=True):
    if not st.session_state.get("metrics"):
        st.warning("請先抓取資料或手動輸入至少部分欄位。")
    else:
        m = Metrics(**st.session_state["metrics"])
        result = analyze(m)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("短線分數", result["short"]["score"])
            st.success(f"短線：{result['short']['decision'][0]} — {result['short']['decision'][1]}")
        with c2:
            st.metric("波段分數", result["swing"]["score"])
            st.info(f"波段：{result['swing']['decision'][0]} — {result['swing']['decision'][1]}")
        with st.expander("判斷依據 / 輸入數據"):
            st.write(result["notes"])
            st.json(result["inputs"])
