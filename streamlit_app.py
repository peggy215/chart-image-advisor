# streamlit_app.py
# -*- coding: utf-8 -*-
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
    MACD: Optional[float] = None    # signal
    DIF: Optional[float] = None     # macd main
    OSC: Optional[float] = None     # histogram
    RSI14: Optional[float] = None
    BB_UP: Optional[float] = None
    BB_MID: Optional[float] = None
    BB_LOW: Optional[float] = None


# ------------------------
# Technicals
# ------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    # Wilder's RSI
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
    out = out.rename(columns=str.title)  # 'Open','High','Low','Close','Adj Close','Volume'

    # MAs & MVs
    for n in [5, 10, 20, 60, 120, 240]:
        out[f"MA{n}"] = out["Close"].rolling(n).mean()
    for n in [5, 20]:
        out[f"MV{n}"] = out["Volume"].rolling(n).mean()

    # Stochastic (%K, %D) 9,3
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

    # RSI(14)
    out["RSI14"] = rsi(out["Close"], 14)

    # Bollinger Bands (20, 2σ)
    bb_mid = out["Close"].rolling(20).mean()
    bb_std = out["Close"].rolling(20).std(ddof=0)
    out["BB_MID"] = bb_mid
    out["BB_UP"]  = bb_mid + 2 * bb_std
    out["BB_LOW"] = bb_mid - 2 * bb_std

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
        RSI14=float(last["RSI14"]),
        BB_UP=float(last["BB_UP"]),
        BB_MID=float(last["BB_MID"]),
        BB_LOW=float(last["BB_LOW"]),
    )
    return m


def analyze(m: Metrics) -> Dict:
    notes = []

    def gt(a, b):
        return (a is not None and b is not None and a > b)
    def lt(a, b):
        return (a is not None and b is not None and a < b)

    # ---- 短線評分 ----
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

    # ---- 波段評分 ----
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
# Support / Resistance estimation
# ------------------------
def recent_levels(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    d = df.dropna().tail(lookback)
    return {
        "recent_high": float(d["High"].max()) if not d.empty else None,
        "recent_low": float(d["Low"].min()) if not d.empty else None,
    }

def pick_levels(price: float, candidates_below: list, candidates_above: list, k: int = 2):
    supports = [x for x in candidates_below if x is not None and x < price]
    resistances = [x for x in candidates_above if x is not None and x > price]
    supports = sorted(supports, key=lambda x: price - x)[:k]
    resistances = sorted(resistances, key=lambda x: x - price)[:k]
    return supports, resistances

def estimate_levels(tech: pd.DataFrame, m: Metrics) -> Dict[str, list]:
    lv20 = recent_levels(tech, 20)
    lv60 = recent_levels(tech, 60)

    short_below = [m.MA5, m.MA10, lv20.get("recent_low")]
    short_above = [m.MA20, lv20.get("recent_high")]

    swing_below = [m.MA20, m.MA60, lv60.get("recent_low")]
    swing_above = [m.MA60, m.MA120, lv60.get("recent_high")]

    s_sup, s_res = pick_levels(m.close, short_below, short_above, k=2)
    w_sup, w_res = pick_levels(m.close, swing_below, swing_above, k=2)

    return {
        "short_supports": s_sup,
        "short_resistances": s_res,
        "swing_supports": w_sup,
        "swing_resistances": w_res,
    }


# ------------------------
# RSI / Bollinger signals
# ------------------------
def rsi_status(rsi_value: Optional[float]) -> str:
    if rsi_value is None:
        return "—"
    if rsi_value >= 70: return f"{rsi_value:.2f}（超買）"
    if rsi_value <= 30: return f"{rsi_value:.2f}（超賣）"
    return f"{rsi_value:.2f}（中性）"

def bollinger_signal(m: Metrics) -> str:
    if any(v is None for v in [m.close, m.BB_UP, m.BB_LOW, m.BB_MID]):
        return "—"
    msg = []
    if m.close > m.BB_UP:
        msg.append("收盤在上軌外（強勢突破）")
    elif m.close < m.BB_LOW:
        msg.append("收盤在下軌外（恐慌/超跌）")
    else:
        # 判斷貼軌：距離小於 0.5% 視為貼軌
        up_gap = (m.BB_UP - m.close) / m.close
        low_gap = (m.close - m.BB_LOW) / m.close
        if 0 <= up_gap <= 0.005:
            msg.append("貼近上軌（偏多）")
        if 0 <= low_gap <= 0.005:
            msg.append("貼近下軌（偏空）")
        if not msg:
            msg.append("軌道內整理")
    return "；".join(msg)


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
    if code.isdigit():
        code = code + ".TW"
    try:
        hist = yf.download(code, period="2y" if period=="6mo" else period, interval="1d", progress=False)
        if hist is None or hist.empty:
            st.error("抓不到此代碼的資料，請確認代碼是否正確（例如 2330 或 2330.TW）。")
        else:
            tech = calc_technicals(hist)
            m = latest_metrics(tech)
            st.session_state["metrics"] = asdict(m)
            st.session_state["tech_df"] = tech
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

        # 支撐/壓力估算
        tech = st.session_state.get("tech_df")
        if tech is not None:
            st.subheader("📍 支撐 / 壓力 估算")
            lv = estimate_levels(tech, m)
            colS, colR = st.columns(2)
            with colS:
                st.markdown("**短線支撐**： " + (", ".join([f"{x:.2f}" for x in lv["short_supports"]]) if lv["short_supports"] else "-"))
                st.markdown("**波段支撐**： " + (", ".join([f"{x:.2f}" for x in lv["swing_supports"]]) if lv["swing_supports"] else "-"))
            with colR:
                st.markdown("**短線壓力**： " + (", ".join([f"{x:.2f}" for x in lv["short_resistances"]]) if lv["short_resistances"] else "-"))
                st.markdown("**波段壓力**： " + (", ".join([f"{x:.2f}" for x in lv["swing_resistances"]]) if lv["swing_resistances"] else "-"))

            st.subheader("🧭 RSI / 布林通道 訊號")
            colX, colY = st.columns(2)
            with colX:
                st.markdown(f"**RSI(14)**：{rsi_status(m.RSI14)}")
            with colY:
                st.markdown(f"**布林帶**：{bollinger_signal(m)}")
        else:
            st.info("尚未抓取技術序列，僅顯示建議分數。")


