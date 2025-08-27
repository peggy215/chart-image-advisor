# streamlit_app.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st


# ------------------------
# 資料結構
# ------------------------
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

def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # True Range
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    # Wilder's ATR（這裡用 SMA 也可）
    atr = tr.rolling(n).mean()
    return atr

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

    # ATR(14) 與 ATR%
    out["ATR14"] = calc_atr(out, 14)
    out["ATR14_pct"] = (out["ATR14"] / out["Close"]) * 100

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
# 技術面分析（評分/結論）
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
# RSI / 布林 訊號文字 & 風控建議（動態）
# ------------------------
def rsi_status(rsi_value: Optional[float]) -> str:
    if rsi_value is None: return "—"
    if rsi_value >= 70: return f"{rsi_value:.2f}（超買）"
    if rsi_value <= 30: return f"{rsi_value:.2f}（超賣）"
    return f"{rsi_value:.2f}（中性）"

def bollinger_signal(m: Metrics) -> str:
    if any(v is None for v in [m.close, m.BB_UP, m.BB_LOW, m.BB_MID]): return "—"
    if m.close > m.BB_UP: return "收盤在上軌外（強勢突破）"
    if m.close < m.BB_LOW: return "收盤在下軌外（超跌/恐慌）"
    # 貼軌提示（±0.5%）
    up_gap = (m.BB_UP - m.close) / m.close
    low_gap = (m.close - m.BB_LOW) / m.close
    tips = []
    if 0 <= up_gap <= 0.005: tips.append("貼近上軌（偏多）")
    if 0 <= low_gap <= 0.005: tips.append("貼近下軌（偏空）")
    return "；".join(tips) if tips else "軌道內整理"

def risk_budget_hint(atr_pct: Optional[float]) -> str:
    if atr_pct is None or np.isnan(atr_pct):
        return "風控：建議單筆風險 1%–2%（波動度無法取得）"
    if atr_pct >= 5:
        return "風控：波動大（ATR≈{:.1f}%），建議單筆風險 **0.5%–0.8%**".format(atr_pct)
    if atr_pct >= 3:
        return "風控：波動偏大（ATR≈{:.1f}%），建議單筆風險 **0.8%–1.2%**".format(atr_pct)
    if atr_pct >= 1.5:
        return "風控：波動中等（ATR≈{:.1f}%），建議單筆風險 **1.0%–1.5%**".format(atr_pct)
    return "風控：波動低（ATR≈{:.1f}%），建議單筆風險 **1.5%–2.0%**".format(atr_pct)


# ------------------------
# 個人倉位與動作建議（含張數判斷 & 代碼顯示）
# ------------------------
def position_analysis(m: Metrics, avg_cost: Optional[float], lots: Optional[float]) -> Dict[str, float]:
    if avg_cost is None or avg_cost <= 0 or lots is None or lots <= 0:
        return {}
    diff = m.close - avg_cost
    ret_pct = diff / avg_cost * 100
    shares = lots * 1000.0  # 台股 1 張 = 1000 股
    unrealized = diff * shares
    return {"ret_pct": ret_pct, "unrealized": unrealized, "shares": shares, "lots": lots}

def personalized_action(symbol: str,
                        short_score: int, swing_score: int,
                        m: Metrics, pa: Dict[str, float],
                        atr_pct: Optional[float]) -> str:
    lots = pa.get("lots", 0) if pa else 0
    # 開頭標的
    header = f"標的— "

    # 未輸入持倉時
    if not pa:
        return header + "未輸入成本/庫存：先依技術面執行。 " + risk_budget_hint(atr_pct)

    ret = pa["ret_pct"]
    msg = [header]

    # 依張數做不同分批語氣
    def sell_phrase():
        if lots >= 3:
            return "逢壓力**分批減碼 20%–30%**"
        if lots >= 2:
            return "逢壓力**先賣 1 張**，其餘續抱"
        return "逢壓力**可考慮全數賣出**或視情況續抱"

    def buy_phrase():
        if lots >= 3:
            return "**逢回測支撐不破小幅加碼（不追高）**"
        if lots == 2:
            return "**回測支撐不破可小量加碼**"
        return "**先觀察支撐，必要時再加碼**（單筆勿過重）"

    # 先看損益狀態
    if ret >= 15:
        msg.append(f"目前獲利約 {ret:.1f}%，{sell_phrase()}。")
    elif ret >= 8:
        msg.append(f"目前獲利約 {ret:.1f}%，{sell_phrase()}，其餘續抱看趨勢。")
    elif ret > 0:
        msg.append(f"小幅獲利 {ret:.1f}%，優先**守 MA5/MA10**；跌破則降風險。")
    elif ret <= -10:
        if lots >= 2:
            msg.append(f"虧損 {ret:.1f}%，建議**嚴設停損**或反彈**大幅減碼（至少 1 張）**。")
        else:
            msg.append(f"虧損 {ret:.1f}%，建議**嚴設停損**或反彈**出清**。")
    elif ret <= -5:
        if lots >= 2:
            msg.append(f"虧損 {ret:.1f}%，建議**反彈先減 1 張**，避免擴大。")
        else:
            msg.append(f"虧損 {ret:.1f}%，建議**反彈減碼或出清**，避免擴大。")
    else:
        msg.append(f"小幅虧損 {ret:.1f}%，依短線趨勢彈性調整，{buy_phrase()}。")

    # 再加上技術總結
    if short_score >= 65 and swing_score >= 65:
        msg.append("技術面：短線/波段皆偏多，可**續抱**或" + buy_phrase() + "。")
    elif short_score < 50 and swing_score < 50:
        msg.append("技術面：短線/波段皆偏弱，建議**逢反彈減碼**或換股。")
    else:
        msg.append("技術面：訊號分歧，採**分批操作**並嚴守支撐/停損。")

    # 動態風控建議（依 ATR%）
    msg.append(risk_budget_hint(atr_pct))

    return " ".join(msg)


# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Chart Advisor — 台股代碼直抓（含持倉）", layout="centered")
st.title("📈 Chart Advisor — 台股代碼直抓版（含持倉分析）")
st.caption("輸入台股代碼（如 2330），自動抓 Yahoo 數據；保留手動覆寫；可輸入平均成本與庫存張數，產生個人化建議。")

symbol = st.text_input("台股代碼 / Yahoo 代碼", value="2330", help="台股四位數代碼，例如 2330；或輸入完整 Yahoo 代碼，如 2330.TW")
period = st.selectbox("抓取區間", ["6mo", "1y", "2y"], index=0, help="用來計算均線/指標的歷史天數")

cA, cB, cC = st.columns(3)
with cA:
    if st.button("🔎 抓取資料", use_container_width=True):
        st.session_state["fetch"] = True
with cB:
    if st.button("🧹 清空/重置", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
with cC:
    st.write("")

st.markdown("---")
st.markdown("### ⌨️ 手動輸入 / 覆寫（可留空） & 個人倉位")

left, right = st.columns(2)

# 手動覆寫技術欄位（保留）
with left:
    st.markdown("**技術欄位**（留空則使用自動計算值）")
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

# 個人倉位：純文字輸入
with right:
    st.markdown("**個人倉位（可選）**")
    avg_cost_str = st.text_input("平均成本價（每股）", value="")
    lots_str = st.text_input("庫存張數（1張=1000股）", value="")
    avg_cost = None
    lots = None
    try:
        if avg_cost_str.strip():
            avg_cost = float(avg_cost_str.replace(",", ""))
    except:
        avg_cost = None
    try:
        if lots_str.strip():
            lots = float(lots_str.replace(",", ""))
    except:
        lots = None

# 抓資料
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
            st.session_state["symbol_final"] = code  # 存代碼以供建議顯示
            st.success("已自動擷取最新技術數據 ✅")
            st.dataframe(tech.tail(5))
    except Exception as e:
        st.error(f"擷取失敗：{e}")

st.markdown("---")
if st.button("🚀 產生建議", type="primary", use_container_width=True):
    if not st.session_state.get("metrics"):
        st.warning("請先抓取資料或手動輸入至少部分欄位。")
    else:
        m = Metrics(**st.session_state["metrics"])
        result = analyze(m)
        code_display = st.session_state.get("symbol_final", symbol)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("短線分數", result["short"]["score"])
            st.success(f"標的短線：{result['short']['decision'][0]} — {result['short']['decision'][1]}")
        with c2:
            st.metric("波段分數", result["swing"]["score"])
            st.info(f"標的波段：{result['swing']['decision'][0]} — {result['swing']['decision'][1]}")

        with st.expander("判斷依據 / 輸入數據"):
            st.write(result["notes"])
            st.json(result["inputs"])

        # 支撐/壓力估算 + RSI/布林 + ATR%
        tech = st.session_state.get("tech_df")
        atr_pct = None
        if tech is not None and "ATR14_pct" in tech.columns:
            try:
                atr_pct = float(tech["ATR14_pct"].dropna().iloc[-1])
            except:
                atr_pct = None

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

            st.subheader("🧭 RSI / 布林通道 / 波動度")
            colX, colY, colZ = st.columns(3)
            with colX:
                st.markdown(f"**RSI(14)**：{rsi_status(m.RSI14)}")
            with colY:
                st.markdown(f"**布林帶**：{bollinger_signal(m)}")
            with colZ:
                st.markdown(f"**ATR(14)%**：{('-' if atr_pct is None else f'{atr_pct:.2f}%')}")

        else:
            st.info("尚未抓取技術序列，僅顯示建議分數。")

        # 個人化持倉輸出（含張數邏輯 + 代碼）
        pa = position_analysis(m, avg_cost, lots)
        st.subheader("👤 個人持倉評估（依你輸入的成本/張數）")
        if pa:
            st.write(f"- 標的：**{code_display}**")
            st.write(f"- 平均成本：{avg_cost:.2f}，現價：{m.close:.2f}，**報酬率：{pa['ret_pct']:.2f}%**")
            st.write(f"- 庫存：{int(pa['shares']):,} 股（約 {pa['lots']} 張），未實現損益：約 **{pa['unrealized']:.0f} 元**")
            suggestion = personalized_action(code_display,
                                            result["short"]["score"], result["swing"]["score"],
                                            m, pa, atr_pct)
            st.success(suggestion)
        else:
            st.write("（如要得到個人化建議，請於右側輸入平均成本與庫存張數）")





