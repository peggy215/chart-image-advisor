# streamlit_app.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple


import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

# 🔤 K 線形態對照表（英文 → 中文 + 解釋）
CANDLE_TRANSLATE = {
    "Bull_Engulfing": ("多頭吞噬", "紅棒完全包住前一天綠棒，代表買盤強勁，常見於反轉起漲點"),
    "Bear_Engulfing": ("空頭吞噬", "綠棒完全包住前一天紅棒，代表賣壓沉重，常見於反轉下跌點"),
    "MorningStar": ("晨星", "三根 K 線組合，常見底部反轉，意味買盤介入"),
    "EveningStar": ("暮星", "三根 K 線組合，常見高檔反轉，意味賣壓出現"),
    "Hammer/HS": ("錘子線/上吊線", "下影線很長，若在低檔 → 止跌；若在高檔 → 轉弱"),
    "ShootingStar": ("射擊之星", "上影線很長，出現在高檔時常見反轉向下"),
    "Doji": ("十字星", "開盤與收盤接近，代表多空僵持，需看前後 K 棒決定方向"),
    "Bull_Marubozu": ("大陽棒", "實體很長幾乎沒影線，代表買方強勢"),
    "Bear_Marubozu": ("大陰棒", "實體很長幾乎沒影線，代表賣方強勢")
}


# =============================
# 資料結構
# =============================
@dataclass
class Metrics:
    # 價量
    close: Optional[float] = None
    volume: Optional[float] = None
    # 均線 / 均量
    MA5: Optional[float] = None
    MA10: Optional[float] = None
    MA20: Optional[float] = None
    MA60: Optional[float] = None
    MA120: Optional[float] = None
    MA240: Optional[float] = None
    MV5: Optional[float] = None
    MV20: Optional[float] = None
    # 指標
    K: Optional[float] = None
    D: Optional[float] = None
    MACD: Optional[float] = None    # signal
    DIF: Optional[float] = None     # macd main
    OSC: Optional[float] = None     # histogram
    RSI14: Optional[float] = None
    BB_UP: Optional[float] = None
    BB_MID: Optional[float] = None
    BB_LOW: Optional[float] = None
    # 當日價量（含 VWAP 近似）
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    chg_pct: Optional[float] = None   # (Close/PrevClose-1)*100
    vol_r5: Optional[float] = None
    vol_r20: Optional[float] = None
    vol_z20: Optional[float] = None
    range_pct: Optional[float] = None
    close_pos: Optional[float] = None
    gap_pct: Optional[float] = None
    vwap_approx: Optional[float] = None  # (H+L+C)/3


# =============================
# 技術指標計算
# =============================
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
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def flatten_columns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        try:
            out.columns = out.columns.get_level_values(0)
        except Exception:
            first_symbol = out.columns.levels[1][0]
            out = out.xs(key=first_symbol, axis=1, level=1, drop_level=True)
    return out

def calc_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = flatten_columns_if_needed(df)
    out = out.rename(columns=lambda s: str(s).strip().title())  # Open/High/Low/Close/Volume

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

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

    # ===== 當日價量 =====
    out["PrevClose"] = out["Close"].shift(1)
    out["ChgPct"] = (out["Close"] / out["PrevClose"] - 1.0) * 100.0
    out["VolR5"] = out["Volume"].div(out["MV5"]).replace([np.inf, -np.inf], np.nan)
    out["VolR20"] = out["Volume"].div(out["MV20"]).replace([np.inf, -np.inf], np.nan)
    vol_mean20 = out["Volume"].rolling(20).mean()
    vol_std20 = out["Volume"].rolling(20).std(ddof=0).replace(0, np.nan)
    out["VolZ20"] = (out["Volume"] - vol_mean20) / vol_std20
    out["RangePct"] = (out["High"] - out["Low"]) / out["Close"] * 100.0
    rng = (out["High"] - out["Low"])
    out["ClosePos"] = np.where(rng > 0, (out["Close"] - out["Low"]) / rng, np.nan)
    out["GapPct"] = (out["Open"] / out["PrevClose"] - 1.0) * 100.0

    # 成交量加權平均價（近似：HLC3）
    out["VWAP_Approx"] = (out["High"] + out["Low"] + out["Close"]) / 3.0

    return out

def latest_metrics(df: pd.DataFrame) -> Metrics:
    last = df.iloc[-1]
    def g(col: str) -> Optional[float]:
        try:
            v = last[col]
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    return Metrics(
        close=g("Close"), volume=g("Volume"),
        MA5=g("MA5"), MA10=g("MA10"), MA20=g("MA20"),
        MA60=g("MA60"), MA120=g("MA120"), MA240=g("MA240"),
        MV5=g("MV5"), MV20=g("MV20"),
        K=g("K"), D=g("D"),
        MACD=g("MACD"), DIF=g("DIF"), OSC=g("OSC"),
        RSI14=g("RSI14"),
        BB_UP=g("BB_UP"), BB_MID=g("BB_MID"), BB_LOW=g("BB_LOW"),
        open=g("Open"), high=g("High"), low=g("Low"),
        chg_pct=g("ChgPct"),
        vol_r5=g("VolR5"), vol_r20=g("VolR20"), vol_z20=g("VolZ20"),
        range_pct=g("RangePct"), close_pos=g("ClosePos"), gap_pct=g("GapPct"),
        vwap_approx=g("VWAP_Approx"),
    )


# =============================
# POC：當日（分時）/ 區間（日線）
# =============================
def session_poc_from_intraday(symbol: str, bins: int = 40, tz: str = "Asia/Taipei") -> Optional[float]:
    """以分時資料計算『當日 POC』。"""
    try:
        for interval in ["1m", "5m"]:
            df = yf.download(symbol, period="7d", interval=interval, progress=False)
            if df is None or df.empty:
                continue

            idx = df.index
            if getattr(idx, "tz", None) is None:
                idx = idx.tz_localize("UTC")
            df.index = idx.tz_convert(tz)

            today = pd.Timestamp.now(tz).normalize()
            dft = df[(df.index >= today) & (df.index < today + pd.Timedelta(days=1))]
            if dft.empty:
                continue

            tp = (dft["High"] + dft["Low"] + dft["Close"]) / 3.0
            vol = dft["Volume"].fillna(0)

            hist, edges = np.histogram(tp, bins=bins, weights=vol)
            if hist.sum() <= 0:
                continue
            centers = (edges[:-1] + edges[1:]) / 2.0
            return float(centers[np.argmax(hist)])
    except Exception:
        pass
    return None

def volume_profile(df: pd.DataFrame, lookback: int = 60, bins: int = 24) -> Optional[Dict[str, float]]:
    """近 N 日（日線）量價分布：POC / VAL / VAH。"""
    try:
        d = df.dropna().tail(lookback)
        if d.empty:
            return None
        typical_price = (d["High"] + d["Low"] + d["Close"]) / 3.0
        vol = d["Volume"].fillna(0)

        hist, edges = np.histogram(typical_price, bins=bins, weights=vol)
        if hist.sum() <= 0:
            return None
        centers = (edges[:-1] + edges[1:]) / 2.0

        poc_idx = int(np.argmax(hist))
        poc = float(centers[poc_idx])

        total = hist.sum()
        target = total * 0.7
        picked = hist[poc_idx]
        left, right = poc_idx, poc_idx
        while picked < target:
            left_val = hist[left - 1] if left - 1 >= 0 else -1
            right_val = hist[right + 1] if right + 1 < len(hist) else -1
            if right_val >= left_val and right + 1 < len(hist):
                right += 1; picked += hist[right]
            elif left - 1 >= 0:
                left -= 1; picked += hist[left]
            else:
                break
        val = float(centers[left]); vah = float(centers[right])
        return {"POC": poc, "VAL": val, "VAH": vah}
    except Exception:
        return None


# =============================
# 評分（含 POC）
# =============================
def analyze(m: Metrics,
            poc_today: Optional[float] = None,
            poc_60: Optional[float] = None) -> Dict:
    notes: List[str] = []
    def gt(a, b): return (a is not None and b is not None and a > b)
    def lt(a, b): return (a is not None and b is not None and a < b)

    short_score, swing_score = 50, 50
    # 短線
    if gt(m.close, m.MA5): short_score += 8; notes.append("收盤>MA5 (+8)")
    if gt(m.close, m.MA10): short_score += 8; notes.append("收盤>MA10 (+8)")
    if gt(m.MA5, m.MA10): short_score += 6; notes.append("MA5>MA10 (+6)")
    if gt(m.volume, m.MV5): short_score += 6; notes.append("量>MV5 (+6)")
    if m.K is not None and m.D is not None and m.K > m.D: short_score += 8; notes.append("K>D (+8)")
    if m.K is not None and m.K < 30: short_score += 4; notes.append("K<30 (+4)")
    if m.DIF is not None and m.MACD is not None and m.DIF > m.MACD: short_score += 6; notes.append("DIF>MACD (+6)")
    if lt(m.close, m.MA20): short_score -= 6; notes.append("收盤<MA20 (-6)")
    if lt(m.volume, m.MV20): short_score -= 4; notes.append("量<MV20 (-4)")
    if poc_today is not None:
        if m.close is not None and m.close > poc_today:
            short_score += 6; notes.append("收盤>當日POC (+6)")
        elif m.close is not None and m.close < poc_today:
            short_score -= 6; notes.append("收盤<當日POC (-6)")

    # 波段
    if gt(m.close, m.MA20): swing_score += 10; notes.append("收盤>MA20 (+10)")
    if gt(m.close, m.MA60): swing_score += 10; notes.append("收盤>MA60 (+10)")
    if gt(m.MA20, m.MA60): swing_score += 10; notes.append("MA20>MA60 (+10)")
    if gt(m.close, m.MA120): swing_score += 8; notes.append("收盤>MA120 (+8)")
    if m.DIF is not None and m.MACD is not None and m.DIF > m.MACD: swing_score += 6; notes.append("DIF>MACD (+6)")
    if m.DIF is not None and m.DIF > 0: swing_score += 4; notes.append("DIF>0 (+4)")
    if lt(m.close, m.MA60): swing_score -= 8; notes.append("收盤<MA60 (-8)")
    if lt(m.MA20, m.MA60): swing_score -= 8; notes.append("MA20<MA60 (-8)")
    if m.DIF is not None and m.MACD is not None and m.DIF < m.MACD: swing_score -= 6; notes.append("DIF<MACD (-6)")
    if poc_60 is not None:
        if m.close is not None and m.close > poc_60:
            swing_score += 6; notes.append("收盤>60日POC (+6)")
        elif m.close is not None and m.close < poc_60:
            swing_score -= 6; notes.append("收盤<60日POC (-6)")

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


# =============================
# 支撐 / 壓力（含 POC）
# =============================
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

def estimate_levels(tech: pd.DataFrame, m: Metrics,
                    poc_today: Optional[float], poc_60: Optional[float]) -> Dict[str, list]:
    lv20 = recent_levels(tech, 20)
    lv60 = recent_levels(tech, 60)

    short_below = [m.MA5, m.MA10, poc_today, lv20.get("recent_low")]
    short_above = [m.MA20, poc_today, lv20.get("recent_high")]

    swing_below = [m.MA20, m.MA60, poc_60, lv60.get("recent_low")]
    swing_above = [m.MA60, m.MA120, poc_60, lv60.get("recent_high")]

    s_sup, s_res = pick_levels(m.close, short_below, short_above, k=2)
    w_sup, w_res = pick_levels(m.close, swing_below, swing_above, k=2)

    return {
        "short_supports": s_sup,
        "short_resistances": s_res,
        "swing_supports": w_sup,
        "swing_resistances": w_res,
    }


# =============================
# 跳空解讀
# =============================
def interpret_gap(gap_pct: Optional[float], vol_r5: Optional[float]) -> str:
    if gap_pct is None:
        return "無法計算跳空。"
    s = "跳空上漲（Gap Up）" if gap_pct > 0 else ("跳空下跌（Gap Down）" if gap_pct < 0 else "無跳空")
    mag = abs(gap_pct)
    strength = "輕微"
    if mag >= 2: strength = "強烈"
    elif mag >= 1: strength = "偏強"
    elif mag >= 0.3: strength = "輕微"
    else: strength = "極小"

    extra = ""
    if vol_r5 is not None:
        if vol_r5 >= 1.3:
            extra = "，且有放量，延續機率提高。"
        elif vol_r5 <= 0.8:
            extra = "，但量縮，隔日回補缺口機率較高。"

    return f"{s}：{gap_pct:.2f}%（{strength}）{extra}"

# =============================
# 均線站穩檢查
# =============================
def check_stand_ma(m: Metrics, tech: pd.DataFrame, ma_key: str = "MA20", days: int = 2) -> str:
    """
    檢查是否『站穩』MA20 / MA60
    條件：
      1. 最近收盤價連續 days 天都 >= 該均線
      2. 成交量 >= MV20
      3. 該均線斜率 >= 0 （均線翻揚或走平）
    """
    if tech is None or tech.empty:
        return "❓ 無法判斷"

    if getattr(m, ma_key) is None or m.close is None:
        return "❓ 無法判斷"

    # 最近 N 天收盤 >= 均線
    cond_close = (tech["Close"].tail(days) >= tech[ma_key].tail(days)).all()

    # 量能條件
    cond_vol = (m.volume is not None and m.MV20 is not None and m.volume >= m.MV20)

    # 均線斜率：最近 3 天
    ma_series = tech[ma_key].dropna().tail(3)
    cond_slope = False
    if len(ma_series) >= 2:
        cond_slope = (ma_series.iloc[-1] - ma_series.iloc[0]) >= 0

    # 判斷
    if cond_close and cond_vol and cond_slope:
        return f"✅ 已站穩 {ma_key}（連續 {days} 日收盤在上方，放量，均線翻揚）"
    elif cond_close and (cond_vol or cond_slope):
        return f"⚠️ 剛突破 {ma_key}，需觀察量能與均線是否翻揚"
    else:
        return f"❌ 尚未站穩 {ma_key}（假突破風險高）"




# =============================
# 🎯 目標價模組
# =============================
def dedup_levels(levels, tol=0.3):
    xs = sorted([float(x) for x in levels if x is not None and np.isfinite(x)])
    out = []
    for x in xs:
        if not out or abs(x - out[-1]) > tol:
            out.append(x)
    return out

def box_breakout_targets(df: pd.DataFrame, lookback: int = 60, base: int = 20) -> Dict:
    d = df.dropna().tail(lookback)
    if d.empty:
        return {}
    prior = d.iloc[:-1]
    last = d.iloc[-1]

    box_high = float(prior["High"].tail(base).max())
    box_low  = float(prior["Low"].tail(base).min())
    box_h    = max(box_high - box_low, 0.0)
    breakout = (float(last["Close"]) > box_high * 1.003)

    t1 = box_high + 0.618 * box_h if box_h > 0 else None
    t2 = box_high + 1.000 * box_h if box_h > 0 else None
    return {
        "breakout_line": box_high,
        "base_low": box_low,
        "box_range": box_h,
        "t1_box": t1,
        "t2_box": t2,
        "is_breakout": breakout
    }

def atr_targets(df: pd.DataFrame, ref_price: Optional[float] = None, mults=(1, 2)) -> Dict:
    if "ATR14" not in df.columns:
        df["ATR14"] = calc_atr(df, 14)
    atr_series = df["ATR14"].dropna()
    if atr_series.empty:
        return {}
    atr = float(atr_series.iloc[-1])
    p = float(ref_price) if ref_price is not None else float(df["Close"].iloc[-1])

    out = {"atr": atr, "ref": p}
    for i, m in enumerate(mults, start=1):
        out[f"t{i}_atr"] = p + m * atr
    return out

def fib_extension_targets(df: pd.DataFrame, lookback: int = 180) -> Dict:
    d = df.dropna().tail(lookback)
    if d.empty:
        return {}
    low_pos  = int(np.argmin(d["Low"].values))
    high_pos = int(np.argmax(d["High"].values[low_pos:])) + low_pos

    low  = float(d["Low"].iloc[low_pos])
    high = float(d["High"].iloc[high_pos])
    rng  = max(high - low, 0.0)
    if rng == 0:
        return {}

    t1 = low + 1.272 * rng
    t2 = low + 1.618 * rng
    return {"swing_low": low, "swing_high": high, "t1_fib": t1, "t2_fib": t2}

def build_targets(m: Metrics,
                  tech: pd.DataFrame,
                  poc_today: Optional[float],
                  vp60: Optional[Dict[str, float]]) -> Dict:
    """
    回傳三層目標：
      - short_targets：近距離（短線）目標
      - swing_targets：中距離（波段）目標
      - mid_targets  ：較長距離（中長）目標，擴大時間窗（含 52週/2年高點、120/250日價值區、整數關卡、可選強制價位）
    """
    def dedup(xs, tol):
        xs = sorted([float(x) for x in xs if x is not None and np.isfinite(x)])
        out = []
        for x in xs:
            if not out or abs(x - out[-1]) > tol:
                out.append(x)
        return out

    def next_rounds(px: float, step: float = 5.0, n: int = 3):
        base = np.ceil(px / step) * step
        return [base + i * step for i in range(n)]

    close = m.close if m.close is not None else float(tech["Close"].iloc[-1])

    # ----- 量度升幅、ATR、斐波延伸（沿用）
    box = box_breakout_targets(tech)
    atr = atr_targets(tech, ref_price=box.get("breakout_line") or close)
    fib = fib_extension_targets(tech)

    # ----- 60 日價值區（沿用呼入參數 vp60）
    vp60 = vp60 or {}
    vp60_poc = vp60.get("POC"); vp60_val = vp60.get("VAL"); vp60_vah = vp60.get("VAH")

    # ----- 更長時間窗的價值區（新增 120 / 250 日）
    vp120 = volume_profile(tech, lookback=120, bins=30) or {}
    vp250 = volume_profile(tech, lookback=250, bins=36) or {}
    vp120_poc = vp120.get("POC"); vp120_vah = vp120.get("VAH")
    vp250_poc = vp250.get("POC"); vp250_vah = vp250.get("VAH")

    # ----- 高點：60 / 120 / 252（52週）/ 500（日約2年）
    recent60_high  = float(tech["High"].tail(60).max())
    recent120_high = float(tech["High"].tail(120).max())
    recent252_high = float(tech["High"].tail(252).max()) if len(tech) >= 60 else recent120_high
    recent500_high = float(tech["High"].tail(500).max()) if len(tech) >= 120 else recent252_high

    # ----- 心理整數關卡：抓 3 階
    round_candidates = next_rounds(close, step=5, n=3)

    # ----- 可選：強制納入特定長期關鍵價（例如 50），限距離 +30% 內
    force_levels = []
    for hard in [50.0]:
        if hard > close and (hard / close - 1.0) * 100.0 <= 30.0:
            force_levels.append(hard)

    # ---- 短線目標（近）
    short_candidates = []
    for v in [m.MA20, m.MA60, poc_today, vp60_poc, vp60_val, vp60_vah, box.get("t1_box")]:
        if v is not None and v > close:
            short_candidates.append(float(v))
    short_targets = dedup(short_candidates, tol=0.3)[:2]

    # ---- 波段目標（中）
    swing_candidates = []
    for v in [box.get("t2_box"),
              fib.get("t1_fib"), fib.get("t2_fib"),
              atr.get("t1_atr"), atr.get("t2_atr"),
              vp60_vah, vp60_poc]:
        if v is not None and v > close:
            swing_candidates.append(float(v))
    swing_targets = dedup(swing_candidates, tol=0.5)[:3]

    # ---- 中長距離目標（遠）：擴大時間窗 + 更大價值區 + 整數關卡 + 強制價位
    mid_candidates = []
    for v in [recent60_high, recent120_high, recent252_high, recent500_high,
              vp120_vah, vp120_poc, vp250_vah, vp250_poc] + round_candidates + force_levels:
        if v is not None and v > close:
            mid_candidates.append(float(v))

    # 避免與 swing 過度重疊；顯示最多 5 個，較不會被近端目標擠掉
    mid_targets = dedup(mid_candidates + swing_targets, tol=0.6)
    mid_targets = [x for x in mid_targets if all(abs(x - s) > 0.6 for s in swing_targets)][:5]

    # ---- 說明
    explain = []
    if box:
        if box.get("is_breakout"):
            explain.append("量度升幅：已突破箱頂，T1=箱頂+0.618×箱高、T2=箱頂+1.0×箱高")
        else:
            explain.append(f"量度升幅：箱頂在 {box.get('breakout_line', float('nan')):.2f}，待突破再看 T1/T2")
    if atr:
        t1a, t2a = atr.get("t1_atr"), atr.get("t2_atr")
        explain.append(f"ATR(14)≈{atr['atr']:.2f}，位移目標：{('-' if t1a is None else f'{t1a:.2f}')} / {('-' if t2a is None else f'{t2a:.2f}')}")
    if fib:
        t1f, t2f = fib.get("t1_fib"), fib.get("t2_fib")
        explain.append(f"斐波延伸：1.272→{('-' if t1f is None else f'{t1f:.2f}')}、1.618→{('-' if t2f is None else f'{t2f:.2f}')}")
    if vp60:
        explain.append(f"60日價值區：POC≈{(vp60_poc or float('nan')):.2f}、VAH≈{(vp60_vah or float('nan')):.2f}")
    if vp120:
        explain.append(f"120日價值區：POC≈{(vp120_poc or float('nan')):.2f}、VAH≈{(vp120_vah or float('nan')):.2f}")
    if vp250:
        explain.append(f"250日價值區：POC≈{(vp250_poc or float('nan')):.2f}、VAH≈{(vp250_vah or float('nan')):.2f}")
    explain.append(f"高點參考：60/120/52週/2年 → {recent60_high:.2f}/{recent120_high:.2f}/{recent252_high:.2f}/{recent500_high:.2f}")
    explain.append("心理整數關卡（上方三階）：{}".format(", ".join([f"{r:.2f}" for r in round_candidates])))
    if force_levels:
        explain.append("強制關鍵價（距離 +30% 內）：{}".format(", ".join([f"{x:.2f}" for x in force_levels])))

    return {
        "short_targets": short_targets,
        "swing_targets": swing_targets,
        "mid_targets": mid_targets,
        "components": {
            "box": box, "atr": atr, "fib": fib,
            "vp60": vp60, "vp120": vp120, "vp250": vp250,
            "recent60_high": recent60_high, "recent120_high": recent120_high,
            "recent252_high": recent252_high, "recent500_high": recent500_high,
            "rounds": round_candidates, "force_levels": force_levels
        },
        "explain": explain
    }

def build_targets_weekly(m: Metrics,
                         tech: pd.DataFrame,
                         poc_today: Optional[float]) -> Dict:
    """
    以『週線』資料計算中長距離目標：
      - 週線 volume profile（60/120 週）→ 更容易抓到大型壓力（如 50）
      - 週線高點（52 週/2 年）
      - 週線箱體/ATR/斐波延伸
    """
    if tech is None or tech.empty:
        return {"short_targets": [], "swing_targets": [], "mid_targets": [], "explain": [], "components": {}}

    # 轉週線
    tw = tech.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    # 週線技術
    tw["ATR14"] = calc_atr(tw, 14)
    # 週線 volume profile（以「週K」計算 60/120 週價值區）
    def vp_week(dfw, lookback, bins):
        try:
            d = dfw.dropna().tail(lookback)
            if d.empty: return None
            tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
            vol = d["Volume"].fillna(0)
            hist, edges = np.histogram(tp, bins=bins, weights=vol)
            if hist.sum() <= 0: return None
            centers = (edges[:-1] + edges[1:]) / 2.0
            poc_idx = int(np.argmax(hist)); poc = float(centers[poc_idx])
            total = hist.sum(); target = total * 0.7
            picked = hist[poc_idx]; left = right = poc_idx
            while picked < target:
                lv = hist[left-1] if left-1 >= 0 else -1
                rv = hist[right+1] if right+1 < len(hist) else -1
                if rv >= lv and right+1 < len(hist):
                    right += 1; picked += hist[right]
                elif left-1 >= 0:
                    left -= 1; picked += hist[left]
                else:
                    break
            val = float(centers[left]); vah = float(centers[right])
            return {"POC": poc, "VAL": val, "VAH": vah}
        except Exception:
            return None

    vp60w  = vp_week(tw, 60, 24)  or {}
    vp120w = vp_week(tw, 120, 30) or {}

    # 高點（週線）
    r52w_high  = float(tw["High"].tail(52).max()) if len(tw) >= 52 else float(tw["High"].max())
    r104w_high = float(tw["High"].tail(104).max()) if len(tw) >= 104 else r52w_high

    # 箱體、ATR（用週線）
    def box_week(dfw, lookback=60, base=12):
        d = dfw.dropna().tail(lookback)
        if d.empty: return {}
        prior = d.iloc[:-1]; last = d.iloc[-1]
        box_high = float(prior["High"].tail(base).max())
        box_low  = float(prior["Low"].tail(base).min())
        box_h    = max(box_high - box_low, 0.0)
        breakout = float(last["Close"]) > box_high * 1.003
        t1 = box_high + 0.618 * box_h if box_h > 0 else None
        t2 = box_high + 1.000 * box_h if box_h > 0 else None
        return {"breakout_line": box_high, "box_range": box_h, "t1_box": t1, "t2_box": t2, "is_breakout": breakout}

    def atr_week(dfw, ref=None, mults=(1,2)):
        atr_series = dfw["ATR14"].dropna()
        if atr_series.empty: return {}
        atr = float(atr_series.iloc[-1])
        p = float(ref) if ref is not None else float(dfw["Close"].iloc[-1])
        out = {"atr": atr, "ref": p}
        for i, mlt in enumerate(mults, start=1):
            out[f"t{i}_atr"] = p + mlt * atr
        return out

    def fib_week(dfw, lookback=180):
        d = dfw.dropna().tail(lookback)
        if d.empty: return {}
        low_pos  = int(np.argmin(d["Low"].values))
        high_pos = int(np.argmax(d["High"].values[low_pos:])) + low_pos
        low  = float(d["Low"].iloc[low_pos]); high = float(d["High"].iloc[high_pos])
        rng = max(high - low, 0.0)
        if rng == 0: return {}
        return {"t1_fib": low + 1.272 * rng, "t2_fib": low + 1.618 * rng}

    bw  = box_week(tw)
    aw  = atr_week(tw, ref=bw.get("breakout_line"))
    fw  = fib_week(tw)

    close = float(tw["Close"].iloc[-1])

    # 週線目標：更長期，直接視為「mid」
    candidates = []
    for v in [
        # 週線價值區
        vp60w.get("VAH"), vp60w.get("POC"),
        vp120w.get("VAH"), vp120w.get("POC"),
        # 週線箱體/ATR/斐波
        bw.get("t2_box"), aw.get("t2_atr"), fw.get("t2_fib"),
        # 週線高點
        r52w_high, r104w_high,
        # 心理整數（以 5 為級距，抓 3 階）
        *[np.ceil(close/5.0)*5.0 + i*5.0 for i in range(3)]
    ]:
        if v is not None and v > close:
            candidates.append(float(v))

    # 去重
    def dedup(xs, tol=0.6):
        xs = sorted([x for x in xs if np.isfinite(x)])
        out = []
        for x in xs:
            if not out or abs(x - out[-1]) > tol:
                out.append(x)
        return out

    mid_targets_w = dedup(candidates, tol=0.8)[:5]

    explain = []
    if vp60w:  explain.append(f"週線 60 週價值區：POC≈{vp60w.get('POC', float('nan')):.2f}、VAH≈{vp60w.get('VAH', float('nan')):.2f}")
    if vp120w: explain.append(f"週線 120 週價值區：POC≈{vp120w.get('POC', float('nan')):.2f}、VAH≈{vp120w.get('VAH', float('nan')):.2f}")
    if bw:
        if bw.get("is_breakout"):
            explain.append("週線量度升幅：已突箱頂，T2=箱頂+1.0×箱高")
        else:
            explain.append(f"週線量度升幅：箱頂≈{bw.get('breakout_line', float('nan')):.2f}，待突破")
    if aw: explain.append(f"週線 ATR≈{aw.get('atr', float('nan')):.2f}，位移 T2≈{aw.get('t2_atr', float('nan')):.2f}")
    if fw: explain.append(f"週線斐波：1.618→{fw.get('t2_fib', float('nan')):.2f}")

    return {
        "mid_targets_weekly": mid_targets_w,
        "components": {"vp60w": vp60w, "vp120w": vp120w, "box_w": bw, "atr_w": aw, "fib_w": fw},
        "explain": explain
    }

# ===== K 線形態偵測（單根/組合，簡化版） =====
def detect_candles(df: pd.DataFrame, lookback: int = 3) -> dict:
    """
    回傳最近一根K線的形態標記與多空傾向。
    規則保守：只偵測常見而直觀的形態，避免過度干擾。
    """
    d = df.dropna().tail(max(lookback, 3)).copy()
    if d.empty:
        return {}

    def body(o, c): return abs(c - o)
    def upper(h, o, c): return h - max(o, c)
    def lower(l, o, c): return min(o, c) - l

    res = {"last": []}
    o = float(d["Open"].iloc[-1]); h = float(d["High"].iloc[-1])
    l = float(d["Low"].iloc[-1]);  c = float(d["Close"].iloc[-1])

    rng = max(h - l, 1e-8)
    b = body(o, c); u = upper(h, o, c); w = lower(l, o, c)
    b_pct = b / rng; u_pct = u / rng; w_pct = w / rng

    # 1) Doji
    if b_pct <= 0.1 and u_pct >= 0.2 and w_pct >= 0.2:
        res["last"].append("Doji")

    # 2) Hammer / Hanging Man（下影長、上影短、實體小）
    if w_pct >= 0.5 and u_pct <= 0.2 and b_pct <= 0.3:
        res["last"].append("Hammer/HS")

    # 3) Shooting Star（上影長、下影短、實體小）
    if u_pct >= 0.5 and w_pct <= 0.2 and b_pct <= 0.3:
        res["last"].append("ShootingStar")

    # 4) Marubozu（大陽/大陰，實體占比大，影線短）
    if b_pct >= 0.7 and u_pct <= 0.15 and w_pct <= 0.15:
        res["last"].append("Bull_Marubozu" if c > o else "Bear_Marubozu")

    # 5) Engulfing（吞噬，簡化）
    if len(d) >= 2:
        o1 = float(d["Open"].iloc[-2]); c1 = float(d["Close"].iloc[-2])
        b1 = abs(c1 - o1)
        if b > b1 * 1.05:
            if c > o and c1 < o1 and c >= max(o1, c1) and o <= min(o1, c1):
                res["last"].append("Bull_Engulfing")
            if c < o and c1 > o1 and o >= min(o1, c1) and c <= min(o1, c1):
                res["last"].append("Bear_Engulfing")

    # 6) Morning/Evening Star（簡化三根）
    if len(d) >= 3:
        o2, c2 = float(d["Open"].iloc[-3]), float(d["Close"].iloc[-3])
        o1, c1 = float(d["Open"].iloc[-2]), float(d["Close"].iloc[-2])
        cond_morning = (c2 < o2 and abs(c1 - o1) < abs(c2 - o2) * 0.6 and c > o and c >= (o2 + c2) / 2)
        cond_evening = (c2 > o2 and abs(c1 - o1) < abs(c2 - o2) * 0.6 and c < o and c <= (o2 + c2) / 2)
        if cond_morning: res["last"].append("MorningStar")
        if cond_evening: res["last"].append("EveningStar")

    res["bullish"] = any(x in res["last"] for x in ["Bull_Marubozu","Bull_Engulfing","MorningStar","Hammer/HS"])
    res["bearish"] = any(x in res["last"] for x in ["Bear_Marubozu","Bear_Engulfing","EveningStar","ShootingStar"])
    return res


def adjust_scores_with_candles_filtered(
    result: dict,
    patt: dict,
    m: Metrics,
    levels: dict,
    *,
    vol_ratio_need: float = 1.2,   # 量能門檻：Vol / MV20 >= 1.2
    near_pct: float = 2.0          # 位置門檻：距支撐/壓力 <= 2%
) -> Tuple[dict, str]:
    """
    形態加權（含過濾）：
      - 量能過濾：Vol / MV20 >= vol_ratio_need 才具備參考價值
      - 位置過濾：距最近支撐/壓力 <= near_pct% 才具備參考價值
      - 加分幅度：
          * 量能 + 位置皆符合：短線 ±4、波段 ±3
          * 只符合其中一項：短線 ±2、波段 ±1
          * 都不符合：不加分（只顯示中性訊息）
    輸出會回傳（更新後的 result, 使用者可讀的說明文字）
    """
    # 無資料/無形態 → 中性
    if not result or not patt:
        return result, "🕯️ 形態加權：中性（無明顯偏多/偏空形態）"

    # 取當前分數（複製 dict，避免就地修改）
    res = {
        "short": dict(result.get("short", {})),
        "swing": dict(result.get("swing", {})),
        "notes": list(result.get("notes", [])),
        "inputs": result.get("inputs", {}),
    }
    short_score = int(res["short"].get("score", 50))
    swing_score = int(res["swing"].get("score", 50))

    # 輔助：距離百分比
    def pct_diff(a: float, b: float) -> float:
        if a is None or b is None or b == 0:
            return float("inf")
        return abs(a / b - 1.0) * 100.0

    close, mv20, vol = m.close, m.MV20, m.volume

    # === 量能過濾 ===
    vol_ok = False
    if vol is not None and mv20 is not None and mv20 > 0:
        vol_ok = (vol / mv20) >= vol_ratio_need

    # === 位置過濾（用最近支撐/壓力） ===
    supports = (levels.get("short_supports", []) or []) + (levels.get("swing_supports", []) or [])
    resistances = (levels.get("short_resistances", []) or []) + (levels.get("swing_resistances", []) or [])

    near_support = max([s for s in supports if s is not None and close is not None and s < close], default=None)
    near_resist  = min([r for r in resistances if r is not None and close is not None and r > close], default=None)

    d_sup = pct_diff(close, near_support) if near_support is not None else float("inf")
    d_res = pct_diff(close, near_resist)  if near_resist  is not None else float("inf")
    near_ok = min(d_sup, d_res) <= near_pct

    # === 形態方向 ===
    is_bull = bool(patt.get("bullish"))
    is_bear = bool(patt.get("bearish"))

    # === 加權 ===
    delta_s = 0
    delta_w = 0
    if is_bull or is_bear:
        if vol_ok and near_ok:
            delta_s, delta_w = 4, 3
        elif vol_ok or near_ok:
            delta_s, delta_w = 2, 1
        if is_bear:
            delta_s, delta_w = -delta_s, -delta_w
        short_score += delta_s
        swing_score += delta_w

    # 更新決策
    def decision(score: int):
        if score >= 65:
            return "BUY / 加碼", "偏多，可分批買進或續抱"
        elif score >= 50:
            return "HOLD / 觀望", "中性，等突破或訊號"
        else:
            return "SELL / 減碼", "偏空，逢反彈減碼或停損"

    res["short"]["score"] = short_score
    res["short"]["decision"] = decision(short_score)
    res["swing"]["score"] = swing_score
    res["swing"]["decision"] = decision(swing_score)

    # === 精簡輸出（你要的文案） ===
    passed = (is_bull or is_bear) and (vol_ok or near_ok)
    if passed:
        if vol_ok and near_ok:
            note_text = (
                "✅ 形態加權：有效（有量、靠近支撐/壓力）\n"
                "量能：符合（大於 20 日均量）\n"
                "位置：符合（股價接近支撐/壓力）\n"
                "📌 說明：這個 K 線形態是可信的，因為今天有放量，股價又剛好靠在支撐/壓力附近。"
            )
        elif vol_ok:
            note_text = (
                "✅ 形態加權：部分成立（有量）\n"
                "量能：符合（大於 20 日均量）\n"
                "位置：不符合（離支撐/壓力較遠）\n"
                "📌 說明：僅有放量，參考性普通。"
            )
        else:
            note_text = (
                "✅ 形態加權：部分成立（靠近支撐/壓力）\n"
                "量能：不符合（量不足）\n"
                "位置：符合（股價接近支撐/壓力）\n"
                "📌 說明：僅位置貼近，參考性普通。"
            )
    else:
        note_text = "🕯️ 形態加權：中性（條件不足，未採納形態加分）"

    # 讓「判斷依據」也看得到結論第一行
    res["notes"].append(note_text.splitlines()[0])
    return res, note_text






# =============================
# 風控 / 個人化動作（已接上目標價）
# =============================
def position_analysis(m: Metrics, avg_cost: Optional[float], lots: Optional[float]) -> Dict[str, float]:
    if avg_cost is None or avg_cost <= 0 or lots is None or lots <= 0:
        return {}
    diff = m.close - avg_cost
    ret_pct = diff / avg_cost * 100
    shares = lots * 1000.0
    unrealized = diff * shares
    return {"ret_pct": ret_pct, "unrealized": unrealized, "shares": shares, "lots": lots}

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

def pct_diff(a: float, b: float) -> float:
    if a is None or b is None or b == 0: return np.inf
    return (a / b - 1.0) * 100.0

def personalized_action(symbol: str,
                        short_score: int, swing_score: int,
                        m: Metrics, pa: Dict[str, float],
                        atr_hint_pct: Optional[float],       # 👈 參數名改掉，避免衝突
                        targets: Dict,
                        weekly_targets: Optional[Dict] = None) -> str:
    """
    個人化建議（已整合週線目標、停利拉高與風控說明）
    - 逼近短線/波段目標 → 依張數減碼
    - 若有週線中長目標且在 +8% 內 → 小幅減碼、續抱挑戰
    """
    def _pct_diff(a: float, b: float) -> float:
        if a is None or b is None or b == 0:
            return np.inf
        return (a / b - 1.0) * 100.0

    lots = pa.get("lots", 0) if pa else 0
    header = f"標的：**{symbol}**。"

    if not pa:
        return header + "未輸入成本/庫存：先依技術面執行。 " + risk_budget_hint(atr_hint_pct)

    close = m.close
    ret = pa["ret_pct"]
    msg = [header]

    # —— 目標距離 —— #
    s_targets = targets.get("short_targets") or []
    w_targets = targets.get("swing_targets") or []
    near_short = next((t for t in s_targets if abs(_pct_diff(close, t)) <= 1.0), None)
    near_swing = next((t for t in w_targets if abs(_pct_diff(close, t)) <= 1.5), None)

    # —— 週線目標（抓最近且在 +8% 內） —— #
    wk_list = (weekly_targets or {}).get("mid_targets_weekly") or []
    wk_within = None
    if wk_list:
        wk_above = [t for t in wk_list if t is not None and t > close]
        wk_above.sort(key=lambda t: t - close)
        for t in wk_above:
            if _pct_diff(t, close) <= 8.0:
                wk_within = t
                break

    # —— 依張數的文字模板 —— #
    def reduce_phrase(weight="20%"):
        if lots >= 3: return f"**分批減碼 {weight}**"
        if lots >= 2: return "**先賣 1 張**"
        return "**可考慮出清**或視情況續抱"

    def small_reduce_phrase():
        if lots >= 3: return "**小幅減碼 10%–20%**"
        if lots >= 2: return "**先賣 1 張或更少**"
        return "**小量賣出或續抱觀察**"

    def add_phrase():
        if lots >= 3: return "**回測支撐不破小幅加碼（不追高）**"
        if lots == 2: return "**回測支撐不破可小量加碼**"
        return "**先觀察支撐，必要時再加碼**"

    # —— 淨損益情境 —— #
    if ret >= 15:
        msg.append(f"目前獲利約 {ret:.1f}%，遇壓力位建議 {reduce_phrase('20%–30%')}。")
    elif ret >= 8:
        msg.append(f"目前獲利約 {ret:.1f}%，逢壓力 {reduce_phrase()}，其餘續抱看趨勢。")
    elif ret > 0:
        msg.append(f"小幅獲利 {ret:.1f}%，優先守 **MA5/MA10**；跌破則降風險。")
    elif ret <= -10:
        if lots >= 2: msg.append(f"虧損 {ret:.1f}%，**嚴設停損**或反彈**大幅減碼（至少 1 張）**。")
        else:         msg.append(f"虧損 {ret:.1f}%，**嚴設停損**或反彈**出清**。")
    elif ret <= -5:
        if lots >= 2: msg.append(f"虧損 {ret:.1f}%，**反彈先減 1 張**，避免擴大。")
        else:         msg.append(f"虧損 {ret:.1f}%，**反彈減碼或出清**，避免擴大。")
    else:
        msg.append(f"小幅虧損 {ret:.1f}%，依短線趨勢彈性調整，{add_phrase()}。")

    # —— 目標價觸發 —— #
    if near_short is not None:
        if wk_within is not None and short_score >= 65 and swing_score >= 65:
            msg.append(f"**已逼近短線目標 {near_short:.2f}（±1%）**，且週線目標 **{wk_within:.2f}** 在 +8% 內。"
                       f"建議{small_reduce_phrase()}，續抱觀察量能挑戰週線目標。")
        else:
            msg.append(f"**已逼近短線目標 {near_short:.2f}（±1%）**，建議 {reduce_phrase()}，"
                       f"停利拉高至 **前一日低點 / MA5**。")
    elif near_swing is not None:
        if wk_within is not None and swing_score >= 65:
            msg.append(f"**已逼近波段目標 {near_swing:.2f}（±1.5%）**，但週線目標 **{wk_within:.2f}** 在 +8% 內，"
                       f"可先{small_reduce_phrase()}；若量價健康再續抱挑戰。")
        else:
            msg.append(f"**已逼近波段目標 {near_swing:.2f}（±1.5%）**，建議 {reduce_phrase('30%–50%')}，其餘視量能續抱。")
    else:
        if short_score >= 65 and swing_score >= 65:
            if wk_within is not None:
                msg.append(f"技術面偏多，且 **週線目標 {wk_within:.2f}** 在 +8% 內，傾向**小幅減碼、續抱挑戰週線目標**。")
            else:
                msg.append("技術面：短線/波段皆偏多，可**續抱**或" + add_phrase() + "。")
        elif short_score < 50 and swing_score < 50:
            msg.append("技術面：短線/波段皆偏弱，建議**逢反彈減碼**或換股。")
        else:
            msg.append("技術面：訊號分歧，採**分批操作**並嚴守支撐/停損。")

    # 風控提示 + 說明
    msg.append(risk_budget_hint(atr_hint_pct))
    msg.append("📘 說明：")
    msg.append("・**停利拉高**：若股價上漲，建議將停利線上移，例如以『前一日低點』或『MA5』作為防守位，確保已獲利不被回吐。")
    msg.append("・**風控比例**：ATR 反映波動度，例如 ATR≈2.9% 屬於中等波動，建議單筆交易風險控制在總資金的 **1%–1.5%**。")

    return ' '.join(msg)




# =============================
# UI
# =============================
st.set_page_config(page_title="Chart Advisor — 台股（含 POC / 目標價動作）", layout="centered")
st.title("📈 Chart Advisor — 台股（含 POC、支撐壓力、目標價與個人化建議）")
st.caption("輸入台股代碼（如 2330），自動抓 Yahoo 數據；整合當日/60日 POC、支撐/壓力、VWAP（近似）、跳空解讀、🎯目標價與個人化倉位建議。")

symbol = st.text_input("台股代碼 / Yahoo 代碼", value="2330", help="台股四位數代碼，例如 2330；或輸入完整 Yahoo 代碼，如 2330.TW")
period = st.selectbox("抓取區間", ["6mo", "1y", "2y"], index=0, help="用來計算均線/指標的歷史天數")

cA, cB, cC = st.columns(3)
with cA:
    fetch_now = st.button("🔎 抓取資料", use_container_width=True)
with cB:
    if st.button("🧹 清空/重置", use_container_width=True):
        # 一次清掉所有用到的 session_state 值
        for k in [
            "metrics", "tech_df", "symbol_final",
            "avg_cost_input", "lots_input"
        ]:
            st.session_state.pop(k, None)
        st.success("已重置")
        st.rerun()  # 立刻重新載入，輸入框會回到空白

with cC:
    st.write("")

st.markdown("---")
st.markdown("### ⌨️ 手動輸入 / 覆寫（可留空） & 個人倉位")

left, right = st.columns(2)

# 手動覆寫（可留空）
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

# 個人倉位
with right:
    st.markdown("**個人倉位（可選）**")

    # 讓輸入框與 session_state 綁定，清空時會跟著被清掉
    avg_cost_str = st.text_input(
        "平均成本價（每股）",
        value=st.session_state.get("avg_cost_input", ""),
        key="avg_cost_input",
    )
    lots_str = st.text_input(
        "庫存張數（1張=1000股）",
        value=st.session_state.get("lots_input", ""),
        key="lots_input",
    )

# 統一的安全轉換
def _to_float(s: str | None) -> Optional[float]:
    if not s:
        return None
    try:
        s = s.strip().replace(",", "")
        return float(s) if s else None
    except Exception:
        return None

avg_cost = _to_float(st.session_state.get("avg_cost_input"))
lots     = _to_float(st.session_state.get("lots_input"))


# 抓資料
if fetch_now:
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
            st.session_state["symbol_final"] = code
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
        code_display = st.session_state.get("symbol_final", symbol)

        # 取得 POC：優先當日，其次 60 日
        poc_today = session_poc_from_intraday(code_display)
        tech = st.session_state.get("tech_df")
        poc_60 = None
        if tech is not None:
            vp = volume_profile(tech, lookback=60, bins=24)
            if vp and "POC" in vp:
                poc_60 = vp["POC"]
        else:
            vp = None

# === 🚀 產生建議（中文術語 + 解釋版） ===
st.subheader("🚀 產生建議")

# 先從 session 取得必要物件
tech = st.session_state.get("tech_df")
metrics_in_state = st.session_state.get("metrics")
code_display = st.session_state.get("symbol_final", symbol)

if not metrics_in_state or tech is None or tech.empty:
    st.warning("請先點選「🔎 抓取資料」，或手動輸入最基本欄位。")
    st.stop()

# 建立 Metrics 物件
m = Metrics(**metrics_in_state)

# 取得 POC：優先當日，其次 60 日（日線量價分布）
poc_today = session_poc_from_intraday(code_display)
vp_60 = volume_profile(tech, lookback=60, bins=24) or {}
# 兼容大小寫 key
poc_60 = vp_60.get("POC", None) if isinstance(vp_60, dict) else None
if poc_60 is None and isinstance(vp_60, dict):
    poc_60 = vp_60.get("poc", None)

# 技術分數（帶 POC）
try:
    result = analyze(m, poc_today=poc_today, poc_60=poc_60)
except TypeError:
    # 如果你的 analyze 簽名不接受 poc_*，退回舊版呼叫
    result = analyze(m)

# ===== K 線形態加權（中文名稱 + 解釋） =====
patt = detect_candles(tech) if tech is not None else {}


# 支撐/壓力（若前面已算過 levels 就略過這行）
levels = estimate_levels(tech, m, poc_today, poc_60)

# 形態偵測（你原本就有）
patt = detect_candles(tech) if tech is not None else {}

# 使用「過濾後」的形態加權 + 精簡說明
result, candle_note = adjust_scores_with_candles_filtered(
    result, patt, m, levels,
    vol_ratio_need=1.2,   # 想更嚴格可改 1.3~1.5
    near_pct=2.0          # 更短線 1.5；波段 3.0
)
st.caption(candle_note)




# 顯示分數與決策
c1, c2 = st.columns(2)
with c1:
    st.metric("短線分數", result["short"]["score"])
    st.success(f"標的短線：{result['short']['decision'][0]} — {result['short']['decision'][1]}")
with c2:
    st.metric("波段分數", result["swing"]["score"])
    st.info(f"標的波段：{result['swing']['decision'][0]} — {result['swing']['decision'][1]}")

# 最近形態（中文 + 解釋）
last_patterns = patt.get("last", [])
translated = [CANDLE_TRANSLATE.get(p, (p, "")) for p in last_patterns]
for name, desc in translated:
    if desc:
        st.caption(f"🕯️ 最近形態：{name} — {desc}")
    else:
        st.caption(f"🕯️ 最近形態：{name}")


with st.expander("判斷依據 / 輸入數據"):
    st.write(result["notes"])
    st.json(result["inputs"])

# ===== 目標價（自動）：日線 + 週線 =====
try:
    vp_full = volume_profile(tech, lookback=60, bins=24) or {}
except Exception:
    vp_full = {}

targets = build_targets(m, tech, poc_today, vp_full)
wk      = build_targets_weekly(m, tech, poc_today)

# ======================================================================
# 目標價（自動）顯示區塊 —— 你應該已經算好 targets / wk 在上方
# ======================================================================
st.markdown("**短線目標**：{}".format(
    "-" if not targets.get("short_targets") else ", ".join([f"{x:.2f}" for x in targets["short_targets"]])
))
st.markdown("**波段目標**：{}".format(
    "-" if not targets.get("swing_targets") else ", ".join([f"{x:.2f}" for x in targets["swing_targets"]])
))
st.markdown("**中長距離（日線延伸）**：{}".format(
    "-" if not targets.get("mid_targets") else ", ".join([f"{x:.2f}" for x in targets["mid_targets"]])
))
st.markdown("**中長距離（週線延伸）**：{}".format(
    "-" if not wk.get("mid_targets_weekly") else ", ".join([f"{x:.2f}" for x in wk["mid_targets_weekly"]])
))

# ===== 趨勢燈號（狀態判斷 + 行動建議）========================================
def _s(val, default=None):
    try:
        return float(val) if val is not None else default
    except Exception:
        return default

def compute_trend_state(tech: pd.DataFrame, m: Metrics, vp60: dict | None = None) -> dict:
    """
    回傳：
      state: one of ["range_neutral","range_up","range_down","range_end",
                    "down_trend","baseing","turning_up",
                    "up_trend","up_warning","turning_down"]
      facts: 指標摘要（給說明用）
    """
    if tech is None or tech.empty:
        return {"state": "unknown", "facts": {}}

    close   = _s(m.close)
    ma5     = _s(m.MA5);   ma10 = _s(m.MA10); ma20 = _s(m.MA20)
    ma60    = _s(m.MA60);  dif  = _s(m.DIF);  macd = _s(m.MACD)
    rsi     = _s(m.RSI14); vol  = _s(m.volume); mv20 = _s(m.MV20)
    bb_up   = _s(m.BB_UP); bb_mid = _s(m.BB_MID); bb_low = _s(m.BB_LOW)

    # 波動（ATR%）與布林寬度
    atr_pct = None
    if "ATR14_pct" in tech.columns:
        s = tech["ATR14_pct"].dropna()
        if not s.empty: atr_pct = float(s.iloc[-1])
    bb_width = None
    if bb_up and bb_low and close:
        bb_width = (bb_up - bb_low) / close * 100.0

    # 量價/價值區
    vp60 = vp60 or {}
    poc60 = vp60.get("POC")

    # 條件
    ma_knit = all(x is not None for x in [ma5, ma10, ma20]) and max(ma5, ma10, ma20) - min(ma5, ma10, ma20) <= (close * 0.01)  # 均線糾結 ~1%
    bb_tight = (bb_width is not None) and (bb_width <= 5.0)   # 布林很窄
    low_vol  = (vol is not None and mv20 is not None and mv20 > 0 and (vol / mv20) < 0.9)
    up_vol   = (vol is not None and mv20 is not None and mv20 > 0 and (vol / mv20) >= 1.2)

    up_bias   = (close is not None and ma20 is not None and close > ma20) and (dif is not None and macd is not None and dif > macd) and (rsi is not None and rsi >= 50)
    down_bias = (close is not None and ma20 is not None and close < ma20) and (dif is not None and macd is not None and dif < macd) and (rsi is not None and rsi <= 45)

    # 趨勢框架
    up_trend   = (ma20 is not None and ma60 is not None and ma20 > ma60) and (rsi is not None and rsi >= 55)
    down_trend = (ma20 is not None and ma60 is not None and ma20 < ma60) and (rsi is not None and rsi <= 50)

    # 盤整（均線糾結 + 布林收斂）
    if ma_knit and bb_tight:
        if atr_pct is not None and atr_pct <= 2.0 and low_vol:
            state = "range_end"          # 尾聲：隨時出方向
        elif up_bias:
            state = "range_up"           # 盤整偏上
        elif down_bias:
            state = "range_down"         # 盤整偏下
        else:
            state = "range_neutral"      # 標準盤整
        return {"state": state, "facts": {
            "ATR%": atr_pct, "BB寬%": bb_width, "量能比": (vol / mv20) if (vol and mv20) else None,
            "RSI14": rsi, "DIF>MACD": bool(dif is not None and macd is not None and dif > macd),
            "close>MA20": bool(close and ma20 and close > ma20), "POC60": poc60
        }}

    # 下跌趨勢族群
    if down_trend:
        # 築底：指標正背離 或 連續站回MA20
        pos_div = False
        try:
            c = tech["Close"].tail(30)
            d = (ema(c, 12) - ema(c, 26)).tail(30)  # 簡化用 DIF 當動能
            pos_div = (c.idxmin() < c.index[-1]) and (d.iloc[-1] > d.min()*0.9)  # 粗略：價創新低後動能未再破底
        except Exception:
            pass
        stand_ma20 = bool(close and ma20 and close > ma20)
        if pos_div or stand_ma20:
            return {"state": "baseing", "facts": {"RSI14": rsi, "站回MA20": stand_ma20, "正背離?": pos_div, "POC60": poc60}}
        # 轉強：站上MA60、MA20上穿MA60
        cross_up = bool(ma20 and ma60 and ma20 > ma60 and (tech["MA20"].iloc[-2] <= tech["MA60"].iloc[-2]))
        if (close and ma60 and close > ma60) or cross_up:
            return {"state": "turning_up", "facts": {"RSI14": rsi, "站上MA60?": bool(close and ma60 and close > ma60), "MA20上穿MA60?": cross_up}}
        return {"state": "down_trend", "facts": {"RSI14": rsi, "close<POC60": bool(poc60 and close and close < poc60)}}

    # 上升趨勢族群
    if up_trend:
        warn = False
        # 警訊：跌破MA20、頂背離、上漲縮量/回檔放量
        below_ma20 = bool(close and ma20 and close < ma20)
        top_div = False
        try:
            c = tech["Close"].tail(40); r = rsi if rsi is not None else 50
            top_div = (c.iloc[-1] >= c.max()*0.995) and (r <= 55)  # 近新高但 RSI 不強
        except Exception:
            pass
        if below_ma20 or top_div:
            warn = True
        if warn:
            return {"state": "up_warning", "facts": {"跌破MA20?": below_ma20, "頂背離?": top_div, "量能比": (vol / mv20) if (vol and mv20) else None}}
        # 轉跌：MA20下彎且跌破MA60
        turn_down = bool(ma20 and ma60 and ma20 < ma60)
        if (close and ma60 and close < ma60) and turn_down:
            return {"state": "turning_down", "facts": {"MA20<MA60?": turn_down}}
        return {"state": "up_trend", "facts": {"RSI14": rsi, "close>POC60": bool(poc60 and close and close > poc60)}}

    # 其它：視為一般震盪
    return {"state": "range_neutral", "facts": {"ATR%": atr_pct, "BB寬%": bb_width, "量能比": (vol / mv20) if (vol and mv20) else None}}

def check_volume_breakout(m: Metrics) -> Optional[str]:
    """
    偵測「價漲 + 放量」情境。
    - 條件：收盤價 > 前一日收盤價，且 Volume > MV20
    """
    if m.close is None or m.volume is None or m.MV20 is None:
        return None
    if m.volume > m.MV20 and m.chg_pct is not None and m.chg_pct > 0:
        return "✅ 脫離盤整 → 偏多（價漲 + 放量）"
    return None

def trend_action_text(ts: dict) -> tuple[str, str]:
    """依 state 回傳 (燈號文字, 行動建議)"""
    s = ts.get("state", "unknown")
    f = ts.get("facts", {})
    if s == "range_end":
        return "盤整（尾聲）", "等『放量 + 布林擴張』再跟；可先小倉佈局，突破確立再加碼，停損放箱底/MA20。"
    if s == "range_up":
        return "盤整（偏上）", "分批佈局；突破箱頂且量能≥1.5×MV20時加碼，防守 MA20 / 箱底。"
    if s == "range_down":
        return "盤整（偏下）", "保守或減碼；跌破箱底且放量時出清弱勢，僅留守 MA60 或具題材標的。"
    if s == "range_neutral":
        return "盤整（中性）", "等待方向，觀察 POC/箱頂箱底；僅做區間短打，嚴守停損。"
    if s == "down_trend":
        return "下跌趨勢", "以反彈減碼為主；除非看到明確築底訊號（站回MA20/正背離/放量）再考慮低接。"
    if s == "baseing":
        return "下跌→築底", "先觀察，站穩 MA20 與量能回升再小量佈局；分批進場，停損設近期低點下。"
    if s == "turning_up":
        return "下跌→轉強", "站回 MA60 或 MA20黃金交叉後可偏多；守 MA20/POC，目標看前高或波段目標。"
    if s == "up_trend":
        return "上升趨勢", "順勢操作、拉回偏多；守 MA20/前低，量能健康可追蹤加碼點。"
    if s == "up_warning":
        return "上升→警訊", "先減碼 20–30%，停利拉高至 MA5/前低；若 2–3 日內無法收復 MA20，續降風險。"
    if s == "turning_down":
        return "上升→翻轉下跌", "逢反彈大幅減碼或出清，先保留現金；等下一次築底/轉強再進。"
    return "未知", "資料不足，請先抓取行情或縮小區間再試。"

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 插入開始：支撐 / 壓力 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# === 🧭 支撐 / 壓力（短線 / 波段） ===
# 依賴：estimate_levels(), tech, m, poc_today, poc_60
st.subheader("🧭 支撐 / 壓力")

try:
    lv = estimate_levels(tech, m, poc_today, poc_60)
except Exception as e:
    lv = {}
    st.warning(f"支撐/壓力計算失敗：{e}")

def _mark_with_poc(values, poc_t=None, poc_60d=None, tol=0.3):
    """把等於當日/60日 POC 的價位加註，提升可讀性。"""
    out = []
    for v in (values or []):
        tag = ""
        if poc_t is not None and abs(v - poc_t) <= tol:
            tag = "（當日POC）"
        elif poc_60d is not None and abs(v - poc_60d) <= tol:
            tag = "（60日POC）"
        out.append(f"{v:.2f}{tag}")
    return "、".join(out) if out else "-"

cA, cB = st.columns(2)
with cA:
    st.markdown("**短線（≈ 1–3 週）**")
    st.markdown("• 支撐： " + _mark_with_poc(lv.get("short_supports"), poc_today, poc_60))
    st.markdown("• 壓力： "  + _mark_with_poc(lv.get("short_resistances"), poc_today, poc_60))

with cB:
    st.markdown("**波段（≈ 1–3 個月）**")
    st.markdown("• 支撐： " + _mark_with_poc(lv.get("swing_supports"), poc_today, poc_60))
    st.markdown("• 壓力： "  + _mark_with_poc(lv.get("swing_resistances"), poc_today, poc_60))

with st.expander("支撐/壓力計算說明"):
    st.write("""
- **短線**：就近的 MA5 / MA10（支撐）、MA20（壓力）＋『當日 POC』＋近 20 日高低點。
- **波段**：MA20 / MA60（支撐）、MA60 / MA120（壓力）＋『60 日 POC』＋近 60 日高低點。
- 旁註 **（當日POC）** 或 **（60日POC）** 代表該價位與 POC 重疊，成交密集、有效性更高。
""")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 插入結束：支撐 / 壓力 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ===== 趨勢燈號（狀態 + 行動） ==============================================
st.subheader("🚦 趨勢燈號（狀態與建議）")

vp60_for_trend = volume_profile(tech, lookback=60, bins=24) or {}
ts = compute_trend_state(tech, m, vp60_for_trend)
label, act = trend_action_text(ts)

colA, colB = st.columns([1,2])
with colA:
    st.metric("狀態", label)
with colB:
    st.write("**行動建議**：", act)

with st.expander("判斷依據（重點數據）"):
    facts = ts.get("facts", {})
    nice = {k: (None if v is None else (f"{v:.2f}" if isinstance(v, (int,float)) else v)) for k,v in facts.items()}
    st.json(nice)

# 👉 額外檢查「價漲 + 放量」
def check_volume_breakout(m: Metrics) -> Optional[str]:
    if m.close is None or m.volume is None or m.MV20 is None:
        return None
    if m.volume > m.MV20 and m.chg_pct is not None and m.chg_pct > 0:
        return "✅ 脫離盤整 → 偏多（價漲 + 放量）"
    return None

extra_signal = check_volume_breakout(m)
if extra_signal:
    st.success(extra_signal + " 👉 可小量試單，突破確認後再加碼")

# === 🛠️ 實務操作建議 ===
st.subheader("🛠️ 實務操作建議")

def practical_advice(m: Metrics, result: dict, lv: dict) -> str:
    """
    根據技術面 & 均線位置，給出持有 / 空手 / 風控建議
    """
    msg = []
    close = m.close or 0.0
    ma20, ma60, ma5 = m.MA20, m.MA60, m.MA5
    res_short = result.get("short", {}).get("decision", ["",""])[0]
    res_swing = result.get("swing", {}).get("decision", ["",""])[0]

    # ===== 已持有 =====
    hold_msg = "若已持有："
    if close > (ma20 or 0) and close > (ma60 or 0):
        hold_msg += "續抱，觀察能否站穩 MA20 / MA60，突破後可續抱挑戰波段壓力。"
    else:
        hold_msg += "守住 MA5 / MA10，若跌破需減碼或停損。"
    msg.append(hold_msg)

    # ===== 空手 =====
    empty_msg = "若空手："
    if res_short.startswith("BUY") or res_swing.startswith("BUY"):
        empty_msg += "可小量切入，設好停損（如回跌到 5 日均線或當日低點）。"
    else:
        empty_msg += "先觀望，等突破壓力或明確轉強再進場。"
    msg.append(empty_msg)

    # ===== 風險控管 =====
    risk_msg = "風險控管："
    if res_short.startswith("BUY") and res_swing.startswith("BUY"):
        risk_msg += "剛轉強，失敗機率仍有 → 建議先小倉位，避免過度槓桿。"
    else:
        risk_msg += "以支撐位為防守線，單筆風險控制在總資金 1%–2%。"
    msg.append(risk_msg)

    return "\n\n".join(msg)

advice_text = practical_advice(m, result, lv)
st.info(advice_text)

# 額外顯示 MA20 / MA60 是否站穩
st.caption(check_stand_ma(m, tech, "MA20"))
st.caption(check_stand_ma(m, tech, "MA60"))

# =============================
# 💡 當沖建議
# =============================
def daytrade_suggestion(df_intraday: pd.DataFrame, vwap: float, poc: float) -> str:
    """
    簡單的當沖建議：
    - 進場：靠近 VWAP 或 POC 附近，且量能放大
    - 出場：日內壓力（前高 ±0.5%）
    - 停損：跌破 VWAP 或當日低點
    """
    if df_intraday is None or df_intraday.empty:
        return "❓ 無法計算當沖建議（缺少分時資料）"

    last = df_intraday.iloc[-1]
    close = float(last["Close"])
    high = float(df_intraday["High"].max())
    low  = float(df_intraday["Low"].min())

    entry = vwap if vwap else poc
    stop  = max(low, entry * 0.99)       # 停損：低點或 VWAP-1%
    target = min(high, entry * 1.01)     # 出場：高點或 VWAP+1%

    return (
        f"🎯 當沖建議：\n"
        f"- **進場價**：{entry:.2f}（VWAP/POC）\n"
        f"- **停損價**：{stop:.2f}（跌破支撐止損）\n"
        f"- **出場價**：{target:.2f}（前高或 VWAP+1%）\n"
        f"📌 說明：靠近 VWAP 或 POC 買進，守停損，逢壓力或 +1% 獲利出場。"
    )

# === 在畫面中顯示 ===
st.subheader("💡 當沖建議（僅供參考）")
try:
    intraday = yf.download(code_display, period="7d", interval="5m", progress=False)
    if intraday is not None and not intraday.empty:
        poc_intraday = session_poc_from_intraday(code_display)
        vwap_today = float(intraday["Close"].mean())  # 近似 VWAP
        suggestion = daytrade_suggestion(intraday, vwap_today, poc_intraday)
        st.info(suggestion)
    else:
        st.warning("抓不到分時資料，無法提供當沖建議。")
except Exception as e:
    st.error(f"當沖建議計算失敗：{e}")

# =============================
# 💡 當沖建議（分時 VWAP / 當日 POC）
# =============================

def _intraday_vwap(df: pd.DataFrame) -> float | None:
    """以分時資料計算 VWAP：sum(price*vol)/sum(vol)。用 Close 近似 price。"""
    if df is None or df.empty:
        return None
    v = df["Volume"].fillna(0)
    if float(v.sum()) <= 0:
        return None
    p = df["Close"].fillna(method="ffill")
    return float((p * v).sum() / v.sum())

def _fmt(p):
    return "-" if p is None or not np.isfinite(p) else f"{p:.2f}"

def daytrade_suggestion_auto(symbol: str) -> tuple[str, dict]:
    """
    自動抓 5 分鐘分時資料，產出當沖建議（做多視角；若市價落在 VWAP 下方則建議觀望或等反彈再說）
    回傳：(建議文字, 參考數據)
    """
    try:
        intraday = yf.download(symbol, period="7d", interval="5m", progress=False)
        if intraday is None or intraday.empty:
            return "❓ 無法計算當沖建議（抓不到分時資料）", {}

        # 只取「今天」
        tz = "Asia/Taipei"
        idx = intraday.index
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize("UTC")
        intraday = intraday.copy()
        intraday.index = idx.tz_convert(tz)
        today = pd.Timestamp.now(tz).normalize()
        dft = intraday[(intraday.index >= today) & (intraday.index < today + pd.Timedelta(days=1))]
        if dft.empty:
            return "❓ 無法計算當沖建議（今日尚無分時資料）", {}

        last = dft.iloc[-1]
        close = float(last["Close"])
        day_high = float(dft["High"].max())
        day_low  = float(dft["Low"].min())

        vwap_today = _intraday_vwap(dft)
        poc_intraday = session_poc_from_intraday(symbol)  # 你前面已定義
        ref = vwap_today if vwap_today is not None else poc_intraday

        if ref is None:
            return "❓ 無法計算當沖建議（VWAP/POC 皆無法取得）", {}

        # 基本參考價位
        entry = ref                                     # 以 VWAP/POC 當進場基準
        stop  = max(day_low, entry * 0.99)             # -1% 或日內低點
        target = min(day_high, entry * 1.01)           # +1% 或日內前高

        # 價位相對 VWAP 的狀態
        diff_pct = (close / ref - 1.0) * 100.0

        # 規則分支（做多視角）
        # 1) 市價遠高於 VWAP（> +0.6%）：傾向追高風險，等回檔靠近 VWAP 再說
        if diff_pct >= 0.6:
            text = (
                f"🎯 當沖建議（偏強；避免追高）：\n"
                f"- **當前價**：{close:.2f}（高於 VWAP/POC {diff_pct:.2f}%）\n"
                f"- **計畫買點**：{entry:.2f}（VWAP/POC 回檔靠近再考慮）\n"
                f"- **停損**：{stop:.2f}（跌破支撐止損）\n"
                f"- **出場**：{target:.2f}（前高或 +1%）\n"
                f"📌 說明：走勢偏強，但**不建議追價**；等回測 VWAP/POC 附近、量縮不破再低風險切入。"
            )
        # 2) 市價略高於 VWAP（0 ~ +0.6%）：順勢偏多，拉回靠近 VWAP 小試
        elif 0.0 < diff_pct < 0.6:
            buy_zone_low  = entry * 0.998   # -0.2%
            buy_zone_high = entry * 1.001   # +0.1%
            text = (
                f"🎯 當沖建議（順勢偏多）：\n"
                f"- **當前價**：{close:.2f}（略高於 VWAP/POC {diff_pct:.2f}%）\n"
                f"- **進場區**：{buy_zone_low:.2f} ~ {buy_zone_high:.2f}（VWAP 附近回檔小試）\n"
                f"- **停損**：{stop:.2f}（跌破 VWAP/POC 或日內低點）\n"
                f"- **出場**：{target:.2f}（前高或 +1%）\n"
                f"📌 說明：以 VWAP 為支撐的順勢交易；若回測失敗跌破，立即認錯退出。"
            )
        # 3) 市價貼近 VWAP（-0.3% ~ 0%）：盤整邊緣，等突破或回測成功再進
        elif -0.3 <= diff_pct <= 0.0:
            text = (
                f"🎯 當沖建議（中性盤整）：\n"
                f"- **當前價**：{close:.2f}（貼近 VWAP/POC {diff_pct:.2f}%）\n"
                f"- **計畫買點**：{entry:.2f}（等『回測不破』或量價突破再進）\n"
                f"- **停損**：{stop:.2f}\n"
                f"- **出場**：{target:.2f}\n"
                f"📌 說明：VWAP 附近容易震盪洗單；**要等確認**（如回測不破、放量紅K）再進場。"
            )
        # 4) 市價低於 VWAP（< -0.3%）：偏空，不建議做多；若要多，需等站回 VWAP
        else:  # diff_pct < -0.3
            text = (
                f"🎯 當沖建議（偏空／觀望）：\n"
                f"- **當前價**：{close:.2f}（低於 VWAP/POC {abs(diff_pct):.2f}%）\n"
                f"- **多單進場**：不建議（籌碼在空方）。若強要做多，請等**站回 VWAP**再說。\n"
                f"- **空方思路**（進階）：反彈至 VWAP 附近、量縮轉弱再尋找做空點；嚴控風險。\n"
                f"📌 說明：市價位於 VWAP 下方表示當日偏弱；**多單勝率低**，建議先觀望。"
            )

        info = {
            "price": close,
            "vwap": vwap_today,
            "poc_intraday": poc_intraday,
            "day_high": day_high,
            "day_low": day_low,
            "diff_vs_vwap_%": diff_pct,
            "entry": entry,
            "stop": stop,
            "target": target,
        }
        return text, info

    except Exception as e:
        return f"❌ 當沖建議計算失敗：{e}", {}

# === 畫面顯示（放在『🧭 支撐 / 壓力』之後、『👤 個人持倉評估』之前） ===
st.subheader("💡 當沖建議（僅供參考）")
try:
    code_for_intraday = st.session_state.get("symbol_final", symbol)
    txt, facts = daytrade_suggestion_auto(code_for_intraday)
    st.info(txt)
    with st.expander("當日關鍵數據（VWAP / POC / 高低點）"):
        if facts:
            show = {k: (None if v is None else (f"{v:.2f}" if isinstance(v,(int,float)) else v)) for k,v in facts.items()}
            st.json(show)
        else:
            st.write("（無可用數據）")
except Exception as e:
    st.error(f"當沖模組出錯：{e}")


# ======================================================================
# 個人化持倉建議（依你輸入的成本/張數）—— 放在支撐/壓力之後
# 依賴：position_analysis(), personalized_action(), result, targets, wk
# ======================================================================
st.subheader("👤 個人持倉評估（依你輸入的成本/張數）")

# 取得 ATR%（給風控文字使用）
atr_pct = None
if tech is not None and "ATR14_pct" in tech.columns:
    _ap = tech["ATR14_pct"].dropna()
    if not _ap.empty:
        atr_pct = float(_ap.iloc[-1])

# 只有 avg_cost & lots 皆有效才做持倉建議
pa = position_analysis(m, avg_cost, lots) if (avg_cost and lots) else {}

if pa:
    st.write(f"- 標的：**{code_display}**")
    st.write(f"- 平均成本：{avg_cost:.2f}，現價：{m.close:.2f}，**報酬率：{pa['ret_pct']:.2f}%**")
    st.write(f"- 庫存：{int(pa['shares']):,} 股（約 {pa['lots']} 張），未實現損益：約 **{pa['unrealized']:.0f} 元**")

    suggestion = personalized_action(
        code_display,
        result["short"]["score"], result["swing"]["score"],
        m, pa, atr_pct,
        targets,
        weekly_targets=wk  # 👈 把週線中長目標一起納入建議判斷
    )
    st.success(suggestion)
else:
    st.write("（如要得到個人化建議，請於右側輸入平均成本與庫存張數）")
















