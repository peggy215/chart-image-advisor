# streamlit_app.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

# 🔤 K 線形態對照表（英文 → 中文）
CANDLE_TRANSLATE = {
    "Bull_Engulfing": "多頭吞噬",
    "Bear_Engulfing": "空頭吞噬",
    "MorningStar": "晨星",
    "EveningStar": "暮星",
    "Hammer": "錘子線",
    "Inverted_Hammer": "倒錘子線",
    "Doji": "十字星",
    "ShootingStar": "射擊之星",
    "Harami": "母子線",
    "Three_White_Soldiers": "三白兵",
    "Three_Black_Crows": "三隻黑鴉"
    # 👆 需要可以再擴充
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


# ===== 將 K 線形態加到分數（日線短/波段） =====
def adjust_scores_with_candles(result: dict, patt: dict) -> tuple[dict, str]:
    """
    在 analyze() 的結果上，根據 K 線形態做小幅加減分，並回傳說明文字。
    （不直接改 analyze()，避免侵入式大改；這裡做後處理）
    """
    if not result or not patt:
        return result, ""

    # 複製 result，避免原 dict 被就地修改
    res = {
        "short": dict(result.get("short", {})),
        "swing": dict(result.get("swing", {})),
        "notes": list(result.get("notes", [])),
        "inputs": result.get("inputs", {}),
    }

    short_score = int(res["short"].get("score", 50))
    swing_score = int(res["swing"].get("score", 50))

    delta_s, delta_w = 0, 0
    last_tags = patt.get("last", [])

    if patt.get("bullish"):
        delta_s += 3; delta_w += 2
        res["notes"].append(f"K線形態偏多 {last_tags} (+3/+2)")
    if patt.get("bearish"):
        delta_s -= 3; delta_w -= 2
        res["notes"].append(f"K線形態偏空 {last_tags} (-3/-2)")

    short_score += delta_s
    swing_score += delta_w

    def decision(score: int):
        if score >= 65: return "BUY / 加碼", "偏多，可分批買進或續抱"
        elif score >= 50: return "HOLD / 觀望", "中性，等突破或訊號"
        else: return "SELL / 減碼", "偏空，逢反彈減碼或停損"

    res["short"]["score"] = short_score
    res["short"]["decision"] = decision(short_score)
    res["swing"]["score"] = swing_score
    res["swing"]["decision"] = decision(swing_score)

    note_text = ""
    if delta_s or delta_w:
        sign = "↑" if (delta_s + delta_w) > 0 else "↓"
        note_text = f"🕯️ K線形態影響：短線 {('+' if delta_s>=0 else '')}{delta_s}、波段 {('+' if delta_w>=0 else '')}{delta_w}（{sign}）"
    else:
        note_text = "🕯️ K線形態影響：中性（無明顯偏多/偏空形態）"

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
                        atr_pct: Optional[float],
                        targets: Dict,
                        weekly_targets: Optional[Dict] = None) -> str:
    """
    已整合『週線目標在 +8% 內 → 減碼少一點、續抱挑戰週線目標』：
      - 逼近短線目標（±1%） → 依張數減碼
      - 逼近波段目標（±1.5%） → 減碼 30–50%
      - 若存在『週線中長目標』且距離當前價 <= +8%：
          建議傾向『先小幅減碼（10–20% / 先賣1張），續抱挑戰週線目標』
    """
    def pct_diff(a: float, b: float) -> float:
        if a is None or b is None or b == 0: return np.inf
        return (a / b - 1.0) * 100.0

    lots = pa.get("lots", 0) if pa else 0
    header = f"標的— "
    close = m.close

    if not pa:
        return header + "未輸入成本/庫存：先依技術面執行。 " + risk_budget_hint(atr_pct)

    ret = pa["ret_pct"]
    msg = [header]

    # —— 日線目標距離 —— #
    s_targets = targets.get("short_targets") or []
    w_targets = targets.get("swing_targets") or []
    near_short = next((t for t in s_targets if abs(pct_diff(close, t)) <= 1.0), None)
    near_swing = next((t for t in w_targets if abs(pct_diff(close, t)) <= 1.5), None)

    # —— 週線目標（抓最近且在 +8% 內） —— #
    wk_list = (weekly_targets or {}).get("mid_targets_weekly") or []
    wk_within = None
    if wk_list:
        wk_above = [t for t in wk_list if t is not None and t > close]
        wk_above.sort(key=lambda t: t - close)
        for t in wk_above:
            if pct_diff(t, close) <= 8.0:
                wk_within = t
                break

    # —— 語句模板（依張數） —— #
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

    # —— 淨損益敘述 —— #
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

    # —— 目標價情境 —— #
    if near_short is not None:
        if wk_within is not None and short_score >= 65 and swing_score >= 65:
            msg.append(f"**已逼近短線目標 {near_short:.2f}（±1%）**，且週線目標 **{wk_within:.2f}** 在 +8% 內。建議{small_reduce_phrase()}，續抱觀察量能挑戰週線目標。")
        else:
            msg.append(f"**已逼近短線目標 {near_short:.2f}（±1%）**，建議 {reduce_phrase()}，停利拉高至 **前一日低點/MA5**。")
    elif near_swing is not None:
        if wk_within is not None and swing_score >= 65:
            msg.append(f"**已逼近波段目標 {near_swing:.2f}（±1.5%）**，但週線目標 **{wk_within:.2f}** 在 +8% 內，可先{small_reduce_phrase()}；若量價健康再續抱挑戰。")
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

    msg.append(risk_budget_hint(atr_pct))
    return " ".join(msg)



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
        for k in list(st.session_state.keys()):
            del st.session_state[k]
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

        # 分數（含 POC）
        result = analyze(m, poc_today=poc_today, poc_60=poc_60)
        # === 在 analyze() 之後，加入 K 線形態加權 ===
        patt = detect_candles(tech) if tech is not None else {}
        result, candle_note = adjust_scores_with_candles(result, patt)

        # 顯示分數與決策（使用調整後的 result）
        c1, c2 = st.columns(2)
        with c1:
             st.metric("短線分數", result["short"]["score"])
             st.success(f"標的短線：{result['short']['decision'][0]} — {result['short']['decision'][1]}")
        with c2:
             st.metric("波段分數", result["swing"]["score"])
             st.info(f"標的波段：{result['swing']['decision'][0]} — {result['swing']['decision'][1]}")

        # 顯示形態與影響說明
        # 將英文形態轉換成中文
        last_patterns = patt.get("last", [])
        translated = [CANDLE_TRANSLATE.get(p, p) for p in last_patterns]
        st.caption(f"🕯️ 最近形態：{', '.join(translated) or '-'}")

        # 顯示加權說明
        st.caption(candle_note)


        with st.expander("判斷依據 / 輸入數據"):
            st.write(result["notes"])
            st.json(result["inputs"])


        # 當日價量：VWAP + POC + 跳空
        st.subheader("📊 當日價量")
        st.caption("成交量加權平均價（VWAP，近似）：{}".format("-" if m.vwap_approx is None else f"{m.vwap_approx:.2f}"))
        st.caption("控制價（POC，當日）：{}".format("-" if poc_today is None else f"{poc_today:.2f}"))
        st.caption("控制價（POC，近60日）：{}".format("-" if poc_60 is None else f"{poc_60:.2f}"))
        st.caption("跳空：{}".format("-" if m.gap_pct is None else f"{m.gap_pct:.2f}%"))
        st.info(interpret_gap(m.gap_pct, m.vol_r5))

        # 支撐 / 壓力
        atr_pct = None
        if tech is not None and "ATR14_pct" in tech.columns:
            try:
                atr_pct = float(tech["ATR14_pct"].dropna().iloc[-1])
            except Exception:
                atr_pct = None

        if tech is not None:
            st.subheader("📍 支撐 / 壓力 估算")
            lv = estimate_levels(tech, m, poc_today, poc_60)
            colS, colR = st.columns(2)
            with colS:
                st.markdown("**短線支撐**： " + (", ".join([f"{x:.2f}" for x in lv["short_supports"]]) if lv["short_supports"] else "-"))
                st.markdown("**波段支撐**： " + (", ".join([f"{x:.2f}" for x in lv["swing_supports"]]) if lv["swing_supports"] else "-"))
            with colR:
                st.markdown("**短線壓力**： " + (", ".join([f"{x:.2f}" for x in lv["short_resistances"]]) if lv["short_resistances"] else "-"))
                st.markdown("**波段壓力**： " + (", ".join([f"{x:.2f}" for x in lv["swing_resistances"]]) if lv["swing_resistances"] else "-"))

        # 🎯 目標價（自動）
        st.subheader("🎯 目標價（自動）")

        # 日線目標價
        vp_full = volume_profile(tech, lookback=60, bins=24)
        targets = build_targets(m, tech, poc_today, vp_full)

        st.markdown("**短線目標**（近）：{}".format(
            "-" if not targets["short_targets"] else ", ".join([f"{x:.2f}" for x in targets["short_targets"]])
        ))
        st.markdown("**波段目標**（中）：{}".format(
            "-" if not targets["swing_targets"] else ", ".join([f"{x:.2f}" for x in targets["swing_targets"]])
        ))
        st.markdown("**中長距離目標（日線延伸）**：{}".format(
            "-" if not targets.get("mid_targets") else ", ".join([f"{x:.2f}" for x in targets["mid_targets"]])
        ))

        with st.expander("目標價計算明細 / 依據（每日線）"):
            st.write(targets["explain"])
            st.json(targets["components"])

        # 週線目標價
        wk = build_targets_weekly(m, tech, poc_today)
        st.markdown("**中長距離目標（週線延伸）**：{}".format(
           "-" if not wk.get("mid_targets_weekly") else ", ".join([f"{x:.2f}" for x in wk["mid_targets_weekly"]])
        ))
        with st.expander("目標價計算明細 / 依據（週線）"):
            st.write(wk["explain"])
            st.json(wk["components"])


        # 個人化持倉建議（已接上日線 + 週線目標條件）
        pa = position_analysis(m, avg_cost, lots)
        st.subheader("👤 個人持倉評估（依你輸入的成本/張數）")

        if pa:
           st.write(f"- 標的：**{code_display}**")
           st.write(f"- 平均成本：{avg_cost:.2f}，現價：{m.close:.2f}，**報酬率：{pa['ret_pct']:.2f}%**")
           st.write(f"- 庫存：{int(pa['shares']):,} 股（約 {pa['lots']} 張），未實現損益：約 **{pa['unrealized']:.0f} 元**")

           # 先算目標價（若你前面已有，可略過重算）
           vp_full = volume_profile(tech, lookback=60, bins=24)
           targets = build_targets(m, tech, poc_today, vp_full)
           wk = build_targets_weekly(m, tech, poc_today)

           # 核心：把週線目標傳入 personalized_action
           suggestion = personalized_action(
              code_display,
              result["short"]["score"], result["swing"]["score"],
              m, pa, atr_pct,
              targets,
              weekly_targets=wk   # 👈 關鍵差異：加入週線目標
           )
           st.success(suggestion)
        else:
           st.write("（如要得到個人化建議，請於右側輸入平均成本與庫存張數）")











