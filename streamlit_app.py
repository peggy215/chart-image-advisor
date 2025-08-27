
# streamlit_app.py
# -*- coding: utf-8 -*-
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

import streamlit as st

# Optional OCR deps
try:
    import cv2
    import numpy as np
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


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


def _to_float(s):
    try:
        return float(str(s).replace(',', ''))
    except Exception:
        return None


KEY_PATTERNS = [
    (r'MA\s*5[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA5'),
    (r'MA5[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA5'),
    (r'MA\s*10[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA10'),
    (r'MA10[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA10'),
    (r'MA\s*20[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA20'),
    (r'MA20[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA20'),
    (r'MA\s*60[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA60'),
    (r'MA60[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA60'),
    (r'MA\s*120[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA120'),
    (r'MA120[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA120'),
    (r'MA\s*240[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA240'),
    (r'MA240[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MA240'),
    (r'MV\s*5[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MV5'),
    (r'MV5[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MV5'),
    (r'MV\s*20[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MV20'),
    (r'MV20[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MV20'),
    (r'K9?[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'K'),
    (r'K值[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'K'),
    (r'D9?[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'D'),
    (r'D值[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'D'),
    (r'MACD9?[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MACD'),
    (r'MACD[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'MACD'),
    (r'DIF[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'DIF'),
    (r'OSC[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'OSC'),
    (r'收盤?[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'close'),
    (r'量(?:\(張\))?[:：]?\s*([\-+]?\d+(?:\.\d+)?)', 'volume'),
]


def ocr_parse_metrics(image_bytes) -> Metrics:
    if not OCR_AVAILABLE:
        return Metrics()
    # Read image bytes to cv2 image
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return Metrics()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        text = pytesseract.image_to_string(
            bw, config=r'--oem 3 --psm 6 -l eng+chi_sim+chi_tra'
        )
    except Exception:
        text = ""
    metrics = Metrics()
    for pat, key in KEY_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = _to_float(m.group(1))
            if val is not None:
                setattr(metrics, key, val)
    return metrics


def analyze(m: Metrics):
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
        "notes": notes
    }


st.set_page_config(page_title="Chart Image Advisor", layout="centered")
st.title("📈 Chart Image Advisor — 圖像讀取 + 短線/波段建議")

uploaded = st.file_uploader("上傳當日股票圖（PNG/JPG）", type=["png", "jpg", "jpeg"])

metrics = Metrics()

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🧠 方式 A：自動 OCR 解析")
    if uploaded is not None:
        st.image(uploaded, caption="已上傳的截圖", use_column_width=True)
        if st.button("嘗試 OCR 解析", use_container_width=True):
            parsed = ocr_parse_metrics(uploaded)
            st.session_state["parsed"] = asdict(parsed)
    parsed_state = st.session_state.get("parsed")
    if parsed_state:
        st.success("OCR 擷取成功：")
        st.json(parsed_state)
        metrics = Metrics(**parsed_state)

with col2:
    st.markdown("### ⌨️ 方式 B：手動輸入/覆寫")
    def num_input(label, value):
        return st.text_input(label, value if value is not None else "", placeholder="例如 936 或 33590")

    for field in metrics.__dataclass_fields__.keys():
        cur = getattr(metrics, field)
        val = num_input(field, "" if cur is None else str(cur))
        if val.strip():
            setattr(metrics, field, _to_float(val))

st.markdown("---")
if st.button("產生建議", type="primary", use_container_width=True):
    result = analyze(metrics)
    st.subheader("🔎 分析結果")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("短線分數", result["short"]["score"])
        st.success(f"短線建議：{result['short']['decision'][0]} — {result['short']['decision'][1]}")
    with c2:
        st.metric("波段分數", result["swing"]["score"])
        st.info(f"波段建議：{result['swing']['decision'][0]} — {result['swing']['decision'][1]}")

    with st.expander("判斷依據（Notes）"):
        for n in result["notes"]:
            st.write("•", n)

    st.download_button(
        "下載 JSON 結果",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="analysis_output.json",
        mime="application/json"
    )

st.caption("提示：OCR 需要本機安裝 Tesseract。若擷取不完整，請於右側輸入框手動補值後再按「產生建議」。")
