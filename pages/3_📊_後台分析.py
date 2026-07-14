"""
後台分析 — 空屋率風險預警與策略沙盒平台
（依 smartaccommodation_imp_new 規格完全改寫；沿用平台原本暖色系）
左欄：策略沙盒與經營控制   右欄：AI 診斷預警與 SHAP 貢獻
"""
import math
import numpy as np
import streamlit as st

from modules.ui_components import inject_css, P, sidebar_nav, ROOM_JP
from modules.vacancy_model import (
    load_data, host_options, host_listings, get_row, predict,
    contributions, diagnose, poi_snapshot, confidence, get_metrics, get_models,
)

st.set_page_config(page_title="後台分析 — 空屋率風險沙盒", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
# 頂部留白，避免房源標題列與信心標籤被工具列切到
st.markdown(
    "<style>.block-container,[data-testid='stMainBlockContainer']"
    "{padding-top:3.5rem !important;}</style>", unsafe_allow_html=True)

RESP = {1: "一小時內 (極速)", 2: "幾小時內 (回覆迅速)",
        3: "一天內 (普通)", 4: "多於一天 (偏慢)"}


def sf(v, d=0.0):
    try:
        v = float(v)
        return d if math.isnan(v) else v
    except Exception:
        return d


def band(vac):
    if vac < 0.40:
        return "低風險", P["low"]
    if vac < 0.70:
        return "中風險", P["medium"]
    return "高風險", P["high"]


def gauge_svg(vac, color):
    """半圓風險儀表：弧長對應空屋率。"""
    cx, cy, r = 130, 120, 96
    theta = math.pi * (1 - max(0.0, min(1.0, vac)))     # 180° -> 0°
    x = cx + r * math.cos(theta)
    y = cy - r * math.sin(theta)
    x0, y0 = cx - r, cy
    pct = int(round(vac * 100))
    return f"""
<svg viewBox="0 0 260 150" width="100%" style="max-width:340px;display:block;margin:0 auto;">
  <path d="M{x0},{y0} A{r},{r} 0 0 1 {cx+r},{cy}" fill="none"
        stroke="{P['border']}" stroke-width="18" stroke-linecap="round"/>
  <path d="M{x0},{y0} A{r},{r} 0 0 1 {x:.1f},{y:.1f}" fill="none"
        stroke="{color}" stroke-width="18" stroke-linecap="round"/>
  <text x="{cx}" y="{cy-14}" text-anchor="middle" font-size="40" font-weight="800"
        fill="{color}">{pct}%</text>
  <text x="{cx}" y="{cy+12}" text-anchor="middle" font-size="14"
        fill="{P['muted']}">預估年空屋率</text>
</svg>"""


def waterfall_html(contribs):
    if not contribs:
        return "<div style='color:#9A9490;'>無足夠特徵可解釋。</div>"
    mx = max(abs(d) for _, _, d in contribs) or 1.0
    rows = []
    for f, zh, d in contribs:
        w = abs(d) / mx * 46          # 半邊最長 46%
        if d >= 0:                    # 推高風險（紅、向右）
            bar = (f"<div style='flex:1;'></div>"
                   f"<div style='flex:1;position:relative;'>"
                   f"<div style='position:absolute;left:0;height:16px;border-radius:0 6px 6px 0;"
                   f"width:{w:.1f}%;background:{P['high']};'></div></div>")
            val = f"<span style='color:{P['high']};font-weight:700;'>+{d:.2f}%</span>"
        else:                         # 降低風險（綠、向左）
            bar = (f"<div style='flex:1;position:relative;'>"
                   f"<div style='position:absolute;right:0;height:16px;border-radius:6px 0 0 6px;"
                   f"width:{w:.1f}%;background:{P['low']};'></div></div>"
                   f"<div style='flex:1;'></div>")
            val = f"<span style='color:{P['low']};font-weight:700;'>{d:.2f}%</span>"
        rows.append(
            f"<div style='display:flex;align-items:center;gap:8px;margin:5px 0;'>"
            f"<div title='{zh}' style='width:120px;text-align:right;font-size:.8rem;color:{P['ink2']};"
            f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{zh}</div>"
            f"<div style='flex:1;display:flex;border-left:2px dashed {P['border2']};'>{bar}</div>"
            f"<div style='width:64px;text-align:left;font-size:.82rem;'>{val}</div></div>")
    return ("<div style='background:" + P['surface'] + ";border:1px solid " + P['border'] +
            ";border-radius:12px;padding:14px 16px;'>"
            "<div style='text-align:center;font-weight:800;margin-bottom:8px;'>"
            "各項經營與 POI 機能指標之空屋率加減分貢獻度</div>"
            "<div style='text-align:center;font-size:.7rem;color:" + P['muted'] +
            ";margin-bottom:8px;'>綠色（左）＝降低風險　紅色（右）＝推高風險</div>"
            + "".join(rows) + "</div>")


# ─── Sidebar：房東 / 房源選擇 ─────────────────────────────────
DF = load_data()
with st.sidebar:
    sidebar_nav()
    st.markdown("#### 🎯 請選擇登入的房東")
    hosts = host_options()
    hlab = [f"房東 ID: {h} (名下 {n} 間房源)" for h, n in hosts]
    hi = st.selectbox("host", range(len(hosts)),
                      format_func=lambda i: hlab[i], label_visibility="collapsed")
    host_id = hosts[hi][0]
    lst_all = host_listings(host_id)
    st.markdown("#### 🗺 區域")
    _dists = sorted({r["neighbourhood_cleansed"] for r in lst_all})
    dsel = st.selectbox("district", _dists, label_visibility="collapsed")
    st.markdown("#### 🏠 切換操作房源")
    lst = [r for r in lst_all if r["neighbourhood_cleansed"] == dsel]
    llab = [f"房源 #{r['id']} ({ROOM_JP.get(r['room_type'], r['room_type'])})" for r in lst]
    li = st.selectbox("listing", range(len(lst)),
                      format_func=lambda i: llab[i], label_visibility="collapsed")
    listing_id = int(lst[li]["id"])
    st.divider()
    st.caption("技術堆疊：HistGradientBoosting 雙軌模型 · "
               "GroupKFold(host_id) 5 折誠實驗證 · POI×7 + NLP 多模態特徵")

row = get_row(listing_id)
conf, conf_why = confidence(row)
conf_color = P["low"] if conf == "極高" else P["medium"]

# ─── 頂部資訊 + 信心標籤 ──────────────────────────────────────
st.markdown(
    f"<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;'>"
    f"<div><span style='font-size:1.5rem;font-weight:800;'>房源 #{listing_id}</span>"
    f"<span style='color:{P['muted']};margin-left:12px;'>📍 {row['neighbourhood_cleansed']} ｜ "
    f"{ROOM_JP.get(row['room_type'], row['room_type'])} ｜ 🧭 {sf(row['latitude']):.4f}, {sf(row['longitude']):.4f}</span></div>"
    f"<div style='background:{conf_color};color:#fff;padding:6px 16px;border-radius:20px;font-weight:800;'>"
    f"預測信心：{conf}</div></div>"
    f"<div style='color:{P['muted']};font-size:.8rem;margin:2px 0 10px;'>{conf_why}</div>",
    unsafe_allow_html=True)

cL, cR = st.columns(2, gap="large")

# ─── 左欄：策略沙盒 ───────────────────────────────────────────
with cL:
    st.markdown(f"### 🎛 策略沙盒與經營控制")
    st.markdown("🔒 **唯讀歷史數據 (不可調整)**")
    a, b = st.columns(2)
    a.text_input("所在行政區", row["neighbourhood_cleansed"], disabled=True)
    b.text_input("歷史性價比評分", f"{sf(row['review_scores_value']):.1f} ★", disabled=True)
    a.text_input("歷史清潔度評分", f"{sf(row['review_scores_cleanliness']):.1f} ★", disabled=True)
    b.text_input("歷史溝通體驗評分", f"{sf(row['review_scores_communication']):.1f} ★", disabled=True)

    ps = poi_snapshot(row)
    st.markdown(
        f"<div style='background:{P['mbg']};border:1px solid {P['border']};border-radius:12px;"
        f"padding:12px 14px;margin:10px 0;font-size:.84rem;line-height:1.9;color:{P['ink2']};'>"
        f"💡 <b>周邊機能快照 (唯讀)</b><br>"
        f"・最近捷運站：{ps['mrt_name']}（{ps['mrt_m']:.0f} 公尺，500m 內有 {ps['mrt_500']:.0f} 個出入口）<br>"
        f"・生活機能：500m 內有超商 {ps['conv_500']:.0f} 家、餐廳 {ps['rest_500']:.0f} 家<br>"
        f"・綠地景觀：最近公園 {ps['park_name']}（{ps['park_m']:.0f} 公尺，500m 內 {ps['park_500']:.0f} 座）<br>"
        f"・住客評價情感得分：{ps['sentiment']:.2f}（歷史平均評論字數 {ps['rev_len']:.0f} 字）</div>",
        unsafe_allow_html=True)

    st.markdown("⚡ **動態參數調整 (即時模擬)**")
    price = st.slider("每晚房價 (NTD $)", 500, 50000,
                      int(min(50000, max(500, sf(row["price"], 1500)))), step=50)
    mn = st.number_input("最低入住天數限制 (晚)", 1, 30,
                         int(min(30, max(1, sf(row["minimum_nights"], 1)))))
    rs_now = int(min(4, max(1, sf(row["response_speed"], 2))))
    rs = st.selectbox("客服平均回覆時間", [1, 2, 3, 4],
                      index=rs_now - 1, format_func=lambda k: RESP[k])
    desc = st.text_area(
        "房源描述文案模擬區",
        value=f"【目前自介字數約 {int(sf(row['desc_len']))} 字】。"
              "調整此文字以模擬房源描述篇幅與用心程度，可在此處輸入新文案…",
        height=110)
    desc_len = len(desc)

overrides = {"price": price, "minimum_nights": mn,
             "response_speed": rs, "desc_len": desc_len}

# ─── 右欄：AI 診斷預警 ────────────────────────────────────────
with cR:
    st.markdown("### 🩺 AI 診斷預警與指標評估")
    vac, risk = predict(row, overrides)
    lbl, color = band(vac)
    st.markdown(gauge_svg(vac, color), unsafe_allow_html=True)
    breathing = "animation:pulse 1.2s ease-in-out infinite;" if vac >= 0.70 else ""
    st.markdown(
        f"<style>@keyframes pulse{{50%{{opacity:.45;}}}}</style>"
        f"<div style='text-align:center;margin:-6px 0 6px;{breathing}'>"
        f"<span style='font-size:1.5rem;font-weight:800;color:{color};'>{lbl}"
        f"{'（空屋警報）' if vac>=0.7 else ''}</span></div>"
        f"<div style='text-align:center;color:{P['muted']};font-size:.9rem;margin-bottom:12px;'>"
        f"高空屋機率 (模型 B)：{risk*100:.1f}%</div>",
        unsafe_allow_html=True)

    tips = diagnose(row, overrides, k=2)
    tip_html = ""
    for t in tips:
        tip_html += (
            f"<div style='border-left:3px solid {P['low']};background:{P['tag_bg']};"
            f"border-radius:0 8px 8px 0;padding:10px 12px;margin:8px 0;font-size:.86rem;line-height:1.7;'>"
            f"💡 <b>{t['zh']}</b>（+{t['delta']:.2f}% 空屋風險）<br>{t['advice']}</div>")
    if not tip_html:
        tip_html = f"<div style='color:{P['low']};'>目前無明顯扣分項，經營狀態良好。</div>"
    st.markdown(
        f"<div style='background:{P['surface']};border:1px solid {P['border']};border-radius:12px;"
        f"padding:14px 16px;margin-bottom:12px;'>"
        f"<div style='font-weight:800;margin-bottom:4px;'>🤖 AI 房源智能診斷報告</div>"
        f"<div style='color:{P['muted']};font-size:.78rem;margin-bottom:6px;'>"
        f"🚨 篩選出推高空屋風險最多的 Top-2 因子並給出優化建議</div>{tip_html}</div>",
        unsafe_allow_html=True)

    st.markdown(waterfall_html(contributions(row, overrides, top=8)),
                unsafe_allow_html=True)

# ─── 底部：誠實模型品質 ──────────────────────────────────────
with st.expander("📐 模型品質（GroupKFold(host_id) 5 折誠實驗證）"):
    m = get_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("模型 A R²", f"{m['R2']:.3f}")
    c2.metric("模型 A MSE", f"{m['MSE']:.4f}")
    c3.metric("模型 B AUC", f"{m['AUC']:.3f}")
    c4.metric("模型 B Recall", f"{m['Recall']:.3f}")
    st.caption(
        f"樣本 {m['n']} 筆 · {m['n_features']} 特徵 · 高風險占比 {m['high_risk_rate']*100:.1f}%。"
        "此為測試房東不出現在訓練集的誠實泛化指標（比隨機切分保守、但可信）。")
