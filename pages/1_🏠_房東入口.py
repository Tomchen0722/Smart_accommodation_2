"""
房東入口 — Landlord Portal
1KM 競爭分析 · 空房預測 · 三維度智慧建議 · NLP 評論分析
"""
import html as _html
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.ui_components import (
    inject_css, P, RC, RTC, ROOM_JP, FEAT_ZH,
    sec, mb, note, risk_badge, stat_card, html_table, apply_theme,
    review_hover_html, sidebar_nav,
)
from modules.data_loader import load_listings, load_reviews
from modules.geo_utils import (
    listings_within_radius, load_all_poi, count_poi_within,
    nearest_poi, poi_points_within, convenience_score, POI_NAMES,
)
from modules.ml_models import (
    train_models, predict_vacancy_prob, generate_landlord_advice, nb_aggregate,
)
from modules.nlp_analysis import listing_review_summary, recent_review_snippets

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="房東入口 — 智慧旅宿", page_icon="🏠",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()

# ─── Dialog support (works across Streamlit versions) ───────────
_DIALOG = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)


def render_poi_list(title, pts):
    """Full detail list for one PoI type (no horizontal scroll)."""
    st.markdown(f"**{title}** — 範圍內共 <b style='color:{P['primary']}'>"
                f"{len(pts)}</b> 筆，依距離排序：", unsafe_allow_html=True)
    if len(pts) == 0:
        st.caption("此範圍內無資料。")
        return
    disp = pts.copy()
    disp["名稱"] = disp["poi_name"]
    disp["地址 / 說明"] = disp["poi_addr"].replace("", "—")
    disp["距離"] = disp["distance_m"].map(lambda d: f"{d:.0f} m")
    html_table(disp[["名稱", "地址 / 說明", "距離"]], wrap=True, scroll=False)


def render_reviews(snips):
    """Review list body used inside the reviews dialog / expander."""
    if not snips:
        st.caption("此房源尚無評論。")
        try:
            st.caption(f"（診斷：資料庫已載入 {len(REVIEWS):,} 則評論、"
                       f"{REVIEWS['listing_id'].nunique():,} 個房源）")
        except Exception:
            pass
        return
    items = "".join(
        f'<div class="rv-item">{_html.escape(str(x))}</div>' for x in snips)
    st.markdown(
        f'<div style="font-size:.78rem;color:{P["ink2"]};line-height:1.7;">'
        f'{items}</div>', unsafe_allow_html=True)


if _DIALOG:
    @_DIALOG("📋 設施明細")
    def poi_dialog(title, pts):
        render_poi_list(title, pts)

    @_DIALOG("💬 房源評論")
    def reviews_dialog(snips):
        render_reviews(snips)
else:
    def poi_dialog(title, pts):
        with st.expander("📋 設施明細", expanded=True):
            render_poi_list(title, pts)

    def reviews_dialog(snips):
        with st.expander("💬 房源評論", expanded=True):
            render_reviews(snips)


# ─── Load data ──────────────────────────────────────────────────
with st.spinner("載入房源資料與訓練模型 …"):
    DF = load_listings()
    REVIEWS = load_reviews()
    MDL = train_models(DF)

# ─── Header ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:6px 0 14px;">
  <h1 style="font-size:1.4rem;font-weight:700;color:{P['ink']};
       margin:0;letter-spacing:-.3px;">🏠 房東智慧分析面板</h1>
  <p style="font-size:.78rem;color:{P['muted']};margin:4px 0 0;">
    選擇房源 → 1KM 競爭分析 → 空房預測 → 智慧建議 → NLP 評論分析
  </p>
</div>
<hr style="margin:0 0 16px;">
""", unsafe_allow_html=True)

# ─── Sidebar: listing selector ──────────────────────────────────
with st.sidebar:
    sidebar_nav()
    st.markdown(f"""
    <div style="padding:4px 0 12px;">
      <div style="font-size:1rem;font-weight:700;color:{P['ink']};">
        🏠 房東面板</div>
      <div style="font-size:.72rem;color:{P['muted']};margin-top:2px;">
        選擇您的房源進行分析</div>
    </div>""", unsafe_allow_html=True)

    # Filter by neighbourhood
    all_nb = sorted(DF["neighbourhood_cleansed"].dropna().unique())
    sel_nb = st.selectbox("🗺 行政區", all_nb, index=0)

    nb_listings = DF[DF["neighbourhood_cleansed"] == sel_nb].sort_values("price")
    listing_options = {
        f"#{r.id} | {r['name'][:30]}… | ${r.price:,.0f}"
         if len(str(r['name'])) > 30
         else f"#{r.id} | {r['name']} | ${r.price:,.0f}": r.id
        for _, r in nb_listings.iterrows()
    }

    sel_label = st.selectbox("🏘 選擇房源", list(listing_options.keys()))
    sel_id = listing_options[sel_label]

    st.divider()
    radius = st.slider("📏 分析半徑 (公尺)", 500, 2000, 1000, step=100)

    st.divider()
    st.caption("© 2026 智慧旅宿 AI 平台")

# ─── Get selected listing ──────────────────────────────────────
listing = DF[DF["id"] == sel_id].iloc[0]
lat, lon = listing["latitude"], listing["longitude"]
snips = recent_review_snippets(REVIEWS, sel_id, n=10)

# ── 1KM competitors ──
nearby = listings_within_radius(DF, lat, lon, radius)
nearby = nearby[nearby["id"] != sel_id]  # Exclude self

# ── PoI ──
poi_all = load_all_poi()

# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════
# ─── Persistent listing banner (visible across all tabs) ────────
st.markdown(f"""
<div style="background:{P['surface']};border:1px solid {P['border']};
     border-left:4px solid {P['landlord']};border-radius:0 12px 12px 0;
     padding:10px 18px;margin:0 0 14px;display:flex;align-items:center;
     justify-content:space-between;flex-wrap:wrap;gap:10px;">
  <div style="font-size:.98rem;font-weight:700;color:{P['ink']};
       max-width:60%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
    🏠 目前分析房源：{listing['name']}</div>
  <div style="font-size:.76rem;color:{P['muted']};">
    #{sel_id} ｜ 🗺 {listing['neighbourhood_cleansed']} ｜
    🛏 {ROOM_JP.get(listing['room_type'], listing['room_type'])} ｜
    💰 ${listing['price']:,.0f}/晚 ｜ {risk_badge(listing['risk_level'])}</div>
</div>
""", unsafe_allow_html=True)

T1, T2, T3, T4 = st.tabs([
    "📊 競爭分析", "🔮 空房預測", "💡 智慧建議", "💬 NLP 評論分析"
])

# ──────────────────────────────────────────────────────────────
# TAB 1: 1KM Competition Analysis
# ──────────────────────────────────────────────────────────────
with T1:
    # ── Listing info card ──
    col_info, col_map = st.columns([1.2, 1.8])
    with col_info:
        sec("房源基本資訊")
        st.markdown(f"""
        <div style="background:{P['surface']};border:1px solid {P['border']};
             border-radius:12px;padding:18px 22px;margin-bottom:12px;">
          <div style="font-size:1.05rem;font-weight:700;color:{P['ink']};
               margin-bottom:8px;">{listing['name']}</div>
          <div style="font-size:.78rem;color:{P['muted']};line-height:1.8;">
            🗺 {listing['neighbourhood_cleansed']} ｜
            🛏 {ROOM_JP.get(listing['room_type'], listing['room_type'])}<br>
            💰 每晚 <b style="color:{P['primary']};">${listing['price']:,.0f}</b> ｜
            ⭐ {listing.get('review_scores_rating', 'N/A')} ｜
            💬 {listing['number_of_reviews']} 則評論<br>
            👥 可住 {int(listing.get('accommodates', 0))} 人 ｜
            🛁 {int(listing.get('bathrooms_count', 0))} 衛浴 ｜
            🛏 {int(listing.get('beds', 0))} 床<br>
            {risk_badge(listing['risk_level'])}
            <span style="font-size:.72rem;color:{P['muted']};margin-left:8px;">
              風險分數: {listing['risk_score']:.3f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"💬 查看 {int(listing['number_of_reviews'])} 則評論",
                     key="rev_btn_landlord", use_container_width=True):
            reviews_dialog(snips)

        # KPI cards
        k1, k2, k3 = st.columns(3)
        median_p = nearby["price"].median() if not nearby.empty else 0
        k1.metric("周邊房源數", f"{len(nearby)}")
        k2.metric("周邊中位價", f"${median_p:,.0f}")
        k3.metric("價格偏差",
                  f"{(listing['price'] - median_p) / median_p * 100:+.1f}%"
                  if median_p > 0 else "–")

    with col_map:
        sec(f"{radius}M 範圍競爭地圖")
        mb(f"地理空間分析 · Haversine 半徑 {radius}m")
        map_data = nearby.copy()
        map_data["type"] = "競爭房源"
        me = pd.DataFrame([{
            "latitude": lat, "longitude": lon, "name": listing["name"],
            "price": listing["price"], "risk_level": listing["risk_level"],
            "risk_score": listing["risk_score"], "room_type_zh": listing["room_type_zh"],
            "type": "📍 我的房源", "neighbourhood_cleansed": listing["neighbourhood_cleansed"],
        }])
        show_df = pd.concat([me, map_data.assign(type="競爭房源")], ignore_index=True)
        # Make my own listing marker markedly larger (about 3x competitors)
        show_df["_sz"] = (show_df["type"] == "📍 我的房源").map({True: 2.6, False: 1.0})

        fig = px.scatter_mapbox(
            show_df, lat="latitude", lon="longitude",
            hover_name="name",
            hover_data={"price": ":,.0f", "risk_level": True, "_sz": False,
                        "latitude": False, "longitude": False, "type": True},
            labels={"price": "每晚價格", "risk_level": "風險等級",
                    "type": "類型", "name": "房源"},
            color="type",
            color_discrete_map={"📍 我的房源": P["high"], "競爭房源": P["primary"]},
            size="_sz", size_max=13, zoom=14, height=380,
            mapbox_style="carto-positron",
            center={"lat": lat, "lon": lon},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(bgcolor=P["surface"], bordercolor=P["border"], borderwidth=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Price distribution comparison ──
    st.divider()
    cp1, cp2 = st.columns(2)
    with cp1:
        sec("周邊價格分佈 vs 我的定價")
        mb("核密度估計 · KDE + 垂直基準線")
        if not nearby.empty:
            fig = px.histogram(
                nearby[nearby["price"] < nearby["price"].quantile(.95)],
                x="price", nbins=30, opacity=0.6,
                color_discrete_sequence=[P["primary"]],
                labels={"price": "每晚價格 (TWD)", "count": "房源數"},
            )
            fig.add_vline(x=listing["price"], line_dash="dash",
                          line_color=P["high"], line_width=2,
                          annotation_text=f"我的定價 ${listing['price']:,.0f}",
                          annotation_font_color=P["high"])
            fig.add_vline(x=median_p, line_dash="dot",
                          line_color=P["low"], line_width=2,
                          annotation_text=f"中位數 ${median_p:,.0f}",
                          annotation_position="top left",
                          annotation_font_color=P["low"])
            apply_theme(fig, h=280, legend=False)
            st.plotly_chart(fig, use_container_width=True)

    with cp2:
        sec("周邊房型分佈")
        mb("佔比分析 · Proportional Analysis")
        if not nearby.empty:
            rt_c = nearby["room_type_zh"].value_counts().reset_index()
            rt_c.columns = ["房型", "數量"]
            fig = px.pie(rt_c, values="數量", names="房型", color="房型",
                         color_discrete_map=RTC, hole=0.55)
            fig.update_traces(textfont_size=11,
                              marker_line_width=2, marker_line_color=P["bg"])
            apply_theme(fig, h=280).update_layout(
                margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(fig, use_container_width=True)

    # ── PoI summary ──
    sec("周邊生活機能")
    mb(f"PoI 分析 · {radius}m 範圍內便利設施統計")
    poi_cols = st.columns(len(poi_all))
    for i, (ptype, pdf) in enumerate(poi_all.items()):
        pts = poi_points_within(lat, lon, pdf, radius)
        cnt = len(pts)
        with poi_cols[i]:
            stat_card(
                f"{cnt}", f"{POI_NAMES[ptype]}",
                color=P["low"] if cnt >= 5 else (P["medium"] if cnt >= 2 else P["high"]),
            )
            if cnt:
                n0 = pts.iloc[0]
                addr = f" · {n0['poi_addr']}" if n0["poi_addr"] else ""
                st.caption(f"最近：{n0['poi_name']}{addr}（{n0['distance_m']:.0f}m）")
            else:
                st.caption("無資料")
            if st.button(f"📋 全部 {cnt} 筆", key=f"poi_ll_{ptype}",
                         use_container_width=True):
                poi_dialog(POI_NAMES[ptype], pts)

# ──────────────────────────────────────────────────────────────
# TAB 2: Vacancy Prediction
# ──────────────────────────────────────────────────────────────
with T2:
    sec("空房預測機率分析")
    mb("機器學習預測 · Logistic Regression × Random Forest")

    vp = predict_vacancy_prob(listing, DF, MDL)

    # Prediction cards
    v1, v2, v3 = st.columns(3)
    for col_, days, lr_k, rf_k in [
        (v1, "30天", "lr_30", "rf_30"),
        (v2, "60天", "lr_60", "rf_60"),
        (v3, "90天", "lr_90", "rf_90"),
    ]:
        with col_:
            avg_p = (vp[lr_k] + vp[rf_k]) / 2
            color = P["high"] if avg_p > 0.6 else (P["medium"] if avg_p > 0.3 else P["low"])
            level = "高風險" if avg_p > 0.6 else ("中風險" if avg_p > 0.3 else "低風險")
            st.markdown(f"""
            <div style="background:{P['surface']};border:1px solid {P['border']};
                 border-radius:12px;padding:20px;text-align:center;
                 border-top:3px solid {color};">
              <div style="font-size:.72rem;color:{P['muted']};
                   letter-spacing:.08em;margin-bottom:8px;">未來 {days} 空房機率</div>
              <div style="font-size:2rem;font-weight:700;color:{color};">
                {avg_p*100:.1f}%</div>
              <div style="font-size:.72rem;color:{P['muted']};margin-top:6px;">
                LR: {vp[lr_k]*100:.1f}% ｜ RF: {vp[rf_k]*100:.1f}%
              </div>
              {risk_badge(level)}
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Gauge chart
    sec("預測趨勢圖")
    base_avg = (vp["base_lr"] + vp["base_rf"]) / 2
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=base_avg * 100,
        delta={"reference": 50, "increasing": {"color": P["high"]},
               "decreasing": {"color": P["low"]}},
        number={"suffix": "%", "font": {"size": 42, "color": P["ink"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": P["border"]},
            "bar": {"color": P["primary"]},
            "steps": [
                {"range": [0, 30], "color": "#EAF5EE"},
                {"range": [30, 60], "color": "#FDF5E4"},
                {"range": [60, 100], "color": "#FDECEA"},
            ],
            "threshold": {
                "line": {"color": P["high"], "width": 3},
                "thickness": 0.75, "value": base_avg * 100,
            },
        },
        title={"text": "基礎空房風險指數", "font": {"size": 14, "color": P["muted"]}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=280,
        margin=dict(l=30, r=30, t=50, b=10),
        font=dict(family="Noto Sans TC,sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    note("模型基於 6,241 筆台北 Airbnb 真實數據訓練。"
         f"LR Recall={MDL['lr']['recall']:.3f}，"
         f"RF AUC={MDL['rf']['auc']:.3f}。"
         "30/60/90 天預測加入短期可訂率趨勢修正。")

# ──────────────────────────────────────────────────────────────
# TAB 3: Smart Advice
# ──────────────────────────────────────────────────────────────
with T3:
    sec("系統智慧建議（三維度分析）")
    mb("決策引擎 · 價格優化 × 動態行銷 × 硬體升級")

    advices = generate_landlord_advice(listing, nearby, vp)

    if not advices:
        st.info("目前無特別建議，房源狀態健康！")
    else:
        for adv in advices:
            severity_color = {
                "high": P["high"], "medium": P["medium"], "low": P["low"]
            }.get(adv["severity"], P["muted"])
            st.markdown(f"""
            <div style="background:{P['surface']};border:1px solid {P['border']};
                 border-left:4px solid {severity_color};border-radius:0 12px 12px 0;
                 padding:16px 20px;margin-bottom:12px;">
              <div style="font-size:.88rem;font-weight:700;color:{P['ink']};
                   margin-bottom:6px;">{adv['type']}</div>
              <div style="font-size:.82rem;color:{P['ink2']};line-height:1.7;">
                {adv['text']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Risk level summary
    st.divider()
    risk_p = max(vp["base_lr"], vp["base_rf"])
    if risk_p < 0.3:
        note("✅ <b>低風險</b>（P < 30%）：房源狀態健康。建議保持現狀，可在續約時考慮小幅調漲。")
    elif risk_p < 0.6:
        note("⚠️ <b>中風險</b>（30% ≤ P < 60%）：出現空房危機信號。建議微調價格、補充照片、優化描述。")
    else:
        note("🔴 <b>高風險</b>（P ≥ 60%）：極可能持續空房。建議立即執行動態定價降價或限時優惠策略。")

# ──────────────────────────────────────────────────────────────
# TAB 4: NLP Review Analysis
# ──────────────────────────────────────────────────────────────
with T4:
    sec("NLP 評論情感分析")
    mb("VADER 情感分析 × jieba 中文分詞 × 關鍵字提取")

    listing_reviews = REVIEWS[REVIEWS["listing_id"] == sel_id]
    total_rev = len(listing_reviews)

    if total_rev == 0:
        st.info("此房源尚無評論資料。")
    else:
        with st.spinner(f"分析 {total_rev} 則評論中 …"):
            nlp_summary = listing_review_summary(REVIEWS, sel_id)

        # Sentiment overview
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("評論總數", f"{nlp_summary['total_reviews']}")
        n2.metric("😊 正面", f"{nlp_summary['pos_pct']}%")
        n3.metric("😐 中立", f"{nlp_summary['neu_pct']}%")
        n4.metric("😞 負面", f"{nlp_summary['neg_pct']}%")

        # Sentiment donut
        sd1, sd2 = st.columns(2)
        with sd1:
            sec("情感分佈")
            sent_data = pd.DataFrame({
                "情感": ["正面", "中立", "負面"],
                "比例": [nlp_summary["pos_pct"], nlp_summary["neu_pct"],
                        nlp_summary["neg_pct"]],
            })
            fig = px.pie(sent_data, values="比例", names="情感",
                         color="情感",
                         color_discrete_map={
                             "正面": P["low"], "中立": P["muted"], "負面": P["high"]
                         }, hole=0.6)
            fig.update_traces(textfont_size=11,
                              marker_line_width=2, marker_line_color=P["bg"])
            apply_theme(fig, h=260).update_layout(
                margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(fig, use_container_width=True)

        with sd2:
            sec("平均情感分數")
            avg_s = nlp_summary["avg_sentiment"]
            color = P["low"] if avg_s > 0.1 else (P["high"] if avg_s < -0.1 else P["muted"])
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_s,
                number={"font": {"size": 36, "color": P["ink"]}},
                gauge={
                    "axis": {"range": [-1, 1]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [-1, -0.05], "color": "#FDECEA"},
                        {"range": [-0.05, 0.05], "color": "#F2F0EC"},
                        {"range": [0.05, 1], "color": "#EAF5EE"},
                    ],
                },
                title={"text": "VADER Compound Score",
                       "font": {"size": 12, "color": P["muted"]}},
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", height=260,
                margin=dict(l=30, r=30, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Keywords
        kw1, kw2 = st.columns(2)
        with kw1:
            sec("✅ 正面關鍵字 TOP 10")
            if nlp_summary["pos_keywords"]:
                kw_df = pd.DataFrame(nlp_summary["pos_keywords"],
                                     columns=["關鍵字", "出現次數"])
                fig = go.Figure(go.Bar(
                    x=kw_df["出現次數"], y=kw_df["關鍵字"],
                    orientation="h",
                    marker=dict(color=P["low"], line=dict(width=0)),
                ))
                apply_theme(fig, h=280, legend=False).update_layout(
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=80, r=20, t=5, b=30))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("無足夠正面評論提取關鍵字")

        with kw2:
            sec("❌ 負面關鍵字 TOP 10")
            if nlp_summary["neg_keywords"]:
                kw_df = pd.DataFrame(nlp_summary["neg_keywords"],
                                     columns=["關鍵字", "出現次數"])
                fig = go.Figure(go.Bar(
                    x=kw_df["出現次數"], y=kw_df["關鍵字"],
                    orientation="h",
                    marker=dict(color=P["high"], line=dict(width=0)),
                ))
                apply_theme(fig, h=280, legend=False).update_layout(
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=80, r=20, t=5, b=30))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("無足夠負面評論提取關鍵字")

        # Sample reviews
        if nlp_summary["sample_pos"]:
            st.markdown(f"""
            <div class="note" style="border-left-color:{P['low']};">
              <b>😊 正面評論摘錄：</b><br>{nlp_summary['sample_pos']}
            </div>""", unsafe_allow_html=True)
        if nlp_summary["sample_neg"]:
            st.markdown(f"""
            <div class="note" style="border-left-color:{P['high']};">
              <b>😞 負面評論摘錄：</b><br>{nlp_summary['sample_neg']}
            </div>""", unsafe_allow_html=True)
