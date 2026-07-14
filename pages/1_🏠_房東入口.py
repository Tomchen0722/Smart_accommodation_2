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
    v2_ready, load_models_v2, load_dataset_v2, local_shap_v2,
)
from modules.feature_engineering import predict_risk_v2, simulate_price_change
from modules.nlp_analysis import listing_review_summary, recent_review_snippets
from modules.image_analysis import (analyze, fake_host_email, compose_email,
                                    listing_photos)

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


@st.cache_data(show_spinner=False)
def _analyze_img(url):
    return analyze(url)


def render_email(listing_name, host_name, email, label, prob):
    subject, body, to = compose_email(listing_name, host_name, email, label, prob)
    st.success(f"✅ 已發送通知信至 {to}")
    st.markdown(f"**主旨：** {subject}")
    st.markdown(f"**收件者：** {to}")
    st.markdown("**內文：**")
    st.code(body)
    st.caption("（示範用：實際部署可串接 SMTP／SendGrid 寄送真實郵件）")


if _DIALOG:
    @_DIALOG("📧 通知信已發送")
    def email_dialog(listing_name, host_name, email, label, prob):
        render_email(listing_name, host_name, email, label, prob)
else:
    def email_dialog(listing_name, host_name, email, label, prob):
        with st.expander("📧 通知信已發送", expanded=True):
            render_email(listing_name, host_name, email, label, prob)


# ─── Load data ──────────────────────────────────────────────────
with st.spinner("載入房源資料與訓練模型 …"):
    DF = load_listings()
    REVIEWS = load_reviews()
    MDL = train_models(DF)
    # ── v2 研究級雙模型（產物缺件時自動退回舊版，不擋頁面）──
    _V2_OK, _ = v2_ready()
    BUNDLE_V2 = load_models_v2() if _V2_OK else None
    DS_V2 = load_dataset_v2().set_index("id") if _V2_OK else None

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
    st.markdown("#### 🎯 請選擇登入的房東")
    _hc = DF.groupby("host_id").size().sort_values(ascending=False)
    _hids = list(_hc.index)
    _hlab = [f"房東 ID: {int(h)} (名下 {int(_hc[h])} 間房源)" for h in _hids]
    _hi = st.selectbox("host", range(len(_hids)),
                       format_func=lambda i: _hlab[i], label_visibility="collapsed")
    host_id = int(_hids[_hi])
    _my = DF[DF["host_id"] == host_id]

    st.markdown("#### 🗺 區域")
    all_nb = sorted(_my["neighbourhood_cleansed"].dropna().unique())
    sel_nb = st.selectbox("district", all_nb, label_visibility="collapsed")

    st.markdown("#### 🏠 切換操作房源")
    nb_listings = _my[_my["neighbourhood_cleansed"] == sel_nb].sort_values("price")
    listing_options = {
        (f"#{r.id} | {str(r['name'])[:24]}… | ${r.price:,.0f}"
         if len(str(r['name'])) > 24
         else f"#{r.id} | {r['name']} | ${r.price:,.0f}"): r.id
        for _, r in nb_listings.iterrows()
    }
    sel_label = st.selectbox("listing", list(listing_options.keys()),
                             label_visibility="collapsed")
    sel_id = listing_options[sel_label]

    st.divider()
    radius = st.slider("📏 分析半徑 (公尺)", 500, 2000, 1000, step=100)

    st.divider()
    st.caption("© 2026 智慧旅宿 AI 平台")

# ─── Get selected listing ──────────────────────────────────────
listing = DF[DF["id"] == sel_id].iloc[0]
lat, lon = listing["latitude"], listing["longitude"]
from modules.geo_utils import nearest_address as _naddr
addr = _naddr(lat, lon)
snips = recent_review_snippets(REVIEWS, sel_id, n=10)
with st.spinner("分析房源照片 …"):
    IMG = _analyze_img(str(listing.get("picture_url", "")))

# ── 1KM competitors ──
nearby = listings_within_radius(DF, lat, lon, radius)
nearby = nearby[nearby["id"] != sel_id]  # Exclude self

# ── PoI ──
poi_all = load_all_poi()

# ── v2 預測（此房源在 5,849 筆協定內才有；未滿一年房源為 None）──
ROW_V2 = (DS_V2.loc[sel_id] if (DS_V2 is not None and sel_id in DS_V2.index)
          else None)
RES_V2 = (predict_risk_v2(ROW_V2, BUNDLE_V2) if ROW_V2 is not None else None)

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
    📍 {addr}<br>#{sel_id} ｜ 🗺 {listing['neighbourhood_cleansed']} ｜
    🛏 {ROOM_JP.get(listing['room_type'], listing['room_type'])} ｜
    💰 ${listing['price']:,.0f}/晚 ｜ {risk_badge(listing['risk_level'])}</div>
</div>
""", unsafe_allow_html=True)

def price_simulator_v2(key):
    """v2 What-if：模型 A 直接回答「調價後預測空屋率變多少」。

    取代舊版 30/60/90 天固定乘數公式。已知限制：價格百分位為市場相對
    排名，單筆模擬不重排整個市場（於下方註明）；樹模型對價格的反應呈
    階梯狀，小幅調價可能不改變預測 —— 這是真實模型行為。
    """
    cur_price = int(listing["price"])
    med = nearby["price"].median() if not nearby.empty else DF["price"].median()
    hi = int(round(med * 1.15))
    lo = 50
    if hi <= lo:
        hi = lo + 500
    step = 100 if hi - lo > 2000 else 50
    default = int(min(max(cur_price, lo), hi))
    st.divider()
    sec("💰 售價模擬器 v2（模型 A What-if）")
    mb("HistGradientBoosting 直接預測 · 非固定乘數公式")
    st.caption(f"周邊 1KM 中位價 ${med:,.0f}｜可模擬區間 ${lo:,} ~ ${hi:,}")
    new_price = st.slider("模擬每晚售價 (TWD)", lo, hi, default, step=step,
                          key=key)

    now = RES_V2["risk_score"]
    sim = simulate_price_change(ROW_V2, BUNDLE_V2, float(new_price))
    pm = st.columns(3)
    pm[0].metric("目前售價", f"${cur_price:,}")
    pm[1].metric("模擬售價", f"${new_price:,}",
                 f"{(new_price - cur_price) / cur_price * 100:+.0f}%",
                 delta_color="off")
    pm[2].metric("預測空屋率（模型A）", f"{sim['risk_score']*100:.1f}%",
                 f"{(sim['risk_score'] - now)*100:+.1f} 百分點",
                 delta_color="inverse")

    # 價格掃描曲線：25 個價位點的模型 A 預測
    prices = np.linspace(lo, hi, 25)
    curve = [simulate_price_change(ROW_V2, BUNDLE_V2, float(p))["risk_score"] * 100
             for p in prices]
    fig = go.Figure(go.Scatter(x=prices, y=curve, mode="lines",
                               line=dict(color=P["primary"], width=2),
                               name="預測空屋率"))
    if lo <= cur_price <= hi:
        fig.add_vline(x=cur_price, line_dash="dot", line_color=P["muted"],
                      annotation_text=f"目前 ${cur_price:,}")
    fig.add_vline(x=new_price, line_dash="dash", line_color=P["ink"],
                  annotation_text=f"模擬 ${new_price:,}")
    apply_theme(fig, h=280, legend=False).update_layout(
        margin=dict(l=50, r=20, t=10, b=36),
        xaxis_title="每晚售價 (TWD)", yaxis_title="模型A 預測空屋率 (%)",
        xaxis_range=[lo, hi])
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_fig")
    note("曲線呈階梯狀是梯度提升樹的真實反應（價格只在越過分裂點時改變預測）。"
         "已知限制：同區價格百分位特徵在模擬中不重排市場，大幅調價時實際效果"
         "可能更明顯。")


def price_simulator(key):
    """Interactive what-if: adjust price, recompute vacancy risk (base + 30/60/90d)."""
    cur_price = int(listing["price"])
    med = nearby["price"].median() if not nearby.empty else DF["price"].median()
    hi = int(round(med * 1.15))                 # 上限 = 周邊中位數 + 15%
    lo = 50                                     # 下限固定 $50
    if hi <= lo:
        hi = lo + 500
    step = 100 if hi - lo > 2000 else 50
    default = int(min(max(cur_price, lo), hi))  # 將目前售價夾進區間內
    st.divider()
    sec("💰 售價模擬器（拖動售價，即時看空房風險變化）")
    mb("What-if 分析 · 售價下限 $50 · 上限＝周邊中位數＋15%")
    st.caption(f"周邊 1KM 中位價 ${med:,.0f}｜可模擬區間 ${lo:,} ~ ${hi:,}（下限 $50、中位數＋15%）")
    new_price = st.slider("模擬每晚售價 (TWD)", lo, hi, default, step=step, key=key)

    sim_row = listing.copy()
    sim_row["price"] = new_price
    vp_sim = predict_vacancy_prob(sim_row, DF, MDL)
    base_now = (vp["base_lr"] + vp["base_rf"]) / 2
    base_sim = (vp_sim["base_lr"] + vp_sim["base_rf"]) / 2

    pm = st.columns(3)
    pm[0].metric("目前售價", f"${cur_price:,}")
    pm[1].metric("模擬售價", f"${new_price:,}",
                 f"{(new_price - cur_price) / cur_price * 100:+.0f}%", delta_color="off")
    pm[2].metric("基礎空房風險", f"{base_sim*100:.1f}%",
                 f"{(base_sim - base_now)*100:+.1f}%", delta_color="inverse")

    hz = st.columns(3)
    for col, (lab, lrk, rfk) in zip(hz, [("30 天", "lr_30", "rf_30"),
                                         ("60 天", "lr_60", "rf_60"),
                                         ("90 天", "lr_90", "rf_90")]):
        now = (vp[lrk] + vp[rfk]) / 2
        sim = (vp_sim[lrk] + vp_sim[rfk]) / 2
        col.metric(f"{lab}空房機率", f"{sim*100:.1f}%",
                   f"{(sim - now)*100:+.1f}%", delta_color="inverse")

    prices = np.linspace(lo, hi, 25)
    base_r, r30, r60, r90 = [], [], [], []
    for pnew in prices:
        r = listing.copy()
        r["price"] = float(pnew)
        v = predict_vacancy_prob(r, DF, MDL)
        base_r.append((v["base_lr"] + v["base_rf"]) / 2 * 100)
        r30.append((v["lr_30"] + v["rf_30"]) / 2 * 100)
        r60.append((v["lr_60"] + v["rf_60"]) / 2 * 100)
        r90.append((v["lr_90"] + v["rf_90"]) / 2 * 100)
    fig = go.Figure()
    for yv, nm, cl in [(base_r, "基礎", P["primary"]), (r30, "30 天", P["low"]),
                       (r60, "60 天", P["medium"]), (r90, "90 天", P["high"])]:
        fig.add_trace(go.Scatter(x=prices, y=yv, mode="lines", name=nm,
                                 line=dict(width=2)))
    if lo <= cur_price <= hi:
        fig.add_vline(x=cur_price, line_dash="dot", line_color=P["muted"],
                      annotation_text=f"目前 ${cur_price:,}")
    fig.add_vline(x=new_price, line_dash="dash", line_color=P["ink"],
                  annotation_text=f"模擬 ${new_price:,}")
    apply_theme(fig, h=300).update_layout(
        margin=dict(l=50, r=20, t=10, b=36),
        xaxis_title="每晚售價 (TWD)", yaxis_title="預估空房風險 (%)",
        xaxis_range=[lo, hi],
        legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_fig")

    _diff = (base_now - base_sim) * 100
    _pdiff = (cur_price - new_price) / cur_price * 100
    if new_price < cur_price and _diff > 0.1:
        note(f"📉 售價自 ${cur_price:,} 調降至 ${new_price:,}（降 {_pdiff:.0f}%），"
             f"基礎空房風險由 {base_now*100:.1f}% 降至 {base_sim*100:.1f}%"
             f"（約降 {_diff:.1f} 個百分點）。")
    elif new_price > cur_price:
        note(f"📈 售價調高至 ${new_price:,}，基礎空房風險上升至 {base_sim*100:.1f}%；"
             f"若追求穩定出租可考慮維持或調降。")
    else:
        note("拖動上方捲軸即可模擬不同售價對空房風險（基礎與 30/60/90 天）的影響。")


T1, T2, T3, T4, T5 = st.tabs([
    "📊 競爭分析", "🔮 空房預測", "💡 智慧建議", "💬 NLP 評論分析", "🖼 房源圖片分析"
])

# 舊版空房機率（供智慧建議等分頁共用）
vp = predict_vacancy_prob(listing, DF, MDL)

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
          <img src="{listing['picture_url']}" alt="房源照片" referrerpolicy="no-referrer"
               style="width:100%;height:150px;object-fit:cover;border-radius:10px;
               margin-bottom:10px;background:{P['tag_bg']};"
               onerror="this.style.display='none'">
          <div style="font-size:1.05rem;font-weight:700;color:{P['ink']};
               margin-bottom:8px;">{listing['name']}</div>
          <div style="font-size:.78rem;color:{P['muted']};line-height:1.8;">
            🗺 {listing['neighbourhood_cleansed']} ｜
            🛏 {ROOM_JP.get(listing['room_type'], listing['room_type'])}<br>
            📍 {addr}<br>
            💰 每晚 <b style="color:{P['primary']};">${listing['price']:,.0f}</b> ｜
            ⭐ {listing.get('review_scores_rating', 'N/A')} ｜
            💬 {listing['number_of_reviews']} 則評論<br>
            👥 可住 {int(listing.get('accommodates', 0))} 人 ｜
            🛁 {int(listing.get('bathrooms_count', 0))} 衛浴 ｜
            🛏 {int(listing.get('beds', 0))} 床<br>
            {risk_badge(listing['risk_level'])}
            <span style="font-size:.72rem;color:{P['muted']};margin-left:8px;">
              風險分數: {listing['risk_score']:.3f}</span><br>
            <a href="{listing['picture_url']}" target="_blank"
               style="color:{P['primary']};font-weight:700;text-decoration:none;">
               🖼 房源照片 ↗（開新視窗）</a>
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
    import modules.vacancy_model as VM
    sec("空屋率風險預警（雙軌模型 · 依 imp_new 規格）")
    mb("模型A 空屋率 × 模型B 校準高風險機率 · GroupKFold(host_id) 誠實驗證 · POI×7 + NLP 多模態")
    _vrow = VM.get_row(int(sel_id))
    if _vrow is None:
        st.info("此房源不在多模態訓練協定範圍（經營未滿一年或缺座標，共 5,849 筆），"
                "暫無法提供空屋率風險評估。")
    else:
        _sc1, _sc2 = st.columns([1, 1])
        with _sc1:
            _pp = st.slider("每晚房價模擬 (NTD $)", 500, 50000,
                            int(min(50000, max(500, VM.sf_price(_vrow)))), 50, key="vm_price")
            _pm = st.number_input("最低入住天數限制 (晚)", 1, 30,
                                  int(min(30, max(1, VM.sf_int(_vrow, "minimum_nights", 1)))),
                                  key="vm_mn")
        _ov = {"price": _pp, "minimum_nights": _pm}
        _vac, _risk = VM.predict(_vrow, _ov)
        _color = P["high"] if _vac >= 0.7 else (P["medium"] if _vac >= 0.4 else P["low"])
        with _sc2:
            _m1, _m2 = st.columns(2)
            _m1.metric("預估年空屋率（模型A）", f"{_vac*100:.1f}%")
            _m2.metric("高空屋機率（模型B）", f"{_risk*100:.1f}%")
        _fig = go.Figure(go.Indicator(
            mode="gauge+number", value=_vac*100,
            number={"suffix": "%", "font": {"size": 40, "color": _color}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": _color},
                   "steps": [{"range": [0, 40], "color": "#EAF5EE"},
                             {"range": [40, 70], "color": "#FDF5E4"},
                             {"range": [70, 100], "color": "#FDECEA"}]},
            title={"text": "預估年空屋率", "font": {"size": 14, "color": P["muted"]}}))
        _fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260,
                           margin=dict(l=30, r=30, t=50, b=10),
                           font=dict(family="Noto Sans TC,sans-serif"))
        st.plotly_chart(_fig, use_container_width=True)

        sec("🤖 AI 智能診斷（Top-2 扣分項）")
        _tips = VM.diagnose(_vrow, _ov, k=2)
        if _tips:
            for _t in _tips:
                note(f"💡 {_t['zh']}（+{_t['delta']:.2f}% 空屋風險）：{_t['advice']}")
        else:
            note("目前無明顯扣分項，經營狀態良好。")

        sec("各特徵之空屋率加減分貢獻（綠＝降風險 / 紅＝推高風險）")
        for _f, _zh, _d in VM.contributions(_vrow, _ov, top=6):
            _bc = P["high"] if _d > 0 else P["low"]
            _w = min(100, abs(_d) / 6 * 100)
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
                f"<div style='width:150px;text-align:right;font-size:.8rem;color:{P['ink2']};'>{_zh}</div>"
                f"<div style='flex:1;background:{P['tag_bg']};border-radius:6px;height:15px;'>"
                f"<div style='width:{_w:.0f}%;height:15px;border-radius:6px;background:{_bc};'></div></div>"
                f"<div style='width:64px;font-size:.8rem;color:{_bc};font-weight:700;'>"
                f"{'+' if _d>0 else ''}{_d:.2f}%</div></div>",
                unsafe_allow_html=True)

        _M = VM.get_metrics()
        note(f"誠實 GroupKFold(host_id) 5 折：模型A R²={_M['R2']:.3f}、模型B AUC={_M['AUC']:.3f} "
             f"（{_M['n']:,} 筆 · {_M['n_features']} 特徵）。完整沙盒見「後台分析」頁。")


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

    # ── v2 單筆 SHAP 歸因：這間房源為什麼有風險 ──
    st.divider()
    sec("為什麼有風險：SHAP 單筆歸因（v2 模型A）")
    mb("紅色推高空屋率 · 綠色降低 · 單位＝空屋率百分點 · 不需 shap")
    import modules.vacancy_model as VM
    _vr = VM.get_row(int(sel_id))
    if _vr is None:
        st.info("此房源不在多模態協定範圍（經營未滿一年或缺座標），無法提供貢獻拆解。")
    else:
        _cons = VM.contributions(_vr, top=8)
        if not _cons:
            st.info("無足夠特徵可解釋。")
        else:
            _mx = max(abs(d) for _, _, d in _cons) or 1.0
            _rows = ""
            for _f, _zh, _d in _cons:
                _w = abs(_d) / _mx * 46
                if _d >= 0:
                    _bar = (f"<div style='flex:1;'></div><div style='flex:1;position:relative;'>"
                            f"<div style='position:absolute;left:0;height:16px;border-radius:0 6px 6px 0;"
                            f"width:{_w:.1f}%;background:{P['high']};'></div></div>")
                    _val = f"<span style='color:{P['high']};font-weight:700;'>+{_d:.2f}%</span>"
                else:
                    _bar = (f"<div style='flex:1;position:relative;'>"
                            f"<div style='position:absolute;right:0;height:16px;border-radius:6px 0 0 6px;"
                            f"width:{_w:.1f}%;background:{P['low']};'></div></div><div style='flex:1;'></div>")
                    _val = f"<span style='color:{P['low']};font-weight:700;'>{_d:.2f}%</span>"
                _rows += (f"<div style='display:flex;align-items:center;gap:8px;margin:5px 0;'>"
                          f"<div title='{_zh}' style='width:130px;text-align:right;font-size:.8rem;"
                          f"color:{P['ink2']};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{_zh}</div>"
                          f"<div style='flex:1;display:flex;border-left:2px dashed {P['border2']};'>{_bar}</div>"
                          f"<div style='width:62px;font-size:.8rem;'>{_val}</div></div>")
            st.markdown(
                f"<div style='background:{P['surface']};border:1px solid {P['border']};border-radius:12px;"
                f"padding:14px 16px;'><div style='text-align:center;font-size:.72rem;color:{P['muted']};"
                f"margin-bottom:8px;'>綠（左）＝降低空屋風險　紅（右）＝推高空屋風險</div>"
                f"{_rows}</div>", unsafe_allow_html=True)
            note("各特徵把這間房源的預測空屋率往上推或往下拉；先處理推高風險最多、"
                 "且房東可控的因素（定價、最低天數、描述、設施）。")

    # Risk level summary
    st.divider()
    risk_p = max(vp["base_lr"], vp["base_rf"])
    if risk_p < 0.3:
        note("✅ <b>低風險</b>（P < 30%）：房源狀態健康。建議保持現狀，可在續約時考慮小幅調漲。")
    elif risk_p < 0.6:
        note("⚠️ <b>中風險</b>（30% ≤ P < 60%）：出現空房危機信號。建議微調價格、補充照片、優化描述。")
    else:
        note("🔴 <b>高風險</b>（P ≥ 60%）：極可能持續空房。建議立即執行動態定價降價或限時優惠策略。")

    if IMG.get("ok") and IMG["label"] == "模糊":
        st.markdown(f"""
        <div style="background:{P['surface']};border:1px solid {P['border']};
             border-left:4px solid {P['high']};border-radius:0 12px 12px 0;
             padding:16px 20px;margin-bottom:12px;">
          <div style="font-size:.88rem;font-weight:700;color:{P['ink']};margin-bottom:6px;">
            🖼 照片品質建議</div>
          <div style="font-size:.82rem;color:{P['ink2']};line-height:1.7;">
            系統偵測您的封面照片<b style="color:{P['high']};">偏模糊／品質不佳</b>
            （清晰機率僅 {IMG['prob']*100:.0f}%）。模糊或昏暗的照片會明顯降低點閱與預訂率，
            建議您<b>重新上傳</b>一張光線充足、對焦清晰的照片。
          </div>
        </div>""", unsafe_allow_html=True)
        _mail = str(listing.get("host_email")
                    or fake_host_email(listing.get("host_name"), listing.get("host_id")))
        if st.button("📧 發送『請重新上傳照片』通知信給房東", key="mail_btn_t3",
                     use_container_width=True):
            email_dialog(listing["name"], listing.get("host_name", "房東"),
                         _mail, IMG["label"], IMG["prob"])

    if RES_V2 is not None:
        price_simulator_v2("price_sim_t3")
    else:
        price_simulator("price_sim_t3")

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

# ──────────────────────────────────────────────────────────────
# TAB 5: 房源圖片分析（CLIP 可選 · Laplacian + 線性 SHAP）
# ──────────────────────────────────────────────────────────────
with T5:
    sec("房源封面照片清晰度分析")
    mb("Laplacian 失焦偵測 × 線性 SHAP 解釋（CLIP 語意判斷為可選）")
    _url = str(listing.get("picture_url", ""))
    ci, cr = st.columns([1, 1.35])
    with ci:
        _photos = listing_photos(listing)
        if _photos:
            _n = len(_photos)
            _idx = 0
            if _n > 1:                                   # 多張照片：上一張/下一張切換
                _k = "ll_photo_idx"
                _idx = int(st.session_state.get(_k, 0)) % _n
                nv = st.columns([1, 2, 1])
                if nv[0].button("◀ 上一張", key="ll_prev", use_container_width=True):
                    _idx = (_idx - 1) % _n
                    st.session_state[_k] = _idx
                nv[1].markdown(
                    f"<div style='text-align:center;color:{P['muted']};'>第 {_idx+1}/{_n} 張</div>",
                    unsafe_allow_html=True)
                if nv[2].button("下一張 ▶", key="ll_next", use_container_width=True):
                    _idx = (_idx + 1) % _n
                    st.session_state[_k] = _idx
            _cur = _photos[_idx]
            st.image(_cur, use_container_width=True,
                     caption=(f"房源照片 {_idx+1}/{_n}" if _n > 1 else "房源封面照片"))
            if _n > 1:
                _links = " ｜ ".join(
                    f'<a href="{u}" target="_blank" style="color:{P["primary"]};">圖{i+1} ↗</a>'
                    for i, u in enumerate(_photos))
            else:
                _links = (f'<a href="{_cur}" target="_blank" '
                          f'style="color:{P["primary"]};font-weight:700;">'
                          f'🖼 開新視窗檢視原圖 ↗</a>')
            st.markdown(_links, unsafe_allow_html=True)
        else:
            st.info("此房源沒有照片 URL。")
    with cr:
        if not IMG.get("ok"):
            st.warning("無法下載此照片進行分析（連結可能失效或環境網路受限）。"
                       "於 Streamlit Cloud 部署時可正常下載分析。")
        else:
            prob, lab = IMG["prob"], IMG["label"]
            col = P["low"] if lab == "清晰" else (P["medium"] if lab == "尚可" else P["high"])
            st.markdown(f"""
            <div style="background:{P['surface']};border:1px solid {P['border']};
                 border-top:3px solid {col};border-radius:12px;padding:16px 20px;
                 margin-bottom:10px;">
              <div style="font-size:.74rem;color:{P['muted']};letter-spacing:.06em;">
                照片清晰度判定</div>
              <div style="font-size:1.8rem;font-weight:800;color:{col};">{lab}</div>
              <div style="font-size:.78rem;color:{P['muted']};">
                清晰機率 {prob*100:.0f}%</div>
            </div>""", unsafe_allow_html=True)
            mm = st.columns(3)
            mm[0].metric("清晰度 Laplacian", f"{IMG['raw']['laplacian_var']:,.0f}")
            mm[1].metric("解析度", f"{IMG['raw']['megapixels']} MP")
            mm[2].metric("邊緣密度", f"{IMG['raw']['edge_density']*100:.1f}%")

    if IMG.get("ok"):
        sec("SHAP 特徵貢獻（為何判定清晰／模糊）")
        mb("線性精確 SHAP · 正值＝推向清晰，負值＝推向模糊")
        sh = IMG["shap"]
        nm = [n for n, _ in sh]
        vv = [v for _, v in sh]
        cols = [P["low"] if v >= 0 else P["high"] for v in vv]
        fig = go.Figure(go.Bar(x=vv, y=nm, orientation="h",
                               marker=dict(color=cols),
                               text=[f"{v:+.2f}" for v in vv], textposition="outside"))
        fig.add_vline(x=0, line_color=P["muted"])
        apply_theme(fig, h=250, legend=False).update_layout(
            margin=dict(l=130, r=40, t=6, b=30),
            xaxis_title="對『清晰』的貢獻（SHAP 值）")
        st.plotly_chart(fig, use_container_width=True)

        if IMG.get("clip"):
            note(f"🧠 CLIP 語意判斷：清晰 {IMG['clip']['clear']*100:.0f}% ／ "
                 f"模糊 {IMG['clip']['blurry']*100:.0f}%")
        else:
            note("🧠 CLIP（torch / open_clip）未安裝，本頁改用輕量 Laplacian 指標運作；"
                 "在支援的環境（已安裝 torch/open_clip）會自動加入 CLIP 語意判斷。")

        sec("評論佐證（照片是否與實際相符）")
        mb("掃描評論中與外觀／整潔／相符相關的關鍵字")
        _lr = REVIEWS[REVIEWS["listing_id"] == sel_id]
        _kw = ["photo", "picture", "as described", "as pictured", "clean",
               "照片", "跟照片", "如圖", "實際", "乾淨", "整潔", "漂亮", "看起來"]
        _txt = _lr["comments"].astype(str).str.lower() if len(_lr) else None
        _hits = int(_txt.apply(lambda x: any(k in x for k in _kw)).sum()) if _txt is not None else 0
        _acc = listing.get("review_scores_accuracy")
        _cln = listing.get("review_scores_cleanliness")
        note(f"此房源 {len(_lr)} 則評論中，約 <b>{_hits}</b> 則提及外觀／整潔／與描述相符；"
             f"『符合描述』評分 <b>{_acc if pd.notna(_acc) else 'N/A'}</b>、"
             f"『整潔度』評分 <b>{_cln if pd.notna(_cln) else 'N/A'}</b>。"
             f"評分越高代表照片與實際越相符，可作為圖片分析的外部佐證。")
