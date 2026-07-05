"""
租客入口 — Tenant Portal
房源搜尋 · 便利性篩選 · 生活圈分析 · 價格評價 · NLP 評論摘要
每個房源可點擊查看詳情彈窗；每個 PoI 總數可點擊查看完整明細。
"""
import ast
import html as _html
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.ui_components import (
    inject_css, P, RTC, ROOM_JP,
    sec, mb, note, stat_card, html_table, apply_theme,
    review_hover_html, sidebar_nav,
)
from modules.data_loader import load_listings, load_reviews
from modules.geo_utils import (
    load_all_poi, count_poi_within, nearest_poi, poi_points_within,
    convenience_score, POI_NAMES,
)
from modules.nlp_analysis import listing_review_summary, recent_review_snippets

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="租客入口 — 智慧旅宿", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()

# ─── Load data ──────────────────────────────────────────────────
with st.spinner("載入房源資料 …"):
    DF = load_listings()
    REVIEWS = load_reviews()
    POI_ALL = load_all_poi()

# ─── Dialog support (works across Streamlit versions) ───────────
_DIALOG = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)


def _amenity_list(raw):
    """Parse the amenities column (JSON-ish string) into a list."""
    try:
        items = ast.literal_eval(raw) if isinstance(raw, str) else []
        return [str(a) for a in items if str(a).strip()]
    except Exception:
        return []


def render_listing_detail(L):
    """Full listing detail body used inside the dialog / expander."""
    lat, lon = L["latitude"], L["longitude"]
    rating = L.get("review_scores_rating")
    rating_s = f"{rating:.2f}" if pd.notna(rating) else "N/A"
    area = L["neighbourhood_cleansed"]
    hood = L.get("neighbourhood", "") if pd.notna(L.get("neighbourhood", "")) else ""
    st.markdown(f"""
    <div style="font-size:1.05rem;font-weight:700;color:{P['ink']};
         margin-bottom:6px;">{L['name']}</div>
    <div style="font-size:.8rem;color:{P['muted']};line-height:1.9;">
      📍 <b>區域位置：</b>{area}{('｜' + str(hood)) if hood else ''}<br>
      🧭 座標：{lat:.5f}, {lon:.5f}<br>
      🛏 {L['room_type_zh']} ｜ 👥 可住 {int(L.get('accommodates', 0))} 人 ｜
      🛁 {int(L.get('bathrooms_count', 0))} 衛浴 ｜ 🛏 {int(L.get('beds', 0))} 床<br>
      💰 <b style="color:{P['tenant']};font-size:1.05rem;">${L['price']:,.0f}</b> / 晚
      ｜ ⭐ {rating_s} ｜ 💬 {int(L['number_of_reviews'])} 則評論
    </div>
    """, unsafe_allow_html=True)

    # ── Amenities ──
    ams = _amenity_list(L.get("amenities", "[]"))
    sec(f"房源設施（共 {len(ams)} 項）")
    if ams:
        chips = "".join(
            f'<span style="display:inline-block;background:{P["tag_bg"]};'
            f'border:1px solid {P["border"]};border-radius:14px;padding:3px 11px;'
            f'margin:3px 4px 0 0;font-size:.74rem;color:{P["ink2"]};">{a}</span>'
            for a in ams
        )
        st.markdown(f"<div style='line-height:2.1;'>{chips}</div>",
                    unsafe_allow_html=True)
    else:
        st.caption("此房源未列出設施資料。")

    # ── Nearby facilities ──
    sec("周遭附近設施（1KM 範圍）")
    rows = []
    for t, pdf in POI_ALL.items():
        pts = poi_points_within(lat, lon, pdf, 1000)
        if len(pts):
            n = pts.iloc[0]
            rows.append({"設施類型": POI_NAMES[t], "1KM 數量": len(pts),
                         "最近地點": n["poi_name"],
                         "最近距離": f"{n['distance_m']:.0f} m"})
        else:
            rows.append({"設施類型": POI_NAMES[t], "1KM 數量": 0,
                         "最近地點": "—", "最近距離": "—"})
    html_table(pd.DataFrame(rows), fmt={"1KM 數量": "{:,.0f}"}, height=280)

    # ── Recent reviews ──
    sec("近期評論")
    _sn = recent_review_snippets(REVIEWS, L["id"], n=8)
    if _sn:
        items = "".join(
            f'<div class="rv-item">{_html.escape(str(x))}</div>' for x in _sn)
        st.markdown(
            f'<div style="max-height:220px;overflow-y:auto;border:1px solid '
            f'{P["border"]};border-radius:10px;padding:8px 12px;font-size:.74rem;'
            f'color:{P["ink2"]};line-height:1.6;">{items}</div>',
            unsafe_allow_html=True)
    else:
        st.caption("此房源尚無評論。")


def render_poi_list(title, pts):
    """Full detail list for one PoI type inside the dialog / expander."""
    st.markdown(f"**{title}** — 1KM 範圍內共 <b style='color:{P['tenant']}'>"
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


# Register dialogs (or expander fallback)
if _DIALOG:
    @_DIALOG("🏠 房源詳情")
    def listing_dialog(L):
        render_listing_detail(L)

    @_DIALOG("📋 設施明細")
    def poi_dialog(title, pts):
        render_poi_list(title, pts)

    @_DIALOG("💬 房源評論")
    def reviews_dialog(snips):
        render_reviews(snips)
else:
    def listing_dialog(L):
        with st.expander("🏠 房源詳情", expanded=True):
            render_listing_detail(L)

    def poi_dialog(title, pts):
        with st.expander("📋 設施明細", expanded=True):
            render_poi_list(title, pts)

    def reviews_dialog(snips):
        with st.expander("💬 房源評論", expanded=True):
            render_reviews(snips)


# ─── Header ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:6px 0 14px;">
  <h1 style="font-size:1.4rem;font-weight:700;color:{P['ink']};
       margin:0;letter-spacing:-.3px;">🔍 智能找房</h1>
  <p style="font-size:.78rem;color:{P['muted']};margin:4px 0 0;">
    自訂便利性權重 → 空間距離計算 → 綜合評分排序 → 智能推薦房源
  </p>
</div>
<hr style="margin:0 0 16px;">
""", unsafe_allow_html=True)

# ─── Sidebar: search filters ────────────────────────────────────
with st.sidebar:
    sidebar_nav()
    st.markdown(f"""
    <div style="padding:4px 0 12px;">
      <div style="font-size:1rem;font-weight:700;color:{P['ink']};">
        🔍 尋房條件</div>
      <div style="font-size:.72rem;color:{P['muted']};margin-top:2px;">
        設定條件與生活機能權重</div>
    </div>""", unsafe_allow_html=True)

    all_nb = sorted(DF["neighbourhood_cleansed"].dropna().unique())
    sel_nbs = st.multiselect("🗺 行政區（可複選）", all_nb, default=all_nb[:1])

    rt_options = sorted(DF["room_type_zh"].dropna().unique())
    sel_rts = st.multiselect("🛏 房型", rt_options, default=rt_options)

    price_min, price_max = int(DF["price"].min()), int(DF["price"].quantile(0.95))
    sel_price = st.slider("💰 每晚預算 (TWD)", price_min, price_max,
                          (price_min, min(5000, price_max)), step=200)

    min_reviews = st.slider("💬 最少評論數", 0, 50, 3, step=1)
    min_rating = st.slider("⭐ 最低評分", 0.0, 5.0, 0.0, step=0.5)

    st.divider()
    st.markdown(f"""<div style="font-size:.78rem;font-weight:700;
         color:{P['tenant']};margin-bottom:2px;">🎚 生活機能權重</div>
      <div style="font-size:.68rem;color:{P['muted']};margin-bottom:6px;">
        數值越高代表您越重視該項便利性</div>""", unsafe_allow_html=True)
    w_mrt = st.slider("🚇 捷運站", 0, 5, 3)
    w_bus = st.slider("🚏 公車站", 0, 5, 2)
    w_conv = st.slider("🏪 超商", 0, 5, 3)
    w_rest = st.slider("🍜 餐廳", 0, 5, 2)
    w_school = st.slider("🏫 學校", 0, 5, 1)
    w_clinic = st.slider("🏥 診所", 0, 5, 1)
    w_park = st.slider("🌳 公園", 0, 5, 2)

    st.divider()
    top_n = st.slider("📋 推薦數量", 5, 30, 12, step=1)
    st.caption("© 2026 智慧旅宿 AI 平台")

WEIGHTS = {"mrt": w_mrt, "bus": w_bus, "convenience": w_conv,
           "restaurant": w_rest, "school": w_school,
           "clinic": w_clinic, "park": w_park}

# ─── Apply filters ──────────────────────────────────────────────
flt = DF.copy()
if sel_nbs:
    flt = flt[flt["neighbourhood_cleansed"].isin(sel_nbs)]
if sel_rts:
    flt = flt[flt["room_type_zh"].isin(sel_rts)]
flt = flt[flt["price"].between(sel_price[0], sel_price[1])]
flt = flt[flt["number_of_reviews"] >= min_reviews]
if min_rating > 0:
    flt = flt[flt["review_scores_rating"].fillna(0) >= min_rating]

if flt.empty:
    st.warning("😢 找不到符合條件的房源，請放寬篩選條件。")
    st.stop()

# Cap the candidate pool for performance, prefer better-reviewed listings
CANDIDATE_CAP = 400
if len(flt) > CANDIDATE_CAP:
    flt = flt.sort_values("number_of_reviews", ascending=False).head(CANDIDATE_CAP)


# ─── Convenience scoring (weighted) ─────────────────────────────
@st.cache_data(show_spinner=False)
def score_candidates(ids, lats, lons, weights):
    """Compute weighted convenience score for candidate listings."""
    rows = []
    for lid, la, lo in zip(ids, lats, lons):
        raw = convenience_score(la, lo, POI_ALL)  # per-type 0/2/5/10
        weighted = sum(raw[t] * weights[t] for t in weights)
        max_w = sum(10 * w for w in weights.values()) or 1
        rows.append({
            "id": lid,
            "conv_raw": raw["total"],
            "conv_weighted": weighted,
            "conv_pct": round(weighted / max_w * 100, 1),
            **{f"s_{t}": raw[t] for t in weights},
        })
    return pd.DataFrame(rows)


with st.spinner(f"計算 {len(flt)} 間候選房源的生活機能分數 …"):
    scores = score_candidates(
        flt["id"].tolist(),
        flt["latitude"].tolist(),
        flt["longitude"].tolist(),
        WEIGHTS,
    )

flt = flt.merge(scores, on="id", how="left")
flt = flt.sort_values("conv_weighted", ascending=False).reset_index(drop=True)
top = flt.head(top_n).copy()

# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════
T1, T2 = st.tabs(["🏘 推薦房源清單", "🔎 房源詳情分析"])

# ──────────────────────────────────────────────────────────────
# TAB 1: Recommended listings
# ──────────────────────────────────────────────────────────────
with T1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("符合房源", f"{len(flt):,} 間")
    k2.metric("中位價格", f"${flt['price'].median():,.0f}")
    k3.metric("平均評分", f"{flt['review_scores_rating'].mean():.2f}"
              if flt['review_scores_rating'].notna().any() else "–")
    k4.metric("平均機能分", f"{flt['conv_pct'].mean():.1f}%")

    st.divider()

    col_list, col_map = st.columns([1.5, 1.5])

    with col_map:
        sec("推薦房源地圖")
        mb("Haversine 空間計算 · 便利性加權評分")
        fig = px.scatter_mapbox(
            top, lat="latitude", lon="longitude",
            hover_name="name",
            hover_data={"price": ":,.0f", "conv_pct": True,
                        "room_type_zh": True,
                        "latitude": False, "longitude": False},
            labels={"price": "每晚價格", "conv_pct": "機能分(%)",
                    "room_type_zh": "房型", "name": "房源"},
            color="conv_pct", size="conv_pct",
            color_continuous_scale=[[0, P["high"]], [0.5, P["medium"]], [1, P["tenant"]]],
            size_max=13, zoom=12, height=520,
            mapbox_style="carto-positron",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="機能分%", len=0.7),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_list:
        sec("智能推薦排序（點按房源查看詳情）")
        mb("綜合評分 = Σ(便利性得分 × 您的權重)")
        region_median = DF.groupby("neighbourhood_cleansed")["price"].median()
        for _, r in top.iterrows():
            reg_med = region_median.get(r["neighbourhood_cleansed"], r["price"])
            diff = (r["price"] - reg_med) / reg_med * 100 if reg_med else 0
            if diff < -5:
                price_tag = f'<span style="color:{P["low"]};font-weight:700;">高 CP 值 ▼{abs(diff):.0f}%</span>'
            elif diff > 15:
                price_tag = f'<span style="color:{P["high"]};font-weight:700;">偏高 ▲{diff:.0f}%</span>'
            else:
                price_tag = f'<span style="color:{P["muted"]};">合理區間</span>'
            rating = r.get("review_scores_rating")
            rating_s = f"⭐ {rating:.2f}" if pd.notna(rating) else "⭐ 無評分"
            st.markdown(f"""
            <div style="background:{P['surface']};border:1px solid {P['border']};
                 border-left:4px solid {P['tenant']};border-radius:0 10px 10px 0;
                 padding:11px 15px;margin-bottom:6px;">
              <div style="display:flex;justify-content:space-between;align-items:baseline;">
                <div style="font-size:.85rem;font-weight:700;color:{P['ink']};
                     max-width:72%;overflow:hidden;text-overflow:ellipsis;
                     white-space:nowrap;">#{r['id']} {r['name']}</div>
                <div style="font-size:.9rem;font-weight:700;color:{P['tenant']};">
                     {r['conv_pct']:.1f}<span style="font-size:.6rem;">分</span></div>
              </div>
              <div style="font-size:.74rem;color:{P['muted']};margin-top:3px;line-height:1.6;">
                🗺 {r['neighbourhood_cleansed']} ｜ 🛏 {r['room_type_zh']} ｜ {rating_s}<br>
                💰 <b style="color:{P['ink']};">${r['price']:,.0f}</b>/晚 · {price_tag}
                ｜ 💬 {int(r['number_of_reviews'])} 則
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"🔍 查看詳情 #{r['id']}", key=f"det_{r['id']}",
                         use_container_width=True):
                listing_dialog(r)

# ──────────────────────────────────────────────────────────────
# TAB 2: Listing detail
# ──────────────────────────────────────────────────────────────
with T2:
    detail_opts = {
        f"#{r.id} | {r['name'][:34]} | ${r.price:,.0f} | {r.conv_pct:.1f}分": r.id
        for _, r in top.iterrows()
    }
    sel_label = st.selectbox("選擇房源查看詳細分析", list(detail_opts.keys()))
    sel_id = detail_opts[sel_label]
    L = flt[flt["id"] == sel_id].iloc[0]
    lat, lon = L["latitude"], L["longitude"]
    snips = recent_review_snippets(REVIEWS, sel_id, n=10)

    if st.button("🗂 開啟完整房源詳情視窗", key="open_detail_dialog"):
        listing_dialog(L)

    # ── Info + amenity radar ──
    ci, cr = st.columns([1.1, 1.3])
    with ci:
        sec("房源核心資訊")
        rating = L.get("review_scores_rating")
        rating_s = f"{rating:.2f}" if pd.notna(rating) else "N/A"
        st.markdown(f"""
        <div style="background:{P['surface']};border:1px solid {P['border']};
             border-radius:12px;padding:18px 22px;margin-bottom:12px;">
          <div style="font-size:1.02rem;font-weight:700;color:{P['ink']};
               margin-bottom:8px;">{L['name']}</div>
          <div style="font-size:.78rem;color:{P['muted']};line-height:1.9;">
            🗺 {L['neighbourhood_cleansed']} ｜ 🛏 {L['room_type_zh']}<br>
            💰 每晚 <b style="color:{P['tenant']};">${L['price']:,.0f}</b> ｜
            ⭐ {rating_s} ｜ 💬 {int(L['number_of_reviews'])} 則評論<br>
            👥 可住 {int(L.get('accommodates', 0))} 人 ｜
            🛁 {int(L.get('bathrooms_count', 0))} 衛浴 ｜
            🛏 {int(L.get('beds', 0))} 床<br>
            🎯 生活機能綜合分：
            <b style="color:{P['tenant']};">{L['conv_pct']:.1f}%</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"💬 查看 {int(L['number_of_reviews'])} 則評論",
                     key="rev_btn_tenant", use_container_width=True):
            reviews_dialog(snips)

        # Price evaluation vs region
        reg_med = DF[DF["neighbourhood_cleansed"] == L["neighbourhood_cleansed"]]["price"].median()
        diff = (L["price"] - reg_med) / reg_med * 100 if reg_med else 0
        if diff < -5:
            note(f"💡 <b>高 CP 值推薦！</b>此房源租金 ${L['price']:,.0f}，"
                 f"低於 {L['neighbourhood_cleansed']}中位數 ${reg_med:,.0f} 約 {abs(diff):.0f}%。")
        elif diff > 15:
            note(f"⚠️ 此房源租金 ${L['price']:,.0f} 高於區域中位數 ${reg_med:,.0f} 約 {diff:.0f}%，"
                 f"可留意是否物有所值。")
        else:
            note(f"✅ 此房源租金 ${L['price']:,.0f} 落在區域合理區間內"
                 f"（中位數 ${reg_med:,.0f}）。")

    with cr:
        sec("便利性雷達圖")
        mb("Amenity Score · 各類生活機能得分")
        cats = [POI_NAMES[t].split(" ")[-1] for t in WEIGHTS]
        vals = [L[f"s_{t}"] for t in WEIGHTS]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor="rgba(91,158,115,.25)",
            line=dict(color=P["tenant"], width=2),
            name="本房源",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 10], showticklabels=True,
                                gridcolor=P["border"], tickfont=dict(size=9)),
                angularaxis=dict(tickfont=dict(size=11, color=P["ink2"])),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(l=40, r=40, t=30, b=30), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── PoI details (clickable totals -> full-list dialog) ──
    sec("周邊生活機能明細（點按總數查看完整清單）")
    mb("1KM 範圍內設施數量與最近距離")
    poi_pts_cache = {t: poi_points_within(lat, lon, pdf, 1000)
                     for t, pdf in POI_ALL.items()}
    poi_cols = st.columns(len(POI_ALL))
    for i, (ptype, pdf) in enumerate(POI_ALL.items()):
        pts = poi_pts_cache[ptype]
        cnt = len(pts)
        nn = pts.iloc[0] if cnt else None
        with poi_cols[i]:
            stat_card(
                f"{cnt}", POI_NAMES[ptype],
                color=P["low"] if cnt >= 5 else (P["medium"] if cnt >= 2 else P["high"]),
            )
            if nn is not None:
                _ad = f" · {nn['poi_addr']}" if nn["poi_addr"] else ""
                st.caption(f"最近：{nn['poi_name']}{_ad}（{nn['distance_m']:.0f}m）")
            else:
                st.caption("無資料")
            if st.button(f"📋 全部 {cnt} 筆", key=f"poi_{ptype}",
                         use_container_width=True):
                poi_dialog(POI_NAMES[ptype], pts)

    # ── Amenity narrative ──
    conv_rank = (flt["conv_pct"] < L["conv_pct"]).mean() * 100
    highlights = [f"{POI_NAMES[t]}僅 {poi_pts_cache[t].iloc[0]['distance_m']:.0f}m"
                  for t in POI_ALL
                  if len(poi_pts_cache[t]) and poi_pts_cache[t].iloc[0]["distance_m"] <= 300]
    hl_txt = "、".join(highlights) if highlights else "周邊生活機能一般"
    note(f"🏆 本房源生活便利性超越篩選範圍內 <b>{conv_rank:.0f}%</b> 的房源。"
         f"亮點：{hl_txt}。")

    # ── Map (hover shows address + distance to listing) ──
    sec("周遭地圖（滑鼠移到設施可見地址與距離）")
    mb("800m 範圍 · hover 顯示地址與距房源距離")
    frames = []
    for t, pdf in POI_ALL.items():
        pts = poi_points_within(lat, lon, pdf, 800)
        if len(pts):
            pts = pts.head(40).copy()
            pts["類型"] = POI_NAMES[t]
            frames.append(pts)
    me = pd.DataFrame([{
        "poi_name": L["name"], "poi_addr": f"{L['neighbourhood_cleansed']}（本房源）",
        "latitude": lat, "longitude": lon, "distance_m": 0.0, "類型": "📍 本房源",
    }])
    map_df = pd.concat([me] + frames, ignore_index=True) if frames else me
    map_df["距離房源"] = map_df["distance_m"].map(lambda d: f"{d:.0f} m")
    # Make the listing marker markedly larger than PoI markers
    map_df["_sz"] = (map_df["類型"] == "📍 本房源").map({True: 2.6, False: 1.0})
    fig = px.scatter_mapbox(
        map_df, lat="latitude", lon="longitude", hover_name="poi_name",
        hover_data={"poi_addr": True, "距離房源": True, "類型": True, "_sz": False,
                    "latitude": False, "longitude": False, "distance_m": False},
        labels={"poi_addr": "地址"},
        color="類型", size="_sz", size_max=13, zoom=15, height=460,
        mapbox_style="carto-positron", center={"lat": lat, "lon": lon},
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(bgcolor=P["surface"], bordercolor=P["border"],
                                  borderwidth=1, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    # ── NLP review summary ──
    st.divider()
    sec("NLP 評論摘要")
    mb("VADER 情感分析 × jieba 中文分詞")
    lr = REVIEWS[REVIEWS["listing_id"] == sel_id]
    if len(lr) == 0:
        st.info("此房源尚無評論資料。")
    else:
        with st.spinner(f"分析 {len(lr)} 則評論 …"):
            summ = listing_review_summary(REVIEWS, sel_id)
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("評論總數", f"{summ['total_reviews']:,}")
        n2.metric("😊 正面", f"{summ['pos_pct']}%")
        n3.metric("😐 中立", f"{summ['neu_pct']}%")
        n4.metric("😞 負面", f"{summ['neg_pct']}%")

        cds, ckw = st.columns([1, 1.3])
        with cds:
            sent_data = pd.DataFrame({
                "情感": ["正面", "中立", "負面"],
                "比例": [summ["pos_pct"], summ["neu_pct"], summ["neg_pct"]],
            })
            fig = px.pie(sent_data, values="比例", names="情感", color="情感",
                         color_discrete_map={"正面": P["low"], "中立": P["muted"],
                                             "負面": P["high"]}, hole=0.6)
            fig.update_traces(textfont_size=11, marker_line_width=2,
                              marker_line_color=P["bg"])
            apply_theme(fig, h=250).update_layout(margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(fig, use_container_width=True)
        with ckw:
            if summ["pos_keywords"]:
                kw_df = pd.DataFrame(summ["pos_keywords"], columns=["關鍵字", "次數"])
                fig = go.Figure(go.Bar(
                    x=kw_df["次數"], y=kw_df["關鍵字"], orientation="h",
                    marker=dict(color=P["low"])))
                apply_theme(fig, h=250, legend=False).update_layout(
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=80, r=20, t=5, b=25),
                    title=dict(text="✅ 正面關鍵字", font=dict(size=12)))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("無足夠正面評論提取關鍵字")

        if summ["sample_pos"]:
            st.markdown(f"""<div class="note" style="border-left-color:{P['low']};">
              <b>😊 正面評論摘錄：</b><br>{summ['sample_pos']}</div>""",
                        unsafe_allow_html=True)
        if summ["sample_neg"]:
            st.markdown(f"""<div class="note" style="border-left-color:{P['high']};">
              <b>😞 貼心提示（負面評論）：</b><br>{summ['sample_neg']}</div>""",
                        unsafe_allow_html=True)
