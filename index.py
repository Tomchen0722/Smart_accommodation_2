"""
智慧旅宿平台 — 動畫首頁（Streamlit 版，全版一頁）

以 components.html 全版嵌入動畫 Hero（來源優先 static/landing_hero.html）。
- 即時從 listings_cleaned.csv.gz 各行政區隨機挑 20 張房源照片。
- 房源統計（房源/評論/行政區/PoI）即時計算後注入 Hero 底部統計列。
- 點照片會開新視窗顯示完整房源詳情（含 AI 照片清晰度）。
執行：  streamlit run index.py
"""
import json
import random
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from modules.data_loader import load_listings, DATA_DIR
from modules.geo_utils import POI_NAMES, load_all_poi, poi_points_within
from modules.image_analysis import analyze, amenities_zh

st.set_page_config(page_title="智慧旅宿平台", page_icon="🏯",
                   layout="wide", initial_sidebar_state="collapsed")

# 全版一頁：隱藏側邊欄／頁首／內距，讓 Hero iframe 佔滿整個視窗
st.markdown("""
<style>
  #MainMenu, footer {display:none !important;}
  header, [data-testid="stHeader"]{display:none !important; height:0 !important;}
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"]{display:none !important;}
  .block-container{padding:0 !important; margin:0 !important; max-width:100% !important;}
  [data-testid="stAppViewContainer"]{padding:0 !important;}
  [data-testid="stMain"] .block-container{padding:0 !important;}
  .main iframe, [data-testid="stAppViewContainer"] iframe{
     height:100vh !important; width:100% !important; border:none !important; display:block;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="準備動畫首頁與房源照片分析 …")
def pick_listings(n=20, seed=7):
    """各行政區隨機挑選房源，並預先分析封面照片清晰度（供詳情視窗顯示）。"""
    DF = load_listings()
    df = DF[DF["picture_url"].astype(str).str.startswith("http")].copy()
    rnd = random.Random(seed)
    picks = []
    for d in sorted(df["neighbourhood_cleansed"].dropna().unique()):
        sub = df[df["neighbourhood_cleansed"] == d]
        if len(sub):
            picks.append(sub.sample(1, random_state=rnd.randint(0, 9999)).iloc[0])
    have = {r["id"] for r in picks}
    pool = df[~df["id"].isin(have)]
    if len(pool):
        for _, r in pool.sample(min(40, len(pool)), random_state=1).iterrows():
            if len(picks) >= n:
                break
            picks.append(r)
    picks = picks[:n]
    POI = load_all_poi()

    def rt(v):
        try:
            return None if str(v) == "nan" else round(float(v), 2)
        except Exception:
            return None

    def gi(v):
        try:
            return int(v) if str(v) != "nan" else 0
        except Exception:
            return 0

    out = []
    for r in picks:
        img = str(r["picture_url"])
        clar = None
        try:
            a = analyze(img)
            if a.get("ok"):
                clar = {"label": a["label"], "prob": round(a["prob"], 3),
                        "lap": a["raw"]["laplacian_var"], "mp": a["raw"]["megapixels"]}
        except Exception:
            clar = None
        ams = amenities_zh(r.get("amenities", "[]"))
        nearby = []
        for t, pdf in POI.items():
            pts = poi_points_within(float(r["latitude"]), float(r["longitude"]), pdf, 1000)
            n0 = pts.iloc[0] if len(pts) else None
            nearby.append({
                "poi": POI_NAMES[t], "count": int(len(pts)),
                "near": (str(n0["poi_name"]) if n0 is not None else "—"),
                "dist": (f"{n0['distance_m']:.0f} m" if n0 is not None else "—")})
        out.append({
            "id": int(r["id"]), "name": str(r["name"])[:70],
            "district": str(r["neighbourhood_cleansed"]), "price": int(r["price"]),
            "img": img, "room": str(r["room_type_zh"]),
            "rating": rt(r.get("review_scores_rating")),
            "lat": round(float(r["latitude"]), 5), "lon": round(float(r["longitude"]), 5),
            "acc": gi(r.get("accommodates")), "bath": gi(r.get("bathrooms_count")),
            "beds": gi(r.get("beds")), "reviews": gi(r.get("number_of_reviews")),
            "clarity": clar, "amenities": ams, "nearby": nearby})
    return out


def _review_count():
    """評論筆數：優先讀 parquet metadata（不載入全部資料）。"""
    try:
        import pyarrow.parquet as pq
        p = DATA_DIR / "reviews_cleaned.parquet"
        if p.exists():
            return int(pq.ParquetFile(p).metadata.num_rows)
    except Exception:
        pass
    try:
        from modules.data_loader import load_reviews
        return int(len(load_reviews()))
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def platform_stats():
    DF = load_listings()
    return {"listings": int(len(DF)), "reviews": _review_count(),
            "districts": int(DF["neighbourhood_cleansed"].nunique()),
            "poi": int(len(POI_NAMES))}


@st.cache_data(show_spinner=False)
def build_hero(listings_json, stats_json):
    base = Path(__file__).parent
    html = ""
    for name in ("static/landing_hero.html", "landing_hero.html", "landing_hero.hmtl"):
        fp = base / name
        if fp.exists():
            html = fp.read_text(encoding="utf-8")
            break
    if not html:
        return "<p style='padding:40px;font-family:sans-serif'>找不到 landing_hero.html</p>"
    html = re.sub(r"const LISTINGS = \[.*?\];",
                  "const LISTINGS = " + listings_json + ";", html, flags=re.S)
    html = re.sub(r"const STATS = \{.*?\};",
                  "const STATS = " + stats_json + ";", html, flags=re.S)
    # iframe 內無法導覽父視窗 → 登入後改為「開新分頁到對應頁面」＋關閉登入視窗
    html = html.replace(
        'if(window.opener && !window.opener.closed){ window.opener.location.href="${APP_URL}"+encodeURIComponent(SLUG); }',
        'window.open("${APP_URL}"+encodeURIComponent(SLUG),"_blank");')
    return html


_hero = build_hero(json.dumps(pick_listings(), ensure_ascii=False),
                   json.dumps(platform_stats(), ensure_ascii=False))
components.html(_hero, height=900, scrolling=False)
