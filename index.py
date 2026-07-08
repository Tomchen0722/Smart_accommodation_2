"""
智慧旅宿平台 — 動畫首頁（Streamlit 版）

以 components.html 嵌入動畫 Hero（來源 landing_hero.html），並在下方放上
平台統計數據（房源／評論／行政區／PoI 資料源）。
執行方式：  streamlit run index.py
"""
import json
import random
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from modules.data_loader import load_listings, DATA_DIR
from modules.geo_utils import POI_NAMES

st.set_page_config(page_title="智慧旅宿平台", page_icon="🏯",
                   layout="wide", initial_sidebar_state="collapsed")

# 首頁走全版、隱藏 Streamlit 預設外框
st.markdown("""
<style>
  #MainMenu, header, footer {visibility:hidden;}
  [data-testid="stSidebarNav"]{display:none;}
  .block-container{padding:0 0 1rem 0 !important; max-width:100% !important;}
  [data-testid="stAppViewBlockContainer"]{padding-top:0 !important;}
  iframe{border:none !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def pick_listings(n=20, seed=7):
    """從各行政區隨機挑選 20 筆有照片的房源（資料：listings_cleaned.csv.gz）。"""
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

    def rt(v):
        try:
            return None if str(v) == "nan" else round(float(v), 2)
        except Exception:
            return None
    return [{"id": int(r["id"]), "name": str(r["name"])[:60],
             "district": str(r["neighbourhood_cleansed"]),
             "price": int(r["price"]), "img": str(r["picture_url"]),
             "room": str(r["room_type_zh"]),
             "rating": rt(r.get("review_scores_rating"))} for r in picks]


@st.cache_data(show_spinner=False)
def build_hero(listings_json):
    """讀取 landing_hero.html，注入即時房源，並調整為 iframe 內可用的導覽。"""
    base = Path(__file__).parent
    html = ""
    for name in ("landing_hero.html", "static/landing_hero.html", "landing_hero.hmtl"):
        fp = base / name
        if fp.exists():
            html = fp.read_text(encoding="utf-8")
            break
    if not html:
        return "<p style='padding:40px;font-family:sans-serif'>找不到 landing_hero.html</p>"
    html = re.sub(r"const LISTINGS = \[.*?\];",
                  "const LISTINGS = " + listings_json + ";", html, flags=re.S)
    # iframe 內無法導覽父視窗 → 登入後改為「開新分頁到對應頁面」＋關閉登入視窗
    html = html.replace(
        'if(window.opener && !window.opener.closed){ window.opener.location.href="${APP_URL}"+encodeURIComponent(SLUG); }',
        'window.open("${APP_URL}"+encodeURIComponent(SLUG),"_blank");')
    return html


hero = build_hero(json.dumps(pick_listings(), ensure_ascii=False))
components.html(hero, height=720, scrolling=False)

def _review_count():
    """評論筆數：優先讀 parquet metadata（不載入全部資料），否則退回 gz。"""
    try:
        import pyarrow.parquet as pq
        pth = DATA_DIR / "reviews_cleaned.parquet"
        if pth.exists():
            return int(pq.ParquetFile(pth).metadata.num_rows)
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
    return (int(len(DF)),
            _review_count(),
            int(DF["neighbourhood_cleansed"].nunique()),
            int(len(POI_NAMES)))


_n_listings, _n_reviews, _n_dist, _n_poi = platform_stats()

# ─── 平台統計數據（做進首頁）──────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:6px 0 24px;font-family:'Noto Sans TC',sans-serif;">
  <div style="display:flex;justify-content:center;gap:46px;flex-wrap:wrap;margin-bottom:16px;">
    <div><div style="font-size:1.7rem;font-weight:800;color:#4E7FB0;">{_n_listings:,}</div>
      <div style="font-size:.72rem;color:#9A9490;letter-spacing:.06em;">房源資料</div></div>
    <div><div style="font-size:1.7rem;font-weight:800;color:#5B9E73;">{_n_reviews:,}</div>
      <div style="font-size:.72rem;color:#9A9490;letter-spacing:.06em;">評論文本</div></div>
    <div><div style="font-size:1.7rem;font-weight:800;color:#8B7BA8;">{_n_dist}</div>
      <div style="font-size:.72rem;color:#9A9490;letter-spacing:.06em;">行政區</div></div>
    <div><div style="font-size:1.7rem;font-weight:800;color:#C49A4A;">{_n_poi}</div>
      <div style="font-size:.72rem;color:#9A9490;letter-spacing:.06em;">PoI 資料源</div></div>
  </div>
  <hr style="border:none;border-top:1px solid #E8E4DE;width:250px;margin:0 auto 12px;">
  <p style="font-size:.72rem;color:#9A9490;line-height:1.9;">
    資料來源：Inside Airbnb · 台北市政府開放資料<br>
    ML：Logistic Regression × Random Forest ｜ NLP：VADER × jieba<br>
    © 2026 智慧旅宿 AI 平台</p>
</div>
""", unsafe_allow_html=True)
