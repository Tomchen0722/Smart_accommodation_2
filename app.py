"""
智慧旅宿平台 — 首頁入口
和風ミニマル · 日系簡約風
"""
import streamlit as st
from modules.ui_components import inject_css, P, sidebar_nav

st.set_page_config(
    page_title="智慧旅宿平台",
    page_icon="🏯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()

with st.sidebar:
    sidebar_nav()

# ─── Landing page ───────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:12px 0 14px;">
  <div style="font-size:3.2rem;margin-bottom:8px;">🏯</div>
  <h1 style="font-size:1.8rem;font-weight:700;color:{P['ink']};
       margin:0;letter-spacing:-.5px;">智慧旅宿平台</h1>
  <p style="font-size:.88rem;color:{P['muted']};margin:8px 0 0;
       letter-spacing:.08em;">
    Smart Accommodation · 台北市 Airbnb 數據分析 · NLP 文本智能
  </p>
  <div style="width:60px;height:2px;background:{P['primary']};
       margin:16px auto 0;border-radius:1px;"></div>
</div>
""", unsafe_allow_html=True)

# ─── Hero banner (split landlord / tenant, redesigned) ─────────
st.markdown(f"""
<div class="hero">
  <div class="hero-seam"></div>
  <div class="hero-half hero-l">
    <div class="hero-txt">
      <span class="hero-tag">房東 · LANDLORD</span>
      <h2>我的房子<br>租得出去嗎？</h2>
      <p>空房預測 · 1KM 競爭分析 · 智慧定價與經營建議</p>
    </div>
    <svg class="hero-sky" viewBox="0 0 500 134"
         preserveAspectRatio="xMidYMax slice"
         xmlns="http://www.w3.org/2000/svg"><g fill="#D2CFC9"><rect x="6" y="52" width="46" height="82"/><rect x="60" y="30" width="34" height="104"/><rect x="104" y="66" width="40" height="68"/><rect x="152" y="44" width="30" height="90"/><rect x="190" y="72" width="54" height="62"/><rect x="252" y="24" width="30" height="110"/><rect x="290" y="58" width="44" height="76"/><rect x="342" y="40" width="34" height="94"/><rect x="384" y="66" width="50" height="68"/><rect x="442" y="50" width="42" height="84"/></g><g fill="#BCB8B2"><rect x="60" y="30" width="10" height="104"/><rect x="252" y="24" width="9" height="110"/><rect x="342" y="40" width="9" height="94"/></g><g fill="#FBFAF8" opacity=".75"><rect x="66" y="42" width="6" height="7"/><rect x="78" y="42" width="6" height="7"/><rect x="66" y="58" width="6" height="7"/><rect x="78" y="58" width="6" height="7"/><rect x="258" y="38" width="6" height="7"/><rect x="270" y="38" width="6" height="7"/><rect x="258" y="54" width="6" height="7"/><rect x="270" y="54" width="6" height="7"/><rect x="298" y="70" width="6" height="6"/><rect x="310" y="70" width="6" height="6"/></g><rect x="0" y="130" width="500" height="4" fill="#2A2A2A" opacity=".55"/></svg>
  </div>
  <div class="hero-half hero-r">
    <div class="hero-txt">
      <span class="hero-tag">租客 · TENANT</span>
      <h2>我想要<br>租個好房子</h2>
      <p>生活圈媒合 · 便利性篩選 · 評價與價格透明</p>
    </div>
    <svg class="hero-sky" viewBox="0 0 500 134"
         preserveAspectRatio="xMidYMax slice"
         xmlns="http://www.w3.org/2000/svg"><g><rect x="6" y="54" width="46" height="80" fill="#F28FB0"/><rect x="60" y="28" width="34" height="106" fill="#F4A64B"/><rect x="104" y="64" width="40" height="70" fill="#A98BD0"/><rect x="152" y="42" width="30" height="92" fill="#6FA8DC"/><rect x="190" y="70" width="54" height="64" fill="#F6C445"/><rect x="252" y="22" width="30" height="112" fill="#EE6F97"/><rect x="290" y="56" width="44" height="78" fill="#F49A3C"/><rect x="342" y="38" width="34" height="96" fill="#7E9BE0"/><rect x="384" y="64" width="50" height="70" fill="#B58BD6"/><rect x="442" y="48" width="42" height="86" fill="#F2C13E"/></g><g fill="#ffffff" opacity=".5"><rect x="66" y="40" width="6" height="8"/><rect x="80" y="40" width="6" height="8"/><rect x="66" y="58" width="6" height="8"/><rect x="80" y="58" width="6" height="8"/><rect x="258" y="36" width="6" height="8"/><rect x="270" y="36" width="6" height="8"/><rect x="258" y="54" width="6" height="8"/><rect x="270" y="54" width="6" height="8"/><rect x="350" y="52" width="6" height="7"/><rect x="362" y="52" width="6" height="7"/></g><rect x="0" y="130" width="500" height="4" fill="#1f1f1f" opacity=".5"/></svg>
  </div>
</div>
""", unsafe_allow_html=True)

hcta1, hcta2 = st.columns(2, gap="large")
with hcta1:
    if st.button("↓ 進入房東面板", key="hero_go_landlord", use_container_width=True):
        st.switch_page("pages/1_🏠_房東入口.py")
with hcta2:
    if st.button("↓ 開始找房", key="hero_go_tenant", use_container_width=True):
        st.switch_page("pages/2_🔍_租客入口.py")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ─── Three portal cards ────────────────────────────────────────
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown(f"""
    <div class="portal-card" style="border-top:3px solid {P['landlord']};">
      <div class="portal-icon">🏠</div>
      <div class="portal-title">房東入口</div>
      <div class="portal-desc">
        房源管理與智慧分析<br>
        1KM 競爭數據 · 空房預測<br>
        三維度經營建議 · NLP 評論
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("進入房東面板 →", key="btn_landlord", use_container_width=True):
        st.switch_page("pages/1_🏠_房東入口.py")

with c2:
    st.markdown(f"""
    <div class="portal-card" style="border-top:3px solid {P['tenant']};">
      <div class="portal-icon">🔍</div>
      <div class="portal-title">租客入口</div>
      <div class="portal-desc">
        智能房源搜尋與推薦<br>
        便利性篩選 · 生活圈分析<br>
        價格評價 · NLP 評論摘要
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("開始找房 →", key="btn_tenant", use_container_width=True):
        st.switch_page("pages/2_🔍_租客入口.py")

with c3:
    st.markdown(f"""
    <div class="portal-card" style="border-top:3px solid {P['admin']};">
      <div class="portal-icon">📊</div>
      <div class="portal-title">後台分析系統</div>
      <div class="portal-desc">
        全站 KPI 監控儀表板<br>
        AI 模型分析 · 市場熱圖<br>
        NLP 全站文本分析
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("進入後台 →", key="btn_admin", use_container_width=True):
        st.switch_page("pages/3_📊_後台分析.py")

# ─── Bottom info ────────────────────────────────────────────────
st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;padding:20px 0;">
  <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;margin-bottom:16px;">
    <div style="text-align:center;">
      <div style="font-size:1.6rem;font-weight:700;color:{P['primary']};">6,241</div>
      <div style="font-size:.72rem;color:{P['muted']};letter-spacing:.06em;">房源資料</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:1.6rem;font-weight:700;color:{P['tenant']};">210,288</div>
      <div style="font-size:.72rem;color:{P['muted']};letter-spacing:.06em;">評論文本</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:1.6rem;font-weight:700;color:{P['admin']};">12</div>
      <div style="font-size:.72rem;color:{P['muted']};letter-spacing:.06em;">行政區</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:1.6rem;font-weight:700;color:{P['medium']};">5</div>
      <div style="font-size:.72rem;color:{P['muted']};letter-spacing:.06em;">PoI 資料源</div>
    </div>
  </div>
  <hr style="border:none;border-top:1px solid {P['border']};width:240px;margin:0 auto 12px;">
  <p style="font-size:.70rem;color:{P['muted']};letter-spacing:.04em;">
    資料來源：Inside Airbnb · 台北市政府開放資料<br>
    ML：Logistic Regression × Random Forest ｜ NLP：VADER × jieba<br>
    © 2026 智慧旅宿 AI 平台
  </p>
</div>
""", unsafe_allow_html=True)
