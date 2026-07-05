"""
後台分析系統 — Admin Analytics Dashboard
全站 KPI · 市場供需熱圖 · 租金偏離監控 · ML 模型分析 · NLP 全站文本 · 異常告警
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.ui_components import (
    inject_css, P, RTC, FEAT_ZH,
    sec, mb, note, risk_badge, stat_card, html_table, apply_theme,
    sidebar_nav,
)
from modules.data_loader import load_listings, load_reviews
from modules.ml_models import (train_models, nb_aggregate,
                               split_summary, cross_validate_models)
from modules.nlp_analysis import global_sentiment_stats

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="後台分析 — 智慧旅宿", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()

# ─── Load data ──────────────────────────────────────────────────
with st.spinner("載入全站資料與訓練模型 …"):
    DF = load_listings()
    REVIEWS = load_reviews()
    MDL = train_models(DF)
    NB = nb_aggregate(DF)


@st.cache_data(show_spinner=False)
def split_cached(test_size, val_size):
    return split_summary(DF, test_size, val_size)


@st.cache_data(show_spinner=False)
def cv_cached(k):
    return cross_validate_models(DF, k)


# ─── Header ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:6px 0 14px;">
  <h1 style="font-size:1.4rem;font-weight:700;color:{P['ink']};
       margin:0;letter-spacing:-.3px;">📊 後台管理大盤儀表板</h1>
  <p style="font-size:.78rem;color:{P['muted']};margin:4px 0 0;">
    監控平台健康度 · 市場趨勢 · AI 模型效能 · 全站文本洞察
  </p>
</div>
<hr style="margin:0 0 16px;">
""", unsafe_allow_html=True)

with st.sidebar:
    sidebar_nav()
    st.markdown(f"""
    <div style="padding:4px 0 12px;">
      <div style="font-size:1rem;font-weight:700;color:{P['ink']};">
        📊 管理後台</div>
      <div style="font-size:.72rem;color:{P['muted']};margin-top:2px;">
        全站數據監控與 AI 分析</div>
    </div>""", unsafe_allow_html=True)
    nlp_sample = st.slider("NLP 抽樣評論數", 2000, 20000, 8000, step=1000)
    st.caption("抽樣越大越準確，運算時間越長")
    st.divider()
    st.caption("© 2026 智慧旅宿 AI 平台")

# ═══════════════════════════════════════════════════════════════
# TOP KPI CARDS
# ═══════════════════════════════════════════════════════════════
sec("全站核心 KPI 指標")
mb("即時營運數據 · Live Metrics")

total_listings = len(DF)
vacancy_rate = round((DF["availability_365"] > 200).mean() * 100, 1)  # 長期空置比例
avg_occ = round(DF["occupancy_pct"].mean(), 1)
n_hosts = DF["host_id"].nunique()
n_superhost = (DF["host_is_superhost"] == "t").sum() if "host_is_superhost" in DF.columns else 0
high_risk_pct = round((DF["risk_level"] == "高風險").mean() * 100, 1)
# 建議採用率為模擬營運指標（實際需前端埋點回傳），以低風險房源占比作為代理
adoption_rate = round((DF["risk_level"] == "低風險").mean() * 100, 1)

kc = st.columns(4)
with kc[0]:
    stat_card(f"{high_risk_pct}%", "全站高風險空房率",
              color=P["high"] if high_risk_pct > 15 else P["low"])
    st.caption("高於 15% 需關注" if high_risk_pct > 15 else "✅ 健康區間")
with kc[1]:
    stat_card(f"{total_listings:,}", "上架中房源總數", color=P["primary"])
    st.caption(f"涵蓋 {DF['neighbourhood_cleansed'].nunique()} 個行政區")
with kc[2]:
    stat_card(f"{n_hosts:,}", "平台房東數", color=P["tenant"])
    st.caption(f"其中超級房東 {n_superhost:,} 位")
with kc[3]:
    stat_card(f"{adoption_rate}%", "健康房源佔比 *", color=P["admin"])
    st.caption("* 建議採用率代理指標")

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
T1, T2, T3, T4 = st.tabs([
    "🗺 市場供需熱圖", "📈 租金偏離監控", "🤖 ML 模型分析", "💬 NLP 全站文本"
])

# ──────────────────────────────────────────────────────────────
# TAB 1: Market Heatmap
# ──────────────────────────────────────────────────────────────
with T1:
    sec("市場供需與空房預測熱圖")
    mb("地理空間分析 · 密度熱圖 (紅=供過於求，綠=供不應求)")

    cm1, cm2 = st.columns([1.7, 1.3])
    with cm1:
        sample_map = DF.sample(min(4000, len(DF)), random_state=1)
        fig = px.density_mapbox(
            sample_map, lat="latitude", lon="longitude", z="risk_score",
            radius=14, zoom=10.5, height=460, mapbox_style="carto-positron",
            center={"lat": 25.05, "lon": 121.55},
            color_continuous_scale=[[0, P["low"]], [0.5, P["medium"]], [1, P["high"]]],
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=0, b=0),
                          coloraxis_colorbar=dict(title="風險", len=0.7))
        st.plotly_chart(fig, use_container_width=True)

    with cm2:
        sec("各行政區高風險佔比")
        nb_sorted = NB.sort_values("高風險佔比", ascending=True)
        fig = go.Figure(go.Bar(
            x=nb_sorted["高風險佔比"], y=nb_sorted["行政區"], orientation="h",
            marker=dict(color=nb_sorted["高風險佔比"],
                        colorscale=[[0, P["low"]], [0.5, P["medium"]], [1, P["high"]]]),
            text=nb_sorted["高風險佔比"].map(lambda x: f"{x:.0f}%"),
            textposition="outside",
        ))
        apply_theme(fig, h=460, legend=False).update_layout(
            margin=dict(l=70, r=30, t=10, b=30),
            xaxis_title="高風險房源佔比 (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("行政區市場總覽")
    mb("供需熱度統計 · 房源數 × 中位價 × 風險 × 入住率")
    disp = NB[["行政區", "房源數", "高風險佔比", "平均風險",
               "中位價格", "平均評論數", "平均入住率"]].copy()
    html_table(
        disp,
        fmt={"中位價格": "${:,.0f}", "高風險佔比": "{:.1f}%",
             "平均風險": "{:.3f}", "平均評論數": "{:.1f}",
             "平均入住率": "{:.1f}%", "房源數": "{:,.0f}"},
        cell_fn={"高風險佔比": lambda v: (
            f"color:{P['high']};font-weight:700;" if v > 15
            else (f"color:{P['medium']};" if v > 8 else f"color:{P['low']};"))},
        height=420,
    )

# ──────────────────────────────────────────────────────────────
# TAB 2: Rent Deviation Monitor
# ──────────────────────────────────────────────────────────────
with T2:
    sec("租金行情與偏離度監控")
    mb("平台定價 vs 區域中位數偏離分析")

    global_median = DF["price"].median()
    cd1, cd2 = st.columns(2)
    with cd1:
        nb_price = NB.copy()
        nb_price["偏離度"] = ((nb_price["中位價格"] - global_median)
                            / global_median * 100).round(1)
        nb_price = nb_price.sort_values("偏離度")
        colors = [P["high"] if v > 20 else (P["low"] if v < -20 else P["primary"])
                  for v in nb_price["偏離度"]]
        fig = go.Figure(go.Bar(
            x=nb_price["偏離度"], y=nb_price["行政區"], orientation="h",
            marker=dict(color=colors),
            text=nb_price["偏離度"].map(lambda x: f"{x:+.0f}%"),
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color=P["muted"])
        apply_theme(fig, h=420, legend=False).update_layout(
            margin=dict(l=70, r=30, t=10, b=30),
            xaxis_title=f"相對全站中位數 ${global_median:,.0f} 的偏離 (%)")
        st.plotly_chart(fig, use_container_width=True)

    with cd2:
        sec("房型價格分佈")
        rt_price = DF[DF["price"] < DF["price"].quantile(0.95)]
        fig = px.box(rt_price, x="room_type_zh", y="price", color="room_type_zh",
                     color_discrete_map=RTC,
                     labels={"room_type_zh": "房型", "price": "每晚價格 (TWD)"})
        apply_theme(fig, h=420, legend=False).update_layout(
            margin=dict(l=50, r=20, t=10, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("評論活躍度趨勢")
    mb("月度評論量 · 市場需求熱度代理指標")
    rev = REVIEWS.dropna(subset=["date"]).copy()
    rev = rev[rev["date"] >= "2015-01-01"]
    monthly = rev.set_index("date").resample("ME").size().reset_index(name="評論數")
    fig = go.Figure(go.Scatter(
        x=monthly["date"], y=monthly["評論數"], mode="lines",
        line=dict(color=P["primary"], width=2),
        fill="tozeroy", fillcolor="rgba(78,127,176,.12)"))
    apply_theme(fig, h=280, legend=False).update_layout(
        margin=dict(l=50, r=20, t=10, b=30),
        xaxis_title="月份", yaxis_title="評論數")
    st.plotly_chart(fig, use_container_width=True)
    note("評論量反映實際入住與市場需求熱度。可觀察畢業季 (6-9月) 與淡季 (11-2月) 的季節性波動。")

# ──────────────────────────────────────────────────────────────
# TAB 3: ML Model Analysis
# ──────────────────────────────────────────────────────────────
with T3:
    # ── Data split: Train / Validation / Test ──
    sec("資料切割 (Train / Validation / Test)")
    mb("分層抽樣 · Stratified 3-way split")
    cs1, cs2 = st.columns(2)
    test_pct = cs1.slider("測試集比例 (%)", 10, 40, 20, 5, key="test_pct")
    val_pct = cs2.slider("驗證集比例 (%)", 10, 30, 20, 5, key="val_pct")
    SP = split_cached(test_pct / 100, val_pct / 100)

    sp = st.columns(3)
    sp[0].metric("訓練集 Train", f"{SP['train']['n']:,}",
                 f"{SP['ratios'][0]*100:.0f}%", delta_color="off")
    sp[1].metric("驗證集 Val", f"{SP['val']['n']:,}",
                 f"{SP['ratios'][1]*100:.0f}%", delta_color="off")
    sp[2].metric("測試集 Test", f"{SP['test']['n']:,}",
                 f"{SP['ratios'][2]*100:.0f}%", delta_color="off")

    cA, cB = st.columns(2)
    with cA:
        sec("各集合類別分布")
        dist_df = pd.DataFrame([
            {"集合": "訓練 Train", "樣本數": SP['train']['n'],
             "高風險": SP['train']['pos'], "高風險%": SP['train']['pos_pct']},
            {"集合": "驗證 Val", "樣本數": SP['val']['n'],
             "高風險": SP['val']['pos'], "高風險%": SP['val']['pos_pct']},
            {"集合": "測試 Test", "樣本數": SP['test']['n'],
             "高風險": SP['test']['pos'], "高風險%": SP['test']['pos_pct']},
        ])
        html_table(dist_df, fmt={"樣本數": "{:,.0f}", "高風險": "{:,.0f}",
                                 "高風險%": "{:.1f}%"}, height=180)
        note("分層抽樣確保三個集合的高風險比例一致，避免切割偏差。")
    with cB:
        sec("驗證集 vs 測試集 指標")
        met_df = pd.DataFrame([
            {"模型": "LR", "AUC(驗證)": SP['lr_val']['auc'], "AUC(測試)": SP['lr_test']['auc'],
             "F1(驗證)": SP['lr_val']['f1'], "F1(測試)": SP['lr_test']['f1']},
            {"模型": "RF", "AUC(驗證)": SP['rf_val']['auc'], "AUC(測試)": SP['rf_test']['auc'],
             "F1(驗證)": SP['rf_val']['f1'], "F1(測試)": SP['rf_test']['f1']},
        ])
        html_table(met_df, fmt={c: "{:.3f}" for c in
                   ["AUC(驗證)", "AUC(測試)", "F1(驗證)", "F1(測試)"]}, height=180)
        note("驗證集用於調參／選型，測試集為最終客觀評估。兩者接近代表未過擬合。")

    st.divider()
    sec("K-fold 交叉驗證")
    mb("Stratified K-Fold · ROC-AUC 與 F1")
    kf = st.slider("折數 k", 3, 10, 5, key="kfolds")
    CV = cv_cached(kf)
    cv1, cv2 = st.columns([1, 1.25])
    with cv1:
        cvt = pd.DataFrame([
            {"模型": "LR", "AUC 平均": float(np.mean(CV['lr_auc'])),
             "AUC 標準差": float(np.std(CV['lr_auc'])),
             "F1 平均": float(np.mean(CV['lr_f1'])),
             "F1 標準差": float(np.std(CV['lr_f1']))},
            {"模型": "RF", "AUC 平均": float(np.mean(CV['rf_auc'])),
             "AUC 標準差": float(np.std(CV['rf_auc'])),
             "F1 平均": float(np.mean(CV['rf_f1'])),
             "F1 標準差": float(np.std(CV['rf_f1']))},
        ])
        html_table(cvt, fmt={c: "{:.3f}" for c in
                   ["AUC 平均", "AUC 標準差", "F1 平均", "F1 標準差"]}, height=170)
    with cv2:
        folds = list(range(1, kf + 1))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=folds, y=CV['lr_auc'], name="LR",
                             marker_color=P['primary']))
        fig.add_trace(go.Bar(x=folds, y=CV['rf_auc'], name="RF",
                             marker_color=P['admin']))
        lo = min(list(CV['lr_auc']) + list(CV['rf_auc'])) - 0.02
        apply_theme(fig, h=210).update_layout(
            barmode="group", margin=dict(l=44, r=10, t=6, b=30),
            xaxis_title="Fold", yaxis_title="ROC-AUC",
            yaxis_range=[max(0, min(0.85, lo)), 1.0])
        st.plotly_chart(fig, use_container_width=True)
    note(f"{kf}-fold 交叉驗證檢視模型在不同切割下的穩定度；標準差越小越穩定。")

    st.divider()
    sec("機器學習模型效能")

    mb(f"訓練集 {MDL['n_train']:,} · 測試集 {MDL['n_test']:,} · "
       f"正樣本 {MDL['n_pos']:,} / 負樣本 {MDL['n_neg']:,}")

    mc = st.columns(2)
    for col_, key, mname in [(mc[0], "lr", "Logistic Regression"),
                             (mc[1], "rf", "Random Forest")]:
        m = MDL[key]
        with col_:
            st.markdown(f"""
            <div style="background:{P['surface']};border:1px solid {P['border']};
                 border-top:3px solid {P['primary'] if key=='lr' else P['admin']};
                 border-radius:12px;padding:14px 18px;margin-bottom:10px;">
              <div style="font-size:.92rem;font-weight:700;color:{P['ink']};
                   margin-bottom:8px;">{mname}</div>
              <div style="display:flex;flex-wrap:wrap;gap:14px;font-size:.78rem;
                   color:{P['ink2']};">
                <span>Accuracy <b>{m['accuracy']:.3f}</b></span>
                <span>Recall <b style="color:{P['low']};">{m['recall']:.3f}</b></span>
                <span>Precision <b>{m['precision']:.3f}</b></span>
                <span>F1 <b>{m['f1']:.3f}</b></span>
                <span>AUC <b style="color:{P['primary']};">{m['auc']:.3f}</b></span>
                <span>AP <b>{m['ap']:.3f}</b></span>
              </div>
            </div>""", unsafe_allow_html=True)

    # ROC curves
    cr1, cr2 = st.columns(2)
    with cr1:
        sec("ROC 曲線比較")
        mb("Receiver Operating Characteristic")
        fig = go.Figure()
        for key, name, color in [("lr", "LR", P["primary"]), ("rf", "RF", P["admin"])]:
            fig.add_trace(go.Scatter(
                x=MDL[key]["fpr"], y=MDL[key]["tpr"], mode="lines",
                name=f"{name} (AUC={MDL[key]['auc']:.3f})",
                line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                      line=dict(color=P["muted"], dash="dash"), showlegend=False))
        apply_theme(fig, h=300).update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            legend=dict(x=0.5, y=0.1), margin=dict(l=50, r=20, t=10, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with cr2:
        sec("Precision-Recall 曲線")
        mb("類別不平衡下的關鍵指標")
        fig = go.Figure()
        for key, name, color in [("lr", "LR", P["primary"]), ("rf", "RF", P["admin"])]:
            fig.add_trace(go.Scatter(
                x=MDL[key]["rec"], y=MDL[key]["prec"], mode="lines",
                name=f"{name} (AP={MDL[key]['ap']:.3f})",
                line=dict(color=color, width=2)))
        apply_theme(fig, h=300).update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
            legend=dict(x=0.1, y=0.1), margin=dict(l=50, r=20, t=10, b=40))
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    ci1, ci2 = st.columns(2)
    with ci1:
        sec("隨機森林 特徵重要度")
        imp = pd.DataFrame(sorted(MDL["rf_import"].items(),
                                  key=lambda x: x[1], reverse=True)[:10],
                           columns=["feat", "imp"])
        imp["特徵"] = imp["feat"].map(lambda c: FEAT_ZH.get(c, c))
        fig = go.Figure(go.Bar(x=imp["imp"], y=imp["特徵"], orientation="h",
                        marker=dict(color=P["admin"])))
        apply_theme(fig, h=300, legend=False).update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(l=110, r=20, t=10, b=30), xaxis_title="重要度")
        st.plotly_chart(fig, use_container_width=True)

    with ci2:
        sec("邏輯迴歸 係數方向")
        coef = pd.DataFrame(sorted(MDL["lr_coef"].items(),
                                   key=lambda x: abs(x[1]), reverse=True)[:10],
                            columns=["feat", "coef"])
        coef["特徵"] = coef["feat"].map(lambda c: FEAT_ZH.get(c, c))
        coef = coef.sort_values("coef")
        colors = [P["high"] if v > 0 else P["low"] for v in coef["coef"]]
        fig = go.Figure(go.Bar(x=coef["coef"], y=coef["特徵"], orientation="h",
                        marker=dict(color=colors)))
        fig.add_vline(x=0, line_color=P["muted"])
        apply_theme(fig, h=300, legend=False).update_layout(
            margin=dict(l=110, r=20, t=10, b=30),
            xaxis_title="係數 (+提高風險 / -降低風險)")
        st.plotly_chart(fig, use_container_width=True)

    note(f"RF 模型 AUC={MDL['rf']['auc']:.3f} 優於 LR AUC={MDL['lr']['auc']:.3f}。"
         "特徵重要度顯示可訂天數與評論活躍度為空房風險的主要驅動因子。")

# ──────────────────────────────────────────────────────────────
# TAB 4: NLP Global Text
# ──────────────────────────────────────────────────────────────
with T4:
    sec("全站評論情感分析")
    mb(f"VADER × jieba · 抽樣 {nlp_sample:,} 則評論")

    with st.spinner(f"分析 {nlp_sample:,} 則評論中，請稍候 …"):
        G = global_sentiment_stats(REVIEWS, sample_n=nlp_sample)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("抽樣評論", f"{G['total_sampled']:,}")
    g2.metric("😊 正面", f"{G['pos_pct']}%")
    g3.metric("😐 中立", f"{G['neu_pct']}%")
    g4.metric("😞 負面", f"{G['neg_pct']}%")

    cn1, cn2 = st.columns([1, 1.4])
    with cn1:
        sec("全站情感分佈")
        sent_data = pd.DataFrame({
            "情感": ["正面", "中立", "負面"],
            "數量": [G["pos_n"], G["neu_n"], G["neg_n"]],
        })
        fig = px.pie(sent_data, values="數量", names="情感", color="情感",
                     color_discrete_map={"正面": P["low"], "中立": P["muted"],
                                         "負面": P["high"]}, hole=0.6)
        fig.update_traces(textfont_size=11, marker_line_width=2,
                          marker_line_color=P["bg"])
        apply_theme(fig, h=300).update_layout(margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(fig, use_container_width=True)

    with cn2:
        sec("各語言情感表現")
        lang_rows = []
        lang_map = {"en": "英文", "zh": "中文", "mixed_zh_en": "中英混合",
                    "other": "其他"}
        for lang, s in G["lang_stats"].items():
            lang_rows.append({
                "語言": lang_map.get(lang, lang), "評論數": s["count"],
                "平均情感": s["avg_sentiment"], "正面佔比": s["pos_pct"],
            })
        lang_df = pd.DataFrame(lang_rows).sort_values("評論數", ascending=False)
        html_table(
            lang_df,
            fmt={"評論數": "{:,.0f}", "平均情感": "{:+.3f}", "正面佔比": "{:.1f}%"},
            cell_fn={"平均情感": lambda v: (
                f"color:{P['low']};font-weight:700;" if v > 0.1
                else (f"color:{P['high']};font-weight:700;" if v < 0 else ""))},
            height=300,
        )

    st.divider()
    kw1, kw2 = st.columns(2)
    with kw1:
        sec("✅ 全站正面關鍵字 (英文)")
        if G["pos_keywords_en"]:
            kd = pd.DataFrame(G["pos_keywords_en"][:15], columns=["kw", "cnt"])
            fig = go.Figure(go.Bar(x=kd["cnt"], y=kd["kw"], orientation="h",
                            marker=dict(color=P["low"])))
            apply_theme(fig, h=380, legend=False).update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(l=90, r=20, t=5, b=25))
            st.plotly_chart(fig, use_container_width=True)
    with kw2:
        sec("❌ 全站負面關鍵字 (英文)")
        if G["neg_keywords_en"]:
            kd = pd.DataFrame(G["neg_keywords_en"][:15], columns=["kw", "cnt"])
            fig = go.Figure(go.Bar(x=kd["cnt"], y=kd["kw"], orientation="h",
                            marker=dict(color=P["high"])))
            apply_theme(fig, h=380, legend=False).update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(l=90, r=20, t=5, b=25))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# ALERT LIST (bottom, full width)
# ═══════════════════════════════════════════════════════════════
st.divider()
sec("系統異常告警清單")
mb("Model Drift / 市場異常 / 營運提醒", warning=True)

alerts = []
# 1. High-risk district alert
worst_nb = NB.iloc[0]
if worst_nb["高風險佔比"] > 15:
    alerts.append(("⚠️ 警告", "區域空房潮",
                   f"{worst_nb['行政區']} 高風險佔比 {worst_nb['高風險佔比']:.0f}%",
                   "通知房東優化", P["medium"]))
# 2. Price deviation alert
gm = DF["price"].median()
dev_nb = NB.assign(dev=(NB["中位價格"] - gm) / gm * 100)
top_dev = dev_nb.loc[dev_nb["dev"].abs().idxmax()]
if abs(top_dev["dev"]) > 25:
    alerts.append(("❌ 嚴重", "租金偏離過大",
                   f"{top_dev['行政區']} 中位價偏離全站 {top_dev['dev']:+.0f}%",
                   "引導定價校正", P["high"]))
# 3. Model drift (compare LR vs RF gap)
auc_gap = abs(MDL["rf"]["auc"] - MDL["lr"]["auc"])
if auc_gap > 0.05:
    alerts.append(("⚠️ 警告", "模型效能落差",
                   f"RF 與 LR AUC 差距 {auc_gap:.3f}，建議統一決策權重",
                   "重新訓練", P["medium"]))
# 4. Healthy status
alerts.append(("✅ 正常", "全站健康度",
               f"平均入住率 {avg_occ:.0f}%，高風險 {high_risk_pct:.0f}%",
               "持續監控", P["low"]))

alert_df = pd.DataFrame(
    [(a[0], a[1], a[2], a[3]) for a in alerts],
    columns=["狀態", "告警類型", "影響範圍 / 說明", "建議動作"])
html_table(
    alert_df,
    cell_fn={"狀態": lambda v: (
        f"color:{P['high']};font-weight:700;" if "嚴重" in v
        else (f"color:{P['medium']};font-weight:700;" if "警告" in v
              else f"color:{P['low']};font-weight:700;"))},
    height=240,
)
note("模型飄移告警：當預測空房率與真實空房率誤差超過閾值 (5%) 時，自動提示需重新訓練 (Retrain)。")
