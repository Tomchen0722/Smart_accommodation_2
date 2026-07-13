# -*- coding: utf-8 -*-
"""
backend_v2_sections.py — 後台分析 v2 分頁（模型與誠實評估 / SHAP 可解釋性）
================================================================
獨立模組設計：pages/3_後台分析.py 只需三行接入，主頁面維持最小侵入修改。
資料來源：models/ 下的訓練產物（App 端不重訓、不重算 SHAP）。
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.ui_components import P, sec, mb, note, html_table, apply_theme
from modules.ml_models import (v2_ready, load_models_v2, load_dataset_v2,
                               load_eval_v2, load_shap_v2, load_tuning_v2,
                               build_shap_explanation, FEAT_ZH_V2)

# SHAP 圖為 matplotlib 繪製：設定繁中字型
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _guard():
    """v2 產物缺件檢查：缺件時顯示指引並中止該分頁渲染。"""
    ok, missing = v2_ready()
    if not ok:
        st.warning("v2 模型產物尚未就緒，請先在專案根目錄依序執行：")
        for m in missing:
            st.code(m)
        st.stop()


def _pyplot():
    """輸出目前 matplotlib 圖到 Streamlit 並清空（避免圖層疊加）。"""
    st.pyplot(plt.gcf(), clear_figure=True)


# ════════════════════════════════════════════════════════════════
# 分頁一：模型架構與誠實評估
# ════════════════════════════════════════════════════════════════
def render_model_tab_v2():
    _guard()
    ev = load_eval_v2()
    full_ev = ev["完整模型_36特徵"]
    cold_ev = ev["冷啟動模型_35特徵"]

    # ── 區塊 1：模型架構總覽（規則 → 舊 ML → v2 演進）──
    sec("模型架構演進：規則計分 → v2 研究級雙模型")
    mb("HistGradientBoosting 雙模型 · isotonic 校準 · Recall≥0.80 門檻")
    arch = pd.DataFrame([
        {"版本": "規則計分（v0）", "目標": "加權公式 risk_score",
         "特徵": "4 欄加權", "評估": "無",
         "已知問題": "權重人工設定，無法學習非線性"},
        {"版本": "舊 ML（v1）", "目標": "預測 risk_level=高風險",
         "特徵": "7 欄 + 房型", "評估": "隨機切分",
         "已知問題": "標籤由特徵計算而來（availability_365 both），屬標籤洩漏"},
        {"版本": "v2 雙模型（現行）", "目標": "Y_vacancy 空屋率（真實未來供給）",
         "特徵": "36（完整）/ 35（冷啟動）", "評估": "單次切分 + GroupKFold 雙軌",
         "已知問題": "誠實 R² 受資訊量天花板限制（見下）"},
    ])
    html_table(arch, height=200)
    note("v1 的標籤（risk_level）由 risk_score 規則計算，而規則的 40% 權重來自 "
         "availability_365 —— 該欄同時也是模型特徵，模型實際在「學習自己的計分公式」，"
         "指標虛高且無預測意義。v2 改以未來一年空屋率為真實目標，並將該欄移出特徵。")

    st.divider()

    # ── 區塊 2：誠實評估雙軌 ──
    sec("誠實評估雙軌制：單次切分 vs GroupKFold")
    mb("同一模型、兩種評估 · 差距即「多房源房東洩漏」的大小")
    c1, c2, c3, c4 = st.columns(4)
    sg, hk = full_ev["single_split"], full_ev["groupkfold"]
    c1.metric("單次切分 R²", f"{sg['reg_r2']:.3f}", "樂觀（含房東洩漏）",
              delta_color="off")
    c2.metric("GroupKFold R²",
              f"{hk['r2']['mean']:.3f} ± {hk['r2']['std']:.3f}",
              "誠實（全新房東）", delta_color="off")
    c3.metric("單次切分 AUC", f"{sg['clf']['auc']:.3f}", "樂觀", delta_color="off")
    c4.metric("GroupKFold AUC",
              f"{hk['auc']['mean']:.3f} ± {hk['auc']['std']:.3f}",
              "誠實", delta_color="off")

    cb1, cb2 = st.columns([1.2, 1])
    with cb1:
        fig = go.Figure()
        cats = ["迴歸 R²", "分類 AUC"]
        fig.add_trace(go.Bar(name="單次切分（樂觀）", x=cats,
                             y=[sg["reg_r2"], sg["clf"]["auc"]],
                             marker_color=P["medium"],
                             text=[f"{sg['reg_r2']:.3f}",
                                   f"{sg['clf']['auc']:.3f}"],
                             textposition="outside"))
        fig.add_trace(go.Bar(
            name="GroupKFold（誠實）", x=cats,
            y=[hk["r2"]["mean"], hk["auc"]["mean"]],
            error_y=dict(type="data",
                         array=[hk["r2"]["std"], hk["auc"]["std"]]),
            marker_color=P["primary"],
            text=[f"{hk['r2']['mean']:.3f}", f"{hk['auc']['mean']:.3f}"],
            textposition="outside"))
        apply_theme(fig, h=320).update_layout(
            barmode="group", margin=dict(l=40, r=20, t=20, b=30),
            yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)
    with cb2:
        note("洩漏機制：平均每位房東擁有約 4.5 筆房源。隨機切分時，同一房東的其他"
             "房源會同時落在訓練與測試集，模型得以「偷看」房東層級的特徵組合，"
             "測試分數因此虛高。GroupKFold 依 host_id 分組切割，"
             "確保測試房東完全未在訓練中出現 —— 這才是模型面對全新房東的真實能力。")
        note("基準線對照：線性迴歸單次切分 R² = "
             f"{sg['baseline_reg_r2']:.3f}、邏輯迴歸 AUC = "
             f"{sg['baseline_clf_auc']:.3f}，HistGB 顯著優於基準（ml-modeling "
             "驗收標準 ①）。")

    st.divider()

    # ── 區塊 3：雙模型策略（老房東 / 新房東冷啟動）──
    sec("雙模型策略：老房東完整模型 × 新房東冷啟動模型")
    mb("冷啟動 = 移除 7 個房東身分特徵 + 加入 6 個地點/房間特徵")
    chk, ck = cold_ev["groupkfold"], cold_ev["single_split"]
    dual = pd.DataFrame([
        {"模型": "完整（36 特徵）", "適用": "老房東（多房源/有歷史）",
         "單次 R²": sg["reg_r2"], "誠實 R²": f"{hk['r2']['mean']:.3f} ± {hk['r2']['std']:.3f}",
         "誠實 AUC": f"{hk['auc']['mean']:.3f} ± {hk['auc']['std']:.3f}"},
        {"模型": "冷啟動（35 特徵）", "適用": "新房東（僅 1 筆房源）",
         "單次 R²": ck["reg_r2"], "誠實 R²": f"{chk['r2']['mean']:.3f} ± {chk['r2']['std']:.3f}",
         "誠實 AUC": f"{chk['auc']['mean']:.3f} ± {chk['auc']['std']:.3f}"},
    ])
    html_table(dual, fmt={"單次 R²": "{:.3f}"}, height=160)
    note("冷啟動模型的 AUC 標準差明顯更低（±"
         f"{chk['auc']['std']:.3f} vs ±{hk['auc']['std']:.3f}）—— 以地段與房間"
         "條件替代房東歷史後，模型對「沒見過的房東」的表現更穩定。房東入口已依"
         "此自動路由：新房東顯示保守估計與信心等級標註。")

    st.divider()

    # ── 區塊 4：調參證偽實驗 ──
    sec("超參數調校（誠實協定）：證偽實驗")
    mb("RandomizedSearchCV 25 次 × GroupKFold 計分")
    tune = load_tuning_v2()
    if tune is None:
        st.info("尚未執行調參實驗：python -X utf8 scripts/tune_hyperparams_honest.py")
    else:
        tr, tc = tune["迴歸_R2"], tune["分類_AUC"]
        tdf = pd.DataFrame([
            {"任務": "迴歸 R²",
             "調參前": f"{tr['調參前']['mean']:.3f} ± {tr['調參前']['std']:.3f}",
             "調參後": f"{tr['調參後']['mean']:.3f} ± {tr['調參後']['std']:.3f}",
             "增益": f"{tr['增益']:+.3f}"},
            {"任務": "分類 AUC",
             "調參前": f"{tc['調參前']['mean']:.3f} ± {tc['調參前']['std']:.3f}",
             "調參後": f"{tc['調參後']['mean']:.3f} ± {tc['調參後']['std']:.3f}",
             "增益": f"{tc['增益']:+.3f}"},
        ])
        html_table(tdf, height=160)
        note("25 次搜尋的增益全數落在 ±1 個標準差內，無統計顯著提升 —— 複現研究"
             "報告結論：瓶頸在資料資訊量與「新舊房東是兩個難度不同的任務」的結構性"
             "限制，而非模型容量。合理的下一步是擴大資料規模（跨城市），而非調參。")


# ════════════════════════════════════════════════════════════════
# 分頁二：SHAP 可解釋性
# ════════════════════════════════════════════════════════════════
def render_shap_tab_v2():
    _guard()
    cache = load_shap_v2()
    full, cold = cache["full"], cache["cold"]

    sec("全域解釋：哪些特徵在驅動空屋風險")
    mb(f"模型 A（迴歸）SHAP · 測試集抽樣 {cache['meta']['取樣筆數']} 筆 · "
       f"方法 {full['method']} · 單位 = 空屋率百分點")

    expl_full = build_shap_explanation(full)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**蜂群圖（方向 + 分布）**")
        plt.figure()
        import shap
        shap.plots.beeswarm(expl_full, max_display=12, show=False)
        _pyplot()
    with g2:
        st.markdown("**平均重要度排序**")
        plt.figure()
        shap.plots.bar(expl_full, max_display=12, show=False)
        _pyplot()
    note("SHAP 值為每個特徵對單筆預測的邊際貢獻（Shapley 分攤），正值推高空屋率、"
         "負值降低。與內建 feature importance 不同：SHAP 有方向、可加總到單筆預測、"
         "且對每筆資料各自成立。")

    st.divider()

    # ── 深度分析：兩張招牌依賴圖 ──
    sec("深度分析：非線性與反直覺訊號")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**{FEAT_ZH_V2['maximum_nights']}：非線性關係**")
        plt.figure()
        shap.plots.scatter(expl_full[:, FEAT_ZH_V2["maximum_nights"]],
                           show=False)
        plt.xlim(0, 1200)
        _pyplot()
        note("約 200 晚以下：可住晚數越短、風險越低（短租定位明確）；"
             "超過 300 晚後貢獻打平 —— 線性模型抓不到這種型態。")
    with d2:
        st.markdown(f"**{FEAT_ZH_V2['hotel_count_1km']}：反直覺發現（冷啟動模型）**")
        expl_cold = build_shap_explanation(cold)
        plt.figure()
        shap.plots.scatter(expl_cold[:, FEAT_ZH_V2["hotel_count_1km"]],
                           show=False)
        _pyplot()
        note("原假設「飯店越密集 → 競爭越兇 → 空屋率越高」被實測推翻：飯店密度與"
             "空屋率呈負相關。飯店紮堆處是觀光精華地段，Airbnb 在那裡也好租 —— "
         "飯店密度是「需求熱度代理」而非「競爭懲罰」。")

    st.divider()

    # ── 單筆診斷 waterfall ──
    sec("單筆房源診斷：這間為什麼有風險")
    mb("Waterfall · 從基準值逐特徵累加到最終預測")
    ids = full["listing_ids"]
    options = {
        f"高風險範例（預測空屋率 {full['risk_pred'][full['example_high_idx']]:.0%}）":
            full["example_high_idx"],
        f"低風險範例（預測空屋率 {full['risk_pred'][full['example_low_idx']]:.0%}）":
            full["example_low_idx"],
    }
    pick = st.selectbox("選擇診斷範例", list(options.keys()))
    i = options[pick]
    st.caption(f"房源 ID：{ids[i]}｜模型預測空屋率 {full['risk_pred'][i]:.1%}")
    plt.figure()
    shap.plots.waterfall(expl_full[i], max_display=12, show=False)
    _pyplot()

    st.divider()

    # ── 雙模型 SHAP 並排 ──
    sec("雙模型對比：模型各自依賴什麼")
    mb("左：完整模型（房東歷史主導） · 右：冷啟動模型（地段/房間補位）")
    s1, s2 = st.columns(2)
    with s1:
        plt.figure()
        shap.plots.bar(expl_full, max_display=10, show=False)
        _pyplot()
    with s2:
        plt.figure()
        shap.plots.bar(expl_cold, max_display=10, show=False)
        _pyplot()
    note("完整模型的頭部特徵集中在房東品質（接受率、回覆速度）；冷啟動模型失去"
         "這些後，由每房單價、飯店密度等地點/房間特徵補位 —— 這正是雙模型策略"
         "「用地段與房間條件替代房東歷史」的視覺證據。")
