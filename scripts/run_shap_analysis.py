# -*- coding: utf-8 -*-
"""
run_shap_analysis.py — SHAP 可解釋性分析（完整模型 + 冷啟動模型）
================================================================
對象：兩個變體的「模型 A（迴歸）」—— 依總結報告 §4.1 設計，迴歸 SHAP 值
單位是空屋率百分點、可直接加總，A 負責「為什麼」。

策略（依報告 §4.3 + shap 版本相容性）：
  1. 優先 shap.TreeExplainer（HistGB 原生支援時最快最精確）
  2. 失敗則降級 PermutationExplainer + model.predict 包裝（報告原做法）
  取樣：測試集固定 500 筆（seed=42），與報告協定一致。

產出：
  models/shap_cache.joblib — SHAP 值/基準值/樣本資料（Streamlit 端重建
                             shap.Explanation 繪圖，App 不需重算）
  docs/shap/*.png          — 靜態圖（簡報用），中文標籤 + 微軟正黑體

執行方式（本機）：
  C:\\Users\\USER\\anaconda3\\python.exe -X utf8 scripts\\run_shap_analysis.py
"""
from pathlib import Path
import sys
import datetime

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")  # 無視窗環境繪圖
import matplotlib.pyplot as plt
import shap

# 讓腳本可匯入專案模組
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from modules.feature_engineering import (  # noqa: E402
    FEAT_ZH_V2, load_dataset_final, load_bundle, SHAP_CACHE_PATH)

DOCS_DIR = PROJECT_ROOT / "docs" / "shap"
N_SAMPLE = 500  # SHAP 取樣筆數（報告協定：500 筆測試樣本）
SEED = 42

# 中文字型設定（Windows 微軟正黑體）
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def compute_shap(model, X_sample):
    """計算 SHAP 值：TreeExplainer 優先，失敗降級 PermutationExplainer。

    回傳 (shap_values ndarray, base_value float, 使用的方法名)。
    """
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_sample)
        return sv.values, float(np.atleast_1d(sv.base_values)[0]), \
            "TreeExplainer"
    except Exception as exc:
        print(f"[降級] TreeExplainer 失敗（{exc}），改用 PermutationExplainer")
        bg = shap.sample(X_sample, 100, random_state=SEED)  # 背景分布
        explainer = shap.PermutationExplainer(model.predict, bg)
        sv = explainer(X_sample)
        return sv.values, float(np.mean(model.predict(X_sample))), \
            "PermutationExplainer"


def to_explanation(values, base, X_sample, feats):
    """以繁中特徵名重建 shap.Explanation 供繪圖。"""
    return shap.Explanation(
        values=values,
        base_values=np.full(len(X_sample), base),
        data=X_sample.to_numpy(),
        feature_names=[FEAT_ZH_V2.get(f, f) for f in feats])


def save_fig(path, tight=True):
    """存圖 + 關閉（避免記憶體累積）。"""
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[圖表] {path.relative_to(PROJECT_ROOT)}")


def plot_variant(tag, expl, X_sample, feats, risk_pred):
    """單一模型變體的標準圖組：beeswarm + bar + waterfall 高/低風險各一。"""
    # 全域：beeswarm（蜂群圖 —— 特徵貢獻方向與分布）
    plt.figure()
    shap.plots.beeswarm(expl, max_display=15, show=False)
    save_fig(DOCS_DIR / f"beeswarm_{tag}.png")

    # 全域：重要度長條
    plt.figure()
    shap.plots.bar(expl, max_display=15, show=False)
    save_fig(DOCS_DIR / f"bar_{tag}.png")

    # 單筆：風險最高與最低各一筆 waterfall（報告 §4.3 單筆解釋範例 ×2）
    hi, lo = int(np.argmax(risk_pred)), int(np.argmin(risk_pred))
    for label, i in [("high", hi), ("low", lo)]:
        plt.figure()
        shap.plots.waterfall(expl[i], max_display=12, show=False)
        save_fig(DOCS_DIR / f"waterfall_{tag}_{label}.png")
    return hi, lo


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset_final()
    bundle = load_bundle()

    # 取樣自測試集（與訓練切分一致，避免解釋訓練集擬合痕跡）
    test_idx = np.array(bundle["split_indices"]["test"])
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(test_idx, size=min(N_SAMPLE, len(test_idx)),
                            replace=False)

    cache = {"meta": {"產出時間":
                      datetime.datetime.now().isoformat(timespec="seconds"),
                      "取樣筆數": int(len(sample_idx)), "seed": SEED}}

    for tag in ["full", "cold"]:
        m = bundle[tag]
        feats = m["feature_names"]
        X_sample = df[feats].iloc[sample_idx].reset_index(drop=True)
        print(f"[計算] {tag} 模型 SHAP（{len(X_sample)} 筆 × {len(feats)} 特徵）")
        values, base, method = compute_shap(m["reg_model"], X_sample)
        print(f"[方法] {method}｜base_value={base:.4f}")

        risk_pred = m["reg_model"].predict(X_sample)
        expl = to_explanation(values, base, X_sample, feats)
        hi, lo = plot_variant(tag, expl, X_sample, feats, risk_pred)

        cache[tag] = {
            "shap_values": values, "base_value": base,
            "X_sample": X_sample, "feature_names": feats,
            "risk_pred": risk_pred, "method": method,
            "listing_ids": df["id"].iloc[sample_idx].to_numpy(),
            "example_high_idx": hi, "example_low_idx": lo,
        }

    # ── 招牌 dependence plot ──
    # 1. maximum_nights 非線性（報告 §4.3 發現：200 晚以下風險隨值降低而降）
    full = cache["full"]
    expl_full = to_explanation(full["shap_values"], full["base_value"],
                               full["X_sample"], full["feature_names"])
    zh_max_nights = FEAT_ZH_V2["maximum_nights"]
    plt.figure()
    shap.plots.scatter(expl_full[:, zh_max_nights], show=False)
    plt.xlim(0, 1200)  # 極端值截尾，聚焦非線性區段
    save_fig(DOCS_DIR / "dependence_maximum_nights_full.png")

    # 2. hotel_count_1km 反直覺圖（冷啟動模型：飯店密度=需求代理，非競爭）
    cold = cache["cold"]
    expl_cold = to_explanation(cold["shap_values"], cold["base_value"],
                               cold["X_sample"], cold["feature_names"])
    plt.figure()
    shap.plots.scatter(expl_cold[:, FEAT_ZH_V2["hotel_count_1km"]], show=False)
    save_fig(DOCS_DIR / "dependence_hotel_count_1km_cold.png")

    joblib.dump(cache, SHAP_CACHE_PATH)
    print(f"[完成] SHAP 快取 → {SHAP_CACHE_PATH.name}｜"
          f"靜態圖 → docs/shap/（{len(list(DOCS_DIR.glob('*.png')))} 張）")


if __name__ == "__main__":
    main()
