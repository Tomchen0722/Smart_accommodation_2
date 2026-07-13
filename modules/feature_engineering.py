# -*- coding: utf-8 -*-
"""
feature_engineering.py — v2 模型特徵層共用模組（Streamlit 無關，腳本可直接匯入）
================================================================
職責：
  1. 特徵繁中標籤 FEAT_ZH_V2（SHAP 圖 / UI 共用，ml-modeling 規範強制）
  2. dataset_final / 模型 bundle 載入器
  3. 新舊房東判斷 + v2 風險預測（自動路由到完整或冷啟動模型）
  4. 價格 what-if 模擬（取代舊版拍腦袋乘數）

設計原則：訓練腳本與 app 端共用同一套特徵邏輯，防止「上線特徵不一致」。
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_CSV = DATA_DIR / "dataset_final.csv"
DATASET_META = DATA_DIR / "dataset_final.meta.json"
BUNDLE_PATH = MODEL_DIR / "backend_models_v2.joblib"
EVAL_PATH = MODEL_DIR / "eval_results.json"
SHAP_CACHE_PATH = MODEL_DIR / "shap_cache.joblib"

# ── 特徵繁中標籤（42 個：36 完整 + 6 地點房間）──────────
FEAT_ZH_V2 = {
    # 結構化 13
    "accommodates": "可住人數", "bedrooms": "臥室數", "beds": "床位數",
    "bathrooms_count": "衛浴數", "is_shared_bath": "共用衛浴",
    "price": "每晚價格", "minimum_nights": "最短入住晚數",
    "maximum_nights": "最長可住晚數", "min_nights_avg_ntm": "近期平均最短入住",
    "instant_bookable": "即時預訂", "self_checkin": "自助入住",
    "room_type_code": "房型", "neighbourhood_code": "行政區",
    # 評分 7
    "review_scores_rating": "總體評分", "review_scores_accuracy": "描述準確度",
    "review_scores_cleanliness": "清潔度評分", "review_scores_checkin": "入住體驗評分",
    "review_scores_communication": "溝通評分", "review_scores_location": "地點評分",
    "review_scores_value": "性價比評分",
    # 競爭 5
    "price_pctl_nbhd": "同區同房型價格百分位", "score_pctl_nbhd": "同區同房型評分百分位",
    "amenities_vs_median": "設施數/周邊中位數", "nbr_density_1km": "1km房源密度",
    "nbr_density_same_type_1km": "1km同房型密度",
    # 房東身分 7（冷啟動模型移除）
    "host_acceptance_rate": "房東接受率", "host_response_rate": "房東回覆率",
    "response_speed": "房東回覆速度", "host_is_superhost": "超讚房東",
    "host_listings_count": "房東房源數",
    "calculated_host_listings_count": "平台計算房源數",
    "host_tenure_days": "房東經營天數",
    # 經營用心度 4
    "desc_len": "房源描述字數", "host_about_len": "房東自介字數",
    "neighborhood_overview_len": "周邊介紹字數", "amenities_count": "設施總數",
    # 地點/房間 6（冷啟動加入）
    "hotel_count_1km": "1km飯店數", "hotel_count_500m": "500m飯店數",
    "airbnb_hotel_supply_ratio": "短租/飯店供給比",
    "price_per_person": "每人單價", "price_per_bedroom": "每房單價",
    "beds_per_person": "每人床位數",
}

# 價格連動特徵：what-if 模擬調價時需同步重算的衍生欄位
PRICE_DERIVED = ["price_per_person", "price_per_bedroom"]


# ── 載入器（Streamlit 端請在 pages 以 @st.cache_resource 包裝）──
def load_dataset_final():
    """載入 dataset_final.csv（5,849 筆 × 50 欄）。"""
    return pd.read_csv(DATASET_CSV, encoding="utf-8-sig")


def load_bundle():
    """載入 v2 模型 bundle；檔案不存在時給出可行動的錯誤訊息。"""
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {BUNDLE_PATH.name}，請先執行："
            f"python -X utf8 scripts/train_backend_models.py")
    return joblib.load(BUNDLE_PATH)


def load_eval_results():
    """載入訓練時產出的雙軌評估結果（單次切分 vs GroupKFold）。"""
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def load_shap_cache():
    """載入 SHAP 快取（run_shap_analysis.py 產出）。"""
    if not SHAP_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {SHAP_CACHE_PATH.name}，請先執行："
            f"python -X utf8 scripts/run_shap_analysis.py")
    return joblib.load(SHAP_CACHE_PATH)


# ── 新舊房東路由 ─────────────────────────────────────────
def is_cold_start(row):
    """判斷是否走冷啟動模型：平台計算房源數 ≤ 1 視為個人/新房東。

    依據總結報告 §5.3：個人房東（僅 1 筆房源）在兩種評估法下分數幾乎不變
    （無洩漏機會），其餘多房源房東走完整模型。
    """
    val = row.get("calculated_host_listings_count", np.nan)
    return bool(pd.isna(val) or val <= 1)


def predict_risk_v2(row, bundle, force_variant=None):
    """v2 風險預測：自動路由到完整/冷啟動模型。

    參數
    ----
    row : pd.Series — dataset_final 的一列（含全部特徵欄）
    force_variant : "full" / "cold" / None — 指定模型（None=自動判斷）

    回傳 dict：
      risk_score   模型 A 迴歸預測（0~1 空屋率 = 風險分數）
      notify_prob  模型 B 校準後高風險機率
      notify       是否觸發 70% 通知（機率 ≥ 驗證集搜出的門檻）
      variant      實際使用的模型（"full"/"cold"）
      confidence   信心等級文字（UI 標註用）
    """
    variant = force_variant or ("cold" if is_cold_start(row) else "full")
    m = bundle[variant]
    X = pd.DataFrame([row[m["feature_names"]]])
    risk = float(np.clip(m["reg_model"].predict(X)[0], 0, 1))
    prob = float(m["clf_model"].predict_proba(X)[0, 1])
    confidence = ("保守估計（新房東冷啟動模型，無房東歷史可依據）"
                  if variant == "cold"
                  else "一般信心（老房東完整模型，含房東歷史特徵）")
    return {"risk_score": risk, "notify_prob": prob,
            "notify": prob >= m["threshold"], "variant": variant,
            "threshold": m["threshold"], "confidence": confidence}


def simulate_price_change(row, bundle, new_price):
    """價格 what-if 模擬：調整價格與連動衍生特徵後重新預測。

    取代舊版 predict_vacancy_prob 的固定乘數公式 —— 直接讓模型 A 回答
    「調價後風險分數變多少」。注意：價格百分位（price_pctl_nbhd）屬群組
    相對排名，單筆模擬不重排整個市場，維持原值（已知限制，於 UI 註明）。
    """
    sim = row.copy()
    sim["price"] = new_price
    acc = sim.get("accommodates", np.nan)
    bed = sim.get("bedrooms", np.nan)
    sim["price_per_person"] = (new_price / acc
                               if acc and not pd.isna(acc) and acc > 0
                               else np.nan)
    sim["price_per_bedroom"] = (new_price / bed
                                if bed and not pd.isna(bed) and bed > 0
                                else np.nan)
    return predict_risk_v2(sim, bundle)
