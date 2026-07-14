"""
空屋率風險雙軌模型 — Vacancy-risk dual-track model (依 smartaccommodation_imp_new 規格).

  • 模型 A（迴歸大腦）：HistGradientBoostingRegressor 擬合 Y_vacancy ∈ [0,1]。
  • 模型 B（分類預警）：CalibratedClassifierCV(HistGradientBoostingClassifier,
      method="isotonic", cv=3) 輸出「未來空屋率 ≥ 70%」的校準機率。
  • 誠實驗證：GroupKFold(host_id) 5 折（測試房東不出現在訓練集）。
  • 解釋：以「基準對照邊際貢獻」把每個特徵對空屋風險的加/減分（百分點）拆解，
      供沙盒 SHAP 式雙向瀑布圖與 Top-2 白話診斷使用。

資料來源：data/dataset_multimodal.csv（dataset_final + POI 距離/密度 + NLP 情感）。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
    cache_data = st.cache_data
    cache_resource = st.cache_resource
except Exception:                       # 允許在無 Streamlit 環境下被匯入/測試
    def cache_data(*a, **k):
        def deco(f): return f
        return deco if not a else a[0]
    cache_resource = cache_data

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_NON_FEATURES = {"id", "host_id", "latitude", "longitude",
                 "neighbourhood_cleansed", "room_type", "Y_vacancy", "Y_high_risk"}

# 房東身分特徵（冷啟動新房東時移除）
_HOST_IDENTITY = ["host_acceptance_rate", "host_response_rate", "response_speed",
                  "host_is_superhost", "host_listings_count",
                  "calculated_host_listings_count", "host_tenure_days"]

# 沙盒可即時調整的特徵
SANDBOX_FEATURES = ["price", "minimum_nights", "response_speed", "desc_len"]

# 特徵中文名（給 SHAP 瀑布 hover / 診斷用）
ZH = {
    "accommodates": "可住人數", "bedrooms": "臥室數", "beds": "床數",
    "bathrooms_count": "衛浴數", "is_shared_bath": "共用衛浴", "price": "每晚房價",
    "minimum_nights": "最低入住天數", "maximum_nights": "最高入住天數",
    "min_nights_avg_ntm": "近期平均最低天數", "instant_bookable": "可即時預訂",
    "self_checkin": "自助入住", "room_type_code": "房型", "neighbourhood_code": "行政區",
    "review_scores_rating": "房源歷史評分", "review_scores_accuracy": "描述準確度評分",
    "review_scores_cleanliness": "清潔度評分", "review_scores_checkin": "入住體驗評分",
    "review_scores_communication": "客服溝通評分", "review_scores_location": "地點評分",
    "review_scores_value": "性價比評分", "price_pctl_nbhd": "同區同房型價格百分位排名",
    "score_pctl_nbhd": "同區評分百分位排名", "amenities_vs_median": "設施數 vs 同區中位數",
    "nbr_density_1km": "1km 內同業密度", "nbr_density_same_type_1km": "1km 內同房型密度",
    "host_acceptance_rate": "房東接受率", "host_response_rate": "房東回覆率",
    "response_speed": "客服回覆速度", "host_is_superhost": "超讚房東",
    "host_listings_count": "名下房源數", "calculated_host_listings_count": "名下獨立房源數",
    "host_tenure_days": "房東經營天數", "desc_len": "房源描述字數",
    "host_about_len": "房東自介字數", "neighborhood_overview_len": "周邊介紹字數",
    "amenities_count": "設施數量", "hotel_count_1km": "1km 內旅宿數",
    "hotel_count_500m": "500m 內旅宿數", "airbnb_hotel_supply_ratio": "旅宿供給比",
    "price_per_person": "每人房價", "price_per_bedroom": "每臥室房價",
    "beds_per_person": "每人床數", "dist_to_nearest_mrt_m": "最近捷運距離(公尺)",
    "mrt_count_500m": "500m 內捷運出入口數", "bus_stops_count_300m": "300m 內公車站數",
    "bus_stops_count_500m": "500m 內公車站數", "conv_stores_count_200m": "200m 內超商數",
    "conv_stores_count_500m": "500m 內超商數", "restaurants_count_500m": "500m 內餐廳數",
    "dist_to_nearest_park_m": "最近公園距離(公尺)", "park_count_500m": "500m 內公園數",
    "dist_to_nearest_clinic_m": "最近診所距離(公尺)", "dist_to_nearest_school_m": "最近學校距離(公尺)",
    "avg_review_sentiment": "住客評論情感得分", "avg_review_length": "評論平均字數",
    "has_no_reviews": "無評論房源",
}


@cache_data(show_spinner=False)
def load_data():
    """載入多模態資料集（POI+NLP 已併入）。"""
    p = DATA_DIR / "dataset_multimodal.csv"
    if not p.exists():                              # 後備：至少用 dataset_final
        p = DATA_DIR / "dataset_final.csv"
    df = pd.read_csv(p)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df


def feature_cols(df):
    return [c for c in df.columns if c not in _NON_FEATURES]


@cache_resource(show_spinner="訓練空屋率雙軌模型 …")
def get_models():
    """在全量資料上訓練 Model A / Model B（供即時推論），並回傳基準中位數。"""
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                   HistGradientBoostingClassifier)
    from sklearn.calibration import CalibratedClassifierCV
    df = load_data()
    feats = feature_cols(df)
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    med = X.median(numeric_only=True)
    Xf = X.fillna(med)
    yv = pd.to_numeric(df["Y_vacancy"], errors="coerce").fillna(0).values
    yr = pd.to_numeric(df["Y_high_risk"], errors="coerce").fillna(0).astype(int).values
    reg = HistGradientBoostingRegressor(random_state=0).fit(Xf, yv)
    clf = CalibratedClassifierCV(
        HistGradientBoostingClassifier(random_state=0),
        method="isotonic", cv=3).fit(Xf, yr)
    return {"reg": reg, "clf": clf, "features": feats,
            "median": med.to_dict(), "high_risk_rate": float(yr.mean())}


@cache_data(show_spinner=False)
def get_metrics():
    """GroupKFold(host_id) 5 折誠實指標；快取到 data/_model_metrics.json。"""
    cache = DATA_DIR / "_model_metrics.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                   HistGradientBoostingClassifier)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (r2_score, mean_squared_error,
                                  roc_auc_score, recall_score, f1_score)
    df = load_data()
    feats = feature_cols(df)
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    yv = pd.to_numeric(df["Y_vacancy"], errors="coerce").fillna(0).values
    yr = pd.to_numeric(df["Y_high_risk"], errors="coerce").fillna(0).astype(int).values
    g = df["host_id"].values
    gkf = GroupKFold(5)
    r2s, mses, aucs, recs, f1s = [], [], [], [], []
    for tr, te in gkf.split(X, yv, g):
        ra = HistGradientBoostingRegressor(random_state=0).fit(X.iloc[tr], yv[tr])
        p = ra.predict(X.iloc[te])
        r2s.append(r2_score(yv[te], p)); mses.append(mean_squared_error(yv[te], p))
        cb = CalibratedClassifierCV(HistGradientBoostingClassifier(random_state=0),
                                    method="isotonic", cv=3).fit(X.iloc[tr], yr[tr])
        pr = cb.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(yr[te], pr))
        recs.append(recall_score(yr[te], (pr >= .5)))
        f1s.append(f1_score(yr[te], (pr >= .5)))
    m = {"n": int(len(df)), "n_features": len(feats),
         "high_risk_rate": float(yr.mean()),
         "R2": float(np.mean(r2s)), "MSE": float(np.mean(mses)),
         "AUC": float(np.mean(aucs)), "Recall": float(np.mean(recs)),
         "F1": float(np.mean(f1s)),
         "R2_folds": [round(float(x), 3) for x in r2s]}
    try:
        cache.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return m


# ─── 房東 / 房源查詢 ───────────────────────────────────────────
@cache_data(show_spinner=False)
def host_options():
    """回傳 [(host_id, 房源數)]，依房源數降序。"""
    df = load_data()
    vc = df.groupby("host_id").size().sort_values(ascending=False)
    return [(int(h), int(n)) for h, n in vc.items()]


@cache_data(show_spinner=False)
def host_listings(host_id):
    df = load_data()
    sub = df[df["host_id"] == host_id]
    return sub[["id", "neighbourhood_cleansed", "room_type"]].to_dict("records")


def get_row(listing_id):
    df = load_data()
    r = df[df["id"] == listing_id]
    return None if r.empty else r.iloc[0]


# ─── 即時推論 + 貢獻度歸因 ─────────────────────────────────────
@cache_data(show_spinner=False)
def _nbhd_prices():
    df = load_data()
    out = {}
    for k, v in df.groupby("neighbourhood_code"):
        out[int(k)] = pd.to_numeric(v["price"], errors="coerce").dropna().values
    return out


def _expand(row, overrides):
    """調整價格時，連動重算價格相關衍生特徵，讓沙盒真正反映影響。"""
    ov = dict(overrides or {})
    if "price" in ov:
        try:
            price = float(ov["price"])
            acc = float(ov.get("accommodates", row.get("accommodates") or 1) or 1)
            bed = float(ov.get("bedrooms", row.get("bedrooms") or 1) or 1)
            ov.setdefault("price_per_person", price / max(acc, 1))
            ov.setdefault("price_per_bedroom", price / max(bed, 1))
            arr = _nbhd_prices().get(int(row.get("neighbourhood_code")))
            if arr is not None and len(arr):
                ov.setdefault("price_pctl_nbhd", float((arr < price).mean()))
        except Exception:
            pass
    return ov


def _vector(row, overrides, feats, med):
    x = {}
    for f in feats:
        v = overrides.get(f, row.get(f))
        try:
            v = float(v)
            if np.isnan(v):
                v = med.get(f, 0.0)
        except Exception:
            v = med.get(f, 0.0)
        x[f] = v
    return pd.DataFrame([x])[feats]


def predict(row, overrides=None):
    """回傳 (空屋率預測 0-1, 高風險機率 0-1)。"""
    M = get_models(); feats, med = M["features"], M["median"]
    X = _vector(row, _expand(row, overrides), feats, med)
    vac = float(np.clip(M["reg"].predict(X)[0], 0, 1))
    risk = float(M["clf"].predict_proba(X)[0, 1])
    return vac, risk


def contributions(row, overrides=None, top=None):
    """
    基準對照邊際貢獻：對每個特徵，計算「現值 vs 全體中位數」對高風險機率的影響
    （百分點）。正值＝推高空屋風險（紅、扣分），負值＝降低風險（綠、加分）。
    回傳依絕對值排序的 [(feature, zh_name, delta_pp)]。
    """
    M = get_models(); feats, med = M["features"], M["median"]
    clf = M["clf"]
    base_vec = _vector(row, _expand(row, overrides), feats, med)
    base_p = clf.predict_proba(base_vec)[0, 1]
    out = []
    for f in feats:
        alt = base_vec.copy()
        alt.iloc[0, alt.columns.get_loc(f)] = med.get(f, 0.0)
        p_alt = clf.predict_proba(alt)[0, 1]
        delta = (base_p - p_alt) * 100.0        # 現值相對基準的加/減風險（百分點）
        if abs(delta) >= 0.05:
            out.append((f, ZH.get(f, f), round(float(delta), 2)))
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return out[:top] if top else out


# ─── 白話診斷規則引擎（規格書 §6）────────────────────────────
def diagnose(row, overrides=None, k=2):
    """取 SHAP 正值最大（推高風險）的 Top-k 特徵，輸出白話優化建議。"""
    ov = _expand(row, overrides)
    pos = [c for c in contributions(row, overrides) if c[2] > 0][:k]

    def val(f):
        v = ov.get(f, row.get(f))
        try:
            return float(v)
        except Exception:
            return None

    rules = {
        "price_pctl_nbhd": lambda: (
            "您的定價高於同區約 {:.0f}% 房源，建議調降 5%–10% 以釋放競爭排位。"
            .format((val("price_pctl_nbhd") or 0) * (100 if (val("price_pctl_nbhd") or 0) <= 1 else 1))),
        "price": lambda: "每晚房價偏高，建議測試調降 5%–10% 觀察空屋率變化。",
        "minimum_nights": lambda: (
            "最低入住天數為 {:.0f} 晚過長，阻擋了短途商旅，建議放寬至 1–2 晚。"
            .format(val("minimum_nights") or 0)),
        "response_speed": lambda: (
            "客服回覆速度偏慢，強烈建議開啟即時預訂或將回覆提升至一小時內。"),
        "dist_to_nearest_mrt_m": lambda: (
            "房源距最近捷運約 {:.0f} 公尺、交通偏遠，建議文案補充公車/叫車指引並於房價讓利。"
            .format(val("dist_to_nearest_mrt_m") or 0)),
        "avg_review_sentiment": lambda: (
            "住客評論情感偏負面，建議針對歷史投訴（清潔、熱水、噪音等）具體改善。"),
        "review_scores_value": lambda: "性價比評分偏低，建議調整定價或增加超值附加服務。",
        "review_scores_cleanliness": lambda: "清潔度評分偏低，建議加強清潔標準或委外深清。",
        "desc_len": lambda: "房源描述偏短，建議補充房源賣點與周邊機能，提升點閱轉換。",
        "amenities_count": lambda: "設施數量偏少，建議補齊常見備品（Wifi、洗衣機、廚房等）。",
    }
    tips = []
    for f, zh, dpp in pos:
        msg = rules.get(f, lambda: "{} 目前推高了空屋風險，建議優化。".format(zh))()
        tips.append({"feature": f, "zh": zh, "delta": dpp, "advice": msg})
    return tips


def poi_snapshot(row):
    """Nearby POI + NLP snapshot (read-only) for the sandbox left column."""
    def g(f, d=0):
        try:
            v = float(row.get(f))
            return d if np.isnan(v) else v
        except Exception:
            return d
    mrt_name, park_name = "—", "—"
    try:
        from modules.geo_utils import load_all_poi, nearest_poi
        _poi = load_all_poi()
        _lat, _lon = float(row.get("latitude")), float(row.get("longitude"))
        mrt_name = str(nearest_poi(_lat, _lon, _poi["mrt"])[0])
        if mrt_name and mrt_name != "未知" and not mrt_name.endswith("站"):
            mrt_name += "站"
        park_name = str(nearest_poi(_lat, _lon, _poi["park"])[0])
    except Exception:
        pass
    return {
        "mrt_m": g("dist_to_nearest_mrt_m"), "mrt_500": g("mrt_count_500m"),
        "mrt_name": mrt_name, "park_name": park_name,
        "conv_500": g("conv_stores_count_500m"), "rest_500": g("restaurants_count_500m"),
        "park_m": g("dist_to_nearest_park_m"), "park_500": g("park_count_500m"),
        "sentiment": g("avg_review_sentiment"), "rev_len": g("avg_review_length"),
    }


def confidence(row):
    """Prediction-confidence label (spec 3.2)."""
    hlc = row.get("host_listings_count")
    rr = row.get("review_scores_rating")
    try:
        has_rating = rr is not None and pd.notna(rr) and float(rr) > 0
    except Exception:
        has_rating = False
    try:
        multi = hlc is not None and pd.notna(hlc) and float(hlc) > 1
    except Exception:
        multi = False
    if multi or has_rating:
        return ("極高", "基於房東歷史經營與評分")
    return ("中等", "新房源冷啟動，基於大數據機能估算")


def sf_price(row):
    try:
        v = float(row.get("price"))
        return 1500.0 if np.isnan(v) else v
    except Exception:
        return 1500.0


def sf_int(row, col, default):
    try:
        v = float(row.get(col))
        return default if np.isnan(v) else v
    except Exception:
        return default
