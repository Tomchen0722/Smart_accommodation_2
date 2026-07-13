# -*- coding: utf-8 -*-
"""
build_dataset_final.py — 重建模型訓練資料集 dataset_final.csv
================================================================
依「模型開發總結報告」§1-§3 規格重建，由 listings_cleaned.csv 產出：
  - 目標變數 Y_vacancy = availability_365 / 365（未來一年空屋率）
  - Y_high_risk = (Y_vacancy > 0.7)（70% 通知門檻）
  - 36 個模型特徵（結構化13 + 評分7 + 競爭5 + 房東身分7 + 經營用心4）
  - 6 個冷啟動地點/房間特徵（外部特徵實驗報告驗證有效組）

與原報告的已知差異（誠實揭露）：
  - 缺 photo_design_sense（CLIP 需重下 5,849 張圖、400+ 分鐘，不重跑）
    → 本版為 36 特徵（報告為 37）
  - 旅宿 POI 為 680 筆（報告 732，OSM 資料時點差異）

缺值策略：數值特徵「保留 NaN」不在此補值 —— HistGradientBoosting 原生支援
缺值；線性基準模型的補值器放在訓練管線內、只 fit 訓練集（ml-modeling 規範，
避免補值統計量洩漏測試集資訊）。

執行方式（本機）：
  C:\\Users\\USER\\anaconda3\\python.exe scripts\\build_dataset_final.py
"""
from pathlib import Path
import json
import datetime

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ── 路徑設定 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_CSV = DATA_DIR / "listings_cleaned.csv"
HOTELS_CSV = DATA_DIR / "hotels_taipei_osm.csv"
OUT_CSV = DATA_DIR / "dataset_final.csv"
OUT_META = DATA_DIR / "dataset_final.meta.json"

EARTH_RADIUS_M = 6371000.0  # 地球半徑（公尺），BallTree haversine 用

# ── 特徵分組定義（後續訓練/SHAP/UI 共用的唯一事實來源）──
FEATURES_STRUCTURAL = [  # 結構化 13
    "accommodates", "bedrooms", "beds", "bathrooms_count", "is_shared_bath",
    "price", "minimum_nights", "maximum_nights", "min_nights_avg_ntm",
    "instant_bookable", "self_checkin", "room_type_code", "neighbourhood_code",
]
FEATURES_SCORES = [      # 評分 7
    "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location",
    "review_scores_value",
]
FEATURES_COMPETITION = [  # 競爭 5（平台核心設計，與地圖 UI 同源）
    "price_pctl_nbhd", "score_pctl_nbhd", "amenities_vs_median",
    "nbr_density_1km", "nbr_density_same_type_1km",
]
FEATURES_HOST = [         # 房東身分 7（冷啟動模型移除這一組）
    "host_acceptance_rate", "host_response_rate", "response_speed",
    "host_is_superhost", "host_listings_count",
    "calculated_host_listings_count", "host_tenure_days",
]
FEATURES_EFFORT = [       # 經營用心度 4
    "desc_len", "host_about_len", "neighborhood_overview_len",
    "amenities_count",
]
FEATURES_LOCATION = [     # 冷啟動地點/房間 6（外部特徵實驗驗證組）
    "hotel_count_1km", "hotel_count_500m", "airbnb_hotel_supply_ratio",
    "price_per_person", "price_per_bedroom", "beds_per_person",
]
FEATURES_FULL = (FEATURES_STRUCTURAL + FEATURES_SCORES +
                 FEATURES_COMPETITION + FEATURES_HOST + FEATURES_EFFORT)
META_COLS = ["id", "host_id", "latitude", "longitude",
             "neighbourhood_cleansed", "room_type"]

ledger = []  # 每步筆數記帳


def log_step(step, before, after, reason):
    """清洗記帳：處理前筆數 → 處理後筆數 → 差異原因。"""
    ledger.append({"步驟": step, "前": int(before), "後": int(after),
                   "差異": int(before - after), "原因": reason})
    print(f"[記帳] {step}: {before} → {after}（{reason}）")


def validate_schema(df):
    """入口驗證：缺必要欄位直接報錯，不默默繼續（data-pipeline 規範）。"""
    required = {"id", "host_id", "host_since", "last_scraped", "price",
                "availability_365", "latitude", "longitude", "room_type",
                "neighbourhood_cleansed", "amenities", "accommodates"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}")
    return df


def parse_percent(series):
    """'95%' 字串 → 0.95；已是數值則直接回傳（>1 視為百分比再除 100）。"""
    if series.dtype == object:
        out = (series.astype(str).str.rstrip("%")
               .replace({"nan": np.nan, "N/A": np.nan}))
        out = pd.to_numeric(out, errors="coerce") / 100.0
        return out
    out = pd.to_numeric(series, errors="coerce")
    return np.where(out > 1.0, out / 100.0, out)


def parse_price(series):
    """'$1,234.00' 字串 → 1234.0；已是數值則直接回傳。"""
    if series.dtype == object:
        return pd.to_numeric(
            series.astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def parse_bool_tf(series):
    """'t'/'f' → 1/0；缺值保留 NaN。"""
    return series.map({"t": 1, "f": 0, True: 1, False: 0})


def build_base_features(df):
    """建構結構化 / 房東 / 評分 / 用心度特徵（不含競爭與地點）。"""
    # ── 價格與比例欄位清洗 ──
    df["price"] = parse_price(df["price"])
    df["host_response_rate"] = parse_percent(df["host_response_rate"])
    df["host_acceptance_rate"] = parse_percent(df["host_acceptance_rate"])

    # ── 布林欄位 ──
    df["instant_bookable"] = parse_bool_tf(df["instant_bookable"])
    df["host_is_superhost"] = parse_bool_tf(df["host_is_superhost"])

    # ── 房東回覆速度：序數編碼（比回覆率多一層「快慢」維度）──
    speed_map = {"within an hour": 4, "within a few hours": 3,
                 "within a day": 2, "a few days or more": 1}
    df["response_speed"] = df["host_response_time"].map(speed_map).fillna(0)

    # ── 自助入住：amenities 關鍵字比對（報告 §3.3 採用特徵）──
    amen_lower = df["amenities"].fillna("[]").astype(str).str.lower()
    kw = ["self check-in", "self-check-in", "lockbox", "keypad", "smart lock"]
    df["self_checkin"] = amen_lower.apply(
        lambda s: int(any(k in s for k in kw)))

    # ── 經營用心度：文字長度 + 設施數 ──
    df["desc_len"] = df["description"].fillna("").astype(str).str.len()
    df["host_about_len"] = df["host_about"].fillna("").astype(str).str.len()
    df["neighborhood_overview_len"] = (
        df["neighborhood_overview"].fillna("").astype(str).str.len())
    # 設施數：以逗號切割計數（amenities 為 JSON 樣式字串）
    df["amenities_count"] = amen_lower.apply(
        lambda s: 0 if s in ("[]", "nan") else s.count(",") + 1)

    # ── 類別編碼（固定排序保證可重現）──
    df["room_type_code"] = pd.Categorical(
        df["room_type"], categories=sorted(df["room_type"].dropna().unique())
    ).codes
    df["neighbourhood_code"] = pd.Categorical(
        df["neighbourhood_cleansed"],
        categories=sorted(df["neighbourhood_cleansed"].dropna().unique())
    ).codes

    # ── 欄位改名對齊報告命名 ──
    df["min_nights_avg_ntm"] = pd.to_numeric(
        df["minimum_nights_avg_ntm"], errors="coerce")
    return df


def build_competition_features(df):
    """競爭特徵 5 個（報告 §3.3.1）：相對定價 / 相對口碑 / 相對設施 / 供給密度。

    - 百分位：同「行政區 × 房型」群組內排名（滯銷是相對的）
    - 密度：BallTree haversine 半徑 1km 鄰居數（減掉自己）
    """
    grp = df.groupby(["neighbourhood_cleansed", "room_type"])
    df["price_pctl_nbhd"] = grp["price"].rank(pct=True)
    df["score_pctl_nbhd"] = grp["review_scores_rating"].rank(pct=True)
    med = grp["amenities_count"].transform("median").replace(0, np.nan)
    df["amenities_vs_median"] = df["amenities_count"] / med

    # ── 地理密度：全房源 1km 鄰居 ──
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    r_1km = 1000.0 / EARTH_RADIUS_M
    df["nbr_density_1km"] = tree.query_radius(
        coords, r=r_1km, count_only=True) - 1  # 減掉自己

    # ── 同房型密度：各房型分別建樹 ──
    same_type = np.zeros(len(df), dtype=int)
    for rt, sub in df.groupby("room_type"):
        sub_coords = np.radians(sub[["latitude", "longitude"]].to_numpy())
        sub_tree = BallTree(sub_coords, metric="haversine")
        cnt = sub_tree.query_radius(sub_coords, r=r_1km, count_only=True) - 1
        same_type[sub.index.to_numpy()] = cnt
    df["nbr_density_same_type_1km"] = same_type
    return df


def build_location_features(df, hotels):
    """冷啟動地點/房間特徵 6 個（外部特徵實驗報告 H1/H3 驗證組）。"""
    # ── 飯店密度（H1：實測為「需求代理」訊號，非競爭懲罰）──
    listing_rad = np.radians(df[["latitude", "longitude"]].to_numpy())
    hotel_rad = np.radians(hotels[["latitude", "longitude"]].to_numpy())
    tree = BallTree(hotel_rad, metric="haversine")
    df["hotel_count_1km"] = tree.query_radius(
        listing_rad, r=1000.0 / EARTH_RADIUS_M, count_only=True)
    df["hotel_count_500m"] = tree.query_radius(
        listing_rad, r=500.0 / EARTH_RADIUS_M, count_only=True)

    # ── 供需比（H2）：Airbnb 供給相對飯店供給（+1 避免除零）──
    df["airbnb_hotel_supply_ratio"] = (
        df["nbr_density_1km"] / (df["hotel_count_1km"] + 1))

    # ── 房間級性價比（H3）──
    acc = df["accommodates"].replace(0, np.nan)
    df["price_per_person"] = df["price"] / acc
    df["price_per_bedroom"] = df["price"] / df["bedrooms"].replace(0, np.nan)
    df["beds_per_person"] = df["beds"] / acc
    return df


def main():
    """主流程：讀檔 → 驗證 → 清洗（記帳）→ 特徵 → 出口核對 → 寫檔。"""
    print(f"[讀取] {SRC_CSV.name}")
    df = pd.read_csv(SRC_CSV, encoding="utf-8", low_memory=False)
    n_input = len(df)
    validate_schema(df)
    print(f"[入口] {n_input} 筆 × {len(df.columns)} 欄")

    # ── 清洗 1：座標缺值剔除（競爭/地點特徵必要條件）──
    n0 = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    log_step("剔除座標缺值", n0, len(df), "無法計算地理特徵")

    # ── 清洗 2：availability_365 缺值剔除（目標變數必要條件）──
    n0 = len(df)
    df = df.dropna(subset=["availability_365"])
    log_step("剔除目標缺值", n0, len(df), "Y_vacancy 無法計算")

    # ── 清洗 3：剔除經營未滿 1 年（報告 §1.2：避免剛上架占位偏差）──
    ref_date = pd.to_datetime(df["last_scraped"], errors="coerce").max()
    host_since = pd.to_datetime(df["host_since"], errors="coerce")
    df["host_tenure_days"] = (ref_date - host_since).dt.days
    n0 = len(df)
    df = df[df["host_tenure_days"] >= 365]
    log_step("剔除經營未滿1年", n0, len(df),
             f"host_tenure_days < 365（基準日 {ref_date.date()}）")

    df = df.reset_index(drop=True)

    # ── 目標變數 ──
    df["Y_vacancy"] = (df["availability_365"] / 365).clip(0, 1)
    df["Y_high_risk"] = (df["Y_vacancy"] > 0.7).astype(int)

    # ── 特徵建構 ──
    df = build_base_features(df)
    df = build_competition_features(df)

    hotels = pd.read_csv(HOTELS_CSV, encoding="utf-8-sig")
    print(f"[讀取] {HOTELS_CSV.name}（{len(hotels)} 筆旅宿 POI）")
    df = build_location_features(df, hotels)

    # ── 數值型別統一（保留 NaN，不在此補值 —— 見檔頭缺值策略）──
    all_feats = FEATURES_FULL + FEATURES_LOCATION
    for col in all_feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 出口核對：等式必須成立（data-pipeline 規範）──
    dropped = sum(item["差異"] for item in ledger)
    assert n_input - dropped == len(df), \
        f"出口等式不成立：{n_input} - {dropped} != {len(df)}（有資料默默消失）"

    out = df[META_COLS + ["Y_vacancy", "Y_high_risk"] + all_feats].copy()
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ── metadata sidecar ──
    nan_stats = {c: int(out[c].isna().sum()) for c in all_feats
                 if out[c].isna().sum() > 0}
    meta = {
        "產出時間": datetime.datetime.now().isoformat(timespec="seconds"),
        "腳本": "scripts/build_dataset_final.py",
        "來源檔": SRC_CSV.name,
        "入口筆數": n_input,
        "出口筆數": int(len(out)),
        "清洗記帳": ledger,
        "特徵數_完整模型": len(FEATURES_FULL),
        "特徵數_冷啟動": len(FEATURES_FULL) - len(FEATURES_HOST)
                        + len(FEATURES_LOCATION),
        "特徵分組": {
            "結構化": FEATURES_STRUCTURAL, "評分": FEATURES_SCORES,
            "競爭": FEATURES_COMPETITION, "房東身分_冷啟動移除": FEATURES_HOST,
            "經營用心度": FEATURES_EFFORT, "地點房間_冷啟動加入": FEATURES_LOCATION,
        },
        "高風險占比": round(float(out["Y_high_risk"].mean()), 4),
        "缺值統計": nan_stats,
        "與報告已知差異": "缺 photo_design_sense（CLIP 不重跑）；旅宿 POI 680 筆",
        "報告基準": {"筆數": 5849, "高風險占比": 0.314},
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[完成] {OUT_CSV.name}：{len(out)} 筆 × {len(out.columns)} 欄"
          f"（完整模型 {len(FEATURES_FULL)} 特徵 / 冷啟動 "
          f"{meta['特徵數_冷啟動']} 特徵）")
    print(f"[核對] 高風險占比 {meta['高風險占比']:.1%}（報告基準 31.4%）")


if __name__ == "__main__":
    main()
