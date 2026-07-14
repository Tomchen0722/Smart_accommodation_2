"""
Geo Utilities — Haversine distance, PoI search, convenience scoring.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ─── Haversine ──────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance in meters.
    Accepts scalars or numpy arrays.
    """
    R = 6_371_000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def listings_within_radius(df, center_lat, center_lon, radius_m=1000):
    """Return subset of df within radius_m meters of center point."""
    dists = haversine(center_lat, center_lon,
                      df["latitude"].values, df["longitude"].values)
    mask = dists <= radius_m
    result = df[mask].copy()
    result["distance_m"] = dists[mask]
    return result


# ─── PoI Loaders ────────────────────────────────────────────────
def _load_mrt():
    """Load MRT station exits with coordinates."""
    path = DATA_DIR / "臺北捷運車站出入口座標.csv"
    try:
        df = pd.read_csv(path, encoding="big5")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8")
    cols = df.columns.tolist()
    # Columns: 序, 出入口名稱, 出入口編號, 經度, 緯度, 是否有無障礙
    df.columns = ["seq", "exit_name", "exit_no", "longitude", "latitude", "accessible"]
    # Extract station name (remove exit info like "出口1")
    df["station_name"] = df["exit_name"].str.replace(r"出口?\d+.*$", "", regex=True).str.strip()
    # Use the exit name as the address/detail field for hover & lists
    df["address"] = df["exit_name"]
    return df[["station_name", "exit_name", "address",
               "latitude", "longitude"]].dropna()


def _load_restaurants():
    """Load restaurant PoI data."""
    path = DATA_DIR / "taipei_restaurants.csv"
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="big5")
    cols = df.columns.tolist()
    # Columns: 共用編號, 商業名稱, 商業地址, Longitude, Latitude
    df.columns = ["seq", "name", "address", "longitude", "latitude"]
    # If an address cell holds multiple addresses, keep only the first
    df["address"] = (df["address"].astype(str)
                     .str.split(r"[、;；\n]", regex=True).str[0].str.strip())
    return df[["name", "address", "latitude", "longitude"]].dropna(subset=["latitude", "longitude"])


def _load_schools():
    """Load school PoI data."""
    path = DATA_DIR / "taipei_school.csv"
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="big5")
    cols = df.columns.tolist()
    # Columns: 序, school, schoolname, postalcode, address, telephone, 經度, 緯度
    df.columns = ["seq", "school_type", "school_name", "postalcode",
                  "address", "telephone", "longitude", "latitude"]
    return df[["school_name", "school_type", "address", "latitude", "longitude"]].dropna(subset=["latitude", "longitude"])


def _load_clinics():
    """Load clinic PoI data."""
    path = DATA_DIR / "taipei_clinics.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Keys: lat, lon, name, address
    df = df.rename(columns={"lon": "longitude", "lat": "latitude"})
    return df[["name", "address", "latitude", "longitude"]].dropna(subset=["latitude", "longitude"])


def _load_parks():
    """Load park PoI data."""
    path = DATA_DIR / "臺北市公園基本資料.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "pm_name": "name", "pm_Latitude": "latitude",
        "pm_Longitude": "longitude", "pm_location": "address",
    })
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df[["name", "address", "latitude", "longitude"]].dropna(subset=["latitude", "longitude"])


# 超商公司名稱 -> 常見品牌短名
_CONV_BRAND = {
    "統一超商": "7-ELEVEN",
    "全家便利商店": "全家",
    "萊爾富": "萊爾富",
    "全聯實業": "全聯",
    "富達零售": "OK超商",
}


def _conv_brand(company):
    """Map a convenience-store company name to a short brand label."""
    s = str(company)
    for key, brand in _CONV_BRAND.items():
        if key in s:
            return brand
    return s.replace("股份有限公司", "").replace("有限公司", "").strip()


def _load_convenience():
    """Load convenience store / supermarket PoI data."""
    path = DATA_DIR / "台北市超商資料集_含經緯度_門牌級.csv"
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")
    # Keep only active branches (狀態 == 1); 3 = 廢止, 6 = other
    if "分公司狀態" in df.columns:
        df = df[df["分公司狀態"] == 1]
    df = df.rename(columns={
        "分公司名稱": "name", "分公司地址": "address",
        "緯度": "latitude", "經度": "longitude", "公司名稱": "brand",
    })
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    # Show the brand (e.g. 全聯 / 7-ELEVEN / 全家) as the display name
    df["branch"] = df["name"]
    df["name"] = df["brand"].map(_conv_brand)
    return df[["name", "address", "latitude", "longitude"]].dropna(
        subset=["latitude", "longitude"])


def _load_busstop():
    """Load Taipei bus stop PoI data (converted from shapefile to CSV)."""
    path = DATA_DIR / "台北市公車站牌.csv"
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df[["name", "address", "latitude", "longitude"]].dropna(
        subset=["latitude", "longitude"])


# Cached PoI loaders
_poi_cache = {}

def load_all_poi():
    """Load all PoI datasets, cached."""
    if not _poi_cache:
        _poi_cache["mrt"] = _load_mrt()
        _poi_cache["bus"] = _load_busstop()
        _poi_cache["convenience"] = _load_convenience()
        _poi_cache["restaurant"] = _load_restaurants()
        _poi_cache["school"] = _load_schools()
        _poi_cache["clinic"] = _load_clinics()
        _poi_cache["park"] = _load_parks()
    return _poi_cache


# ─── PoI Convenience Score ──────────────────────────────────────
POI_NAMES = {
    "mrt": "🚇 捷運站",
    "bus": "🚏 公車站",
    "convenience": "🏪 超商",
    "restaurant": "🍜 餐廳",
    "school": "🏫 學校",
    "clinic": "🏥 診所",
    "park": "🌳 公園",
}


def poi_name_col(poi_df):
    """Return the first available name-like column for a PoI DataFrame."""
    return next(
        (c for c in ("station_name", "name", "school_name", "exit_name")
         if c in poi_df.columns),
        None,
    )


def count_poi_within(lat, lon, poi_df, radius_m=1000):
    """Count PoI within radius."""
    dists = haversine(lat, lon, poi_df["latitude"].values, poi_df["longitude"].values)
    return int((dists <= radius_m).sum())


def nearest_poi(lat, lon, poi_df):
    """Find nearest PoI and its distance."""
    if poi_df.empty:
        return None, float("inf")
    dists = haversine(lat, lon, poi_df["latitude"].values, poi_df["longitude"].values)
    idx = int(np.argmin(dists))
    name_col = poi_name_col(poi_df)
    label = poi_df.iloc[idx][name_col] if name_col else "未知"
    return label, float(dists[idx])


def poi_points_within(lat, lon, poi_df, radius_m=1000):
    """
    Return all PoI points within radius as a tidy DataFrame with unified
    columns: poi_name, poi_addr, latitude, longitude, distance_m -- sorted
    by distance. Used for map hover, detail lists, and counts.
    """
    empty = pd.DataFrame(columns=["poi_name", "poi_addr", "latitude",
                                   "longitude", "distance_m"])
    if poi_df is None or poi_df.empty:
        return empty
    dists = haversine(lat, lon, poi_df["latitude"].values,
                      poi_df["longitude"].values)
    mask = dists <= radius_m
    sub = poi_df[mask].copy()
    if sub.empty:
        return empty
    sub["distance_m"] = dists[mask]
    nc = poi_name_col(poi_df)
    sub["poi_name"] = sub[nc].astype(str) if nc else "未知"
    sub["poi_addr"] = (sub["address"].astype(str)
                       if "address" in sub.columns else "")
    return (sub[["poi_name", "poi_addr", "latitude", "longitude", "distance_m"]]
            .sort_values("distance_m").reset_index(drop=True))


def convenience_score(lat, lon, poi_dict):
    """
    Continuous convenience score (0-10) per PoI type, combining:
      • proximity  — how close the nearest facility is (linear decay to 1200m)
      • density    — how many facilities are within 800m (log-saturating)
    Continuous values (not 0/2/5/10 buckets) so listings differ from each
    other and the user's weights meaningfully change the ranking.
    Returns dict with per-type scores and total.
    """
    scores = {}
    for poi_type, poi_df in poi_dict.items():
        if poi_df.empty:
            scores[poi_type] = 0.0
            continue
        dists = haversine(lat, lon, poi_df["latitude"].values,
                          poi_df["longitude"].values)
        nearest = float(np.min(dists))
        prox = 10.0 * max(0.0, 1.0 - nearest / 1200.0)
        cnt = int((dists <= 800).sum())
        dens = 10.0 * min(1.0, np.log1p(cnt) / np.log1p(25))
        scores[poi_type] = round(0.6 * prox + 0.4 * dens, 2)
    scores["total"] = round(sum(scores.values()), 2)
    return scores


def batch_convenience_scores(df, poi_dict):
    """Calculate convenience scores for all listings in df."""
    results = []
    for _, row in df.iterrows():
        sc = convenience_score(row["latitude"], row["longitude"], poi_dict)
        sc["listing_id"] = row["id"]
        results.append(sc)
    return pd.DataFrame(results).set_index("listing_id")


# ─── 由經緯度推估地址（用最近門牌級超商/餐廳地址，離線、免 API）────
import re as _re
_addr_cache = {}

def nearest_address(lat, lon, road_level=True):
    """
    以房源座標找最近的「有門牌地址」PoI（超商為主、餐廳備援），
    回傳路段層級地址（預設去掉門牌號）。離線、即時、無 API 限制。
    找不到回傳空字串。
    """
    try:
        key = (round(float(lat), 5), round(float(lon), 5), road_level)
    except Exception:
        return ""
    if key in _addr_cache:
        return _addr_cache[key]
    poi = load_all_poi()
    best = ""
    for name in ("convenience", "restaurant"):
        df = poi.get(name)
        if df is None or df.empty or "address" not in df.columns:
            continue
        d = haversine(lat, lon, df["latitude"].values, df["longitude"].values)
        i = int(np.argmin(d))
        addr = str(df.iloc[i].get("address", "")).strip()
        if addr and addr.lower() != "nan":
            best = addr
            break
    if best and road_level:
        best = _re.split(r"[0-9〇零一二三四五六七八九十百千]+號", best)[0]   # 去掉門牌號（含中文數字）
        best = _re.sub(r"[0-9〇零一二三四五六七八九十百千\-、,~－]+$", "", best)
        best = best.rstrip("、,-－~ ").strip()
    _addr_cache[key] = best
    return best
