# -*- coding: utf-8 -*-
"""
fetch_hotels_osm.py — 從 OpenStreetMap Overpass API 抓取台北市旅宿 POI
================================================================
目的：重建外部特徵實驗報告中的 hotels_taipei_osm.csv（約 732 筆）
      供冷啟動模型的地點特徵（hotel_count_1km / hotel_count_500m）使用。

依 data-pipeline 技能規範實作：
  1. 快取：輸出檔已存在則跳過（--force 可強制重抓）
  2. 重試：指數退避，每個鏡像最多 3 次
  3. 備援：三個 Overpass 公開鏡像依序降級
  4. 出口核對：輸出筆數記帳 + metadata sidecar

執行方式（本機）：
  C:\\Users\\USER\\anaconda3\\python.exe scripts\\fetch_hotels_osm.py
"""
from pathlib import Path
import json
import sys
import time
import datetime

import requests
import pandas as pd

# ── 常數設定 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # 專案根目錄
DATA_DIR = PROJECT_ROOT / "data"
OUT_CSV = DATA_DIR / "hotels_taipei_osm.csv"
OUT_META = DATA_DIR / "hotels_taipei_osm.meta.json"

# 台北市外接矩形（south, west, north, east）
BBOX = (24.96, 121.45, 25.22, 121.67)

# Overpass 鏡像備援順序（主 → 備）
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# 旅宿類型（與實驗報告一致：hotel / hostel / guest_house / motel）
TOURISM_TYPES = "hotel|hostel|guest_house|motel"

# Overpass QL 查詢：node + way + relation，way/relation 取幾何中心（out center）
QUERY = f"""
[out:json][timeout:180];
(
  node["tourism"~"^({TOURISM_TYPES})$"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["tourism"~"^({TOURISM_TYPES})$"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  relation["tourism"~"^({TOURISM_TYPES})$"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out center tags;
"""

MAX_RETRY = 3          # 每個鏡像最多重試次數
BASE_BACKOFF = 5       # 指數退避基準秒數（5 → 10 → 20）


def fetch_overpass():
    """依鏡像備援順序查詢 Overpass，回傳 (JSON 結果, 實際使用的鏡像)。

    失敗策略：單一鏡像指數退避重試 3 次仍失敗 → 換下一個鏡像。
    全部鏡像失敗 → 拋出例外並列出各鏡像最終錯誤。
    """
    errors = {}
    for mirror in OVERPASS_MIRRORS:
        for attempt in range(1, MAX_RETRY + 1):
            try:
                print(f"[抓取] 鏡像 {mirror}（第 {attempt}/{MAX_RETRY} 次）...")
                resp = requests.post(mirror, data={"data": QUERY}, timeout=200)
                if resp.status_code == 200:
                    return resp.json(), mirror
                # 429/504 屬可重試錯誤
                errors[mirror] = f"HTTP {resp.status_code}"
                print(f"[警告] HTTP {resp.status_code}，準備退避重試")
            except Exception as exc:  # 連線逾時等
                errors[mirror] = str(exc)
                print(f"[警告] 連線失敗：{exc}")
            if attempt < MAX_RETRY:
                wait = BASE_BACKOFF * (2 ** (attempt - 1))
                print(f"[退避] 等待 {wait} 秒後重試")
                time.sleep(wait)
        print(f"[降級] 鏡像 {mirror} 三次皆失敗，切換下一鏡像")
    raise RuntimeError(f"所有 Overpass 鏡像皆失敗：{errors}")


def parse_elements(raw):
    """將 Overpass 回傳的 elements 轉為 DataFrame。

    座標規則：node 直接取 lat/lon；way/relation 用 center。
    去重規則：以 (osm_type, osm_id) 為唯一鍵。
    """
    rows = []
    skipped_no_coord = 0
    for el in raw.get("elements", []):
        tags = el.get("tags", {})
        # node 有 lat/lon；way/relation 用 center
        if "lat" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            skipped_no_coord += 1  # 記帳：無座標者拒絕
            continue
        rows.append({
            "osm_type": el.get("type"),
            "osm_id": el.get("id"),
            "name": tags.get("name", tags.get("name:zh", "")),
            "tourism_type": tags.get("tourism", ""),
            "latitude": lat,
            "longitude": lon,
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["osm_type", "osm_id"])
    print(f"[記帳] 原始 elements={len(raw.get('elements', []))} → "
          f"無座標剔除={skipped_no_coord} → 去重後={len(df)}")
    return df


def main():
    """主流程：快取檢查 → 抓取 → 解析 → 出口核對 → 寫檔。"""
    force = "--force" in sys.argv
    DATA_DIR.mkdir(exist_ok=True)

    # ── 快取檢查（data-pipeline 四件套之一）──
    if OUT_CSV.exists() and not force:
        cached = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
        print(f"[快取] {OUT_CSV.name} 已存在（{len(cached)} 筆），跳過抓取。"
              f"如需重抓請加 --force")
        return

    raw, mirror_used = fetch_overpass()
    df = parse_elements(raw)

    # ── 出口驗證：筆數需在合理範圍（報告基準 732 筆 ±40%）──
    if not (400 <= len(df) <= 1200):
        print(f"[警告] 筆數 {len(df)} 偏離報告基準 732 筆過多，請人工檢查查詢條件")

    # 型別統計（供 metadata 與人工檢核）
    type_counts = df["tourism_type"].value_counts().to_dict()

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    meta = {
        "產出時間": datetime.datetime.now().isoformat(timespec="seconds"),
        "腳本": "scripts/fetch_hotels_osm.py",
        "資料源": f"Overpass API（{mirror_used}）",
        "查詢範圍_bbox": BBOX,
        "旅宿類型": TOURISM_TYPES,
        "輸出筆數": int(len(df)),
        "類型分布": type_counts,
        "報告基準筆數": 732,
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[完成] 寫出 {OUT_CSV}（{len(df)} 筆）+ metadata sidecar")
    print(f"[分布] {type_counts}")


if __name__ == "__main__":
    main()
