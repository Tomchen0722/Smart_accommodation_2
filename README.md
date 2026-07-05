# 智慧旅宿平台 · Smart Accommodation

台北市 Airbnb 房源智慧分析平台 — 空房預測（ML）、地理空間便利性評分、NLP 評論情感分析。
房東入口 / 租客入口 / 後台分析三端，使用 Streamlit 多頁應用。

## 本機執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 將整個專案推上 GitHub（公開或私有皆可）。
2. 前往 https://share.streamlit.io → New app。
3. 選擇 repo、branch，Main file path 設為 `app.py`。
4. Deploy。相依套件會自動依 `requirements.txt` 安裝。

### 注意事項
- 大型資料以壓縮檔提交：`data/*.csv.gz`（`data_loader` 會自動優先讀取 `.csv.gz`，找不到才讀 `.csv`）。
  原始未壓縮的 `reviews_cleaned.csv`（108MB）與 `listings_cleaned.csv` 已由 `.gitignore` 排除，
  避免超過 GitHub 單檔 100MB 限制。
- 主題設定在 `.streamlit/config.toml`。
- Python 版本可在 Streamlit Cloud 的 Advanced settings 指定（建議 3.11+）。

## 專案結構

```
app.py                     首頁（分流入口 + Hero）
pages/                     房東入口 / 租客入口 / 後台分析
modules/
  data_loader.py           資料載入（自動讀 .csv.gz）
  ml_models.py             風險評分 + 空房預測（LR / RF）
  geo_utils.py             Haversine + 7 類 PoI 便利性評分
  nlp_analysis.py          VADER + jieba 情感分析
  ui_components.py         設計 tokens / CSS / 共用元件 / 側邊導覽
data/                      資料集（房源、評論、7 類 PoI）
```

## 資料來源
Inside Airbnb · 台北市政府開放資料（捷運、公車、超商、餐廳、學校、診所、公園）
