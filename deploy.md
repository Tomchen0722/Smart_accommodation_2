# 部署到 Streamlit Community Cloud

## 一、先把專案推上 GitHub（重點：不要把 108MB 原始 CSV 推上去）

大型原始資料（`data/reviews_cleaned.csv` 108MB、`data/listings_cleaned.csv`）已被 `.gitignore`
排除，實際部署讀取的是壓縮檔 `data/*.csv.gz`。若你之前 push 失敗（GitHub 單檔上限 100MB），
表示舊的 git 歷史裡含有那個大檔，需要重建乾淨的歷史。

### 在你自己的電腦操作（PowerShell，先關閉 VS Code 以免 .git 被鎖住）

```powershell
cd C:\AI\Smart_accommodation_2

# 1) 刪除損壞/含大檔的舊 git 歷史（你的檔案不會被刪，只清 git 內部資料）
Remove-Item -Recurse -Force .git

# 2) 重新初始化（.gitignore 會自動排除 108MB 原始 CSV 與 __pycache__）
git init
git add .
git commit -m "Deploy-ready build"
git branch -M main

# 3) 連到你的 GitHub repo，強制推送（覆蓋空的/失敗的遠端）
git remote add origin https://github.com/Tomchen0722/Smart_accommodation_2.git
git push -u origin main --force
```

推送前確認大檔沒被追蹤（應無輸出）：

```powershell
git ls-files | Select-String "reviews_cleaned.csv$"
```

> 若 `git push` 要求登入，使用 GitHub 帳號 + Personal Access Token（不是密碼）。

## 二、在 share.streamlit.io 部署

前往 https://share.streamlit.io → New app，填入：

| 欄位 | 值 |
| --- | --- |
| Repository | `Tomchen0722/Smart_accommodation_2` |
| Branch | `main` （不是 master） |
| Main file path | `app.py` （不是 streamlit_app.py） |

按 Deploy，相依套件會依 `requirements.txt` 自動安裝。

## 三、部署後檢查

- GitHub 網頁上 `data/` 內應是 `reviews_cleaned.csv.gz`（約 25MB），**不應**出現 `reviews_cleaned.csv`。
- App 開啟後：首頁 → 房東入口／租客入口／後台分析皆可切換；房源評論、地圖、PoI 皆正常。

## 常見問題

- **This branch does not exist**：分支填成 master；改成 `main`，或你尚未成功 push（見上）。
- **This file does not exist**：主檔填成 streamlit_app.py；改成 `app.py`。
- **push 被拒 / 檔案過大**：舊歷史仍含 108MB 檔，代表 `.git` 沒刪乾淨，重做第一節步驟 1–2。
- **記憶體不足**：Streamlit Cloud 免費層約 1GB；本專案冷啟動約 5 秒、記憶體足夠。

## 資料檔說明

| 檔案 | 說明 |
| --- | --- |
| `data/reviews_cleaned.csv.gz` | 評論（精簡欄位、壓縮）；`data_loader` 會自動優先讀 `.csv.gz` |
| `data/listings_cleaned.csv.gz` | 房源（壓縮） |
| `data/台北市公車站牌.csv` 等 | 7 類 PoI 開放資料 |
