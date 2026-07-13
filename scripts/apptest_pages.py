# -*- coding: utf-8 -*-
"""
apptest_pages.py — Streamlit AppTest 無頭整合測試（驗收用）
================================================================
實際執行「房東入口」與「後台分析」兩個頁面腳本（含資料載入、v2 模型載入、
SHAP 圖渲染），回報任何執行期例外。

測試樁說明：st.page_link 需要多頁應用執行環境，AppTest 單頁模式下會拋
KeyError('url_pathname')（測試框架限制、非應用程式 bug），故以 no-op 打樁；
其餘元件皆真實執行。
"""
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

# ── 測試樁：page_link / switch_page 在 AppTest 單頁環境無法運作 ──
st.page_link = lambda *a, **k: None
st.switch_page = lambda *a, **k: None

PAGES = [
    PROJECT_ROOT / "pages" / "1_🏠_房東入口.py",
    PROJECT_ROOT / "pages" / "3_📊_後台分析.py",
]


def run_page(path):
    """執行單一頁面，回傳例外清單。"""
    t0 = time.time()
    at = AppTest.from_file(str(path), default_timeout=900)
    at.run()
    errs = [str(e.value) for e in at.exception]
    print(f"[{path.name}] {time.time()-t0:.0f} 秒｜"
          f"例外 {len(errs)} 個" + ("" if not errs else " ↓"))
    for e in errs:
        print("   ⚠", e[:300])
    return errs


def main():
    all_errs = []
    for p in PAGES:
        all_errs += run_page(p)
    if all_errs:
        sys.exit(f"[失敗] 共 {len(all_errs)} 個執行期例外")
    print("[整合測試] 兩頁皆無執行期例外 ✓")


if __name__ == "__main__":
    main()
