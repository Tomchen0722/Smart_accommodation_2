# -*- coding: utf-8 -*-
"""
smoke_test_v2.py — v2 鏈路煙霧測試（驗收用，可重複執行）
================================================================
測試範圍：載入器 → 新舊房東自動路由 → 價格 what-if → SHAP 重建與可加總性。
任一斷言失敗即非零退出，供 CI / 手動驗收使用。
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import feature_engineering as fe  # noqa: E402


def main():
    df = fe.load_dataset_final()
    bundle = fe.load_bundle()
    cache = fe.load_shap_cache()

    # ── 測 1：新舊房東自動路由 ──
    old_host = df[df["calculated_host_listings_count"] > 3].iloc[0]
    new_host = df[df["calculated_host_listings_count"] <= 1].iloc[0]
    r1 = fe.predict_risk_v2(old_host, bundle)
    r2 = fe.predict_risk_v2(new_host, bundle)
    print(f"老房東 → {r1['variant']}｜風險 {r1['risk_score']:.3f}｜"
          f"通知機率 {r1['notify_prob']:.3f}｜門檻 {r1['threshold']:.3f}")
    print(f"新房東 → {r2['variant']}｜風險 {r2['risk_score']:.3f}｜"
          f"信心標註：{r2['confidence']}")
    assert r1["variant"] == "full" and r2["variant"] == "cold", "路由錯誤"

    # ── 測 2：價格 what-if（調價應改變預測且不炸掉）──
    sim = fe.simulate_price_change(old_host, bundle, old_host["price"] * 0.8)
    print(f"調價模擬：原 {r1['risk_score']:.3f} → 降價20%後 "
          f"{sim['risk_score']:.3f}")
    assert 0.0 <= sim["risk_score"] <= 1.0, "模擬預測超出範圍"

    # ── 測 3：SHAP 可加總性（base + Σshap ≈ 模型預測，SHAP 核心性質）──
    full = cache["full"]
    i = full["example_high_idx"]
    recon = full["base_value"] + full["shap_values"][i].sum()
    print(f"SHAP 可加總性：base+Σshap={recon:.4f} vs "
          f"模型預測={full['risk_pred'][i]:.4f}")
    assert abs(recon - full["risk_pred"][i]) < 1e-4, "SHAP 加總不一致"

    # ── 測 4：兩變體特徵名與模型輸入維度一致 ──
    for tag in ("full", "cold"):
        n_feat = len(bundle[tag]["feature_names"])
        n_cache = cache[tag]["X_sample"].shape[1]
        assert n_feat == n_cache, f"{tag} 特徵維度不一致 {n_feat} vs {n_cache}"
    print("[煙霧測試] v2 鏈路全部通過 ✓")


if __name__ == "__main__":
    main()
