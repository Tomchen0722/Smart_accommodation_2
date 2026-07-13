# -*- coding: utf-8 -*-
"""
train_backend_models.py — 訓練後台雙模型（老房東完整版 + 新房東冷啟動版）
================================================================
依「模型開發總結報告」§4 架構訓練四個模型並序列化：

  完整模型（老房東，36 特徵）：
    A. HistGradientBoostingRegressor  → 風險分數（Y_vacancy 0~1）+ SHAP 解釋
    B. HistGradientBoostingClassifier + isotonic 校準 → 70% 通知決策
  冷啟動模型（新房東，35 特徵 = 36 - 房東身分7 + 地點房間6）：
    同上 A/B 結構

評估雙軌制（報告 §5 方法論）：
  ① 單次切分 60/20/20（train/val/test）—— 樂觀數字（含房東洩漏）
  ② GroupKFold(5) 依 host_id 分組 —— 誠實數字（面對全新房東）

ml-modeling 技能規範落實：
  - seed=42 固定並記錄；前處理（基準線的補值/標準化）只 fit 訓練集
  - HistGB 原生支援 NaN，不需補值器
  - 基準線對照：LinearRegression / LogisticRegression
  - preprocessor + model + feature_names + train_date 同存 joblib
  - 存檔後重新載入、對保留樣本重跑預測，數值一致才算保存成功

執行方式（本機）：
  C:\\Users\\USER\\anaconda3\\python.exe -X utf8 scripts\\train_backend_models.py
"""
from pathlib import Path
import json
import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (r2_score, mean_absolute_error, roc_auc_score,
                             f1_score, recall_score, precision_score,
                             precision_recall_curve, confusion_matrix)

# ── 路徑與常數 ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_CSV = DATA_DIR / "dataset_final.csv"
DATASET_META = DATA_DIR / "dataset_final.meta.json"
OUT_BUNDLE = MODEL_DIR / "backend_models_v2.joblib"
OUT_EVAL = MODEL_DIR / "eval_results.json"

SEED = 42                # 全域隨機種子（可重現性）
TEST_SIZE = 0.20         # 測試集比例
VAL_SIZE = 0.20          # 驗證集比例（用於門檻搜尋與校準）
RECALL_FLOOR = 0.80      # 通知門檻搜尋約束：Recall ≥ 0.80（寧可多抓不可漏抓）
N_FOLDS = 5              # GroupKFold 折數


def load_feature_groups():
    """從 dataset_final.meta.json 讀特徵分組 —— 資料集 metadata 是唯一事實來源。"""
    meta = json.loads(DATASET_META.read_text(encoding="utf-8"))
    g = meta["特徵分組"]
    full_feats = (g["結構化"] + g["評分"] + g["競爭"]
                  + g["房東身分_冷啟動移除"] + g["經營用心度"])
    cold_feats = (g["結構化"] + g["評分"] + g["競爭"]
                  + g["經營用心度"] + g["地點房間_冷啟動加入"])
    return full_feats, cold_feats, g


def search_threshold(y_true, y_prob):
    """在驗證集搜尋通知門檻：Recall ≥ RECALL_FLOOR 下 precision 最高的機率切點。

    找不到滿足約束的切點時退回 0.5 並記錄警告（報告 §4.2 門檻選擇規則）。
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve 的 thr 比 prec/rec 少一個元素，對齊處理
    candidates = [(p, r, t) for p, r, t in zip(prec[:-1], rec[:-1], thr)
                  if r >= RECALL_FLOOR]
    if not candidates:
        return 0.5, {"警告": f"無切點滿足 Recall≥{RECALL_FLOOR}，退回 0.5"}
    best = max(candidates, key=lambda x: x[0])  # precision 最高者
    return float(best[2]), {"precision": float(best[0]),
                            "recall": float(best[1])}


def clf_metrics(y_true, y_prob, threshold):
    """分類指標組：AUC + F1 + Recall + Precision + 混淆矩陣（至少兩指標規範）。"""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred,
                                                 zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": round(float(threshold), 4),
    }


def groupkfold_eval(X, y_reg, y_clf, groups, make_reg, make_clf):
    """GroupKFold 誠實評估：依 host_id 分組，回報平均 ± 標準差。

    分類模型以「未校準機率」計 AUC —— AUC 是排序指標，校準不改變其值。
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    r2s, maes, aucs = [], [], []
    for tr_idx, te_idx in gkf.split(X, y_reg, groups):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        reg = make_reg().fit(X_tr, y_reg.iloc[tr_idx])
        pred = reg.predict(X_te)
        r2s.append(r2_score(y_reg.iloc[te_idx], pred))
        maes.append(mean_absolute_error(y_reg.iloc[te_idx], pred))
        clf = make_clf().fit(X_tr, y_clf.iloc[tr_idx])
        prob = clf.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_clf.iloc[te_idx], prob))
    stat = lambda a: {"mean": round(float(np.mean(a)), 4),
                      "std": round(float(np.std(a)), 4)}
    return {"r2": stat(r2s), "mae": stat(maes), "auc": stat(aucs)}


def train_variant(name, df, feats, idx_tr, idx_va, idx_te):
    """訓練一個模型變體（完整版或冷啟動版），回傳 (模型物件 dict, 評估 dict)。"""
    X = df[feats]
    y_reg, y_clf = df["Y_vacancy"], df["Y_high_risk"]
    X_tr, X_va, X_te = X.iloc[idx_tr], X.iloc[idx_va], X.iloc[idx_te]

    # ── 模型 A：迴歸（風險分數 + SHAP 解釋來源）──
    reg = HistGradientBoostingRegressor(random_state=SEED)
    reg.fit(X_tr, y_reg.iloc[idx_tr])
    pred_te = reg.predict(X_te)

    # ── 模型 B：分類 + isotonic 校準（校準 fit 在驗證集，不碰訓練/測試）──
    clf_raw = HistGradientBoostingClassifier(random_state=SEED)
    clf_raw.fit(X_tr, y_clf.iloc[idx_tr])
    clf_cal = CalibratedClassifierCV(FrozenEstimator(clf_raw),
                                     method="isotonic")
    clf_cal.fit(X_va, y_clf.iloc[idx_va])

    # ── 門檻搜尋（驗證集，Recall≥0.80 下 precision 最高）──
    prob_va = clf_cal.predict_proba(X_va)[:, 1]
    threshold, thr_info = search_threshold(y_clf.iloc[idx_va], prob_va)
    prob_te = clf_cal.predict_proba(X_te)[:, 1]

    # ── 基準線：線性模型（補值+標準化管線，只 fit 訓練集）──
    base_reg = make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(), LinearRegression())
    base_reg.fit(X_tr, y_reg.iloc[idx_tr])
    base_clf = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced",
                           random_state=SEED))
    base_clf.fit(X_tr, y_clf.iloc[idx_tr])

    # ── 評估 ①：單次切分（樂觀數字）──
    single = {
        "reg_r2": round(float(r2_score(y_reg.iloc[idx_te], pred_te)), 4),
        "reg_mae": round(float(mean_absolute_error(y_reg.iloc[idx_te],
                                                   pred_te)), 4),
        "clf": clf_metrics(y_clf.iloc[idx_te], prob_te, threshold),
        "baseline_reg_r2": round(float(r2_score(
            y_reg.iloc[idx_te], base_reg.predict(X_te))), 4),
        "baseline_clf_auc": round(float(roc_auc_score(
            y_clf.iloc[idx_te], base_clf.predict_proba(X_te)[:, 1])), 4),
        "threshold_info": thr_info,
    }

    # ── 評估 ②：GroupKFold（誠實數字）──
    honest = groupkfold_eval(
        X, y_reg, y_clf, df["host_id"],
        lambda: HistGradientBoostingRegressor(random_state=SEED),
        lambda: HistGradientBoostingClassifier(random_state=SEED))

    print(f"[{name}] 單次 R²={single['reg_r2']} AUC={single['clf']['auc']} | "
          f"誠實 R²={honest['r2']['mean']}±{honest['r2']['std']} "
          f"AUC={honest['auc']['mean']}±{honest['auc']['std']}")

    models = {"reg_model": reg, "clf_model": clf_cal,
              "baseline_reg": base_reg, "baseline_clf": base_clf,
              "threshold": threshold, "feature_names": feats}
    return models, {"single_split": single, "groupkfold": honest}


def main():
    """主流程：載入 → 切分 → 訓練兩變體 → 序列化 → 重載驗證。"""
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET_CSV, encoding="utf-8-sig")
    full_feats, cold_feats, groups_def = load_feature_groups()
    print(f"[載入] {len(df)} 筆｜完整 {len(full_feats)} 特徵｜"
          f"冷啟動 {len(cold_feats)} 特徵")

    # ── 三層切分（stratify 高風險標籤，兩變體共用同一份切分）──
    idx = np.arange(len(df))
    idx_tmp, idx_te = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED,
        stratify=df["Y_high_risk"])
    val_rel = VAL_SIZE / (1 - TEST_SIZE)
    idx_tr, idx_va = train_test_split(
        idx_tmp, test_size=val_rel, random_state=SEED,
        stratify=df["Y_high_risk"].iloc[idx_tmp])
    print(f"[切分] train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")

    full_models, full_eval = train_variant("完整模型", df, full_feats,
                                           idx_tr, idx_va, idx_te)
    cold_models, cold_eval = train_variant("冷啟動模型", df, cold_feats,
                                           idx_tr, idx_va, idx_te)

    # ── 序列化（ml-modeling 規範：模型+特徵名+日期同存）──
    bundle = {
        "full": full_models, "cold": cold_models,
        "split_indices": {"train": idx_tr.tolist(), "val": idx_va.tolist(),
                          "test": idx_te.tolist()},
        "feature_groups": groups_def,
        "seed": SEED,
        "train_date": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET_CSV.name,
        "n_rows": int(len(df)),
    }
    joblib.dump(bundle, OUT_BUNDLE)

    # ── 重載驗證：載入後對保留樣本重跑，數值一致才算保存成功 ──
    reloaded = joblib.load(OUT_BUNDLE)
    sample = df[full_feats].iloc[idx_te[:20]]
    ok = np.allclose(bundle["full"]["reg_model"].predict(sample),
                     reloaded["full"]["reg_model"].predict(sample))
    assert ok, "重載驗證失敗：預測值不一致"
    print("[驗證] joblib 重載預測一致 ✓")

    eval_out = {
        "訓練時間": bundle["train_date"],
        "資料集": {"筆數": int(len(df)),
                   "高風險占比": round(float(df["Y_high_risk"].mean()), 4)},
        "完整模型_36特徵": full_eval,
        "冷啟動模型_35特徵": cold_eval,
        "報告基準": {"單次_R2": 0.587, "單次_AUC": 0.900,
                     "誠實_R2": "0.209±0.046", "誠實_AUC": "0.710±0.043",
                     "說明": "報告為 37 特徵含 photo_design_sense，本版 36 特徵"},
    }
    OUT_EVAL.write_text(json.dumps(eval_out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[完成] {OUT_BUNDLE.name} + {OUT_EVAL.name}")


if __name__ == "__main__":
    main()
