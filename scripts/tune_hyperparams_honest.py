# -*- coding: utf-8 -*-
"""
tune_hyperparams_honest.py — 誠實協定下的超參數調校（證偽實驗）
================================================================
目的：複現總結報告 §5.6 的結論 ——「瓶頸在資料資訊量，非模型容量，
調參投資報酬率極低」（報告：R² +0.017、AUC +0.002）。

方法：RandomizedSearchCV 25 次迭代，scoring 以 GroupKFold(host_id)
計分 —— 在單次隨機切分上調參等於對房東洩漏過擬合，必須用誠實協定。

產出：models/tuning_results.json（調參前後對照，供後台分析頁展示）

執行方式（本機，約 3~10 分鐘）：
  C:\\Users\\USER\\anaconda3\\python.exe -X utf8 scripts\\tune_hyperparams_honest.py
"""
from pathlib import Path
import sys
import json
import datetime

import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from modules.feature_engineering import load_dataset_final, load_bundle  # noqa: E402

OUT_JSON = PROJECT_ROOT / "models" / "tuning_results.json"
SEED = 42
N_ITER = 25   # 與報告 §5.6 相同迭代數
N_FOLDS = 5

# 搜尋空間：樹深/葉節點/學習率/正則化/樣本抽樣（HistGB 主要容量旋鈕）
PARAM_DIST = {
    "learning_rate": uniform(0.03, 0.27),      # 0.03 ~ 0.30
    "max_iter": randint(100, 500),
    "max_depth": [None, 3, 5, 8, 12],
    "max_leaf_nodes": randint(15, 63),
    "min_samples_leaf": randint(10, 60),
    "l2_regularization": uniform(0.0, 2.0),
}


def tune_one(model, X, y, groups, scoring):
    """回傳 (調參前分數, 調參後分數, 最佳參數)。皆以 GroupKFold 計分。"""
    gkf = GroupKFold(n_splits=N_FOLDS)
    cv_iter = list(gkf.split(X, y, groups))  # 固定折切分供前後公平比較

    before = cross_val_score(model, X, y, cv=cv_iter, scoring=scoring,
                             n_jobs=-1)
    search = RandomizedSearchCV(
        model, PARAM_DIST, n_iter=N_ITER, cv=cv_iter, scoring=scoring,
        random_state=SEED, n_jobs=-1, refit=False)
    search.fit(X, y)
    # 用最佳參數重跑同一組折，取得可比較的 mean±std
    best_model = model.__class__(random_state=SEED, **search.best_params_)
    after = cross_val_score(best_model, X, y, cv=cv_iter, scoring=scoring,
                            n_jobs=-1)
    fmt = lambda a: {"mean": round(float(np.mean(a)), 4),
                     "std": round(float(np.std(a)), 4)}
    return fmt(before), fmt(after), {
        k: (round(v, 4) if isinstance(v, float) else v)
        for k, v in search.best_params_.items()}


def main():
    df = load_dataset_final()
    feats = load_bundle()["full"]["feature_names"]  # 對完整模型調參
    X, groups = df[feats], df["host_id"]

    print(f"[開始] RandomizedSearchCV {N_ITER} 次 × GroupKFold {N_FOLDS} 折")
    reg_before, reg_after, reg_params = tune_one(
        HistGradientBoostingRegressor(random_state=SEED),
        X, df["Y_vacancy"], groups, "r2")
    print(f"[迴歸] 調參前 R²={reg_before['mean']}±{reg_before['std']} → "
          f"調參後 {reg_after['mean']}±{reg_after['std']}")

    clf_before, clf_after, clf_params = tune_one(
        HistGradientBoostingClassifier(random_state=SEED),
        X, df["Y_high_risk"], groups, "roc_auc")
    print(f"[分類] 調參前 AUC={clf_before['mean']}±{clf_before['std']} → "
          f"調參後 {clf_after['mean']}±{clf_after['std']}")

    result = {
        "執行時間": datetime.datetime.now().isoformat(timespec="seconds"),
        "方法": f"RandomizedSearchCV {N_ITER} 次迭代，GroupKFold({N_FOLDS}) "
                f"依 host_id 分組計分（誠實協定）",
        "迴歸_R2": {"調參前": reg_before, "調參後": reg_after,
                    "增益": round(reg_after["mean"] - reg_before["mean"], 4),
                    "最佳參數": reg_params},
        "分類_AUC": {"調參前": clf_before, "調參後": clf_after,
                     "增益": round(clf_after["mean"] - clf_before["mean"], 4),
                     "最佳參數": clf_params},
        "報告基準_37特徵": {"迴歸增益": 0.017, "分類增益": 0.002,
                            "結論": "瓶頸在資料資訊量，非模型容量"},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[完成] {OUT_JSON.name}")


if __name__ == "__main__":
    main()
