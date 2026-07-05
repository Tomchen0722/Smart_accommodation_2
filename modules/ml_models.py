"""
ML Models — Risk scoring, Logistic Regression, Random Forest.
空房預測 + 1KM 競爭分析
"""
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)

from modules.ui_components import FEAT_ZH

# Feature columns used for ML
FEAT_COLS = [
    "price", "minimum_nights", "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365", "number_of_reviews_ltm",
]


def compute_risk_scores(df):
    """
    Compute weighted risk scores for all listings.
    Score = 0.40×availability_rate + 0.30×review_scarcity
          + 0.20×ltm_activity_loss + 0.10×price_premium
    """
    a = (df["availability_365"] / 365).clip(0, 1)
    b = 1 - np.clip(df["number_of_reviews"] / 100, 0, 1)
    c = 1 - np.clip(df["number_of_reviews_ltm"] / 20, 0, 1)
    d = np.clip(df["price"] / df["price"].quantile(0.95), 0, 1)
    df["risk_score"] = (0.40 * a + 0.30 * b + 0.20 * c + 0.10 * d).clip(0, 1).round(3)
    df["risk_level"] = pd.cut(
        df["risk_score"], bins=[-0.001, 0.35, 0.60, 1.001],
        labels=["低風險", "中風險", "高風險"],
    ).astype(str)
    return df


def train_models(df):
    """
    Train Logistic Regression and Random Forest models.
    Returns dict with metrics, coefficients, importances, etc.
    """
    rt_dum = pd.get_dummies(df["room_type"], prefix="rt")
    X = pd.concat([df[FEAT_COLS].fillna(0), rt_dum], axis=1)
    y = (df["risk_level"] == "高風險").astype(int)
    feat_names = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_tr_s, y_tr)
    lr_prob = lr.predict_proba(X_te_s)[:, 1]
    lr_pred = lr.predict(X_te_s)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:, 1]
    rf_pred = rf.predict(X_te)

    def metrics(yt, yp, yprob):
        fpr, tpr, _ = roc_curve(yt, yprob)
        prec, rec, _ = precision_recall_curve(yt, yprob)
        return dict(
            accuracy=accuracy_score(yt, yp),
            recall=recall_score(yt, yp),
            precision=precision_score(yt, yp),
            f1=f1_score(yt, yp),
            auc=roc_auc_score(yt, yprob),
            ap=average_precision_score(yt, yprob),
            cm=confusion_matrix(yt, yp),
            fpr=fpr, tpr=tpr, prec=prec, rec=rec,
        )

    corr_df = X_tr.copy()
    corr_df.columns = [FEAT_ZH.get(c, c) for c in corr_df.columns]

    return dict(
        lr=metrics(y_te, lr_pred, lr_prob),
        rf=metrics(y_te, rf_pred, rf_prob),
        lr_coef=dict(zip(feat_names, lr.coef_[0])),
        rf_import=dict(zip(feat_names, rf.feature_importances_)),
        feat_names=feat_names,
        corr=corr_df.corr(),
        n_train=len(X_tr), n_test=len(X_te),
        n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
        scaler=sc, lr_model=lr, rf_model=rf,
    )


def predict_vacancy_prob(listing_row, df_all, model_data):
    """
    Predict vacancy probability for a single listing.
    Returns dict with lr_prob and rf_prob for 30/60/90 days.
    """
    rt_dum = pd.get_dummies(
        pd.Series([listing_row["room_type"]]), prefix="rt"
    )
    # Ensure all room type columns exist
    for col in [c for c in model_data["feat_names"] if c.startswith("rt_")]:
        if col not in rt_dum.columns:
            rt_dum[col] = 0

    feat = pd.DataFrame([listing_row[FEAT_COLS].fillna(0).values],
                        columns=FEAT_COLS)
    X = pd.concat([feat, rt_dum[
        [c for c in model_data["feat_names"] if c.startswith("rt_")]
    ]], axis=1)

    # Ensure column order
    for col in model_data["feat_names"]:
        if col not in X.columns:
            X[col] = 0
    X = X[model_data["feat_names"]]

    X_s = model_data["scaler"].transform(X)
    lr_p = model_data["lr_model"].predict_proba(X_s)[0, 1]
    rf_p = model_data["rf_model"].predict_proba(X.values)[0, 1]

    # Estimate 30/60/90 day adjustments based on availability trend
    avail_30 = listing_row.get("availability_30", 15) / 30
    avail_60 = listing_row.get("availability_60", 30) / 60
    avail_90 = listing_row.get("availability_90", 45) / 90

    return {
        "lr_30": min(1.0, lr_p * (0.8 + 0.4 * avail_30)),
        "lr_60": min(1.0, lr_p * (0.8 + 0.4 * avail_60)),
        "lr_90": min(1.0, lr_p * (0.8 + 0.4 * avail_90)),
        "rf_30": min(1.0, rf_p * (0.8 + 0.4 * avail_30)),
        "rf_60": min(1.0, rf_p * (0.8 + 0.4 * avail_60)),
        "rf_90": min(1.0, rf_p * (0.8 + 0.4 * avail_90)),
        "base_lr": lr_p,
        "base_rf": rf_p,
    }


def generate_landlord_advice(listing_row, nearby_df, vacancy_prob):
    """
    Generate smart advice for a landlord based on the 專案.txt spec:
    - Dimension A: Price optimization
    - Dimension B: Marketing/seasonal advice
    - Dimension C: Hardware upgrade suggestions
    """
    advices = []
    price = listing_row["price"]
    risk_prob = max(vacancy_prob["base_lr"], vacancy_prob["base_rf"])

    # ── Dimension A: Price ──
    if not nearby_df.empty:
        median_price = nearby_df["price"].median()
        price_diff_pct = (price - median_price) / median_price * 100

        if price_diff_pct > 10:
            suggested = round(median_price * 1.02, 0)
            new_prob = max(0, risk_prob - 0.15 * (price_diff_pct / 10))
            advices.append({
                "type": "💰 價格優化建議",
                "severity": "high" if price_diff_pct > 20 else "medium",
                "text": (f"您的租金 ${price:,.0f} 高於周邊 1KM 中位數 ${median_price:,.0f} 約 "
                         f"{price_diff_pct:.0f}%。建議調降至 ${suggested:,.0f}，"
                         f"預估可將空房機率從 {risk_prob*100:.0f}% 降至 {new_prob*100:.0f}%。"),
            })
        elif price_diff_pct < -10:
            advices.append({
                "type": "💰 價格調整建議",
                "severity": "low",
                "text": (f"您的租金 ${price:,.0f} 低於周邊中位數 ${median_price:,.0f} 約 "
                         f"{abs(price_diff_pct):.0f}%，有上調空間。"),
            })
        else:
            advices.append({
                "type": "💰 價格狀態",
                "severity": "low",
                "text": f"租金 ${price:,.0f} 與周邊中位數 ${median_price:,.0f} 相近，定價合理。",
            })

    # ── Dimension B: Marketing ──
    if risk_prob > 0.6:
        advices.append({
            "type": "📣 動態行銷建議",
            "severity": "high",
            "text": ("空房風險較高。建議推出「首月租金減免」或「免收管理費」活動，"
                     "可加速在 14 天內出租。"),
        })
    elif risk_prob > 0.3:
        advices.append({
            "type": "📣 行銷建議",
            "severity": "medium",
            "text": "建議更新房源照片、優化描述內容，提升曝光吸引力。",
        })

    # ── Dimension C: Hardware ──
    if not nearby_df.empty:
        amenity_suggestions = _analyze_amenity_gaps(listing_row, nearby_df)
        if amenity_suggestions:
            advices.append({
                "type": "🔧 硬體升級建議",
                "severity": "medium",
                "text": amenity_suggestions,
            })

    return advices


def _analyze_amenity_gaps(listing_row, nearby_df):
    """Analyze amenity gaps versus nearby listings."""
    import ast

    my_amenities = set()
    try:
        raw = listing_row.get("amenities", "[]")
        if isinstance(raw, str):
            items = ast.literal_eval(raw)
            my_amenities = set(str(a).lower() for a in items)
    except Exception:
        pass

    popular = Counter()
    for _, row in nearby_df.iterrows():
        try:
            raw = row.get("amenities", "[]")
            if isinstance(raw, str):
                items = ast.literal_eval(raw)
                for a in items:
                    popular[str(a).lower()] += 1
        except Exception:
            continue

    # Find popular amenities this listing lacks
    threshold = len(nearby_df) * 0.5
    missing = []
    for amenity, count in popular.most_common(20):
        if count >= threshold and amenity not in my_amenities:
            missing.append(amenity.title())

    if missing:
        top3 = missing[:3]
        return (f"周邊 1KM 內有 {len(nearby_df)} 間房源，多數配備「{'、'.join(top3)}」。"
                f"建議增設以提升競爭力與曝光量。")
    return ""


# ─── Neighbourhood aggregation ──────────────────────────────────
def nb_aggregate(df):
    """Aggregate statistics by neighbourhood."""
    g = df.groupby("neighbourhood_cleansed").agg(
        房源數=("id", "count"),
        高風險數=("risk_level", lambda x: (x == "高風險").sum()),
        平均風險=("risk_score", "mean"),
        中位價格=("price", "median"),
        平均評論數=("number_of_reviews", "mean"),
        平均入住率=("occupancy_pct", "mean"),
    ).reset_index()
    g = g.rename(columns={"neighbourhood_cleansed": "行政區"})
    g["高風險佔比"] = (g["高風險數"] / g["房源數"] * 100).round(1)
    g["平均風險"] = g["平均風險"].round(3)
    g["中位價格"] = g["中位價格"].round(0).astype(int)
    g["平均評論數"] = g["平均評論數"].round(1)
    g["平均入住率"] = g["平均入住率"].round(1)
    return g.sort_values("高風險佔比", ascending=False).reset_index(drop=True)
