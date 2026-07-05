"""
Data Loader — Centralized data loading with Streamlit caching.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from modules.ui_components import ROOM_JP
from modules.ml_models import compute_risk_scores

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_csv(stem, **kw):
    """Read <stem>.csv.gz if present (deploy build), else <stem>.csv."""
    gz = DATA_DIR / f"{stem}.csv.gz"
    csv = DATA_DIR / f"{stem}.csv"
    return pd.read_csv(gz if gz.exists() else csv, **kw)


@st.cache_data(show_spinner=False)
def load_listings():
    """Load and preprocess listings data."""
    df = _read_csv("listings_cleaned", low_memory=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].between(200, 80000)].copy()
    df["room_type_zh"] = df["room_type"].map(ROOM_JP).fillna(df["room_type"])
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df["number_of_reviews_ltm"] = df["number_of_reviews_ltm"].fillna(0)
    df["number_of_reviews_l30d"] = df.get("number_of_reviews_l30d", pd.Series(0)).fillna(0)
    df["occupancy_pct"] = (
        (365 - df["availability_365"]) / 365 * 100
    ).clip(0, 100).round(1)

    # Review scores
    for col in ["review_scores_rating", "review_scores_cleanliness",
                "review_scores_location", "review_scores_value",
                "review_scores_communication", "review_scores_checkin",
                "review_scores_accuracy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dates
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["first_review"] = pd.to_datetime(df["first_review"], errors="coerce")

    # Numeric columns
    for col in ["bedrooms", "beds", "bathrooms_count", "accommodates"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Compute risk scores
    df = compute_risk_scores(df)

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_reviews():
    """Load reviews (Parquet preferred — robust dtypes; falls back to .csv.gz/.csv)."""
    pq = DATA_DIR / "reviews_cleaned.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = _read_csv("reviews_cleaned", low_memory=False)
    # Guarantee listing_id is a clean int64 so it always matches listing ids
    df["listing_id"] = pd.to_numeric(df["listing_id"], errors="coerce")
    df = df.dropna(subset=["listing_id"])
    df["listing_id"] = df["listing_id"].astype("int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_neighbourhoods():
    """Load neighbourhood list."""
    try:
        df = pd.read_csv(DATA_DIR / "neighbourhoods.csv", encoding="utf-8")
    except Exception:
        df = pd.read_csv(DATA_DIR / "neighbourhoods.csv", encoding="big5")
    r