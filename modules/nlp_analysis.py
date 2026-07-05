"""
NLP Analysis Module — Sentiment analysis & keyword extraction.
Uses VADER for English, keyword-based rules + jieba for Chinese.
"""
import pandas as pd
import numpy as np
import re
from collections import Counter

# ─── Sentiment Analyzers ────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _vader = None

try:
    import jieba
    _has_jieba = True
except ImportError:
    _has_jieba = False

# ─── Chinese sentiment lexicon (common positive/negative words) ─
_ZH_POS = set([
    "乾淨", "整潔", "舒適", "方便", "便利", "親切", "友善", "溫馨", "安靜",
    "推薦", "喜歡", "滿意", "讚", "棒", "優", "好", "完美", "貼心",
    "寬敞", "明亮", "新", "美", "讚嘆", "值得", "感謝", "開心", "愉快",
    "優秀", "超棒", "很好", "不錯", "划算", "超值", "很棒", "極佳",
    "清潔", "設備齊全", "交通方便", "位置好", "nice", "good", "great",
    "perfect", "excellent", "amazing", "wonderful", "love", "lovely",
    "clean", "comfortable", "convenient", "friendly", "quiet", "recommend",
    "beautiful", "spacious", "helpful", "cozy", "awesome",
])

_ZH_NEG = set([
    "髒", "吵", "噪音", "臭", "差", "破", "舊", "爛", "小",
    "不乾淨", "不方便", "不好", "失望", "難過", "糟", "問題",
    "蟑螂", "漏水", "霉", "黴", "潮濕", "壞", "不推薦", "不舒服",
    "太貴", "不值", "態度差", "冷氣壞", "熱水不夠", "危險",
    "dirty", "noisy", "bad", "poor", "terrible", "horrible", "worst",
    "disappointed", "uncomfortable", "expensive", "broken", "smell",
    "cockroach", "bug", "mold", "cold", "hot", "rude", "dangerous",
])

# ─── Stopwords ──────────────────────────────────────────────────
_EN_STOP = set([
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "don",
    "now", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "it", "its", "they", "them", "their", "this",
    "that", "these", "those", "am", "and", "but", "if", "or", "because",
    "until", "while", "about", "up", "also", "get", "got", "much",
    "really", "even", "one", "two", "like", "go", "going", "went",
    "stay", "stayed", "place", "room", "apartment", "host", "airbnb",
])

_ZH_STOP = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一個", "上", "也", "很", "到", "說", "要",
    "去", "你", "會", "著", "沒有", "看", "好", "自己", "這",
    "他", "她", "們", "那", "這個", "但", "吧", "啊", "呢",
    "嗎", "把", "被", "讓", "給", "跟", "還", "從", "對",
    "得", "過", "可以", "比較", "而且", "如果", "因為", "所以",
    "雖然", "但是", "然後", "之後", "以後", "已經", "正在",
])


# ─── Core Analysis Functions ────────────────────────────────────

def analyze_sentiment(text, lang="en"):
    """
    Analyze sentiment of a single text.
    Returns: dict with 'compound', 'pos', 'neg', 'neu', 'label'
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "label": "中立"}

    if lang in ("en", "mixed_zh_en") and _vader:
        scores = _vader.polarity_scores(text)
        compound = scores["compound"]
    elif lang == "zh":
        compound = _zh_sentiment_score(text)
        scores = {"pos": max(0, compound), "neg": abs(min(0, compound)),
                  "neu": 1 - abs(compound)}
    else:
        if _vader:
            scores = _vader.polarity_scores(text)
            compound = scores["compound"]
        else:
            compound = 0.0
            scores = {"pos": 0, "neg": 0, "neu": 1}

    if compound >= 0.05:
        label = "正面"
    elif compound <= -0.05:
        label = "負面"
    else:
        label = "中立"

    return {
        "compound": round(compound, 4),
        "pos": round(scores.get("pos", 0), 4),
        "neg": round(scores.get("neg", 0), 4),
        "neu": round(scores.get("neu", 0), 4),
        "label": label,
    }


def _zh_sentiment_score(text):
    """Simple Chinese sentiment scoring using keyword matching."""
    if not _has_jieba:
        return 0.0
    words = set(jieba.lcut(text))
    pos_count = len(words & _ZH_POS)
    neg_count = len(words & _ZH_NEG)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / max(total, 1) * 0.8


def batch_sentiment(df, text_col="cleaned_comments", lang_col="language_type",
                    sample_n=None):
    """
    Batch sentiment analysis on a DataFrame.
    Returns DataFrame with sentiment columns added.
    """
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)

    results = []
    for _, row in df.iterrows():
        text = row.get(text_col, "")
        lang = row.get(lang_col, "en")
        s = analyze_sentiment(str(text), lang=str(lang))
        results.append(s)

    sent_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, sent_df], axis=1)


def extract_keywords(texts, lang="en", top_n=20):
    """
    Extract top keywords from a list of texts.
    Returns list of (word, count) tuples.
    """
    word_counts = Counter()

    for text in texts:
        if not isinstance(text, str):
            continue
        text = text.lower().strip()

        if lang == "zh" and _has_jieba:
            words = jieba.lcut(text)
            words = [w for w in words if len(w) >= 2 and w not in _ZH_STOP
                     and not re.match(r'^[\d\s\W]+$', w)]
        else:
            words = re.findall(r'[a-z]+', text)
            words = [w for w in words if len(w) >= 3 and w not in _EN_STOP]

        word_counts.update(words)

    return word_counts.most_common(top_n)


def listing_review_summary(reviews_df, listing_id):
    """
    Generate NLP summary for a specific listing's reviews.
    Returns dict with sentiment stats and keywords.
    """
    lr = reviews_df[reviews_df["listing_id"] == listing_id].copy()
    if lr.empty:
        return {
            "total_reviews": 0,
            "avg_sentiment": 0,
            "pos_pct": 0, "neg_pct": 0, "neu_pct": 0,
            "pos_keywords": [], "neg_keywords": [],
            "sample_pos": "", "sample_neg": "",
        }

    # Analyze sentiment
    analyzed = batch_sentiment(lr, sample_n=200)
    total = len(analyzed)
    pos_n = (analyzed["label"] == "正面").sum()
    neg_n = (analyzed["label"] == "負面").sum()

    # Extract keywords by sentiment
    pos_texts = analyzed[analyzed["label"] == "正面"]["cleaned_comments"].tolist()
    neg_texts = analyzed[analyzed["label"] == "負面"]["cleaned_comments"].tolist()

    # Determine dominant language
    lang_mode = lr["language_type"].mode()
    dominant_lang = lang_mode.iloc[0] if len(lang_mode) > 0 else "en"

    pos_kw = extract_keywords(pos_texts, lang=dominant_lang, top_n=10)
    neg_kw = extract_keywords(neg_texts, lang=dominant_lang, top_n=10)

    # Get sample reviews
    sample_pos = ""
    sample_neg = ""
    if pos_texts:
        sample_pos = max(pos_texts[:10], key=lambda x: len(str(x)) if isinstance(x, str) else 0, default="")
        if isinstance(sample_pos, str) and len(sample_pos) > 150:
            sample_pos = sample_pos[:150] + "…"
    if neg_texts:
        sample_neg = max(neg_texts[:10], key=lambda x: len(str(x)) if isinstance(x, str) else 0, default="")
        if isinstance(sample_neg, str) and len(sample_neg) > 150:
            sample_neg = sample_neg[:150] + "…"

    return {
        "total_reviews": total,
        "avg_sentiment": round(analyzed["compound"].mean(), 3),
        "pos_pct": round(pos_n / total * 100, 1) if total > 0 else 0,
        "neg_pct": round(neg_n / total * 100, 1) if total > 0 else 0,
        "neu_pct": round((total - pos_n - neg_n) / total * 100, 1) if total > 0 else 0,
        "pos_keywords": pos_kw,
        "neg_keywords": neg_kw,
        "sample_pos": sample_pos,
        "sample_neg": sample_neg,
    }


def global_sentiment_stats(reviews_df, sample_n=10000):
    """
    Compute global sentiment statistics for the admin dashboard.
    Samples reviews for performance.
    """
    sampled = reviews_df.sample(min(sample_n, len(reviews_df)), random_state=42)
    analyzed = batch_sentiment(sampled)

    total = len(analyzed)
    pos_n = (analyzed["label"] == "正面").sum()
    neg_n = (analyzed["label"] == "負面").sum()
    neu_n = total - pos_n - neg_n

    # Per-language breakdown
    lang_stats = {}
    for lang in analyzed["language_type"].unique():
        lang_data = analyzed[analyzed["language_type"] == lang]
        lt = len(lang_data)
        lang_stats[str(lang)] = {
            "count": lt,
            "avg_sentiment": round(lang_data["compound"].mean(), 3),
            "pos_pct": round((lang_data["label"] == "正面").sum() / lt * 100, 1) if lt > 0 else 0,
        }

    # All keywords
    all_pos = analyzed[analyzed["label"] == "正面"]["cleaned_comments"].tolist()
    all_neg = analyzed[analyzed["label"] == "負面"]["cleaned_comments"].tolist()

    return {
        "total_sampled": total,
        "avg_sentiment": round(analyzed["compound"].mean(), 3),
        "pos_n": int(pos_n), "neg_n": int(neg_n), "neu_n": int(neu_n),
        "pos_pct": round(pos_n / total * 100, 1),
        "neg_pct": round(neg_n / total * 100, 1),
        "neu_pct": round(neu_n / total * 100, 1),
        "lang_stats": lang_stats,
        "pos_keywords_en": extract_keywords(all_pos, lang="en", top_n=25),
        "neg_keywords_en": extract_keywords(all_neg, lang="en", top_n=25),
        "pos_keywords_zh": extract_keywords(all_pos, lang="zh", top_n=25),
        "neg_keywords_zh": extract_keywords(all_neg, lang="zh", top_n=25),
        "sentiment_series": analyzed,
    }


def recent_review_snippets(reviews_df, listing_id, n=6, maxlen=150):
    """
    Return up to n recent review comment snippets (plain strings) for a
    listing, newest first, for the hover-preview tooltip.
    """
    r = reviews_df[reviews_df["listing_id"] == listing_id]
    if r.empty:
        return []
    if "date" in r.columns:
        r = r.sort_values("date", ascending=False)
    col = "comments" if "comments" in r.columns else "cleaned_comments"
    out = []
    for c in r[col].head(n).tolist():
        s = " ".join(str(c).split()).strip()
        if not s:
            continue
        if len(s) > maxlen:
            s = s[:maxlen] + "…"
        out.append(s)
    return out
