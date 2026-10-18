"""
Voice of Customer AI engine - pure NLP (TF-IDF + VADER)
No API keys, no rate limits, works on Streamlit Cloud forever.
"""
import os, re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

STOPWORDS = {
    "the","a","an","and","was","is","to","i","my","it","of","in","for","that",
    "they","we","very","good","great","this","with","have","not","but","are",
    "so","be","had","from","at","he","she","our","their","were","just","been",
    "would","there","when","all","also","you","your","will","more","than",
    "could","has","its","me","us","get","got","went","back","like","what",
    "which","then","them","said","time","one","out","up","do","did","no",
    "walmart","store","shop","shopping","went","going","comes","come",
}

analyzer = SentimentIntensityAnalyzer()

def get_client():
    return None

def get_groq_client():
    return None

def _sentiment_label(score: float) -> str:
    if score >= 0.15:  return "positive"
    if score <= -0.15: return "negative"
    return "mixed"

def _clean(text: str) -> str:
    return re.sub(r"[^a-z\s]", " ", text.lower()).strip()

def _top_words(texts: list, n: int = 6) -> list:
    words = " ".join([_clean(t) for t in texts]).split()
    freq = {}
    for w in words:
        if len(w) > 4 and w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:n]]

def _best_quote(texts: list, keywords: list) -> str:
    kw_set = set(keywords)
    candidates = [
        t for t in texts
        if any(k in t.lower() for k in kw_set) and 20 < len(t) < 180
    ]
    if candidates:
        return min(candidates, key=len)
    return texts[0][:120] if texts else ""

THEME_LABELS = {
    frozenset(["wait", "line", "checkout", "cashier", "slow", "long", "register"]):
        ("Long Checkout Lines", "negative",
         "Customers consistently report excessive wait times at checkout, especially during peak hours."),
    frozenset(["staff", "rude", "employee", "associate", "helpful", "friendly", "manager"]):
        ("Staff & Service Quality", "mixed",
         "Staff interactions vary widely - praised when helpful and attentive, criticized when dismissive or hard to find."),
    frozenset(["clean", "dirty", "mess", "floor", "bathroom", "smell", "condition"]):
        ("Store Cleanliness & Condition", "negative",
         "Customers frequently flag dirty floors, messy shelves, and poorly maintained restrooms."),
    frozenset(["price", "cheap", "expensive", "cost", "deal", "value", "saving"]):
        ("Pricing & Value", "positive",
         "Walmart's low prices and deals remain a key driver of positive sentiment and loyalty."),
    frozenset(["stock", "empty", "shelf", "available", "inventory", "product", "item"]):
        ("Out of Stock & Inventory Issues", "negative",
         "Empty shelves and unavailable items are a recurring complaint, especially for staple products."),
    frozenset(["easy", "quick", "fast", "convenient", "pickup", "online", "delivery"]):
        ("Convenient Pickup & Delivery", "positive",
         "Curbside pickup and delivery options receive strong praise for saving customers time."),
    frozenset(["self", "checkout", "scan", "machine", "kiosk", "broken", "work"]):
        ("Self-Checkout Experience", "mixed",
         "Self-checkout gets mixed reviews - appreciated for speed when working, frustrating when malfunctioning."),
    frozenset(["pharmacy", "vision", "auto", "tire", "garden", "grocery", "produce"]):
        ("Departmental Experience", "mixed",
         "Specific departments like pharmacy and grocery get targeted feedback separate from general store experience."),
}

def _match_label(keywords: list):
    kw_set = set(keywords)
    best_match, best_score = None, 0
    for label_keys, label_data in THEME_LABELS.items():
        score = len(kw_set & label_keys)
        if score > best_score:
            best_score = score
            best_match = label_data
    return best_match

def cluster_themes(reviews_sample: list, client=None, industry: str = "retail") -> dict:
    texts = [t for t in reviews_sample if isinstance(t, str) and len(t.strip()) > 10]
    if len(texts) < 6:
        return {"themes": [], "error": "Not enough reviews to cluster."}

    n_clusters = min(6, len(texts) // 5)

    try:
        vec = TfidfVectorizer(
            max_features=300,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        X     = vec.fit_transform(texts)
        terms = vec.get_feature_names_out()

        km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        themes = []
        for cluster_id in range(n_clusters):
            cluster_texts = [texts[i] for i, l in enumerate(labels) if l == cluster_id]
            if not cluster_texts:
                continue

            center    = km.cluster_centers_[cluster_id]
            top_idx   = center.argsort()[-8:][::-1]
            keywords  = [terms[i] for i in top_idx]
            top_words = _top_words(cluster_texts, n=5)

            scores    = [analyzer.polarity_scores(t)["compound"] for t in cluster_texts[:50]]
            avg_score = np.mean(scores)
            sentiment = _sentiment_label(avg_score)

            matched = _match_label(keywords + top_words)
            if matched:
                name, _, description = matched
            else:
                name        = " ".join(top_words[:3]).title()
                description = f"Customers frequently discuss {', '.join(top_words[:4])} in their reviews."

            if matched and abs(avg_score) < 0.1:
                sentiment = matched[1]

            percent = round(len(cluster_texts) / len(texts) * 100)
            quote   = _best_quote(cluster_texts, top_words)

            themes.append({
                "name":          name,
                "description":   description,
                "percent":       percent,
                "sentiment":     sentiment,
                "example_quote": quote,
            })

        themes.sort(key=lambda t: (t["sentiment"] != "negative", -t["percent"]))
        return {"themes": themes}

    except Exception as e:
        return {"themes": [], "error": str(e)}


def detect_anomalies(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    df = df.copy()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])
    if df.empty:
        return pd.DataFrame()

    loc_col = "place_name" if "place_name" in df.columns else (
              "business_id" if "business_id" in df.columns else None)
    if loc_col is None:
        return pd.DataFrame()

    df = df[df[loc_col].fillna("").str.strip() != ""]
    if df.empty:
        return pd.DataFrame()

    loc_agg = (
        df.groupby(loc_col)["stars"]
        .agg(avg_rating="mean", total_reviews="count")
        .reset_index()
        .rename(columns={loc_col: "business_id"})
    )

    brand_avg                 = df["stars"].mean()
    loc_agg["brand_avg"]      = round(brand_avg, 2)
    loc_agg["rating_drop"]    = (loc_agg["brand_avg"] - loc_agg["avg_rating"]).round(2)
    loc_agg["historical_avg"] = loc_agg["brand_avg"]
    loc_agg["recent_avg"]     = loc_agg["avg_rating"].round(2)
    loc_agg["recent_reviews"] = loc_agg["total_reviews"]

    anomalies = loc_agg[
        (loc_agg["rating_drop"] >= threshold) &
        (loc_agg["total_reviews"] >= 2)
    ].copy()
    return anomalies.sort_values("rating_drop", ascending=False)


def write_exec_summary(themes, anomaly_stores, total_reviews,
                       avg_rating, date_range, client=None, brand_name="Walmart") -> str:
    if not themes:
        return "Insufficient data to generate an executive summary."

    neg_themes = [t for t in themes if t["sentiment"] == "negative"]
    pos_themes = [t for t in themes if t["sentiment"] == "positive"]

    if avg_rating >= 4.0:
        headline = f"Customer sentiment for {brand_name} is strong, averaging {avg_rating:.2f}⭐ across {total_reviews:,} reviews during {date_range}."
    elif avg_rating >= 3.0:
        headline = f"{brand_name} is delivering a mixed customer experience, averaging {avg_rating:.2f}⭐ across {total_reviews:,} reviews during {date_range}. Operational improvements are needed."
    else:
        headline = f"Customer experience at {brand_name} requires immediate attention - averaging {avg_rating:.2f}⭐ across {total_reviews:,} reviews during {date_range}."

    if neg_themes:
        top        = neg_themes[0]
        issue_para = (
            f"The most urgent issue is **{top['name']}** ({top['percent']}% of reviews): "
            f"{top['description']} "
            f"One customer noted: \"{top['example_quote']}\""
        )
    elif themes:
        top        = themes[0]
        issue_para = (
            f"The dominant theme is **{top['name']}** ({top['percent']}% of reviews): "
            f"{top['description']}"
        )
    else:
        issue_para = "No dominant theme could be identified from the current dataset."

    if pos_themes:
        pos      = pos_themes[0]
        pos_para = f"On the positive side, **{pos['name']}** ({pos['percent']}% of reviews) shows what's working: {pos['description']}"
    else:
        pos_para = "Positive signals are limited - resolving the top complaints would likely improve ratings meaningfully."

    if not anomaly_stores.empty:
        top_loc      = anomaly_stores.iloc[0]
        loc_name     = top_loc.get("city", top_loc.get("business_id", "One location"))
        anomaly_para = (
            f"Field Leaders should prioritize **{loc_name}**, rated "
            f"{top_loc['recent_avg']:.1f}⭐ against a brand average of {top_loc['historical_avg']:.1f}⭐ "
            f"- a gap of {top_loc['rating_drop']:.1f} stars requiring immediate operational review."
        )
    else:
        anomaly_para = "No locations are significantly below the brand average at this time."

    return f"{headline}\n\n{issue_para}\n\n{pos_para}\n\n{anomaly_para}"


def score_sentiment_batch(texts: list, client=None) -> list:
    return [_sentiment_label(analyzer.polarity_scores(str(t))["compound"]) for t in texts]
