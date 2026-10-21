"""Module 4 - Analyst Copilot (pure pandas, no API needed)"""

import os, sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from config import REVIEWS_CSV, BRAND_NAME as APP_NAME


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(REVIEWS_CSV):
        return None
    df = pd.read_csv(REVIEWS_CSV)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars", "text"])
    return df


def top_words(texts: pd.Series, n: int = 8) -> str:
    stopwords = {
        "the","a","an","and","was","is","to","i","my","it","of","in","for",
        "that","they","we","very","good","great","this","with","have","not",
        "but","are","so","be","had","from","at","he","she","our","their",
        "were","just","been","would","there","when","all","also","you","your",
        "will","more","than","could","has","its","me","us","get","got","went",
        "back","like","what","which","then","them","said","time","one","out",
        "store","walmart","shop","shopping",
    }
    words = " ".join(texts.astype(str)).lower().split()
    freq = {}
    for w in words:
        w = w.strip(".,!?;:\"'()")
        if len(w) > 4 and w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:n]
    return ", ".join([f"{w} ({c}x)" for w, c in top])


def answer_question(question: str, df: pd.DataFrame) -> str:
    q = question.lower()

    if any(w in q for w in ["complain", "negative", "bad", "worst", "1 star", "2 star"]):
        low = df[df["stars"] <= 2]["text"].dropna()
        if low.empty:
            return "No 1-2 star reviews found in the dataset."
        pct    = len(low) / len(df) * 100
        words  = top_words(low)
        sample = low.iloc[0][:180]
        return (
            f"With **{pct:.0f}% of reviews being 1-2 stars**, negative feedback is significant. "
            f"The most recurring themes are around **{words}**. "
            f"One customer wrote: *\"{sample}...\"*"
        )

    if any(w in q for w in ["wait", "line", "checkout", "slow", "cashier", "register"]):
        mask = df["text"].str.lower().str.contains(
            "wait|waiting|line|queue|slow|checkout|cashier|register", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews specifically about wait times or checkout were found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        return (
            f"Checkout and wait times appear in **{len(sub):,} reviews** averaging **{avg:.2f}⭐**. "
            f"**{neg_pct:.0f}%** are negative. Customers frequently report long lines and understaffed registers. "
            f"This is {'a critical pain point' if neg_pct > 50 else 'a moderate concern'}."
        )

    if any(w in q for w in ["clean", "dirty", "mess", "bathroom", "floor", "smell"]):
        mask = df["text"].str.lower().str.contains(
            "clean|dirty|mess|floor|bathroom|smell|trash|restroom", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews specifically about cleanliness were found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        return (
            f"Cleanliness comes up in **{len(sub):,} reviews** averaging **{avg:.2f}⭐**. "
            f"**{neg_pct:.0f}%** are negative — customers flag dirty floors, messy shelves, and restrooms. "
            f"This is {'a high-priority operational issue' if neg_pct > 50 else 'worth monitoring'}."
        )

    if any(w in q for w in ["stock", "empty", "shelf", "inventory", "available", "item", "product"]):
        mask = df["text"].str.lower().str.contains(
            "stock|empty|shelf|shelves|inventory|available|product|item|sold out", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews about stock or inventory were found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        return (
            f"Stock and inventory issues appear in **{len(sub):,} reviews** averaging **{avg:.2f}⭐** "
            f"with **{neg_pct:.0f}%** negative. Customers frequently report empty shelves on essentials. "
            f"This is {'a systemic supply chain signal' if neg_pct > 50 else 'an occasional pain point'}."
        )

    if "state" in q and any(w in q for w in ["low", "worst", "bottom", "bad", "poor"]):
        if "state" not in df.columns:
            return "No state data available in this dataset."
        s    = df.groupby("state")["stars"].agg(avg="mean", count="count").sort_values("avg").head(5)
        rows = "\n".join([f"  {st}: {r['avg']:.2f}⭐ ({int(r['count'])} reviews)" for st, r in s.iterrows()])
        return f"**5 lowest rated states:**\n{rows}"

    if "state" in q and any(w in q for w in ["best", "top", "high", "great"]):
        if "state" not in df.columns:
            return "No state data available in this dataset."
        s    = df.groupby("state")["stars"].agg(avg="mean", count="count").sort_values("avg", ascending=False).head(5)
        rows = "\n".join([f"  {st}: {r['avg']:.2f}⭐ ({int(r['count'])} reviews)" for st, r in s.iterrows()])
        return f"**5 highest rated states:**\n{rows}"

    if any(w in q for w in ["staff", "employee", "associate", "service", "rude", "helpful"]):
        mask = df["text"].str.lower().str.contains(
            "staff|employee|associate|service|rude|helpful|friendly|manager|unprofessional", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews about staff or service were found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        pos_pct = (sub["stars"] >= 4).mean() * 100
        return (
            f"Staff and service appear in **{len(sub):,} reviews** averaging **{avg:.2f}⭐**. "
            f"**{pos_pct:.0f}%** praise helpful associates while **{neg_pct:.0f}%** describe poor interactions. "
            f"Staff consistency is a key differentiator between locations."
        )

    if any(w in q for w in ["self", "kiosk", "scan", "machine"]):
        mask = df["text"].str.lower().str.contains(
            "self.checkout|self checkout|kiosk|scan|register|machine", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews about self-checkout were found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        pos_pct = (sub["stars"] >= 4).mean() * 100
        return (
            f"Self-checkout appears in **{len(sub):,} reviews** averaging **{avg:.2f}⭐**. "
            f"**{pos_pct:.0f}%** appreciate the speed, but **{neg_pct:.0f}%** report broken machines. "
            f"It's both a strength and a frustration depending on location."
        )

    if any(w in q for w in ["happy", "positive", "praise", "like", "love", "good", "best"]):
        high = df[df["stars"] >= 4]["text"].dropna()
        if high.empty:
            return "No 4-5 star reviews found."
        pct    = len(high) / len(df) * 100
        words  = top_words(high)
        sample = high.iloc[0][:180]
        return (
            f"**{pct:.0f}% of customers leave 4-5 star reviews.** "
            f"Happy customers most often highlight **{words}**. "
            f"One satisfied customer said: *\"{sample}...\"*"
        )

    if any(w in q for w in ["pickup", "delivery", "curbside", "online", "order"]):
        mask = df["text"].str.lower().str.contains(
            "pickup|curbside|delivery|online order|grocery pickup", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews about pickup or delivery were found."
        avg     = sub["stars"].mean()
        pos_pct = (sub["stars"] >= 4).mean() * 100
        return (
            f"Pickup and delivery appear in **{len(sub):,} reviews** averaging **{avg:.2f}⭐** "
            f"with **{pos_pct:.0f}%** positive. This is one of Walmart's strongest differentiators."
        )

    if any(w in q for w in ["trend", "over time", "month", "year", "recent"]):
        if "date" not in df.columns or df["date"].isna().all():
            return "No date information available."
        monthly   = df.dropna(subset=["date"]).set_index("date")["stars"].resample("ME").mean()
        if len(monthly) < 2:
            return "Not enough time-series data to identify a trend."
        first     = monthly.iloc[0]
        last      = monthly.iloc[-1]
        diff      = last - first
        direction = "improving 📈" if diff > 0.1 else ("declining 📉" if diff < -0.1 else "holding steady ➡️")
        return (
            f"Ratings are **{direction}**. "
            f"Earliest avg: **{first:.2f}⭐** → Most recent: **{last:.2f}⭐** "
            f"(change of {diff:+.2f}⭐ over {len(monthly)} months)."
        )

    # fallback
    total   = len(df)
    avg     = df["stars"].mean()
    neg_pct = (df["stars"] <= 2).mean() * 100
    pos_pct = (df["stars"] >= 4).mean() * 100
    return (
        f"Across **{total:,} reviews**, the average rating is **{avg:.2f}⭐** — "
        f"{pos_pct:.0f}% positive (4-5★) and {neg_pct:.0f}% negative (1-2★). "
        f"Try asking about: complaints, checkout, cleanliness, stock, staff, self-checkout, pickup, or rating trends."
    )


def show():
    st.markdown("## 🤖 Analyst Copilot")
    st.markdown(f"Ask anything about **{APP_NAME}** customer reviews in plain English.")

    with st.spinner("Loading data..."):
        df = load_data()

    if df is None:
        st.error("No data found. Push a change to trigger the scraper.")
        return

    st.markdown("### 💡 Try asking:")
    questions = [
        "What do customers complain about most?",
        "How are checkout wait times rated?",
        "What do customers say about cleanliness?",
        "Are there complaints about out of stock items?",
        "Which states have the lowest ratings?",
        "What do customers say about staff?",
        "How is the self-checkout experience rated?",
        "What do customers praise most?",
    ]
    cols = st.columns(4)
    for i, q in enumerate(questions):
        if cols[i % 4].button(q, key=f"q{i}"):
            st.session_state["pending"] = q

    st.markdown("---")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    for msg in st.session_state["history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending  = st.session_state.pop("pending", "")
    user_in  = st.chat_input("Ask anything about the reviews...")
    question = user_in or pending

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state["history"].append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            answer = answer_question(question, df)
            st.markdown(answer)
            st.session_state["history"].append({"role": "assistant", "content": answer})

    if st.session_state.get("history"):
        if st.button("Clear conversation"):
            st.session_state["history"] = []
            st.rerun()

    with st.expander("📊 Full dataset stats"):
        total = len(df)
        avg   = df["stars"].mean()
        dist  = df["stars"].value_counts().sort_index()
        st.metric("Total Reviews", f"{total:,}")
        st.metric("Avg Rating", f"{avg:.2f} ⭐")
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (star, count) in zip([c1, c2, c3, c4, c5], dist.items()):
            col.metric(f"{int(star)}★", f"{int(count):,}")
