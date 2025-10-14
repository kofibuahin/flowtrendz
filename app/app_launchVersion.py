#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlowTrendz — Launch Version (App)
A polished Streamlit UI for exploring hip‑hop/rap lyrics:
- Clear, non-technical tab names
- Inline "How to use" guidance for each control
- "What am I looking at?" explainers for each chart/table
- Light design polish (scorecards, spacing, icons, emojis)
This app expects a curated Parquet at data/curated/songs_curated.parquet
and (optionally) BERTopic artifacts in data/topics/ (wired later).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Page + global settings
# -----------------------------
st.set_page_config(page_title="FlowTrendz — Hip‑Hop Lyrics Explorer", layout="wide", page_icon="🎧")

# Light CSS polish
st.markdown(
    """
    <style>
      .ftz-muted { color:#6b7280; font-size:0.9rem }
      .ftz-card { background:#fff; border-radius:16px; padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,.06) }
      .metric-row { display:grid; grid-template-columns: repeat(4, minmax(150px,1fr)); gap:12px; }
      .metric { background:#f9fafb; border-radius:14px; padding:12px 14px; border:1px solid #eef2f7; }
      .metric h4 { margin:0 0 2px 0; font-size:0.9rem; color:#6b7280; font-weight:600; }
      .metric div { font-size:1.4rem; font-weight:700; }
      .stTabs [data-baseweb="tab-list"] { gap: 6px; }
      .stTabs [data-baseweb="tab"] { background-color: #f3f4f6; padding: 10px 14px; border-radius: 10px; }
      .stTabs [aria-selected="true"] { background-color: #ffffff; border: 1px solid #e5e7eb; }
      .block-caption { color:#6b7280; font-size:0.9rem; margin-top:-8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_CURATED = Path("data/curated/songs_curated.parquet")
EMO_LABELS = ["anger","disgust","fear","joy","neutral","sadness","surprise"]

@st.cache_data(show_spinner=False)
def load_curated() -> pd.DataFrame:
    if not DATA_CURATED.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_CURATED)
    # Best-effort canonical columns
    if "release_date" in df.columns and "year" not in df.columns:
        df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    # Primary artist col (backwards-compat with v05)
    if "artist_primary" in df.columns:
        df["artist"] = df["artist_primary"]
    elif "artist_name" in df.columns:
        df["artist"] = df["artist_name"]
    return df

songs = load_curated()

def explain(text: str):
    with st.expander("What am I looking at?"):
        st.markdown(text)

def howto(text: str):
    st.caption(text)

def scorecards(df: pd.DataFrame):
    total_tracks = int(df.shape[0])
    artists = int(df["artist"].nunique()) if "artist" in df else 0
    with_lyrics = int(df["lyrics"].notna().sum()) if "lyrics" in df else np.nan
    mean_comp = df.get("compound", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna().mean()
    cols = st.columns(4)
    cols[0].markdown('<div class="metric"><h4>Tracks</h4><div>🎵 ' + f"{total_tracks:,}" + '</div></div>', unsafe_allow_html=True)
    cols[1].markdown('<div class="metric"><h4>Artists</h4><div>👤 ' + f"{artists:,}" + '</div></div>', unsafe_allow_html=True)
    cols[2].markdown('<div class="metric"><h4>Tracks with lyrics</h4><div>✍️ ' + (f"{with_lyrics:,}" if not np.isnan(with_lyrics) else "—") + '</div></div>', unsafe_allow_html=True)
    cols[3].markdown('<div class="metric"><h4>Avg sentiment (compound)</h4><div>🧠 ' + (f"{mean_comp:.2f}" if pd.notnull(mean_comp) else "—") + '</div></div>', unsafe_allow_html=True)

# -----------------------------
# Intro tab
# -----------------------------
intro, overview, emotions, keywords, search, topics_map, fusion, table, story = st.tabs([
    "👋 Intro",
    "📊 Artist Overview",
    "😊 Emotion Explorer",
    "🔎 Keywords & Phrases",
    "🧭 Semantic Search",
    "🗺️ Topics Map",
    "💡 Topic + Emotion",
    "📄 Raw Songs",
    "🧩 Story Mode"
])

with intro:
    st.title("FlowTrendz — Hip‑Hop Lyrics Explorer")
    st.write("Explore sentiment, emotions, keywords, topics and trends across artists and songs. Built for non‑technical users.")
    howto("Tip: use the **Artist Overview** tab first. All charts update as you change filters.")
    if songs.empty:
        st.warning("No curated data found. Please run the bootstrap pipeline to generate `data/curated/songs_curated.parquet`.")
    else:
        scorecards(songs)
        explain("""
        **Scorecards** summarize the dataset currently loaded. *Avg sentiment* uses VADER compound (−1 to +1).
        """)
    st.markdown("---")
    st.subheader("How to use the app")
    st.markdown("""
    **1. Pick an artist** in *Artist Overview* to get tailored charts and summaries.  
    **2. Explore emotions** in *Emotion Explorer* (e.g., which emotions dominate an artist's catalog).  
    **3. Dig into wording** in *Keywords & Phrases* (common terms/phrases; use to compare eras).  
    **4. Try *Semantic Search*** to find songs with similar meaning rather than exact words.  
    **5. Use *Topics Map*** to see clusters of songs (requires topics/embeddings if available).  
    **6. *Topic + Emotion*** overlays emotions by discovered topics.  
    **7. *Raw Songs*** lists the underlying rows with links to Spotify/Genius.  
    **8. *Story Mode*** strings highlights into a narrative for sharing.
    """)

# -----------------------------
# Artist Overview
# -----------------------------
with overview:
    st.header("Artist Overview")
    if songs.empty:
        st.info("No data yet.")
    else:
        left, right = st.columns([1.2, 1])
        with left:
            artists = sorted(songs["artist"].dropna().unique().tolist())
            artist = st.selectbox("Artist (pick one)", artists, index=0, help="Choose the artist to analyze throughout this page.")
            df_a = songs[songs["artist"] == artist].copy()
            howto("You can switch artists anytime; the charts and stats will refresh.")
            
            # Time trend of sentiment
            if "year" in df_a.columns:
                ts = df_a.dropna(subset=["year"]).groupby("year")["compound"].mean().reset_index()
                fig = px.line(ts, x="year", y="compound", title="Average sentiment over time (VADER compound)",
                              template="plotly_white", markers=True)
                st.plotly_chart(fig, use_container_width=True)
                explain("**Line chart**: average VADER compound by release year for the selected artist (−1=negative, +1=positive).")
            
            # Distribution
            if "compound" in df_a.columns:
                fig = px.box(df_a, x="artist", y="compound", title="Song‑level sentiment distribution",
                             points="outliers", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                explain("**Box plot**: spread of per‑song sentiment for the selected artist.")
        
        with right:
            # Emotion mix
            emo_cols = [c for c in df_a.columns if c.startswith("emotion_")]
            if emo_cols:
                avg = df_a[emo_cols].mean(numeric_only=True).rename(lambda x: x.replace("emotion_","")).reset_index()
                avg.columns = ["emotion","score"]
                fig = px.bar(avg.sort_values("score", ascending=False), x="score", y="emotion",
                             orientation="h", title="Average emotion mix", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                explain("**Bar chart**: average predicted emotion scores across the artist’s songs.")
        
        # Songs table
        cols = ["artist","track_name","year","compound","spotify_url","genius_url"]
        present = [c for c in cols if c in df_a.columns]
        st.subheader("Songs for selected artist")
        st.dataframe(df_a[present].sort_values("compound", ascending=False), use_container_width=True)
        st.caption("Click Spotify/Genius links to open the track pages.")
        
# -----------------------------
# Emotion Explorer
# -----------------------------
with emotions:
    st.header("Emotion Explorer")
    if songs.empty:
        st.info("No data yet.")
    else:
        artists = sorted(songs["artist"].dropna().unique().tolist())
        pick = st.multiselect("Artist(s) to compare", artists, default=artists[:3], help="Choose one or more artists to compare emotions.")
        view = songs[songs["artist"].isin(pick)].copy()
        emo_cols = [c for c in view.columns if c.startswith("emotion_")]
        if emo_cols:
            # Average by artist
            avg = view.groupby("artist")[emo_cols].mean(numeric_only=True)
            avg = avg.rename(columns=lambda x: x.replace("emotion_","")).reset_index()
            fig = px.bar(avg.melt(id_vars="artist", var_name="emotion", value_name="score"),
                         x="emotion", y="score", color="artist", barmode="group",
                         title="Average emotion scores by artist", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            explain("**Grouped bars** show mean emotion scores by artist. Useful to compare tones across catalogs.")
        else:
            st.info("No emotion columns found (expected columns like `emotion_joy`, `emotion_sadness`, etc.).")

# -----------------------------
# Keywords & Phrases (N‑grams)
# -----------------------------
with keywords:
    st.header("Keywords & Phrases")
    if songs.empty:
        st.info("No data yet.")
    else:
        artists = sorted(songs["artist"].dropna().unique().tolist())
        pick = st.multiselect("Artist filter (optional)", artists, help="Limit counts to selected artist(s). Leave blank for all songs.")
        n_val = st.select_slider("Phrase length (n‑gram)", options=[1,2,3], value=2, help="1=single words, 2=bigrams, 3=trigrams.")
        top_k = st.slider("How many to show", min_value=10, max_value=50, value=20, step=5, help="Number of most common phrases to display.")
        df_k = songs.copy()
        if pick:
            df_k = df_k[df_k["artist"].isin(pick)]
        if "lyrics" not in df_k.columns:
            st.info("No `lyrics` column found.")
        else:
            texts = df_k["lyrics"].fillna("")
            # simple n‑gram counts
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(lowercase=True, stop_words="english", ngram_range=(n_val, n_val), min_df=2)
            try:
                X = vectorizer.fit_transform(texts)
                vocab = np.array(vectorizer.get_feature_names_out())
                counts = np.asarray(X.sum(axis=0)).ravel()
                top_idx = counts.argsort()[::-1][:top_k]
                top = pd.DataFrame({"phrase": vocab[top_idx], "count": counts[top_idx]})
                fig = px.bar(top, x="count", y="phrase", orientation="h", title=f"Top {top_k} {n_val}-grams", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                explain("**Bar chart**: most frequent phrases across the filtered set. Use to spot themes and repeated motifs.")
            except Exception as e:
                st.error(f"Vectorization failed: {e}")

# -----------------------------
# Semantic Search (cached placeholder)
# -----------------------------
with search:
    st.header("Semantic Lyric Search (Cached)")
    st.caption("Find songs with related meaning. (This placeholder uses simple keyword search until embeddings are wired.)")
    q = st.text_input("Search lyrics", help="Type a keyword or phrase. Example: 'hustle', 'dreams', 'heartbreak'.")
    if q and "lyrics" in songs.columns:
        mask = songs["lyrics"].fillna("").str.contains(re.escape(q), case=False, regex=True)
        results = songs[mask].copy()
        st.write(f"Found {results.shape[0]} matching songs.")
        cols = ["artist","track_name","year","compound","spotify_url","genius_url"]
        st.dataframe(results[[c for c in cols if c in results.columns]].head(200), use_container_width=True)
        explain("This is a **keyword** match. We’ll switch to **semantic** (embedding‑based) search in a later iteration.")
    else:
        st.info("Enter a search term to see results.")

# -----------------------------
# Topics Map (UMAP)
# -----------------------------
with topics_map:
    st.header("Topics Map")
    st.caption("A 2‑D map of songs using topic/embedding coordinates (requires precomputed topic embeddings).")
    xcol = "umap_x" if "umap_x" in songs.columns else None
    ycol = "umap_y" if "umap_y" in songs.columns else None
    colorcol = "topic_label" if "topic_label" in songs.columns else ("artist" if "artist" in songs.columns else None)
    if xcol and ycol:
        fig = px.scatter(songs, x=xcol, y=ycol, color=colorcol, hover_data=[c for c in ["artist","track_name","year"] if c in songs.columns],
                         title="Song clusters", template="plotly_white", opacity=0.75)
        st.plotly_chart(fig, use_container_width=True)
        explain("**Scatter plot**: nearby points share similar lyrical content; colors group by topic (or artist if topics aren’t available).")
    else:
        st.info("No `umap_x/umap_y` columns found yet. Run the topic pipeline first.")

# -----------------------------
# Topic + Emotion Fusion
# -----------------------------
with fusion:
    st.header("Topic + Emotion")
    if "topic_label" in songs.columns:
        emo_cols = [c for c in songs.columns if c.startswith("emotion_")]
        if emo_cols:
            grp = songs.groupby("topic_label")[emo_cols].mean(numeric_only=True).rename(columns=lambda x: x.replace("emotion_","")).reset_index()
            grp = grp.melt(id_vars="topic_label", var_name="emotion", value_name="score")
            fig = px.bar(grp, x="topic_label", y="score", color="emotion", barmode="group", title="Avg emotion by topic", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            explain("**Grouped bars**: emotional profile per discovered topic cluster.")
        else:
            st.info("No emotion columns found.")
    else:
        st.info("No `topic_label` column found yet.")

# -----------------------------
# Raw Songs
# -----------------------------
with table:
    st.header("Raw Songs")
    if songs.empty:
        st.info("No data yet.")
    else:
        cols = ["artist","track_name","year","compound","valence","spotify_url","genius_url","lyrics_snippet"]
        present = [c for c in cols if c in songs.columns]
        st.dataframe(songs[present].sort_values(["artist","year","track_name"]), use_container_width=True)
        explain("**Table**: underlying dataset with links and a short lyric snippet (not full lyrics).")

# -----------------------------
# Story Mode
# -----------------------------
with story:
    st.header("Story Mode")
    st.caption("Auto‑generate a short narrative with highlights for selected artists.")
    if songs.empty:
        st.info("Load data first.")
    else:
        artists = sorted(songs["artist"].dropna().unique().tolist())
        picks = st.multiselect("Artists", artists, default=artists[:2], help="Choose 1–3 artists to summarize.")
        if picks:
            df = songs[songs["artist"].isin(picks)].copy()
            parts = []
            for a in picks:
                d = df[df["artist"] == a]
                mean_c = d.get("compound", pd.Series(dtype=float)).mean()
                emo_cols = [c for c in d.columns if c.startswith("emotion_")]
                emo_top = None
                if emo_cols:
                    avg = d[emo_cols].mean(numeric_only=True).rename(lambda x: x.replace("emotion_",""))
                    emo_top = avg.sort_values(ascending=False).index[0]
                piece = f"**{a}** – average sentiment {mean_c:+.2f}" + (f"; dominant emotion: *{emo_top}*" if emo_top else "")
                parts.append(piece)
            st.markdown(" • " + "\n • ".join(parts))
            explain("**Narrative**: quick takeaways per artist. Expand this section later with more storytelling.")
