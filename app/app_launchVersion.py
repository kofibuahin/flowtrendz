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
import streamlit.components.v1 as components
import os, hashlib, textwrap, datetime as dt
# If you’re using the official OpenAI client (Python >= 1.0):
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # we’ll error nicely if the lib isn’t installed

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
# --- Parse an ID from a Spotify track URL or URI ---
_SPOTIFY_TRACK_RE = re.compile(r"(?:https?://open\.spotify\.com/track/|spotify:track:)([A-Za-z0-9]+)")

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

def _get_openai_key():
    # Try env first (works for non-Streamlit contexts too), then st.secrets
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)  # type: ignore[attr-defined]
        except Exception:
            key = None
    return key

@st.cache_data(show_spinner=False, ttl=60*60*24)  # cache for 24h
def _cached_artist_summary(artist: str, lyrics_blob_hash: str, model: str, temperature: float) -> str:
    # This function runs only on cache misses.
    api_key = _get_openai_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to `.streamlit/secrets.toml` or your environment variables."
        )
    if OpenAI is None:
        raise RuntimeError("The `openai` Python package is not installed. `pip install openai` and retry.")

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are a music analyst. Read the following lyrics excerpts for **{artist}** (hip-hop/rap).
Provide:

1) A concise **overall summary** of themes & motifs (4–6 sentences).
2) A short **style profile label** (3–6 words), e.g., “confessional luxury braggadocio”.
3) 3–5 **recurring motifs** as bullet points.
4) 3 **notable lines** (quote briefly; if duplicates/noisy, say “(line omitted)”).

Write for non-technical readers. Avoid over-indexing on any one song; capture trends across the set.

Lyrics excerpts (may be partial, normalized, or noisy):
---
{{LYRICS}}
---
"""

    # call the API (responses.create for 1.x client)
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=600,
        input=[{"role": "user", "content": prompt}],
    )

    # Pull text (works with current responses API)
    out = ""
    try:
        out = "".join([p.text for msg in resp.output for p in getattr(msg, "content", []) if hasattr(p, "text")])  # type: ignore
    except Exception:
        # fallback for older/newer shapes
        out = str(resp)

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"**Music summary for {artist}** (model: `{model}`, temp: {temperature:.1f}) — _generated {stamp}_\n\n"
    return header + out.strip()

def _make_lyrics_blob(df_artist, max_songs=12, max_chars_per_song=1200):
    """Concatenate a subset of lyrics for cost/speed control."""
    texts = []
    if "lyrics" not in df_artist.columns:
        return ""
    # Prefer the most “typical” songs (middle sentiment), fall back to head
    d = df_artist.dropna(subset=["lyrics"]).copy()
    if d.empty:
        return ""
    if "compound" in d.columns:
        d["__dist"] = (d["compound"] - d["compound"].median()).abs()
        d = d.sort_values("__dist").head(max_songs)
    else:
        d = d.head(max_songs)
    for _, r in d.iterrows():
        t = str(r["lyrics"])[:max_chars_per_song]
        name = r.get("track_name", "Unknown")
        year = r.get("year", "")
        texts.append(f"[{year}] {name}\n{t}")
    blob = "\n\n---\n\n".join(texts)
    return blob

# --- helpers: stat "tile" + tiny CSS to make the selectbox stretch ---
def stat_card(label: str, value: str, emoji: str = "📊"):
    st.markdown(
        f"""
        <div style="
            background: #F8FAFF; border: 1px solid #EEF2FF;
            border-radius: 12px; padding: 18px 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        ">
          <div style="font-size:14px;color:#475569;margin-bottom:6px;">{label}</div>
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:28px;line-height:1;">{emoji}</span>
            <span style="font-size:28px;font-weight:700;color:#111827;">{value}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def extract_spotify_track_id(url: str) -> str | None:
    if not url:
        return None
    m = _SPOTIFY_TRACK_RE.search(str(url))
    return m.group(1) if m else None

def spotify_track_embed_html(track_id: str, height: int = 80) -> str:
    # Compact embed for a single track
    src = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
    return (
        f'<iframe style="border-radius:12px" '
        f'src="{src}" width="100%" height="{height}" frameBorder="0" '
        f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
        f'loading="lazy"></iframe>'
    )

# (Optional) nudge selectbox to fill its container
st.markdown(
    """
    <style>
      /* makes the select container use all available width */
      div[data-baseweb="select"]{ min-width: 100% !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        # --- full-width artist picker ---
        with st.container():
            artists = sorted(songs["artist"].dropna().unique().tolist())
            artist = st.selectbox(
                "Artist (pick one)",
                artists,
                index=0,
                help="Choose the artist to analyze throughout this page."
            )
            st.caption("You can switch artists anytime; the charts and stats will refresh.")
        df_a = songs[songs["artist"] == artist].copy()

        # --- scorecards row (tiles) ---
        col_a, col_b, col_c, col_d = st.columns(4)

        # Tracks
        tracks_n = len(df_a)
        with col_a:
            stat_card("Tracks Analyzed", f"{tracks_n:,}", "🎵")

        # Lines of Lyrics
        if "lyrics" in df_a.columns:
            lines_total = int(
                df_a["lyrics"]
                .dropna()
                .astype(str)
                .apply(lambda s: s.count("\n") + 1)
                .sum()
            )
        else:
            lines_total = 0
        with col_b:
            stat_card("Lines of Lyrics Analyzed", f"{lines_total:,}", "✍️")

        # Avg sentiment
        avg_sent = float(df_a["compound"].mean()) if "compound" in df_a.columns else float("nan")
        with col_c:
            stat_card("Average Sentiment", f"{avg_sent:.2f}" if avg_sent == avg_sent else "—", "🧠")

        # Dominant emotion (name)
        emo_cols = [c for c in df_a.columns if c.startswith("emotion_")]
        if emo_cols:
            emo_means = df_a[emo_cols].mean(numeric_only=True)
            dom_emo = emo_means.idxmax().replace("emotion_", "")
            dom_val = emo_means.max()
            dom_label = f"{dom_emo} ({dom_val:.2f})"
        else:
            dom_label = "—"

        with col_d:
            stat_card("Dominant Emotion", dom_label, "💫")

        st.markdown("")  # small spacer

        
        # === TOP ROW ===  Avg emotion mix (LEFT)  |  AI summary (RIGHT)
        top_left, top_right = st.columns([1, 1])

        with top_left:
            emo_cols = [c for c in df_a.columns if c.startswith("emotion_")]
            if emo_cols:
                avg = (
                    df_a[emo_cols]
                    .mean(numeric_only=True)
                    .rename(lambda x: x.replace("emotion_", ""))
                    .reset_index()
                )
                avg.columns = ["emotion", "score"]
                fig = px.bar(
                    avg.sort_values("score", ascending=False),
                    x="score",
                    y="emotion",
                    orientation="h",
                    title="Average emotion mix",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
                explain(
                    "**Bar chart**: average predicted emotion scores across the artist’s songs."
                )
            else:
                st.info("No emotion columns found.")

        with top_right:
            # --- AI Artist Summary panel (unchanged logic; just moved up) ---
            st.subheader("🧠 Artist Summary")
            st.caption(
                "Click to generate a concise summary of themes, motifs, and a style label for this artist. "
                "Powered by OpenAI and cached for 24h."
            )

            default_model = None
            try:
                default_model = st.secrets.get("OPENAI_MODEL", None)  # type: ignore[attr-defined]
            except Exception:
                default_model = None
            model = st.selectbox(
                "Model",
                options=[default_model or "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
                index=0,
            )
            temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.6, 0.1)

            lyrics_blob = _make_lyrics_blob(df_a, max_songs=12, max_chars_per_song=1200)
            if not lyrics_blob:
                st.info("No lyrics available for this artist in the current dataset.")
            else:
                blob_hash = hashlib.md5(lyrics_blob.encode("utf-8")).hexdigest()
                gen = st.button("✨ Generate summary", type="primary", use_container_width=True)
                if gen:
                    try:
                        payload = _cached_artist_summary(
                            artist, blob_hash, model, temperature
                        ).replace(
                            "{ {LYRICS} }",
                            textwrap.shorten(lyrics_blob, width=18000, placeholder="…"),
                        )
                        st.markdown(payload)
                        st.download_button(
                            "Download summary (Markdown)",
                            payload.encode("utf-8"),
                            file_name=f"{artist}_ai_summary.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    except RuntimeError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.exception(e)
                else:
                    st.caption(
                        "The model isn’t called until you press the button. "
                        "Summaries are cached for 24 hours per artist + dataset."
                    )

        # === SECOND ROW ===  Sentiment over time (LEFT)  |  Distribution (RIGHT)
        bot_left, bot_right = st.columns([1, 1])

        with bot_left:
            if "year" in df_a.columns and "compound" in df_a.columns:
                ts = df_a.dropna(subset=["year"]).groupby("year")["compound"].mean().reset_index()
                fig = px.line(
                    ts,
                    x="year",
                    y="compound",
                    title="Average sentiment over time (VADER compound)",
                    template="plotly_white",
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)
                explain(
                    "**Line chart**: average VADER compound by release year for the selected artist "
                    "(−1=negative, +1=positive)."
                )

        with bot_right:
            if "compound" in df_a.columns:
                fig = px.box(
                    df_a,
                    x="artist",
                    y="compound",
                    title="Song-level sentiment distribution",
                    points="outliers",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
                explain("**Box plot**: spread of per-song sentiment for the selected artist.")

         # --- Spotify Artist iFrame ---
        st.subheader("🎵 Listen on Spotify - Artist's Top Songs")
        artist_spotify_id = df_a["artist_id"].dropna().unique()[0] if "artist_id" in df_a.columns else None

        if artist_spotify_id:
            embed_html = f"""
                <iframe style="border-radius:12px"
                        src="https://open.spotify.com/embed/artist/{artist_spotify_id}"
                        width="100%" height="200" frameBorder="0"
                        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                        loading="lazy"></iframe>
            """
            components.html(embed_html, height=220)
        else:
            st.info("No Spotify artist ID available for this artist.")
        
        # --- Embedded Spotify players for this artist ---
        st.subheader("🎧 Listen to songs used for analysis!")

        ids = []
        if "spotify_url" in df_a.columns:
            ids = [
                extract_spotify_track_id(u)
                for u in df_a["spotify_url"].dropna().unique().tolist()
            ]
            ids = [i for i in ids if i]  # keep only valid IDs

        # Limit and display as a small grid of players
        max_players = 20  # tweak as you like
        ids = ids[:max_players]

        if not ids:
            st.info("No Spotify links available for this artist.")
        else:
            ncols = 2  # 2 columns looks nice; try 3 if you have lots of space
            cols = st.columns(ncols)
            for i, tid in enumerate(ids):
                with cols[i % ncols]:
                    html = spotify_track_embed_html(tid, height=80)
                    components.html(html, height=100)  # a touch taller to avoid clipping


        # --- Songs table (clickable links) ---
        st.subheader("Songs for selected artist")
        cols = ["artist", "track_name", "year", "compound", "spotify_url", "genius_url"]
        present = [c for c in cols if c in df_a.columns]

        # Use Streamlit's LinkColumn to make URLs clickable
        column_cfg = {}
        if "spotify_url" in present:
            column_cfg["spotify_url"] = st.column_config.LinkColumn(
                "Spotify", display_text="Open"
            )
        if "genius_url" in present:
            column_cfg["genius_url"] = st.column_config.LinkColumn(
                "Genius", display_text="Open"
            )

        st.dataframe(
            df_a[present].sort_values("compound", ascending=False),
            use_container_width=True,
            column_config=column_cfg,
            hide_index=True,
        )
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
