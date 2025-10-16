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

import re, html as html_lib, itertools, math, random, tempfile

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os, hashlib, textwrap, json, datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple
from collections import Counter, defaultdict
from textwrap import shorten


# Sentencetransformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


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
_WORD_RE = re.compile(r"[a-zA-Z']+")
WORD_RE = re.compile(r"\b[\w']+\b", flags=re.IGNORECASE)


EMOTION_COLS = ["emotion_joy","emotion_sadness","emotion_anger","emotion_fear","emotion_disgust","emotion_surprise","emotion_neutral"]
EMO_LABELS = {
    "emotion_joy": "joy",
    "emotion_sadness": "sadness",
    "emotion_anger": "anger",
    "emotion_fear": "fear",
    "emotion_disgust": "disgust",
    "emotion_surprise": "surprise",
    "emotion_neutral": "neutral",
}
# A pleasant, distinct palette (adjust if you like)
EMO_COLORS = {
    "joy":        "#4CAF50",
    "sadness":    "#5DADE2",
    "anger":      "#E74C3C",
    "fear":       "#AF7AC5",
    "disgust":    "#27AE60",
    "surprise":   "#F4D03F",
    "neutral":    "#95A5A6",
}

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

# --------------------------------------------
# V3 Embedding & Similarity Helpers (cached)
# --------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load SentenceTransformer once per session.
    """
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Add `sentence-transformers` to requirements.txt"
        )
    return SentenceTransformer(model_name)

@st.cache_data(show_spinner=False)
def _song_embeddings(df_songs: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute embeddings for all songs with lyrics (cached).
    Returns: (embeddings, idx_rows, mask_has_lyrics)
      - embeddings: shape [N_with_lyrics, 384]
      - idx_rows: integer index positions mapping back to df_songs
      - mask_has_lyrics: boolean mask over df_songs rows
    """
    if df_songs.empty:
        return np.zeros((0, 384)), np.array([], dtype=int), np.array([], dtype=bool)

    mask = df_songs["lyrics"].fillna("").str.strip().ne("")
    if mask.sum() == 0:
        return np.zeros((0, 384)), np.where(mask)[0], mask.values

    mdl = _load_embedder(model_name)
    texts = df_songs.loc[mask, "lyrics"].fillna("").astype(str).tolist()
    embs = mdl.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embs), np.where(mask)[0], mask.values

@st.cache_data(show_spinner=False)
def _artist_centroids(df_songs: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compute a mean embedding per artist.
    Returns (df_stats, centroid_map)
      - df_stats: per-artist counts and availability
      - centroid_map: dict {artist -> 384-dim vector}
    """
    embs, idx_rows, mask = _song_embeddings(df_songs, model_name=model_name)
    centroid_map: Dict[str, np.ndarray] = {}
    if embs.shape[0] == 0:
        # No lyrics
        stats = (df_songs[["artist"]]
                 .dropna()
                 .value_counts()
                 .rename("track_count")
                 .reset_index())
        return stats, centroid_map

    sub = df_songs.iloc[idx_rows]
    # compute mean embedding per artist
    for a, block in sub.groupby("artist"):
        if block.empty: 
            continue
        # rows in sub for a
        rows = block.index
        # map to positions in embs by aligning indices
        pos = [np.where(idx_rows == r)[0][0] for r in rows if r in idx_rows]
        if len(pos) == 0: 
            continue
        centroid_map[a] = embs[pos].mean(axis=0)

    stats = (df_songs.groupby("artist")
             .size().rename("track_count")
             .reset_index())
    stats["has_centroid"] = stats["artist"].isin(centroid_map.keys())
    return stats, centroid_map

def _similar_artists(artist: str, centroid_map: Dict[str, np.ndarray], top_k: int = 8) -> pd.DataFrame:
    """
    Cosine similarity from chosen artist to others.
    """
    if artist not in centroid_map:
        return pd.DataFrame(columns=["artist", "similarity"])

    anchor = centroid_map[artist].reshape(1, -1)
    names, mats = [], []
    for a, v in centroid_map.items():
        if a == artist:
            continue
        names.append(a)
        mats.append(v.reshape(1, -1))
    if not names:
        return pd.DataFrame(columns=["artist", "similarity"])

    others = np.vstack([m for m in mats])
    sims = cosine_similarity(anchor, others).ravel()
    out = pd.DataFrame({"artist": names, "similarity": sims}).sort_values("similarity", ascending=False)
    return out.head(top_k)

# ------------- AI Critic (OpenAI) -------------
@st.cache_data(show_spinner=False, ttl=60*60*24)  # cache 24h per pair
def _cached_ai_critic(pair_key: str, content: str, model: str, temperature: float) -> str:
    """
    Call OpenAI once per (pair_key, content hash, model, temp).
    """
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)  # type: ignore[attr-defined]
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .streamlit/secrets.toml or env.")
    client = OpenAI(api_key=api_key)

    sys = (
        "You are a concise hip-hop critic. Compare the two artists using the provided"
        " metrics and lyric excerpts. Be specific but compact. Structure as:\n"
        "1) Overall contrast\n2) Lyrical themes & motifs\n3) Emotional tone"
        "\n4) Flow/Delivery observations\n5) Who each artist appeals to\n"
        "Limit to ~180-220 words total."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": content},
        ],
    )
    return resp.choices[0].message.content.strip()


# ------------- Lyric DNA Helpers -------------
@st.cache_data(show_spinner=False)
def _tokenize_lines(lyrics_list):
    """Split lyrics into lines and tokenize to simple word tokens."""
    lines = []
    for lyr in lyrics_list:
        if not isinstance(lyr, str) or not lyr.strip():
            continue
        for ln in lyr.splitlines():
            toks = [t.lower() for t in _WORD_RE.findall(ln)]
            if toks:
                lines.append(toks)
    return lines

def _build_cooccurrence(lines, window=8, stopwords=None):
    """Return unigram counts and pair counts from tokenized lines."""
    stop = set(stopwords or [])
    unigram = Counter()
    pair = Counter()

    for toks in lines:
        toks = [t for t in toks if t not in stop]
        if not toks: 
            continue
        # local window co-occurrence
        for i, w in enumerate(toks):
            unigram[w] += 1
            start = max(0, i - window)
            end   = min(len(toks), i + window + 1)
            for j in range(start, end):
                if j <= i:
                    continue
                v = toks[j]
                if v == w: 
                    continue
                if v in stop:
                    continue
                a, b = (w, v) if w < v else (v, w)
                pair[(a, b)] += 1

    return unigram, pair

def _pmi(pair_count, a_count, b_count, total):
    """PMI-like association score; clipped at 0 to avoid negatives."""
    # add tiny eps to be safe
    num = pair_count * total
    den = a_count * b_count or 1
    val = math.log2(max(num / den, 1e-12))
    return max(val, 0.0)

@st.cache_data(show_spinner=False)
def compute_lyric_network(
    df, 
    top_n_terms=80,        # limit nodes by frequency
    min_pair=6,            # minimum pair co-occurrence
    min_pmi=0.6,           # association threshold
    window=8,
    use_artist=None,       # None => all, else filter
):
    """Return (nodes, edges) for PyVis from a lyrics dataframe."""
    if use_artist:
        df = df[df["artist"] == use_artist]
    lyrics_list = df["lyrics"].dropna().tolist()
    lines = _tokenize_lines(lyrics_list)

    # Simple stoplist (tune as needed)
    stop = {
        "the","and","a","to","of","in","i","it","you","my","on","for","is","me","that",
        "we","be","with","your","this","not","im","all","do","no","so","but","got","ya",
        "uh","yeah","oh","la","like","just","go","up","out","get","gotta","aint","na"
    }

    unigram, pair = _build_cooccurrence(lines, window=window, stopwords=stop)
    if not unigram or not pair:
        return [], []

    # pick top terms by frequency
    vocab = [w for w, _ in unigram.most_common(top_n_terms)]
    vocab_set = set(vocab)

    total = sum(unigram[w] for w in vocab)
    nodes = []
    edges = []

    # node sizes by frequency
    for w in vocab:
        freq = unigram[w]
        size = 8 + 22 * (freq / max(1, unigram[vocab[0]]))  # 8..30
        nodes.append({"id": w, "label": w, "size": size, "title": f"{w}: {freq} occurrences"})

    # edges filtered by thresholds
    for (a, b), c in pair.items():
        if a not in vocab_set or b not in vocab_set:
            continue
        if c < min_pair:
            continue
        pmi = _pmi(c, unigram[a], unigram[b], total)
        if pmi < min_pmi:
            continue
        weight = 1 + 4 * min(1.0, pmi / 3.0)  # stroke width ~ PMI
        edges.append({"from": a, "to": b, "value": weight, "title": f"co-occurs: {c} | pmi: {pmi:.2f}"})

    return nodes, edges

def render_pyvis_network(nodes, edges, height="680px", physics=True):
    """Create a PyVis HTML and return it as a string (Streamlit-safe)."""
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#222")
    net.toggle_physics(physics)
    net.barnes_hut(gravity=-20000, central_gravity=0.3,
                   spring_length=110, spring_strength=0.01, damping=0.8)

    # ⬇️ Pass color/group if present
    for n in nodes:
        net.add_node(
            n["id"],
            label=n.get("label", n["id"]),
            size=n.get("size", 10),
            title=n.get("title", ""),
            color=n.get("color"),     # <—
        )

    for e in edges:
        net.add_edge(
            e["from"], e["to"],
            value=e.get("value", 1.0),
            title=e.get("title", ""),
            width=e.get("width"),          # << NEW
            color=e.get("color")           # << NEW
        )


    # Only style edges here; don't set node color in options.
    options = {
        "nodes": {"shape": "dot"},
        "edges": {"color": {"color": "#9aa6b2"}, "smooth": {"type": "dynamic"}},
        "interaction": {"tooltipDelay": 150, "hover": True},
        "physics": {"stabilization": {"iterations": 80}},
    }
    net.set_options(json.dumps(options))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    try:
        net.write_html(tmp.name, notebook=False, open_browser=False)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html_str = f.read()

        # --- Make the vis tooltip render our node titles as HTML, not plain text ---
        # 1) slightly nicer tooltip styling (optional)
        inject_css = """
        <style>
        .vis-tooltip {
            white-space: normal !important;
            max-width: 520px;        /* wrap long lines */
            font-size: 13px;
            line-height: 1.25;
            box-shadow: 0 6px 18px rgba(0,0,0,.12);
            border-radius: 8px;
            padding: 10px 12px;
            background: #fff;
        }
        .vis-tooltip b { color: #111; }
        .vis-tooltip ul { margin: 6px 0 0 18px; color:#111; }
        .vis-tooltip li { margin: 0 0 2px 0; }
        </style>
        """

        # 2) on popup show, convert textContent -> innerHTML so our <br>, <ul>, etc. render
        inject_js = """
        <script>
        (function () {
          // Wait until vis creates the Network and the tooltip node exists
          const tryHook = () => {
            const canvases = document.querySelectorAll('div.vis-network');
            if (!canvases.length) { requestAnimationFrame(tryHook); return; }
            // Find the first network container on this page (the component we render)
            const container = canvases[canvases.length - 1];
            if (!container || !container.parentNode) { requestAnimationFrame(tryHook); return; }

            // The tooltip node is created by vis-network with class '.vis-tooltip'
            let tooltip = document.querySelector('.vis-tooltip');
            if (!tooltip) { requestAnimationFrame(tryHook); return; }

            // Monkey-patch showPopup: rewrite the tooltip's text as HTML
            const orig = tooltip.__setTextAsHtmlApplied ? null : tooltip;
            if (orig) {
              tooltip.__setTextAsHtmlApplied = true;
              // Use MutationObserver to convert *any* text writes into HTML
              const mo = new MutationObserver(() => {
                // If library set textContent, swap to innerHTML
                if (tooltip.textContent && tooltip.textContent.indexOf('<') !== -1) {
                  tooltip.innerHTML = tooltip.textContent;
                }
              });
              mo.observe(tooltip, { childList: true, characterData: true, subtree: true });
            }
          };
          tryHook();
        })();
        </script>
        """

        # Inject CSS+JS just before </body> to keep the file valid
        html_str = html_str.replace("</body>", inject_css + inject_js + "</body>")

        return html_str
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass



def _dominant_emotion_for_terms(df, terms, min_occurrences=1):
    """
    For each term, find songs whose lyrics contain that term, average emotion columns,
    and pick the dominant emotion. Returns {term: {"label": str, "score": float}}.
    """
    out = {}
    if not set(EMOTION_COLS).issubset(df.columns):
        return out

    # Pre-ensure we have a fast lowercase lyrics column
    if "_lyrics_lc" not in df.columns:
        df["_lyrics_lc"] = df["lyrics"].fillna("").str.lower()

    for t in terms:
        if not t:
            continue
        # whole-word match; tweak if you want stem/lemmatized behavior
        pat = r"\b" + re.escape(t.lower()) + r"\b"
        mask = df["_lyrics_lc"].str.contains(pat, regex=True)
        if mask.sum() < min_occurrences:
            continue

        emo_means = df.loc[mask, EMOTION_COLS].mean(numeric_only=True)
        if emo_means.isna().all():
            continue

        dom_col = emo_means.idxmax()
        dom_label = EMO_LABELS.get(dom_col, "neutral")
        dom_score = float(emo_means.max())
        out[t] = {"label": dom_label, "score": dom_score}
    return out


def _safe_html(s: str) -> str:
    return html_lib.escape(s, quote=False)

def example_snippets_for_term(df, term: str, max_snippets=3, max_chars=120):
    """Return a few lyric lines that contain the term (case-insensitive)."""
    term_lc = term.lower()
    hits = []
    for txt in df["lyrics"].dropna().astype(str):
        for line in txt.splitlines():
            # token check so 'at' doesn't match 'that'
            tokens = [t.lower() for t in WORD_RE.findall(line)]
            if term_lc in tokens:
                line = line.strip()
                if line and line not in hits:
                    hits.append(line)
    if not hits:
        return []
    # de-duplicate and sample
    random.shuffle(hits)
    hits = hits[:max_snippets]
    # shorten + HTML escape for vis-network tooltip
    return [ _safe_html(shorten(h, width=max_chars, placeholder="…")) for h in hits ]

def scale_edges_for_fade(edges):
    """Add width + rgba color based on normalized value (strength)."""
    if not edges:
        return edges
    vals = [e.get("value", 1.0) for e in edges]
    vmin, vmax = min(vals), max(vals)
    denom = (vmax - vmin) or 1.0
    for e in edges:
        v = e.get("value", 1.0)
        t = (v - vmin) / denom
        # width 1..4, alpha 0.15..0.95
        e["width"] = 1.0 + 3.0 * t
        alpha = 0.15 + 0.80 * t
        e["color"] = f"rgba(154,166,178,{alpha:.2f})"  # soft gray w/ alpha
    return edges


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
intro, overview, emotions, keywords, search, topics_map, fusion, table, story, dna, critic_tab = st.tabs([
    "👋 Intro",
    "📊 Artist Overview",
    "😊 Emotion Explorer",
    "🔎 Keywords & Phrases",
    "🧭 Semantic Search",
    "🗺️ Topics Map",
    "💡 Topic + Emotion",
    "📄 Raw Songs",
    "🧩 Story Mode",
    "🎯 Lyric DNA", 
    "🤖 AI Critic"

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
                [default_model or "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
                index=0,
                key="ai_summary_model"      # <-- unique key
            )
            temperature = st.slider(
                "Creativity (temperature)", 0.0, 1.2, 0.6, 0.1,
                key="ai_summary_temp"       # <-- unique key
            )

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
                    iframe_html = spotify_track_embed_html(tid, height=80)
                    components.html(iframe_html, height=100) # a touch taller to avoid clipping

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


# -----------------------------
# Lyric DNA (Similarity + PyVis network)
# -----------------------------

with dna:
    st.header("Lyric DNA")

    sub_sim, sub_net = st.tabs(["🎯 Similarity", "🕸️ Word Network"])

    with sub_sim:
        # -- paste everything that was under: with dna_tab: (the embedding/centroid similarity UI) --
        # keep widget keys unique (e.g., "sim_model", "sim_k") so they don't collide with others
        st.header("Lyric DNA — Artist Similarity")
        if songs.empty:
            st.info("No data yet.")
        else:
            # Controls
            model_name = st.selectbox("Embedding model", ["all-MiniLM-L6-v2"], index=0, help="Used to compute lyric embeddings.")
            stats, centroids = _artist_centroids(songs, model_name=model_name)

            if len(centroids) == 0:
                st.warning("No lyrics available to compute embeddings.")
                st.dataframe(stats, use_container_width=True)
            else:
                # Pick artist + k
                artists_avail = sorted(list(centroids.keys()))
                colA, colB = st.columns([1, 1])
                with colA:
                    artist_pick = st.selectbox("Anchor artist", artists_avail, index=0)
                with colB:
                    top_k = st.slider("Top similar artists (k)", 3, 15, 8)

                # Compute neighbors
                sims = _similar_artists(artist_pick, centroids, top_k=top_k)

                left, right = st.columns([1.25, 1])
                with left:
                    st.subheader(f"Most similar to **{artist_pick}**")
                    if sims.empty:
                        st.info("Not enough embeddings for similarity.")
                    else:
                        fig = px.bar(
                            sims.sort_values("similarity"),
                            x="similarity", y="artist",
                            orientation="h", title="Cosine similarity",
                            template="plotly_white", range_x=[0, 1]
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with right:
                    st.subheader("Artist coverage")
                    st.caption("Tracks per artist and whether we computed a centroid.")
                    st.dataframe(stats.sort_values(["has_centroid", "track_count"], ascending=[False, False]),
                                use_container_width=True)

                with st.expander("What am I looking at?"):
                    st.write(
                        "We embed each song’s lyrics with MiniLM and average per artist to get a **centroid**."
                        " Cosine similarity between centroids yields the closeness shown here."
                        " Similarity improves as you add more songs per artist."
                    )
    with sub_net:
        # -- paste everything that was under: with lyric_dna: (the PyVis co-occurrence network UI) --
        # widget keys already unique: "lydna_*"
        st.header("Lyric DNA — Word Co-occurrence Network")

        if songs.empty or "lyrics" not in songs.columns:
            st.info("No lyrics available to build the network.")
        else:
            # Controls (unique keys to avoid clashes)
            scope = st.radio("Scope", ["Selected artist", "All artists"], horizontal=True, key="lydna_scope")
            window = st.slider("Co-occurrence window (tokens)", 3, 12, 8, 1, key="lydna_window")
            top_n = st.slider("Max terms (most frequent)", 30, 200, 100, 10, key="lydna_topn")
            min_pair = st.slider("Min co-occurrences", 2, 15, 6, 1, key="lydna_minpair")
            min_pmi = st.slider("Min PMI score", 0.0, 2.0, 0.6, 0.05, key="lydna_minpmi")
            physics = st.checkbox("Enable physics (force simulation)", True, key="lydna_physics")
            # Focus/spotlight which emotion to emphasize
            _focus_choices = ["All", "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
            focus_emo = st.selectbox("Focus on emotion", _focus_choices, index=0, key="lydna_focus")


            # If you already have the currently selected artist from the Overview tab, reuse it.
            # Otherwise we’ll give a local selector:
            try:
                current_artist  # from your overview tab if in same script scope
                artist = current_artist
            except NameError:
                artists = sorted(songs["artist"].dropna().unique().tolist())
                artist = st.selectbox("Artist", artists, index=0, key="lydna_artist")

            use_artist = None if scope == "All artists" else artist

            with st.spinner("Building network..."):
                nodes, edges = compute_lyric_network(
                    songs, 
                    top_n_terms=top_n,
                    min_pair=min_pair,
                    min_pmi=min_pmi,
                    window=window,
                    use_artist=use_artist
                )

            if not nodes or not edges:
                st.warning("Not enough signal to draw a network with the current thresholds. Try lowering the thresholds or widening the window.")
            else:
                # --- NEW: color nodes by dominant emotion ---
                terms = [n["label"] for n in nodes]  # assuming your node "label" is the term string
                emo_map = _dominant_emotion_for_terms(songs, terms, min_occurrences=1)

                # Attach color + legend label to each node; fall back to neutral if unknown
                for n in nodes:
                    t = n.get("label", "")
                    emo_info = emo_map.get(t, {"label": "neutral", "score": 0.0})
                    emo = emo_info["label"]
                    n["color"] = EMO_COLORS.get(emo, EMO_COLORS["neutral"])
                    n["title"] = f"{n.get('title','')}" + (f"<br><b>Emotion:</b> {emo} ({emo_info['score']:.2f})" if emo_info["score"] else "")
                    n["group"] = emo  # (optional) useful for future legends/grouping

                # 1) Fade low-weight edges (sets width + rgba color)
                edges = scale_edges_for_fade(edges)

                # 2) Add lyric snippet examples into node tooltips
                _snip_cache = {}
                for n in nodes:
                    term = n.get("label", "")
                    if term:
                        if term not in _snip_cache:
                            _snip_cache[term] = example_snippets_for_term(songs, term, max_snippets=3, max_chars=110)
                        snips = _snip_cache[term]
                        if snips:
                            # Append as a small list in the vis-network tooltip (HTML)
                            n["title"] = f"{n.get('title','')}<br><b>Examples:</b><ul><li>" + "</li><li>".join(snips) + "</li></ul>"

                # 3) Spotlight a single emotion, gently dim the rest
                if focus_emo != "All":
                    focused_terms = {n["id"] for n in nodes if n.get("group") == focus_emo}
                    # Dim node color/size when not focused
                    for n in nodes:
                        if n.get("group") != focus_emo:
                            n["color"] = "#d1d5db"             # light gray
                            n["size"] = max(6, int(n.get("size", 10) * 0.65))

                    # Soften edges that don't touch a focused node
                    for e in edges:
                        if e["from"] not in focused_terms and e["to"] not in focused_terms:
                            e["width"] = 0.6
                            # very soft stroke
                            e["color"] = "rgba(200,200,200,0.25)"

                html_str = render_pyvis_network(nodes, edges, height="720px", physics=physics)
                components.html(html_str, height=760, scrolling=False)

                # Optional: a tiny legend under the graph
                legend_cols = st.columns(len(EMO_COLORS))
                for (emo, col) in zip(EMO_COLORS.keys(), legend_cols):
                    col.markdown(
                        f"""<div style="display:flex;align-items:center;gap:8px;">
                            <div style="width:14px;height:14px;border-radius:3px;background:{EMO_COLORS[emo]};"></div>
                            <span style="font-size:0.9rem;">{emo.title()}</span>
                        </div>""",
                        unsafe_allow_html=True
            )

                # Small KPIs + download
                colA, colB, colC = st.columns(3)
                colA.metric("Nodes", f"{len(nodes)}")
                colB.metric("Edges", f"{len(edges)}")
                colC.metric("Scope", "All artists" if use_artist is None else artist)

                st.download_button(
                    "Download network (HTML)",
                    data=html_str.encode("utf-8"),
                    file_name=f"lyric_dna_{('all' if use_artist is None else use_artist)}.html",
                    mime="text/html",
                    use_container_width=True
                )

        with st.expander("What am I looking at?"):
            st.markdown("""
    **Lyric DNA** visualizes a word co-occurrence network built from the lyrics.
    - **Nodes** = frequent words (after removing common stopwords).
    - **Edges** = words that appear within a token *window* of each other; thicker edges ≈ stronger association (PMI).
    - Use the sliders to control vocabulary size, window, and thresholds.
    - Hover nodes/edges for counts; drag to explore; download as a standalone HTML.
            """)


# -----------------------------
# AI Critic Tab
# -----------------------------

with critic_tab:
    st.header("AI Critic — Compare Two Artists")
    if songs.empty:
        st.info("No data yet.")
    else:
        # Controls
        artists = sorted(songs["artist"].dropna().unique().tolist())
        col1, col2 = st.columns([1, 1])
        with col1:
            a1 = st.selectbox("Artist A", artists, index=0)
        with col2:
            a2 = st.selectbox("Artist B", [a for a in artists if a != a1], index=0)

        # Model controls (reuse your default model secret if available)
        default_model = None
        try:
            default_model = st.secrets.get("OPENAI_MODEL", None)  # type: ignore[attr-defined]
        except Exception:
            default_model = None
        
        model = st.selectbox(
            "Model",
            [default_model or "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
            index=0,
            key="critic_model"          # <-- unique key
        )
        temperature = st.slider(
            "Creativity (temperature)", 0.0, 1.0, 0.3, 0.05,
            key="critic_temp"           # <-- unique key
        )


        # Build compact stats + small lyric excerpts for each artist
        def pack_artist_blob(name: str) -> Dict[str, object]:
            df = songs[songs["artist"] == name].copy()
            out = {
                "artist": name,
                "tracks": int(df.shape[0]),
                "avg_compound": float(df["compound"].mean()) if "compound" in df else None,
            }
            # emotion mix
            emo_cols = [c for c in df.columns if c.startswith("emotion_")]
            if emo_cols:
                emo_avg = df[emo_cols].mean(numeric_only=True).to_dict()
                dom = max(emo_avg.items(), key=lambda kv: kv[1]) if emo_avg else (None, None)
                out["dominant_emotion"] = {"label": (dom[0] or "").replace("emotion_", ""), "score": float(dom[1] or 0)}
                # keep top 3 for context
                top3 = sorted([(k.replace("emotion_", ""), float(v)) for k, v in emo_avg.items()],
                              key=lambda kv: kv[1], reverse=True)[:3]
                out["emotion_top3"] = top3

            # few short lyric snippets (kept brief to control tokens)
            L = []
            for t, lyr in df[["track_name", "lyrics"]].dropna().head(6).itertuples(index=False, name=None):
                if not isinstance(lyr, str) or len(lyr.strip()) == 0:
                    continue
                # get first ~200 chars (avoid huge payloads)
                L.append({"track": t, "snippet": textwrap.shorten(lyr.replace("\n", " "), width=200, placeholder="…")})
            out["snippets"] = L
            return out

        blobA = pack_artist_blob(a1)
        blobB = pack_artist_blob(a2)

        # Build a compact JSON payload for GPT
        content_dict = {
            "artist_A": blobA,
            "artist_B": blobB,
            "instructions": "Compare A vs B using stats and snippets. Keep it balanced and insightful."
        }
        content_json = json.dumps(content_dict, ensure_ascii=False)
        pair_key = hashlib.md5((a1 + "||" + a2 + "||" + str(len(content_json))).encode("utf-8")).hexdigest()

        gen = st.button("✨ Generate AI comparison", type="primary", use_container_width=True)
        if gen:
            try:
                result = _cached_ai_critic(pair_key, content_json, model=model, temperature=temperature)
                st.markdown(result)
                st.caption(f"AI comparison for **{a1}** vs **{a2}**  •  model: `{model}`  •  temp: {temperature:.2f}")
                st.download_button(
                    "Download comparison (Markdown)",
                    result.encode("utf-8"),
                    file_name=f"critic_{a1}_vs_{a2}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.exception(e)

        with st.expander("What am I looking at?"):
            st.write(
                "AI Critic mixes **quantitative signals** (sentiment, emotions, track counts) with **lyric snippets**"
                " to produce a concise, human-readable comparison. Results are cached for 24 hours per artist pair."
            )
