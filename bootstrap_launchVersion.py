#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlowTrendz — Launch v2 Bootstrap (always-on Emotions + Topics, all-artists run)
- Spotify top tracks + (optional) audio features
- Genius lyrics (defensive against API shape)
- Sentiment (VADER) -> Emotions (HF if available, else fast proxy)
- BERTopic topics + UMAP (version-agnostic)
Outputs: data/curated/songs_curated.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(".")
DATA = ROOT / "data"
RAW = DATA / "raw"
CURATED = DATA / "curated"
SUMMARIES = DATA / "summaries"
LOGS = ROOT / "logs"
for p in (RAW, CURATED, SUMMARIES, LOGS):
    p.mkdir(parents=True, exist_ok=True)

ENV_FILE = ROOT / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "bootstrap_v2.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)

def load_env():
    try:
        load_dotenv(dotenv_path=ENV_FILE if ENV_FILE.exists() else None, encoding="utf-8")
    except TypeError:
        load_dotenv(dotenv_path=ENV_FILE if ENV_FILE.exists() else None)

def read_secrets() -> Dict[str,str]:
    """Accept SPOTIFY_* or SPOTIPY_*; GENIUS_TOKEN or GENIUS_API_TOKEN."""
    load_env()
    genius = os.getenv("GENIUS_TOKEN") or os.getenv("GENIUS_API_TOKEN")
    cid = os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("SPOTIPY_CLIENT_ID")
    csecret = os.getenv("SPOTIFY_CLIENT_SECRET") or os.getenv("SPOTIPY_CLIENT_SECRET")
    cfg = {"GENIUS_TOKEN": genius, "SPOTIFY_CLIENT_ID": cid, "SPOTIFY_CLIENT_SECRET": csecret}
    missing = [k for k,v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}. Add them to .env or Streamlit secrets.")
    return cfg

# -------- Spotify helpers --------
def spotify_client(client_id: str, client_secret: str):
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth, requests_timeout=20, retries=3, status_forcelist=(429,500,502,503,504))

def fetch_top_tracks(sp, artist_name: str, market: str="US", limit: int=10):
    results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
    if not results["artists"]["items"]:
        return []
    artist_id = results["artists"]["items"][0]["id"]
    tracks = sp.artist_top_tracks(artist_id, country=market)["tracks"]
    out = []
    for t in tracks[:limit]:
        out.append({
            "artist": results["artists"]["items"][0]["name"],
            "artist_id": artist_id,
            "track_id": t["id"],
            "track_name": t["name"],
            "album": t["album"]["name"],
            "release_date": t["album"].get("release_date"),
            "spotify_url": t["external_urls"]["spotify"],
            "popularity": t.get("popularity"),
        })
    return out

def add_audio_features(sp, df: pd.DataFrame, enabled: bool=True) -> pd.DataFrame:
    if not enabled:
        logging.info("[Spotify] Skipping audio features (--skip-audio).")
        return df
    ids = pd.Series(df["track_id"].dropna().astype(str).unique()).tolist()
    feats = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        try:
            f = sp.audio_features(chunk) or []
        except Exception as e:
            logging.error(f"[Spotify] audio_features batch error ({i}:{i+50}): {e}")
            for tid in chunk:
                try:
                    single = sp.audio_features([tid]) or []
                    feats += single
                    time.sleep(0.05)
                except Exception as e2:
                    logging.error(f"[Spotify] audio_features single error for {tid}: {e2}")
            continue
        feats += f
        time.sleep(0.1)
    fdf = pd.DataFrame([x for x in feats if isinstance(x, dict) and x.get("id")])
    if not fdf.empty:
        fdf = fdf.rename(columns={"id":"track_id"})
        df = df.merge(fdf, on="track_id", how="left")
    else:
        logging.warning("[Spotify] No audio features returned; columns will be NaN.")
    return df

# -------- Genius helpers --------
def genius_client(token: str):
    import lyricsgenius
    g = lyricsgenius.Genius(
        token,
        timeout=15,
        retries=3,
        remove_section_headers=True,
        skip_non_songs=True,
        excluded_terms=["(Remix)","(Live)"]
    )
    return g

def clean_title(title: str) -> str:
    t = re.sub(r"\s*-\s*[^-]+$", "", title or "")
    t = re.sub(r"\(feat\..*?\)", "", t, flags=re.I)
    t = re.sub(r"\(with .*?\)", "", t, flags=re.I)
    t = re.sub(r'\s*\[.*?\]\s*', " ", t)
    t = re.sub(r'["“”]', "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def fuzzy_ok(a: str, b: str, cutoff: int=88) -> bool:
    from rapidfuzz import fuzz
    return fuzz.token_set_ratio((a or "").lower(), (b or "").lower()) >= cutoff

def fetch_lyrics_for_tracks(genius, df_tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_tracks.iterrows():
        artist = r["artist"]
        raw_title = r["track_name"]
        query = clean_title(raw_title)
        try:
            song = genius.search_song(query, artist)
            time.sleep(1)
        except Exception as e:
            logging.warning(f"[Genius] search error for {artist} - {query}: {e}")
            song = None

        if song:
            try:
                title_ok = fuzzy_ok(query, getattr(song, "title", "") or getattr(song, "full_title", ""))
                lyrics = getattr(song, "lyrics", None)
                gid = (
                    getattr(song, "id", None)
                    or getattr(song, "_id", None)
                    or (getattr(song, "to_dict", lambda: {})() or {}).get("id")
                    or getattr(getattr(song, "song", {}), "get", lambda *_: None)("id")
                )
                gurl = (
                    getattr(song, "url", None)
                    or (getattr(song, "to_dict", lambda: {})() or {}).get("url")
                    or getattr(getattr(song, "song", {}), "get", lambda *_: None)("url")
                )
            except Exception as e:
                logging.info(f"[Genius] unexpected song shape for: {artist} - {raw_title}: {e}")
                title_ok, lyrics, gid, gurl = False, None, None, None
        else:
            title_ok, lyrics, gid, gurl = False, None, None, None

        if title_ok and lyrics:
            rows.append({**r.to_dict(), "genius_id": gid, "genius_url": gurl, "lyrics": lyrics})
        else:
            logging.info(f"[Genius] No acceptable lyrics for: {artist} - {raw_title} (query='{query}')")
    return pd.DataFrame(rows)

# -------- Sentiment & Emotions --------
def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    def _sc(text):
        try: return sia.polarity_scores(text or "")
        except Exception: return {"neg":0,"neu":0,"pos":0,"compound":0}
    scores = df["lyrics"].fillna("").map(_sc).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), scores], axis=1)

def add_emotions_fast_proxy(df: pd.DataFrame) -> pd.DataFrame:
    if {"neg","pos","neu"}.issubset(df.columns) is False:
        return df
    neg = df["neg"].fillna(0.0)
    pos = df["pos"].fillna(0.0)
    neu = df["neu"].fillna(0.0)
    df["emotion_joy"] = pos.clip(0,1)
    df["emotion_sadness"] = (neg * 0.6).clip(0,1)
    df["emotion_anger"] = (neg * 0.4).clip(0,1)
    df["emotion_fear"] = (neg * 0.2).clip(0,1)
    df["emotion_disgust"] = (neg * 0.2).clip(0,1)
    df["emotion_surprise"] = (pos * 0.2 + neu * 0.1).clip(0,1)
    df["emotion_neutral"] = neu.clip(0,1)
    return df

def add_emotions_hf(df: pd.DataFrame, model_name: str="j-hartmann/emotion-english-distilroberta-base") -> pd.DataFrame:
    try:
        from transformers import pipeline
        clf = pipeline("text-classification", model=model_name, return_all_scores=True, truncation=True)
    except Exception as e:
        logging.warning(f"[Emotions] transformers not available ({e}); falling back to fast proxy.")
        return add_emotions_fast_proxy(df)
    labels = ["anger","disgust","fear","joy","neutral","sadness","surprise"]
    vecs = []
    texts = df["lyrics"].fillna("").astype(str).tolist()
    for t in texts:
        try:
            scores = clf(t[:512])
            d = {x["label"].lower(): x["score"] for x in scores[0]}
        except Exception:
            d = {}
        vecs.append([d.get(k, 0.0) for k in labels])
        time.sleep(0.005)
    emo = pd.DataFrame(vecs, columns=[f"emotion_{k}" for k in labels])
    return pd.concat([df.reset_index(drop=True), emo], axis=1)

# -------- Topics (BERTopic + UMAP) --------
def add_topics(df: pd.DataFrame, text_col: str="lyrics") -> pd.DataFrame:
    """
    Version-agnostic topics + 2-D UMAP coordinates.
    """
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        import umap
        import hdbscan  # noqa: F401
    except Exception as e:
        logging.warning(f"[Topics] BERTopic stack not available ({e}). Skipping topics.")
        return df

    texts = df[text_col].fillna("").astype(str).tolist()
    if sum(1 for t in texts if t.strip()) < 3:
        logging.info("[Topics] Not enough texts to model; skipping.")
        return df

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    topic_model = BERTopic(
        embedding_model=None,
        verbose=False,
        calculate_probabilities=False,
        min_topic_size=3
    )
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)

    out = df.copy()
    out["topic"] = topics

    try:
        info = topic_model.get_topic_info()
        label_map = dict(zip(info["Topic"].tolist(), info["Name"].tolist()))
        out["topic_label"] = out["topic"].map(label_map).fillna("Outliers")
    except Exception as e:
        logging.warning(f"[Topics] Could not fetch topic labels: {e}")
        out["topic_label"] = out["topic"].astype(str)

    try:
        n_docs = len(texts)
        n_neighbors = max(2, min(15, n_docs - 1))
        um = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.05,
            metric="cosine",
            random_state=42,
        )
        coords = um.fit_transform(embeddings)
        out["umap_x"] = coords[:, 0]
        out["umap_y"] = coords[:, 1]
    except Exception as e:
        logging.warning(f"[Topics] UMAP computation failed: {e}")

    return out

# -------- Artist list --------
def load_artists(default_limit: int) -> List[str]:
    yaml_path = ROOT / "artists.yaml"
    if yaml_path.exists():
        try:
            import yaml
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            artists = data if isinstance(data, list) else data.get("artists", [])
            artists = [str(a) for a in artists if str(a).strip()]
            if artists:
                logging.info(f"Loaded {len(artists)} artists from artists.yaml")
                return artists
        except Exception as e:
            logging.warning(f"Could not parse artists.yaml ({e}); using defaults.")
    defaults = [
        "Jay-Z","Nas","Kanye West","Drake","Kendrick Lamar",
        "Nicki Minaj","J. Cole","Lil Wayne","Outkast","Travis Scott",
        "A Tribe Called Quest","The Notorious B.I.G.","2Pac","Future","Doja Cat",
        "Yeat", "Juice WRLD", "Gunna", "Cardi B", "Rapsody", "Latto", "Megan Thee Stallion",
        "Rick Ross", "Big Sean", "Meek Mill"
    ]
    if default_limit and default_limit < len(defaults):
        return defaults[:default_limit]
    return defaults

# -------- Pipeline --------
def run_pipeline(market: str="US", limit_per_artist: int=10, skip_audio: bool=True,
                 checkpoint_every: int=10) -> Path:
    """
    Spotify → Genius → Sentiment → Emotions (HF fallback) → Topics + UMAP → Parquet
    Always runs emotions + topics.
    """
    secrets = read_secrets()
    sp = spotify_client(secrets["SPOTIFY_CLIENT_ID"], secrets["SPOTIFY_CLIENT_SECRET"])
    g = genius_client(secrets["GENIUS_TOKEN"])

    artists = load_artists(default_limit=0)
    all_rows = []
    for idx, a in enumerate(artists, 1):
        logging.info(f"[Spotify] Top tracks for: {a}")
        rows = fetch_top_tracks(sp, a, market=market, limit=limit_per_artist)
        all_rows += rows
        if idx % checkpoint_every == 0:
            tmp = pd.DataFrame(all_rows).drop_duplicates(subset=["track_id"])
            tmp.to_parquet(CURATED / "songs_checkpoint.parquet", index=False)
        time.sleep(0.3)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["track_id"])
    if df.empty:
        raise RuntimeError("No tracks found. Check artist names and credentials.")
    logging.info(f"Collected {df.shape[0]} unique tracks across {len(artists)} artists.")

    df = add_audio_features(sp, df, enabled=not skip_audio)

    lyr = fetch_lyrics_for_tracks(g, df)
    if lyr.empty:
        logging.warning("No lyrics found. Proceeding without sentiment.")
        curated = df.copy()
    else:
        curated = compute_sentiment(lyr)
        curated = add_emotions_hf(curated)    # always-on, with HF fallback
        curated = add_topics(curated, text_col="lyrics")

    curated["artist"] = curated.get("artist", curated.get("artist_name", ""))
    curated["year"] = pd.to_datetime(curated.get("release_date"), errors="coerce").dt.year
    if "lyrics" in curated.columns:
        curated["lyrics_snippet"] = curated["lyrics"].fillna("").str.slice(0, 200)

    out = CURATED / "songs_curated.parquet"
    curated.to_parquet(out, index=False)
    logging.info(f"Saved curated parquet -> {out.resolve()}")
    return out

# -------- CLI --------
def parse_args():
    p = argparse.ArgumentParser(description="FlowTrendz — Launch v2 bootstrap (always emotions + topics)")
    p.add_argument("--market", default="US")
    p.add_argument("--limit", type=int, default=10, help="Top tracks per artist")
    p.add_argument("--skip-audio", action="store_true", help="Skip Spotify audio features")
    p.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint frequency (artists)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out = run_pipeline(market=args.market, limit_per_artist=args.limit,
                       skip_audio=args.skip_audio, checkpoint_every=args.checkpoint_every)
    # quick summary
    try:
        df = pd.read_parquet(out)
        print(f"Rows: {df.shape[0]:,} | Artists: {df['artist'].nunique() if 'artist' in df else '—'}")
        print("Emotions present:", any(c.startswith("emotion_") for c in df.columns))
        print("Topics present:", {"topic_label","umap_x","umap_y"}.issubset(df.columns))
    except Exception:
        pass
