#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlowTrendz — Launch Version (Bootstrap)
Clean, documented pipeline for:
- Loading secrets
- Spotify top-tracks + audio features
- Genius lyrics (with cleaned titles + fuzzy validation)
- Sentiment (VADER)
- Curated export compatible with the launch app
This is a slimmed, readable launch script that wraps v05 logic and
adds clearer logs and docstrings.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# -----------------------------
# Paths & logging
# -----------------------------
ROOT = Path(".")
DATA = ROOT / "data"
RAW = DATA / "raw"
CURATED = DATA / "curated"
LOGS = ROOT / "logs"
RAW.mkdir(parents=True, exist_ok=True)
CURATED.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
ENV_FILE = ROOT / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "flowtrendz_launch.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)

def load_env():
    """Load .env in UTF‑8; allow shell env as fallback."""
    try:
        load_dotenv(dotenv_path=ENV_FILE if ENV_FILE.exists() else None, encoding="utf-8")
    except TypeError:
        load_dotenv(dotenv_path=ENV_FILE if ENV_FILE.exists() else None)

def read_secrets() -> Dict[str,str]:
    """Read credentials from env; raise if missing."""
    load_env()
    req = ["GENIUS_TOKEN","SPOTIFY_CLIENT_ID","SPOTIFY_CLIENT_SECRET"]
    cfg = {k: os.getenv(k) for k in req}
    missing = [k for k,v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}. Create a .env with those keys.")
    return cfg

# -----------------------------
# Spotify helpers
# -----------------------------
def spotify_client(client_id: str, client_secret: str):
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth, requests_timeout=20, retries=3, status_forcelist=(429,500,502,503,504))

def fetch_top_tracks(sp, artist_name: str, market: str="US", limit: int=10) -> List[Dict]:
    """Top tracks for an artist name (best match)."""
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

def add_audio_features(sp, df: pd.DataFrame) -> pd.DataFrame:
    """Robust audio-features fetch with chunking + single-ID fallback."""
    ids = (
        pd.Series(df["track_id"].dropna().astype(str).unique())
        .loc[lambda s: s.str.len() > 0]
        .tolist()
    )
    feats = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        try:
            f = sp.audio_features(chunk) or []
        except Exception as e:
            logging.error(f"[Spotify] audio_features batch error ({i}:{i+50}): {e}")
            # Fallback per-ID
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

# -----------------------------
# Genius helpers
# -----------------------------
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
    """Search by cleaned title + artist; accept if titles match by fuzzy threshold.
    Defensive against LyricsGenius versions where Song.id/url live in different places.
    """
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

        # Pull fields defensively
        if song:
            try:
                title_ok = fuzzy_ok(query, getattr(song, "title", "") or getattr(song, "full_title", ""))
                lyrics = getattr(song, "lyrics", None)
                # Try multiple places for id/url
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
            rows.append({
                **r.to_dict(),
                "genius_id": gid,
                "genius_url": gurl,
                "lyrics": lyrics
            })
        else:
            logging.info(f"[Genius] No acceptable lyrics for: {artist} - {raw_title} (query='{query}')")

    return pd.DataFrame(rows)

# -----------------------------
# Sentiment
# -----------------------------
def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add VADER sentiment columns (neg/neu/pos/compound)."""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    def _sc(text):
        try:
            return sia.polarity_scores(text or "")
        except Exception:
            return {"neg":0,"neu":0,"pos":0,"compound":0}
    scores = df["lyrics"].fillna("").map(_sc).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), scores], axis=1)

# -----------------------------
# Pipeline
# -----------------------------
def run_pipeline(artists: List[str], market: str="US", limit_per_artist: int=10) -> Path:
    """End‑to‑end run: Spotify → Genius → Sentiment → Curated file."""
    secrets = read_secrets()
    sp = spotify_client(secrets["SPOTIFY_CLIENT_ID"], secrets["SPOTIFY_CLIENT_SECRET"])
    g = genius_client(secrets["GENIUS_TOKEN"])

    all_rows = []
    for a in artists:
        logging.info(f"[Spotify] Top tracks for: {a}")
        rows = fetch_top_tracks(sp, a, market=market, limit=limit_per_artist)
        all_rows += rows
        time.sleep(0.3)
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["track_id"])
    if df.empty:
        raise RuntimeError("No tracks found. Check artist names and credentials.")
    logging.info(f"Collected {df.shape[0]} unique tracks across {len(artists)} artists.")

    df = add_audio_features(sp, df)
    lyr = fetch_lyrics_for_tracks(g, df)
    if lyr.empty:
        logging.warning("No lyrics found. Proceeding without sentiment.")
        curated = df.copy()
    else:
        curated = compute_sentiment(lyr)

    # Minimal columns expected by the app
    curated["artist"] = curated.get("artist", curated.get("artist_name", ""))
    curated["year"] = pd.to_datetime(curated.get("release_date"), errors="coerce").dt.year
    # Short snippet for display safety
    if "lyrics" in curated.columns:
        curated["lyrics_snippet"] = curated["lyrics"].fillna("").str.slice(0, 200)

    out = CURATED / "songs_curated.parquet"
    curated.to_parquet(out, index=False)
    logging.info(f"Saved curated parquet -> {out.resolve()}")
    return out

# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FlowTrendz — Launch bootstrap")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run pipeline and write curated Parquet")
    r.add_argument("--market", default="US")
    r.add_argument("--limit", type=int, default=10, help="Top tracks per artist")
    r.add_argument("--artists", nargs="+", help="Override artist list (space‑separated)")
    r.set_defaults(func=_do_run)

    return p.parse_args()

DEFAULT_ARTISTS = [
    "Jay-Z","Nas","Kanye West","Drake","Kendrick Lamar",
    "Nicki Minaj","J. Cole","Lil Wayne","Outkast","Travis Scott",
    "A Tribe Called Quest","The Notorious B.I.G.","2Pac","Future","Doja Cat"
]

def _do_run(args: argparse.Namespace):
    artists = args.artists or DEFAULT_ARTISTS
    out = run_pipeline(artists=artists, market=args.market, limit_per_artist=args.limit)
    print("\\n=== FlowTrendz (Launch) — Quick Summary ===")
    try:
        df = pd.read_parquet(out)
        print(f"Rows: {df.shape[0]:,} | Artists: {df['artist'].nunique() if 'artist' in df else '—'}")
        if 'compound' in df:
            print(f"Compound mean: {df['compound'].replace([np.inf,-np.inf], np.nan).dropna().mean():.3f}")
    except Exception:
        pass
    print(f"Curated file: {out.resolve()}")

if __name__ == "__main__":
    args = parse_args()
    args.func(args)
