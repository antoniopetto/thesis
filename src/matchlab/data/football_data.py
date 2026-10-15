"""Fonte: football-data.co.uk (risultati, tiri, corner, falli, cartellini, quote).

Una riga per partita. Per i campionati italiani la colonna `Referee` non è presente:
gli arbitri si possono aggiungere con `data/manual/referees.csv` (vedi manual.py).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..config import League, season_code
from .cache import HttpCache

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Colonne football-data -> nomi interni (formato "partita").
COLUMN_MAP = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HTHG": "home_ht_goals",
    "HTAG": "away_ht_goals",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    # Quote medie di mercato (pre-partita e chiusura), utili come riferimento.
    "AvgH": "odds_home",
    "AvgD": "odds_draw",
    "AvgA": "odds_away",
    "Avg>2.5": "odds_over25",
    "Avg<2.5": "odds_under25",
    "AvgCH": "close_odds_home",
    "AvgCD": "close_odds_draw",
    "AvgCA": "close_odds_away",
    "AvgC>2.5": "close_odds_over25",
    "AvgC<2.5": "close_odds_under25",
}


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV illeggibile")


def normalize_matches(raw: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    df = raw.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in raw.columns})
    keep = [c for c in COLUMN_MAP.values() if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["home", "away"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    for col in df.columns:
        if col not in ("date", "time", "home", "away", "referee"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    if "referee" in df.columns:
        df["referee"] = df["referee"].astype("string").str.strip()
    if season is not None:
        df["season"] = season
    return df.sort_values("date").reset_index(drop=True)


def load_season(league: League, season: int, cache: HttpCache, csv_path: Path | None = None) -> pd.DataFrame:
    """Partite giocate di una stagione (season = anno di inizio, es. 2026 per 2026/27)."""
    if csv_path is not None:
        raw = Path(csv_path).read_bytes()
    else:
        url = BASE_URL.format(season=season_code(season), div=league.football_data_code)
        raw = cache.get_bytes(url, suffix=".csv")
    df = normalize_matches(_read_csv_bytes(raw), season)
    return df.dropna(subset=["home_goals", "away_goals"])


def load_fixtures(league: League, cache: HttpCache, csv_path: Path | None = None) -> pd.DataFrame:
    """Prossime partite (con quote, quando disponibili)."""
    raw = Path(csv_path).read_bytes() if csv_path else cache.get_bytes(FIXTURES_URL, suffix=".csv")
    df = _read_csv_bytes(raw)
    if "Div" in df.columns:
        df = df[df["Div"].astype(str).str.strip() == league.football_data_code]
    return normalize_matches(df)
