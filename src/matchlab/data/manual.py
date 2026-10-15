"""Dati inseriti a mano o esportati da altre fonti (Opta, SofaScore, FBref, ...).

File letti da `data/manual/` (tutti opzionali):
- referees.csv:     date,home,away,referee         -> designazioni arbitrali
- coaches.csv:      team,coach[,since]             -> allenatore attuale
- player_stats.csv: player,team,minutes,<colonne>  -> statistiche giocatori extra
  (es. dribbles_completed, dribbles_attempted, tackles, fouls_committed, fouls_won, ...)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    return df if not df.empty else None


def load_referees(manual_dir: Path) -> pd.DataFrame | None:
    df = _read(Path(manual_dir) / "referees.csv")
    if df is None:
        return None
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.normalize()
    for c in ("home", "away", "referee"):
        df[c] = df[c].astype(str).str.strip()
    return df.dropna(subset=["date"])


def load_coaches(manual_dir: Path) -> dict[str, str]:
    df = _read(Path(manual_dir) / "coaches.csv")
    if df is None:
        return {}
    return {str(r.team).strip(): str(r.coach).strip() for r in df.itertuples()}


def load_player_stats(manual_dir: Path) -> pd.DataFrame | None:
    df = _read(Path(manual_dir) / "player_stats.csv")
    if df is None:
        return None
    for c in df.columns:
        if c not in ("player", "team", "position"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
