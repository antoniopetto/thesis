"""Fonte: understat.com (xG, xGA, PPDA, deep completions, statistiche giocatori).

Understat espone i dati in due modi a seconda del periodo:
- endpoint AJAX `getLeagueData/<league>/<season>` (JSON con teams/players/dates);
- variabili `JSON.parse('...')` incorporate nella pagina HTML.
Proviamo il primo e ripieghiamo sul secondo.
"""
from __future__ import annotations

import codecs
import json
import re

import pandas as pd

from ..config import League
from .cache import FetchError, HttpCache

AJAX_URL = "https://understat.com/getLeagueData/{league}/{season}"
PAGE_URL = "https://understat.com/league/{league}/{season}"


def _extract_var(html: str, var: str):
    m = re.search(rf"var\s+{var}\s*=\s*JSON\.parse\('(.*?)'\)", html, re.S)
    if not m:
        return None
    text = codecs.decode(m.group(1), "unicode_escape")
    try:  # le sequenze \xNN sono byte UTF-8
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return json.loads(text)


def fetch_league(league: League, season: int, cache: HttpCache) -> dict:
    if not league.understat_code:
        raise FetchError(f"Understat non copre {league.name}")
    try:
        raw = cache.get_bytes(
            AJAX_URL.format(league=league.understat_code, season=season),
            suffix=".json",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        data = json.loads(raw)
        if isinstance(data, dict) and "teams" in data:
            return data
    except (FetchError, ValueError):
        pass
    html = cache.get_bytes(PAGE_URL.format(league=league.understat_code, season=season), suffix=".html")
    html = html.decode("utf-8", errors="replace")
    data = {"teams": _extract_var(html, "teamsData"), "players": _extract_var(html, "playersData")}
    if not data["teams"]:
        raise FetchError("Formato Understat non riconosciuto")
    return data


def team_matches(data: dict) -> pd.DataFrame:
    """Una riga per squadra per partita con le metriche avanzate."""
    rows = []
    for team in (data.get("teams") or {}).values():
        for h in team.get("history", []):
            ppda = h.get("ppda") or {}
            ppda_a = h.get("ppda_allowed") or {}
            rows.append(
                {
                    "team_us": team["title"],
                    "date": pd.to_datetime(h["date"]).normalize(),
                    "venue": "H" if h.get("h_a") == "h" else "A",
                    "xg_for": float(h["xG"]),
                    "xg_against": float(h["xGA"]),
                    "npxg_for": float(h.get("npxG", h["xG"])),
                    "npxg_against": float(h.get("npxGA", h["xGA"])),
                    "ppda": _ratio(ppda),
                    "ppda_allowed": _ratio(ppda_a),
                    "deep_for": float(h.get("deep", 0)),
                    "deep_against": float(h.get("deep_allowed", 0)),
                    "xpts": float(h.get("xpts", 0)),
                }
            )
    return pd.DataFrame(rows)


def _ratio(d: dict) -> float | None:
    try:
        att, dfn = float(d["att"]), float(d["def"])
        return att / dfn if dfn else None
    except (KeyError, TypeError, ValueError):
        return None


PLAYER_COLUMNS = {
    "player_name": "player",
    "team_title": "team",
    "position": "position",
    "games": "games",
    "time": "minutes",
    "goals": "goals",
    "npg": "np_goals",
    "xG": "xg",
    "npxG": "npxg",
    "assists": "assists",
    "xA": "xa",
    "shots": "shots",
    "key_passes": "key_passes",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "xGChain": "xg_chain",
    "xGBuildup": "xg_buildup",
}


def players(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data.get("players") or [])
    if df.empty:
        return df
    df = df.rename(columns=PLAYER_COLUMNS)
    df = df[[c for c in PLAYER_COLUMNS.values() if c in df.columns]].copy()
    for col in df.columns:
        if col not in ("player", "team", "position"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Giocatori trasferiti a metà stagione: Understat scrive "Squadra A,Squadra B"; teniamo l'ultima.
    df["team"] = df["team"].astype(str).str.split(",").str[-1].str.strip()
    return df
