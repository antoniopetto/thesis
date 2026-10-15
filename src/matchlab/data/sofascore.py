"""Fonte: SofaScore (API pubblica non ufficiale) — arbitri delle partite giocate e designati.

Usata per gli arbitri perché football-data non li fornisce per la Serie A.
Le partite concluse si mettono in cache a lungo (non cambiano più); le prossime si
riscaricano spesso perché la designazione arriva pochi giorni prima.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from ..config import League
from .cache import FetchError, HttpCache

BASE = "https://api.sofascore.com/api/v1"
TOURNAMENT_IDS = {"serie_a": 23, "serie_b": 53, "premier_league": 17, "la_liga": 8, "bundesliga": 35, "ligue_1": 34}
LONG_CACHE_H = 24 * 60
POLITE_DELAY_S = 0.25


def _get(cache: HttpCache, path: str, max_age_hours: float | None = None) -> dict:
    url = f"{BASE}/{path}"
    fresh_before = cache._path(url, ".json").exists()
    data = json.loads(cache.get_bytes(url, suffix=".json", max_age_hours=max_age_hours))
    if not fresh_before:
        time.sleep(POLITE_DELAY_S)  # non martellare l'API
    return data


def season_id(cache: HttpCache, tournament: int, season: int) -> int:
    year = f"{season % 100:02d}/{(season + 1) % 100:02d}"
    for s in _get(cache, f"unique-tournament/{tournament}/seasons", max_age_hours=24)["seasons"]:
        if s.get("year") == year:
            return int(s["id"])
    raise FetchError(f"Stagione {year} non trovata su SofaScore")


def _events(cache: HttpCache, tournament: int, sid: int, direction: str, max_pages: int = 20) -> list[dict]:
    out = []
    for page in range(max_pages):
        try:
            d = _get(cache, f"unique-tournament/{tournament}/season/{sid}/events/{direction}/{page}", max_age_hours=6)
        except FetchError:
            break
        out += d.get("events", [])
        if not d.get("hasNextPage"):
            break
    return out


def _row(e: dict, detail: dict) -> dict:
    ref = detail.get("referee") or {}
    ts = pd.Timestamp(e["startTimestamp"], unit="s", tz="UTC").tz_convert("Europe/Rome")
    return {
        "event_id": e["id"],
        "date": ts.tz_localize(None).normalize(),
        "home": e["homeTeam"]["name"],
        "away": e["awayTeam"]["name"],
        "referee": ref.get("name"),
        "ref_career_games": ref.get("games"),
        "ref_career_yellows": ref.get("yellowCards"),
        "ref_career_reds": (ref.get("redCards") or 0) + (ref.get("yellowRedCards") or 0) if ref else None,
        "status": e.get("status", {}).get("type"),
    }


def referees(league: League, season: int, cache: HttpCache, upcoming: bool = True) -> pd.DataFrame:
    """Una riga per partita (giocate + prossime) con l'arbitro e le sue statistiche di carriera."""
    tournament = TOURNAMENT_IDS.get(league.key)
    if tournament is None:
        raise FetchError(f"SofaScore non configurato per {league.name}")
    sid = season_id(cache, tournament, season)
    rows = []
    for e in _events(cache, tournament, sid, "last"):
        if e.get("status", {}).get("type") != "finished":
            continue
        detail = _get(cache, f"event/{e['id']}", max_age_hours=LONG_CACHE_H).get("event", {})
        rows.append(_row(e, detail))
    if upcoming:
        for e in _events(cache, tournament, sid, "next", max_pages=1)[:20]:
            detail = _get(cache, f"event/{e['id']}", max_age_hours=3).get("event", {})
            rows.append(_row(e, detail))
    return pd.DataFrame(rows)


def _decimal(frac: str | None) -> float | None:
    try:
        num, den = frac.split("/")
        return round(float(num) / float(den) + 1, 3)
    except (AttributeError, ValueError, ZeroDivisionError):
        return None


def odds(cache: HttpCache, event_id: int) -> dict:
    """Quote pre-partita (un solo bookmaker, usato come riferimento di mercato).

    Ritorna {"1x2": {"1","X","2"}, "btts": {"Yes","No"}, "goals"/"cards"/"corners": {linea: {"Over","Under"}}}.
    """
    try:
        d = _get(cache, f"event/{event_id}/odds/1/all", max_age_hours=3)
    except FetchError:
        return {}
    out: dict = {}
    kinds = {"Match goals": "goals", "Cards in match": "cards", "Corners 2-Way": "corners"}
    for m in d.get("markets", []):
        choices = {c["name"]: _decimal(c.get("fractionalValue")) for c in m.get("choices", [])}
        name = m.get("marketName")
        if name == "Full time":
            out["1x2"] = choices
        elif name == "Both teams to score":
            out["btts"] = choices
        elif name in kinds and m.get("choiceGroup") is not None:
            try:
                line = float(m["choiceGroup"])
            except (TypeError, ValueError):
                continue
            out.setdefault(kinds[name], {})[line] = choices
    return out
