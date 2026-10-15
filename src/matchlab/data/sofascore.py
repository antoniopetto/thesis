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


# Statistiche giocatore SofaScore -> nomi interni (totali di stagione).
PLAYER_FIELDS = {
    "minutesPlayed": "minutes_sofa",
    "appearances": "appearances",
    "successfulDribbles": "dribbles_completed",
    "tackles": "tackles",
    "interceptions": "interceptions",
    "fouls": "fouls_committed",
    "wasFouled": "fouls_won",
    "shotsOnTarget": "shots_on_target",
    "accurateCrosses": "crosses",
    "bigChancesCreated": "big_chances_created",
    "clearances": "clearances",
    "saves": "saves",
    "rating": "rating",
}


def player_stats(league: League, season: int, cache: HttpCache) -> pd.DataFrame:
    """Statistiche di stagione di tutti i giocatori del campionato."""
    tournament = TOURNAMENT_IDS.get(league.key)
    if tournament is None:
        raise FetchError(f"SofaScore non configurato per {league.name}")
    sid = season_id(cache, tournament, season)
    fields = ",".join(PLAYER_FIELDS)
    rows, offset = [], 0
    for _ in range(20):
        d = _get(cache, f"unique-tournament/{tournament}/season/{sid}/statistics?limit=100&offset={offset}"
                        f"&order=-minutesPlayed&accumulation=total&fields={fields}", max_age_hours=12)
        for r in d.get("results", []):
            row = {"player": r["player"]["name"], "team": r["team"]["name"]}
            row.update({PLAYER_FIELDS[k]: r.get(k) for k in PLAYER_FIELDS})
            rows.append(row)
        if d.get("page", 1) >= d.get("pages", 1):
            break
        offset += 100
    df = pd.DataFrame(rows)
    for c in PLAYER_FIELDS.values():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


INJURY_IT = {
    "cruciate ligament": "legamento crociato", "meniscus": "menisco", "hamstring": "bicipite femorale",
    "muscle": "muscolare", "thigh": "coscia", "groin": "inguine", "knee": "ginocchio", "ankle": "caviglia",
    "calf": "polpaccio", "back": "schiena", "shoulder": "spalla", "foot": "piede", "hip": "anca",
    "achilles": "tendine d'Achille", "head": "testa", "concussion": "commozione cerebrale", "finger": "dito",
    "hand": "mano", "knock": "botta", "physical discomfort": "affaticamento", "illness": "malattia",
    "gastrointestinal": "problema gastrointestinale", "toe": "dito del piede", "rib": "costola",
}


def _reason(desc: str | None, kind: str | None) -> str:
    d = (desc or "").lower()
    if "suspen" in d or "card" in d:
        label = "squalifica"
    else:
        where = next((it for en, it in INJURY_IT.items() if en in d), None)
        broken = "broken" in d or "fracture" in d
        if where in ("affaticamento", "malattia", "problema gastrointestinale"):
            label = where
        elif where:
            label = f"{'frattura' if broken else 'infortunio'}: {where}"
        else:
            label = desc or "assente"
    return label + (" — in dubbio" if kind == "doubtful" else "")


def missing_players(cache: HttpCache, event_id: int) -> dict[str, list[dict]]:
    """Indisponibili (infortunati, squalificati, in dubbio) per casa e ospite."""
    try:
        d = _get(cache, f"event/{event_id}/lineups", max_age_hours=3)
    except FetchError:
        return {}
    out = {}
    for side in ("home", "away"):
        out[side] = [
            {"player": m["player"]["name"], "doubtful": m.get("type") == "doubtful",
             "reason": _reason(m.get("description"), m.get("type"))}
            for m in d.get(side, {}).get("missingPlayers", [])
        ]
    out["confirmed"] = bool(d.get("confirmed"))
    return out
