"""Assemblaggio del dataset di campionato a partire dalle varie fonti.

Il formato centrale è `team_matches`: UNA RIGA PER SQUADRA PER PARTITA, con colonne
"_for" (fatti dalla squadra) e "_against" (subiti). Tutti gli agenti lavorano su questo.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DEFAULT_CACHE_DIR, DEFAULT_MANUAL_DIR, League
from . import football_data, manual, sofascore, understat
from .cache import FetchError, HttpCache
from .names import canonical, match_players

log = logging.getLogger(__name__)

# (colonna home, colonna away, nome base): genera <base>_for / <base>_against
PAIRED = [
    ("home_goals", "away_goals", "goals"),
    ("home_ht_goals", "away_ht_goals", "ht_goals"),
    ("home_shots", "away_shots", "shots"),
    ("home_sot", "away_sot", "sot"),
    ("home_fouls", "away_fouls", "fouls"),
    ("home_corners", "away_corners", "corners"),
    ("home_yellows", "away_yellows", "yellows"),
    ("home_reds", "away_reds", "reds"),
]


@dataclass
class LeagueData:
    league: League
    season: int
    matches: pd.DataFrame  # una riga per partita
    team_matches: pd.DataFrame  # una riga per squadra per partita
    players: pd.DataFrame = field(default_factory=pd.DataFrame)
    fixtures: pd.DataFrame = field(default_factory=pd.DataFrame)
    coaches: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    referee_career: pd.DataFrame = field(default_factory=pd.DataFrame)
    coach_info: dict = field(default_factory=dict)  # squadra -> scheda allenatore SofaScore  # statistiche carriera (SofaScore)
    previous: "LeagueData | None" = None  # stagione precedente (stesso campionato), se disponibile

    @property
    def teams(self) -> list[str]:
        return sorted(self.team_matches["team"].unique())

    def has(self, column: str) -> bool:
        return column in self.team_matches.columns and self.team_matches[column].notna().any()


def to_team_matches(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.reset_index(drop=True).copy()
    matches["match_id"] = np.arange(len(matches))
    sides = []
    for venue, me, opp in (("H", "home", "away"), ("A", "away", "home")):
        d = pd.DataFrame(
            {
                "match_id": matches["match_id"],
                "date": matches["date"].dt.normalize(),
                "season": matches.get("season"),
                "team": matches[me],
                "opponent": matches[opp],
                "venue": venue,
            }
        )
        for h, a, base in PAIRED:
            mine, theirs = (h, a) if venue == "H" else (a, h)
            if mine in matches.columns:
                d[f"{base}_for"] = matches[mine].values
                d[f"{base}_against"] = matches[theirs].values
        # Altre coppie home_X / away_X (statistiche di gioco SofaScore, allenatore, modulo).
        done = {c for h, a, _ in PAIRED for c in (h, a)}
        for col in matches.columns:
            if not col.startswith("home_") or col in done:
                continue
            base = col[len("home_"):]
            other = f"away_{base}"
            if other not in matches.columns or other in done:
                continue
            mine, theirs = (col, other) if venue == "H" else (other, col)
            if pd.api.types.is_numeric_dtype(matches[col]):
                d[f"{base}_for"] = matches[mine].values
                d[f"{base}_against"] = matches[theirs].values
            else:
                d[base] = matches[mine].values
                d[f"opp_{base}"] = matches[theirs].values
        if "referee" in matches.columns:
            d["referee"] = matches["referee"].values
        # Probabilità implicita di vittoria dalle quote (senza margine), se presenti.
        if {"odds_home", "odds_draw", "odds_away"} <= set(matches.columns):
            inv = 1 / matches[["odds_home", "odds_draw", "odds_away"]]
            norm = inv.div(inv.sum(axis=1), axis=0)
            d["mkt_win_prob"] = (norm["odds_home"] if venue == "H" else norm["odds_away"]).values
            d["mkt_draw_prob"] = norm["odds_draw"].values
        sides.append(d)
    tm = pd.concat(sides, ignore_index=True)
    tm["result"] = np.select(
        [tm["goals_for"] > tm["goals_against"], tm["goals_for"] == tm["goals_against"]], ["W", "D"], "L"
    )
    tm["points"] = tm["result"].map({"W": 3, "D": 1, "L": 0})
    if "yellows_for" in tm.columns:
        # Conteggio cartellini "da scommessa": giallo = 1, rosso = 1.
        tm["cards_for"] = tm["yellows_for"].fillna(0) + tm.get("reds_for", 0).fillna(0)
        tm["cards_against"] = tm["yellows_against"].fillna(0) + tm.get("reds_against", 0).fillna(0)
    if "ht_goals_for" in tm.columns:
        tm["sh_goals_for"] = tm["goals_for"] - tm["ht_goals_for"]
        tm["sh_goals_against"] = tm["goals_against"] - tm["ht_goals_against"]
    # Totali di partita (utili per i mercati Over/Under).
    for base in ("goals", "corners", "cards", "shots", "fouls", "ht_goals"):
        if f"{base}_for" in tm.columns:
            tm[f"{base}_total"] = tm[f"{base}_for"] + tm[f"{base}_against"]
    return tm.sort_values(["date", "match_id", "venue"]).reset_index(drop=True)


def merge_understat(tm: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    if us.empty:
        return tm
    known = sorted(tm["team"].unique())
    us = us.copy()
    us["team"] = us["team_us"].map(lambda n: canonical(n, known))
    us = us.drop(columns=["team_us", "venue"])
    us["date"] = us["date"].astype("datetime64[ns]")
    us = us.sort_values("date")
    left = tm.assign(date=tm["date"].astype("datetime64[ns]")).sort_values("date")
    merged = pd.merge_asof(
        left, us, on="date", by="team", tolerance=pd.Timedelta(days=1), direction="nearest"
    )
    return merged.sort_values(["date", "match_id", "venue"]).reset_index(drop=True)


def merge_referees(matches: pd.DataFrame, refs: pd.DataFrame | None, by_date: bool = True) -> pd.DataFrame:
    """Aggiunge la colonna `referee` (senza sovrascrivere quelle già presenti).

    by_date=False abbina solo per casa/ospite: va bene quando `refs` copre una sola
    stagione, in cui ogni accoppiamento casa-ospite capita una volta sola.
    """
    if refs is None or refs.empty or matches.empty:
        return matches
    known = sorted(set(matches["home"]) | set(matches["away"]))
    keys = ["home", "away"] + (["_d"] if by_date else [])
    refs = refs.assign(
        home=refs["home"].map(lambda n: canonical(n, known)),
        away=refs["away"].map(lambda n: canonical(n, known)),
        _d=pd.to_datetime(refs["date"]).dt.normalize(),
    )[keys + ["referee"]].dropna(subset=["referee"]).drop_duplicates(keys, keep="last")
    refs = refs.rename(columns={"referee": "_ref"})
    out = matches.assign(_d=matches["date"].dt.normalize()).merge(refs, on=keys, how="left")
    out["referee"] = out["referee"].fillna(out["_ref"]) if "referee" in out.columns else out["_ref"]
    return out.drop(columns=["_d", "_ref"])


def merge_sofascore_players(players: pd.DataFrame, sofa: pd.DataFrame, known_teams: list[str]) -> pd.DataFrame:
    """Aggiunge le statistiche SofaScore ai giocatori Understat (abbinando i nomi squadra per squadra).

    I giocatori presenti solo su SofaScore vengono aggiunti con i minuti SofaScore.
    """
    if sofa.empty:
        return players
    sofa = sofa.copy()
    sofa["team"] = sofa["team"].map(lambda n: canonical(n, known_teams))
    if players.empty:
        return sofa.rename(columns={"minutes_sofa": "minutes"})
    extra_cols = [c for c in sofa.columns if c not in ("player", "team")]
    parts = []
    for team, g in players.groupby("team"):
        s = sofa[sofa["team"] == team]
        mapping = match_players(
            g["player"].tolist(), s["player"].tolist(),
            left_minutes=dict(zip(g["player"], g["minutes"])), right_minutes=dict(zip(s["player"], s["minutes_sofa"])),
        )
        g = g.assign(_sofa=g["player"].map(mapping))
        merged = g.merge(s[["player"] + extra_cols].rename(columns={"player": "_sofa"}), on="_sofa", how="left")
        only_sofa = s[~s["player"].isin(mapping.values())].rename(columns={"minutes_sofa": "minutes"})
        parts += [merged.drop(columns=["_sofa"]), only_sofa]
    out = pd.concat(parts, ignore_index=True)
    out["minutes"] = out["minutes"].fillna(out.get("minutes_sofa"))
    return out


def merge_event_columns(matches: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge alle partite le colonne home_*/away_* degli eventi SofaScore (abbinando casa/ospite)."""
    cols = [c for c in ev.columns if (c.startswith("home_") or c.startswith("away_")) and c not in matches.columns]
    if ev.empty or not cols:
        return matches
    known = sorted(set(matches["home"]) | set(matches["away"]))
    ev = ev.assign(
        home=ev["home"].map(lambda n: canonical(n, known)), away=ev["away"].map(lambda n: canonical(n, known)),
    )
    ev = ev[ev["status"] == "finished"].drop_duplicates(["home", "away"], keep="last")[["home", "away"] + cols]
    out = matches.merge(ev, on=["home", "away"], how="left")
    for c in cols:
        if c.split("_", 1)[1].startswith(("coach", "formation")):
            out[c] = out[c].astype("object").where(out[c].notna(), None).map(lambda v: v if v is None else str(v))
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def merge_player_extras(players: pd.DataFrame, extras: pd.DataFrame | None, known_teams: list[str]) -> pd.DataFrame:
    if extras is None:
        return players
    extras = extras.copy()
    if "team" in extras.columns:
        extras["team"] = extras["team"].astype(str).map(lambda n: canonical(n, known_teams))
    if players.empty:
        return extras
    extra_cols = [c for c in extras.columns if c not in ("player", "team") and c not in players.columns]
    key = ["player", "team"] if "team" in extras.columns else ["player"]
    return players.merge(extras[key + extra_cols], on=key, how="left")


def build(
    league: League,
    season: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    manual_dir: Path = DEFAULT_MANUAL_DIR,
    offline: bool = False,
    matches_csv: Path | None = None,
    fixtures_csv: Path | None = None,
    use_understat: bool = True,
    use_sofascore: bool = True,
    with_previous: bool = True,
    prev_matches_csv: Path | None = None,
    _is_previous: bool = False,
) -> LeagueData:
    cache = HttpCache(cache_dir, offline=offline)
    warnings: list[str] = []
    sources = ["football-data.co.uk"]

    matches = football_data.load_season(league, season, cache, csv_path=matches_csv)
    referees = manual.load_referees(manual_dir)
    matches = merge_referees(matches, referees)
    sofa_refs = pd.DataFrame()
    if use_sofascore:
        try:
            sofa_refs = sofascore.events(league, season, cache, upcoming=not _is_previous)
            matches = merge_referees(matches, sofa_refs, by_date=False)
            matches = merge_event_columns(matches, sofa_refs)
            sources.append("sofascore.com (arbitri)")
        except (FetchError, ValueError, KeyError) as exc:
            warnings.append(f"SofaScore non disponibile ({exc}): arbitri solo da data/manual/referees.csv.")
    tm = to_team_matches(matches)
    known = sorted(tm["team"].unique())

    players = pd.DataFrame()
    if use_understat and league.understat_code:
        try:
            raw = understat.fetch_league(league, season, cache)
            tm = merge_understat(tm, understat.team_matches(raw))
            players = understat.players(raw)
            if not players.empty:
                players["team"] = players["team"].map(lambda n: canonical(n, known))
            sources.append("understat.com")
        except FetchError as exc:
            warnings.append(f"Understat non disponibile ({exc}): niente xG/PPDA/giocatori.")
    if use_sofascore and not _is_previous:
        try:
            sofa_players = sofascore.player_stats(league, season, cache)
            players = merge_sofascore_players(players, sofa_players, known)
            sources.append("sofascore.com (giocatori)")
        except (FetchError, ValueError, KeyError) as exc:
            warnings.append(f"Statistiche giocatori SofaScore non disponibili ({exc}).")
    extras = None if _is_previous else manual.load_player_stats(manual_dir)
    if extras is not None:
        sources.append("data/manual/player_stats.csv")
    players = merge_player_extras(players, extras, known)

    fixtures = pd.DataFrame()
    if _is_previous:
        return LeagueData(league, season, matches, tm, players, sources=sources, warnings=warnings)
    try:
        fixtures = football_data.load_fixtures(league, cache, csv_path=fixtures_csv)
        last_played = matches["date"].max()
        fixtures = fixtures[fixtures["date"] >= last_played.normalize()]
        fixtures = fixtures.assign(
            home=fixtures["home"].map(lambda n: canonical(n, known)),
            away=fixtures["away"].map(lambda n: canonical(n, known)),
        )
        fixtures = merge_referees(fixtures, referees)
        fixtures = merge_referees(fixtures, sofa_refs, by_date=False)
    except (FetchError, ValueError, KeyError) as exc:
        warnings.append(f"Calendario prossime partite non disponibile ({exc}).")
    if fixtures.empty and not sofa_refs.empty:
        # Prossima giornata non ancora su football-data: calendario (senza quote) da SofaScore.
        nxt = sofa_refs[sofa_refs["status"] == "notstarted"].copy()
        if not nxt.empty:
            nxt["home"] = nxt["home"].map(lambda n: canonical(n, known))
            nxt["away"] = nxt["away"].map(lambda n: canonical(n, known))
            fixtures = nxt[["date", "home", "away", "referee"]].sort_values("date").reset_index(drop=True)
    if not fixtures.empty:
        # Solo la prossima giornata (le successive non hanno ancora arbitro né quote affidabili).
        fixtures = fixtures[fixtures["date"] <= fixtures["date"].min() + pd.Timedelta(days=5)].reset_index(drop=True)
    if not fixtures.empty and not sofa_refs.empty:
        ids = sofa_refs.assign(
            home=sofa_refs["home"].map(lambda n: canonical(n, known)),
            away=sofa_refs["away"].map(lambda n: canonical(n, known)),
        ).drop_duplicates(["home", "away"], keep="last")[["home", "away", "event_id"]]
        fixtures = fixtures.drop(columns=["event_id"], errors="ignore").merge(ids, on=["home", "away"], how="left")
        fixtures["book_odds"] = [
            sofascore.odds(cache, int(e)) if pd.notna(e) else {} for e in fixtures["event_id"]
        ]
        fixtures["missing"] = [
            sofascore.missing_players(cache, int(e)) if pd.notna(e) else {} for e in fixtures["event_id"]
        ]

    coaches = manual.load_coaches(manual_dir)
    coaches = {canonical(t, known): c for t, c in coaches.items()}

    previous = None
    if with_previous:
        try:
            previous = build(
                league, season - 1, cache_dir=cache_dir, manual_dir=manual_dir, offline=offline,
                matches_csv=prev_matches_csv, use_understat=use_understat, use_sofascore=use_sofascore,
                _is_previous=True,
            )
            if previous.matches.empty:
                previous = None
        except (FetchError, ValueError, KeyError) as exc:
            warnings.append(f"Stagione precedente non disponibile ({exc}): niente confronto né prior.")

    career = pd.DataFrame()
    if not sofa_refs.empty:
        career = (
            sofa_refs.dropna(subset=["referee"]).drop_duplicates("referee", keep="last")
            .set_index("referee")[["ref_career_games", "ref_career_yellows", "ref_career_reds"]]
        )

    coach_info = {}
    if use_sofascore and "home_coach_id" in matches.columns:
        last = tm.sort_values("date").dropna(subset=["coach_id"]).groupby("team").tail(1)
        for r in last.itertuples():
            try:
                coach_info[r.team] = sofascore.coach_profile(cache, int(float(r.coach_id)))
            except (ValueError, TypeError):
                pass

    return LeagueData(
        coach_info=coach_info,
        referee_career=career,
        previous=previous,
        league=league,
        season=season,
        matches=matches,
        team_matches=tm,
        players=players,
        fixtures=fixtures,
        coaches=coaches,
        sources=sources,
        warnings=warnings,
    )
