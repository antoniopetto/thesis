"""Registro delle metriche di squadra e calcolo dei profili (con shrinkage verso la media)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metric:
    key: str
    label: str  # etichetta in italiano per il report
    category: str  # attacco, difesa, corner, disciplina, stile, risultati
    column: str | None = None  # media per partita di questa colonna...
    num: str | None = None  # ...oppure rapporto somma(num)/somma(den)
    den: str | None = None
    # True = più alto è meglio, False = più basso è meglio, None = neutro (solo tendenza)
    higher_is_better: bool | None = True
    fmt: str = "{:.2f}"
    is_rate: bool = False  # percentuale (0-1)

    @property
    def required(self) -> list[str]:
        return [c for c in (self.column, self.num, self.den) if c]

    def format(self, v: float) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "–"
        return f"{v * 100:.0f}%" if self.is_rate else self.fmt.format(v)


TEAM_METRICS: list[Metric] = [
    # Risultati
    Metric("points", "Punti a partita", "risultati", column="points"),
    Metric("xpts", "Punti attesi (xPts) a partita", "risultati", column="xpts"),
    # Attacco
    Metric("goals_for", "Gol fatti a partita", "attacco", column="goals_for"),
    Metric("xg_for", "xG a partita", "attacco", column="xg_for"),
    Metric("shots_for", "Tiri a partita", "attacco", column="shots_for", fmt="{:.1f}"),
    Metric("sot_for", "Tiri in porta a partita", "attacco", column="sot_for", fmt="{:.1f}"),
    Metric("conversion", "Conversione (gol/tiri)", "attacco", num="goals_for", den="shots_for", is_rate=True),
    Metric("xg_per_shot", "Qualità media tiro (xG/tiro)", "attacco", num="xg_for", den="shots_for", fmt="{:.3f}"),
    Metric("deep_for", "Passaggi pericolosi vicino all'area (deep)", "attacco", column="deep_for", fmt="{:.1f}"),
    Metric("ht_goals_for", "Gol fatti nel 1° tempo", "attacco", column="ht_goals_for"),
    Metric("sh_goals_for", "Gol fatti nel 2° tempo", "attacco", column="sh_goals_for"),
    Metric("scored_rate", "Partite in cui segna", "attacco", column="_scored", is_rate=True),
    # Difesa
    Metric("goals_against", "Gol subiti a partita", "difesa", column="goals_against", higher_is_better=False),
    Metric("xg_against", "xG concessi a partita", "difesa", column="xg_against", higher_is_better=False),
    Metric("shots_against", "Tiri concessi a partita", "difesa", column="shots_against", higher_is_better=False, fmt="{:.1f}"),
    Metric("sot_against", "Tiri in porta concessi", "difesa", column="sot_against", higher_is_better=False, fmt="{:.1f}"),
    Metric("deep_against", "Deep concessi", "difesa", column="deep_against", higher_is_better=False, fmt="{:.1f}"),
    Metric("clean_sheet_rate", "Partite senza subire gol", "difesa", column="_clean_sheet", is_rate=True),
    Metric("ht_goals_against", "Gol subiti nel 1° tempo", "difesa", column="ht_goals_against", higher_is_better=False),
    Metric("sh_goals_against", "Gol subiti nel 2° tempo", "difesa", column="sh_goals_against", higher_is_better=False),
    # Corner
    Metric("corners_for", "Corner battuti a partita", "corner", column="corners_for", fmt="{:.1f}"),
    Metric("corners_against", "Corner concessi a partita", "corner", column="corners_against", higher_is_better=False, fmt="{:.1f}"),
    Metric("corners_total", "Corner totali nelle sue partite", "corner", column="corners_total", higher_is_better=None, fmt="{:.1f}"),
    # Disciplina
    Metric("fouls_for", "Falli commessi a partita", "disciplina", column="fouls_for", higher_is_better=None, fmt="{:.1f}"),
    Metric("fouls_against", "Falli subiti a partita", "disciplina", column="fouls_against", higher_is_better=None, fmt="{:.1f}"),
    Metric("cards_for", "Cartellini presi a partita", "disciplina", column="cards_for", higher_is_better=None),
    Metric("cards_against", "Cartellini fatti prendere agli avversari", "disciplina", column="cards_against", higher_is_better=None),
    Metric("cards_total", "Cartellini totali nelle sue partite", "disciplina", column="cards_total", higher_is_better=None),
    # Stile (allenatore)
    Metric("ppda", "PPDA (più basso = pressing più alto)", "stile", column="ppda", higher_is_better=None, fmt="{:.1f}"),
    Metric("ppda_allowed", "PPDA avversari (più basso = la squadra viene pressata)", "stile", column="ppda_allowed", higher_is_better=None, fmt="{:.1f}"),
    Metric("goals_total", "Gol totali nelle sue partite", "stile", column="goals_total", higher_is_better=None),
]

METRICS_BY_KEY = {m.key: m for m in TEAM_METRICS}
VENUES = {"all": "Totale", "H": "In casa", "A": "Fuori casa"}


def add_derived(tm: pd.DataFrame) -> pd.DataFrame:
    tm = tm.copy()
    tm["_scored"] = (tm["goals_for"] > 0).astype(float)
    tm["_clean_sheet"] = (tm["goals_against"] == 0).astype(float)
    return tm


def available_metrics(tm: pd.DataFrame) -> list[Metric]:
    return [m for m in TEAM_METRICS if all(c in tm.columns and tm[c].notna().any() for c in m.required)]


def _value(df: pd.DataFrame, m: Metric) -> tuple[float, float, float, int]:
    """(valore grezzo, somma numeratore, somma denominatore, n partite)."""
    if m.column:
        s = df[m.column].dropna()
        return (s.mean() if len(s) else np.nan), s.sum(), float(len(s)), len(s)
    sub = df[[m.num, m.den]].dropna()
    num, den = sub[m.num].sum(), sub[m.den].sum()
    return (num / den if den else np.nan), num, den, len(sub)


def profile_table(
    tm: pd.DataFrame,
    shrinkage: float = 4.0,
    form_window: int = 5,
    prior: pd.DataFrame | None = None,
    prior_weight: float = 0.6,
) -> pd.DataFrame:
    """Tabella lunga: team, venue, metric, n, value, shrunk, league, prev, z, rank, last_n.

    - value: media grezza della stagione.
    - shrunk: media "prudente" che, quando le partite sono poche, tira verso un punto di
      partenza: la media di lega, spostata verso il valore della stagione scorsa della
      squadra se disponibile (`prior`, di solito l'output di profile_table sulla stagione
      precedente). `prior_weight` < 1 perché tra una stagione e l'altra cambiano rose e
      allenatori: si eredita solo una parte dello scarto dalla media.
    - prev: valore della stagione scorsa (NaN per le neopromosse).
    - z: quante deviazioni standard la squadra è lontana dalle altre (sul valore shrunk).
    - rank: 1 = valore più alto del campionato.
    """
    tm = add_derived(tm)
    metrics = available_metrics(tm)
    prior_map: dict[tuple[str, str, str], tuple[float, float]] = {}
    if prior is not None and not prior.empty:
        for r in prior.itertuples():
            prior_map[(r.team, r.venue, r.metric)] = (r.value, r.league)
    rows = []
    for venue in VENUES:
        sub = tm if venue == "all" else tm[tm["venue"] == venue]
        for m in metrics:
            lg_val, lg_num, lg_den, lg_n = _value(sub, m)
            avg_den_per_match = lg_den / lg_n if lg_n else 0
            for team, g in sub.groupby("team"):
                val, num, den, n = _value(g, m)
                prev, prev_lg = prior_map.get((team, venue, m.key), (np.nan, np.nan))
                center = lg_val
                if not (np.isnan(prev) or np.isnan(prev_lg)):
                    center = lg_val + prior_weight * (prev - prev_lg)
                if m.column:
                    shrunk = (num + shrinkage * center) / (n + shrinkage) if n + shrinkage else np.nan
                else:
                    pseudo_den = shrinkage * avg_den_per_match
                    shrunk = (num + center * pseudo_den) / (den + pseudo_den) if den + pseudo_den else np.nan
                last = _value(g.sort_values("date").tail(form_window), m)[0]
                rows.append(
                    dict(team=team, venue=venue, metric=m.key, n=n, value=val, shrunk=shrunk, league=lg_val,
                         prev=prev, last_n=last)
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    grp = out.groupby(["venue", "metric"])["shrunk"]
    std = grp.transform("std")
    std = std.where(std > 1e-9 * (1 + grp.transform("mean").abs()))  # metrica costante: niente z
    out["z"] = (out["shrunk"] - grp.transform("mean")) / std
    out["rank"] = grp.rank(ascending=False, method="min")
    out["n_teams"] = grp.transform("count")
    return out


def display_rank(m: Metric, rank: float, n_teams: float) -> str:
    """Posizione leggibile: per metriche buone/cattive 1° = migliore, per quelle neutre 1° = più alto."""
    if m.higher_is_better is None:
        return f"{int(rank)}° più alto su {int(n_teams)}"
    pos = rank if m.higher_is_better else n_teams - rank + 1
    return f"{int(pos)}° migliore su {int(n_teams)}"


def lookup(table: pd.DataFrame, team: str, metric: str, venue: str = "all") -> pd.Series | None:
    row = table[(table.team == team) & (table.metric == metric) & (table.venue == venue)]
    return None if row.empty else row.iloc[0]
