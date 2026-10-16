"""Agente Contesto avversario: come cambia ogni squadra in base al tipo di avversario.

Usa le partite della stagione in corso + precedente. Per ogni categoria di avversario
(big, media, piccole, chiuse, di possesso, che pressano) confronta la squadra con sé
stessa nelle altre partite, al netto dell'effetto "normale" di quella categoria su tutto
il campionato (contro le big tutti segnano meno: si segnala solo chi cambia molto più,
o molto meno, degli altri).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..stats.context import OpponentContext
from .base import Agent, AgentOutput, Context, Insight


@dataclass(frozen=True)
class SplitMetric:
    key: str
    label: str
    column: str | None  # media della colonna; None = calcolata
    fmt: str = "{:.2f}"
    rate: bool = False
    higher_is_better: bool | None = None


SPLIT_METRICS = [
    SplitMetric("points", "Punti/partita", "points"),
    SplitMetric("goals_for", "Gol fatti", "goals_for"),
    SplitMetric("goals_against", "Gol subiti", "goals_against"),
    SplitMetric("xg_for", "xG", "xg_for"),
    SplitMetric("xg_against", "xG concessi", "xg_against"),
    SplitMetric("possession", "Possesso %", "possession_for", fmt="{:.0f}"),
    SplitMetric("shots_for", "Tiri", "shots_for", fmt="{:.1f}"),
    SplitMetric("corners_for", "Corner fatti", "corners_for", fmt="{:.1f}"),
    SplitMetric("corners_against", "Corner subiti", "corners_against", fmt="{:.1f}"),
    SplitMetric("cards_for", "Cartellini", "cards_for"),
    SplitMetric("over25", "Over 2.5", "_over25", rate=True),
    SplitMetric("btts", "Gol/NoGol: Gol", "_btts", rate=True),
]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_over25"] = (df["goals_total"] > 2.5).astype(float)
    df["_btts"] = ((df["goals_for"] > 0) & (df["goals_against"] > 0)).astype(float)
    return df


def _fmt(m: SplitMetric, v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v * 100:.0f}%" if m.rate else m.fmt.format(v)


def category_mask(df: pd.DataFrame, cat) -> pd.Series:
    col = df[cat.column]
    return (col == cat.value) if cat.column == "opp_tier" else col.astype(bool)


def split_table(oc: OpponentContext, team: str) -> pd.DataFrame:
    df = _prep(oc.matches[oc.matches["team"] == team])
    metrics = [m for m in SPLIT_METRICS if m.column in df.columns and df[m.column].notna().any()]
    rows = []
    for label, mask in [("Tutte le partite", pd.Series(True, index=df.index))] + \
                       [(c.label, category_mask(df, c)) for c in oc.categories]:
        sub = df[mask]
        row = {"Avversario": label, "Partite": len(sub)}
        for m in metrics:
            row[m.label] = _fmt(m, sub[m.column].mean()) if len(sub) else "–"
        rows.append(row)
    return pd.DataFrame(rows)


class OpponentAgent(Agent):
    name = "Contesto avversario"
    role = ("Come cambia ogni squadra contro big, medie, piccole, squadre chiuse, di possesso o che pressano "
            "(stagione in corso + precedente).")

    MIN_N = 4
    T_THRESHOLD = 2.0

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        oc: OpponentContext | None = ctx.opponents
        if oc is None or oc.matches.empty:
            out.notes.append("Contesto avversario non disponibile.")
            return out
        df = _prep(oc.matches)
        if "possession_for" not in df.columns or df["possession_for"].isna().all():
            out.notes.append("Possesso palla non disponibile (SofaScore): categorie 'chiuse/di possesso' disattivate.")
        metrics = [m for m in SPLIT_METRICS if m.column in df.columns and df[m.column].notna().any()]
        out.tables["per_team"] = {t: split_table(oc, t) for t in ctx.data.teams}
        out.tables["classes"] = oc.classes

        # Effetto medio di ogni categoria su tutto il campionato (rapporto categoria / resto).
        league_ratio: dict[tuple[str, str], float] = {}
        for cat in oc.categories:
            mask = category_mask(df, cat)
            for m in metrics:
                a, b = df.loc[mask, m.column].mean(), df.loc[~mask, m.column].mean()
                league_ratio[(cat.key, m.key)] = a / b if b else 1.0

        for team, g in df.groupby("team"):
            for cat in oc.categories:
                mask = category_mask(g, cat)
                inside, outside = g[mask], g[~mask]
                if len(inside) < self.MIN_N or len(outside) < self.MIN_N:
                    continue
                for m in metrics:
                    x, y = inside[m.column].dropna(), outside[m.column].dropna()
                    if len(x) < self.MIN_N or len(y) < self.MIN_N:
                        continue
                    ratio = league_ratio[(cat.key, m.key)]
                    expected = y.mean() * ratio
                    se = np.sqrt(x.var(ddof=1) / len(x) + ratio**2 * y.var(ddof=1) / len(y))
                    if not se or np.isnan(se):
                        continue
                    t = (x.mean() - expected) / se
                    rel = (x.mean() - expected) / expected if expected else 0
                    big_enough = abs(x.mean() - expected) >= 0.15 if m.rate else abs(rel) >= 0.2
                    if abs(t) < self.T_THRESHOLD or not big_enough:
                        continue
                    more = "più" if x.mean() > expected else "meno"
                    text = (f"{cat.label}: {m.label.lower()} {_fmt(m, x.mean())} contro {_fmt(m, y.mean())} nelle altre "
                            f"partite ({len(x)} gare). Ci si aspetterebbe {_fmt(m, expected)} visto come cambiano "
                            f"le altre squadre: qui ne fa {more} del previsto.")
                    out.insights.append(Insight(
                        self.name, "contesto", "tendenza", text, team=team, metric=f"{cat.key}:{m.key}",
                        value=float(x.mean()), league=float(expected), n=len(x), score=min(abs(t), 5) * 0.8,
                    ))
        return out


def matchup_lines(oc: OpponentContext, team: str, opponent: str) -> list[dict]:
    """Per l'anteprima: come gioca `team` contro squadre del tipo di `opponent`."""
    if oc is None:
        return []
    df = _prep(oc.matches[oc.matches["team"] == team])
    keys = [("goals_for", "GF"), ("goals_against", "GS"), ("possession", "possesso"),
            ("corners_for", "corner"), ("cards_for", "cartellini"), ("over25", "Over 2.5")]
    by_key = {m.key: m for m in SPLIT_METRICS}
    lines = []
    for cat in oc.categories_of(opponent):
        sub = df[category_mask(df, cat)]
        if sub.empty:
            continue
        parts = []
        for key, short in keys:
            m = by_key[key]
            if m.column in sub.columns and sub[m.column].notna().any():
                parts.append(f"{short} {_fmt(m, sub[m.column].mean())}")
        ppg = sub["points"].mean()
        lines.append({"label": cat.label.replace("Contro", "contro"), "n": len(sub),
                      "text": f"{ppg:.2f} punti/partita, " + ", ".join(parts)})
    return lines
