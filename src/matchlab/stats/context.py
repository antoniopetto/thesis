"""Contesto avversario: classifica ogni squadra per forza e stile e annota ogni partita
con il "tipo" di avversario affrontato.

Forza (rating = differenza xG a partita, o differenza reti se mancano gli xG):
    big = primo 30% (6 squadre su 20), piccola = ultimo 35% (7 su 20), media = le altre.
Stile (dalle medie di stagione):
    chiusa = possesso nel terzo più basso del campionato
    possesso = possesso nel terzo più alto
    pressing = PPDA nel terzo più basso (pressa di più)

Per la stagione in corso, con poche giornate, i valori sono mescolati con quelli della
stagione scorsa (shrinkage); le neopromosse usano solo i dati di quest'anno.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIER_LABELS = {"big": "big", "media": "media classifica", "piccola": "piccole"}
BIG_SHARE, SMALL_SHARE = 0.30, 0.35


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    column: str
    value: object = True


CATEGORIES = [
    Category("big", "Contro le big", "opp_tier", "big"),
    Category("media", "Contro squadre di media classifica", "opp_tier", "media"),
    Category("piccola", "Contro le piccole", "opp_tier", "piccola"),
    Category("chiusa", "Contro squadre chiuse (poco possesso)", "opp_chiusa"),
    Category("possesso", "Contro squadre di possesso", "opp_possesso"),
    Category("pressing", "Contro squadre che pressano alto", "opp_pressing"),
]


def _season_means(tm: pd.DataFrame) -> pd.DataFrame:
    g = tm.groupby("team")
    out = pd.DataFrame({"n": g.size()})
    if "xg_for" in tm.columns and tm["xg_for"].notna().any():
        out["rating"] = g["xg_for"].mean() - g["xg_against"].mean()
    else:
        out["rating"] = g["goals_for"].mean() - g["goals_against"].mean()
    out["possession"] = g["possession_for"].mean() if "possession_for" in tm.columns else np.nan
    out["ppda"] = g["ppda"].mean() if "ppda" in tm.columns else np.nan
    return out


def classify(tm: pd.DataFrame, prev_tm: pd.DataFrame | None = None, k: float = 6.0) -> pd.DataFrame:
    """Una riga per squadra: rating, tier, possesso, ppda, flag di stile."""
    cur = _season_means(tm)
    if prev_tm is not None and not prev_tm.empty:
        prev = _season_means(prev_tm).reindex(cur.index)
        for c in ("rating", "possession", "ppda"):
            has_prev = prev[c].notna()
            blended = (cur["n"] * cur[c] + k * prev[c]) / (cur["n"] + k)
            cur[c] = cur[c].where(~has_prev | cur[c].isna(), blended)
            cur[c] = cur[c].fillna(prev[c])
    rank = cur["rating"].rank(ascending=False, method="first")
    n = len(cur)
    big_n, small_n = round(n * BIG_SHARE), round(n * SMALL_SHARE)
    cur["tier"] = np.where(rank <= big_n, "big", np.where(rank > n - small_n, "piccola", "media"))
    for c, flag, low in (("possession", "chiusa", True), ("possession", "possesso", False), ("ppda", "pressing", True)):
        if cur[c].notna().sum() >= 6:
            pct = cur[c].rank(pct=True)
            cur[flag] = pct <= 1 / 3 if low else pct > 2 / 3
        else:
            cur[flag] = False
    return cur


def annotate(tm: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    tm = tm.copy()
    tm["opp_tier"] = tm["opponent"].map(classes["tier"])
    for flag in ("chiusa", "possesso", "pressing"):
        tm[f"opp_{flag}"] = tm["opponent"].map(classes[flag]).fillna(False).astype(bool)
    return tm


def team_tags(classes: pd.DataFrame, team: str) -> list[str]:
    if team not in classes.index:
        return []
    r = classes.loc[team]
    tags = [{"big": "big", "media": "squadra di media classifica", "piccola": "piccola"}[r["tier"]]]
    if r["chiusa"]:
        tags.append("chiusa (poco possesso)")
    if r["possesso"]:
        tags.append("di possesso")
    if r["pressing"]:
        tags.append("pressing alto")
    return tags


@dataclass
class OpponentContext:
    classes: pd.DataFrame  # classificazione attuale (stagione in corso + prior)
    matches: pd.DataFrame  # partite delle due stagioni annotate con il tipo di avversario
    categories: list[Category] = field(default_factory=lambda: list(CATEGORIES))

    def categories_of(self, team: str) -> list[Category]:
        """Le categorie in cui rientra `team` come avversario (es. Bologna -> media, pressing)."""
        if team not in self.classes.index:
            return []
        r = self.classes.loc[team]
        out = []
        for c in self.categories:
            if c.column == "opp_tier" and r["tier"] == c.value:
                out.append(c)
            elif c.column != "opp_tier" and bool(r[c.column.removeprefix("opp_")]):
                out.append(c)
        return out


def build(tm: pd.DataFrame, prev_tm: pd.DataFrame | None) -> OpponentContext:
    classes = classify(tm, prev_tm)
    frames = [annotate(tm, classes).assign(stagione="corrente")]
    if prev_tm is not None and not prev_tm.empty:
        frames.append(annotate(prev_tm, classify(prev_tm)).assign(stagione="precedente"))
    return OpponentContext(classes=classes, matches=pd.concat(frames, ignore_index=True))
