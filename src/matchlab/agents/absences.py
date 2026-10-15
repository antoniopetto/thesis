"""Assenze: peso offensivo dei giocatori indisponibili per la prossima partita.

Il modello statistico vede solo le medie di squadra; se manca chi produce gran parte
degli xG/xA, quelle medie sopravvalutano l'attacco. Qui si stima quanto "vale" ogni
assente (quota di xG + xA della squadra) e si riduce il valore atteso dei gol di
conseguenza, assumendo che il sostituto recuperi in parte il contributo.
"""
from __future__ import annotations

import pandas as pd

from ..data.names import match_players

# Quota del contributo di un assente che il sostituto NON recupera.
REPLACEMENT_LOSS = 0.5
MAX_REDUCTION = 0.25  # al massimo -25% dei gol attesi
DOUBTFUL_WEIGHT = 0.5


def attacking_shares(players: pd.DataFrame, team: str) -> pd.DataFrame:
    """Per ogni giocatore della squadra: quota di xG+xA e quota di minuti."""
    g = players[players["team"] == team].copy()
    if g.empty or not {"xg", "xa"} <= set(g.columns):
        return pd.DataFrame(columns=["player", "share", "minutes_share"])
    g["contrib"] = g["npxg"].fillna(g["xg"]).fillna(0) + g["xa"].fillna(0) if "npxg" in g else g["xg"] + g["xa"]
    total = g["contrib"].sum()
    max_min = g["minutes"].max() or 1
    g["share"] = g["contrib"] / total if total else 0.0
    g["minutes_share"] = g["minutes"] / max_min
    return g[["player", "share", "minutes_share"]]


def team_absences(players: pd.DataFrame, team: str, missing: list[dict]) -> tuple[list[dict], float]:
    """Lista assenti con il loro peso e moltiplicatore dei gol attesi della squadra."""
    if not missing:
        return [], 1.0
    shares = attacking_shares(players, team)
    names = [m["player"] for m in missing]
    mapping = match_players(names, shares["player"].tolist()) if not shares.empty else {}
    by_name = shares.set_index("player") if not shares.empty else pd.DataFrame()
    rows, reduction = [], 0.0
    for m in missing:
        share = 0.0
        regular = False
        if m["player"] in mapping:
            r = by_name.loc[mapping[m["player"]]]
            share = float(r["share"])
            regular = float(r["minutes_share"]) >= 0.5
        w = DOUBTFUL_WEIGHT if m["doubtful"] else 1.0
        reduction += share * REPLACEMENT_LOSS * w
        rows.append({**m, "share": share, "regular": regular})
    rows.sort(key=lambda r: -r["share"])
    return rows, max(1 - MAX_REDUCTION, 1 - reduction)
