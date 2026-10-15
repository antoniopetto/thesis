"""Agente Scout: profilo dei singoli giocatori (tiri, xG, passaggi chiave, cartellini, dribbling...)."""
from __future__ import annotations

import pandas as pd

from .base import Agent, AgentOutput, Context, Insight

# Statistiche da riportare "per 90 minuti". Le colonne extra numeriche presenti in
# data/manual/player_stats.csv vengono aggiunte automaticamente.
PER90_LABELS = {
    "goals": "Gol/90",
    "npxg": "npxG/90",
    "xg": "xG/90",
    "xa": "xA/90",
    "shots": "Tiri/90",
    "key_passes": "Passaggi chiave/90",
    "yellow_cards": "Gialli/90",
    "xg_chain": "xGChain/90",
    "dribbles_completed": "Dribbling riusciti/90",
    "tackles": "Contrasti/90",
    "interceptions": "Intercetti/90",
    "fouls_committed": "Falli commessi/90",
    "fouls_won": "Falli subiti/90",
    "shots_on_target": "Tiri in porta/90",
    "crosses": "Cross/90",
    "saves": "Parate/90",
    "big_chances_created": "Grandi occasioni create/90",
    "clearances": "Respinte/90",
}
NOT_PER90 = {"games", "minutes", "red_cards", "assists", "np_goals", "xg_buildup", "age", "rating", "minutes_sofa",
             "appearances"}
# Metriche dove un valore alto è da segnalare come tendenza (non necessariamente "bravo").
HIGHLIGHT = ["shots", "xg", "xa", "key_passes", "yellow_cards", "dribbles_completed", "fouls_committed",
             "fouls_won", "tackles", "shots_on_target"]


def _label(col: str) -> str:
    return PER90_LABELS.get(col, col.replace("_", " ").capitalize() + "/90")


class PlayersAgent(Agent):
    name = "Scout giocatori"
    role = "Rendimento individuale per 90': chi tira, chi crea, chi prende cartellini, chi dribbla."

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        p = ctx.data.players
        s = ctx.settings
        if p is None or p.empty or "minutes" not in p.columns:
            out.notes.append("Statistiche giocatori non disponibili (Understat o data/manual/player_stats.csv).")
            return out
        p = p[p["minutes"] >= s.player_min_minutes].copy()
        if p.empty:
            out.notes.append(f"Nessun giocatore con almeno {s.player_min_minutes} minuti.")
            return out
        numeric = [c for c in p.columns if c not in ("player", "team", "position") and c not in NOT_PER90
                   and pd.api.types.is_numeric_dtype(p[c]) and p[c].notna().any()]
        for c in numeric:
            p[f"{c}_p90"] = p[c] / p["minutes"] * 90
        if "xg" in p.columns:
            team_xg = ctx.data.players.groupby("team")["xg"].sum()
            p["xg_share"] = p["xg"] / p["team"].map(team_xg)
        # Percentili di lega tra i giocatori qualificati.
        for c in numeric:
            p[f"{c}_pct"] = p[f"{c}_p90"].rank(pct=True)

        per_team = {}
        sort_key = (p["xg"].fillna(0) + p["xa"].fillna(0)) if {"xg", "xa"} <= set(p.columns) else p["minutes"]
        p = p.assign(_sort=sort_key)
        for team, g in p.groupby("team"):
            g = g.sort_values("_sort", ascending=False).head(s.players_per_team)
            cols = {"player": "Giocatore", "position": "Ruolo", "games": "PG", "minutes": "Min",
                    "goals": "Gol", "xg": "xG", "assists": "Assist", "xa": "xA"}
            t = g[[c for c in cols if c in g.columns]].rename(columns=cols)
            for c in [x for x in PER90_LABELS if x in numeric and x not in ("goals",)] + \
                     [x for x in numeric if x not in PER90_LABELS]:
                t[_label(c)] = g[f"{c}_p90"].round(2)
            if "rating" in g.columns:
                t["Voto medio"] = g["rating"].round(2)
            if "xg_share" in g.columns:
                t["% xG squadra"] = (g["xg_share"] * 100).round(0)
            for c in ("xG", "xA"):
                if c in t.columns:
                    t[c] = t[c].round(2)
            per_team[team] = t.reset_index(drop=True)
        out.tables["per_team"] = per_team

        for r in p.itertuples(index=False):
            d = r._asdict()
            for c in [c for c in HIGHLIGHT if c in numeric]:
                pct = d.get(f"{c}_pct")
                if pct is not None and pct >= 0.95:
                    out.insights.append(Insight(
                        self.name, "giocatori", "tendenza", team=d["team"], metric=c,
                        value=float(d[f"{c}_p90"]), n=int(d["minutes"]), score=1 + (pct - 0.95) * 20,
                        text=f"{d['player']}: {_label(c)} {d[f'{c}_p90']:.2f} — top {max(1, round((1 - pct) * 100))}% "
                             f"del campionato ({int(d['minutes'])}').",
                    ))
            if "xg" in d and "goals" in d and pd.notna(d["xg"]):
                diff = d["goals"] - d["xg"]
                if diff <= -2:
                    out.insights.append(Insight(
                        self.name, "giocatori", "tendenza", team=d["team"], metric="player_finishing",
                        value=float(diff), score=abs(diff) / 2,
                        text=f"{d['player']}: {int(d['goals'])} gol con {d['xg']:.1f} xG — sotto le attese, "
                             "candidato marcatore 'in ritardo'.",
                    ))
                elif diff >= 3:
                    out.insights.append(Insight(
                        self.name, "giocatori", "tendenza", team=d["team"], metric="player_finishing",
                        value=float(diff), score=diff / 3,
                        text=f"{d['player']}: {int(d['goals'])} gol con {d['xg']:.1f} xG — segna molto più "
                             "del previsto, ritmo difficile da mantenere.",
                    ))
        return out
