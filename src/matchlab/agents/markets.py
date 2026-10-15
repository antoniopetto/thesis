"""Agente Mercati: frequenze storiche degli esiti più giocati (Over/Under, Gol/NoGol, corner, cartellini)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..stats.metrics import VENUES
from .base import Agent, AgentOutput, Context, Insight


@dataclass(frozen=True)
class Market:
    key: str
    label: str
    category: str
    requires: tuple[str, ...]
    hit: Callable[[pd.DataFrame], pd.Series]


def _over(col: str, line: float) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda d: d[col] > line


MARKETS: list[Market] = [
    Market("win", "Vittoria (1 in casa / 2 fuori)", "esito", ("result",), lambda d: d["result"] == "W"),
    Market("draw", "Pareggio (X)", "esito", ("result",), lambda d: d["result"] == "D"),
    Market("loss", "Sconfitta", "esito", ("result",), lambda d: d["result"] == "L"),
    Market("over15", "Over 1.5 gol (partita)", "gol", ("goals_total",), _over("goals_total", 1.5)),
    Market("over25", "Over 2.5 gol (partita)", "gol", ("goals_total",), _over("goals_total", 2.5)),
    Market("over35", "Over 3.5 gol (partita)", "gol", ("goals_total",), _over("goals_total", 3.5)),
    Market("btts", "Gol (entrambe segnano)", "gol", ("goals_for", "goals_against"),
           lambda d: (d["goals_for"] > 0) & (d["goals_against"] > 0)),
    Market("team_over15", "Squadra segna 2+ gol", "gol", ("goals_for",), _over("goals_for", 1.5)),
    Market("ht_over05", "Over 0.5 gol 1° tempo", "gol", ("ht_goals_total",), _over("ht_goals_total", 0.5)),
    Market("corners_over85", "Over 8.5 corner (partita)", "corner", ("corners_total",), _over("corners_total", 8.5)),
    Market("corners_over95", "Over 9.5 corner (partita)", "corner", ("corners_total",), _over("corners_total", 9.5)),
    Market("corners_over105", "Over 10.5 corner (partita)", "corner", ("corners_total",), _over("corners_total", 10.5)),
    Market("team_corners_over45", "Squadra batte 5+ corner", "corner", ("corners_for",), _over("corners_for", 4.5)),
    Market("corners_more", "Batte più corner dell'avversario", "corner", ("corners_for", "corners_against"),
           lambda d: d["corners_for"] > d["corners_against"]),
    Market("cards_over35", "Over 3.5 cartellini (partita)", "cartellini", ("cards_total",), _over("cards_total", 3.5)),
    Market("cards_over45", "Over 4.5 cartellini (partita)", "cartellini", ("cards_total",), _over("cards_total", 4.5)),
    Market("cards_over55", "Over 5.5 cartellini (partita)", "cartellini", ("cards_total",), _over("cards_total", 5.5)),
    Market("team_cards_over15", "Squadra prende 2+ cartellini", "cartellini", ("cards_for",), _over("cards_for", 1.5)),
]


class MarketsAgent(Agent):
    name = "Mercati"
    role = "Quanto spesso, storicamente, si verificano gli esiti delle schedine nelle partite di ogni squadra."

    def run(self, ctx: Context) -> AgentOutput:
        s = ctx.settings
        tm = ctx.data.team_matches
        out = AgentOutput(self.name, self.role)
        markets = [m for m in MARKETS if all(c in tm.columns and tm[c].notna().any() for c in m.requires)]
        rows = []
        for venue in VENUES:
            sub = tm if venue == "all" else tm[tm.venue == venue]
            for m in markets:
                valid = sub.dropna(subset=list(m.requires))
                if valid.empty:
                    continue
                hits = m.hit(valid).astype(float)
                league = hits.mean()
                for team, h in hits.groupby(valid["team"]):
                    rows.append(dict(team=team, venue=venue, market=m.key, label=m.label, category=m.category,
                                     n=len(h), hits=int(h.sum()), freq=h.mean(), league=league))
        freq = pd.DataFrame(rows)
        out.tables["frequencies"] = freq
        if freq.empty:
            return out

        per_team = {}
        for team, g in freq.groupby("team"):
            piv = g.pivot_table(index=["market", "label"], columns="venue", values="freq", sort=False)
            cnt = g.pivot_table(index=["market", "label"], columns="venue", values="n", sort=False)
            lg = g[g.venue == "all"].set_index("market")["league"]
            table = []
            for (key, label), r in piv.iterrows():
                row = {"Esito": label}
                for v, vl in VENUES.items():
                    row[vl] = f"{r.get(v, float('nan')) * 100:.0f}% ({int(cnt.loc[(key, label)].get(v, 0))})" \
                        if pd.notna(r.get(v)) else "–"
                row["Media lega"] = f"{lg.get(key, float('nan')) * 100:.0f}%"
                table.append(row)
            per_team[team] = pd.DataFrame(table)
        out.tables["per_team"] = per_team

        # Prima il dato complessivo, poi casa/trasferta solo se aggiungono un esito nuovo.
        seen = set()
        order = {"all": 0, "H": 1, "A": 2}
        for r in sorted(freq.itertuples(), key=lambda x: order[x.venue]):
            if r.n < s.min_matches:
                continue
            notable = (r.freq >= s.market_high or r.freq <= s.market_low) and abs(r.freq - r.league) >= 0.15
            if not notable or (r.team, r.market) in seen:
                continue
            seen.add((r.team, r.market))
            where = "" if r.venue == "all" else f" {VENUES[r.venue].lower()}"
            text = (f"{r.label}{where}: {r.hits}/{r.n} partite ({r.freq * 100:.0f}%), "
                    f"media lega {r.league * 100:.0f}%.")
            out.insights.append(Insight(self.name, r.category, "tendenza", text, team=r.team, metric=r.market,
                                        venue=r.venue, value=float(r.freq), league=float(r.league), n=int(r.n),
                                        score=abs(r.freq - r.league) * 5))
        return out
