"""Agente Arbitro: tendenze dei direttori di gara (cartellini, falli, sbilanciamento casa/ospite)."""
from __future__ import annotations

import pandas as pd

from .base import Agent, AgentOutput, Context, Insight


class RefereeAgent(Agent):
    name = "Arbitri"
    role = "Severità degli arbitri: cartellini e falli fischiati a partita, differenza casa/ospite."

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        m = ctx.data.matches
        if "referee" not in m.columns or m["referee"].isna().all() or "home_yellows" not in m.columns:
            out.notes.append(
                "Dati arbitri non disponibili. Compila data/manual/referees.csv (date,home,away,referee) "
                "con le designazioni (anche quelle della prossima giornata) per attivare questo agente."
            )
            return out
        m = m.dropna(subset=["referee"]).copy()
        m["cards"] = m["home_yellows"] + m["away_yellows"] + m.get("home_reds", 0) + m.get("away_reds", 0)
        m["home_cards"] = m["home_yellows"] + m.get("home_reds", 0)
        m["away_cards"] = m["away_yellows"] + m.get("away_reds", 0)
        agg = {"Partite": ("cards", "size"), "Cartellini/partita": ("cards", "mean"),
               "Casa": ("home_cards", "mean"), "Ospiti": ("away_cards", "mean")}
        if "home_fouls" in m.columns:
            m["fouls"] = m["home_fouls"] + m["away_fouls"]
            agg["Falli/partita"] = ("fouls", "mean")
        if "home_reds" in m.columns:
            m["reds"] = m["home_reds"] + m["away_reds"]
            agg["Rossi totali"] = ("reds", "sum")
        m["over45"] = (m["cards"] > 4.5).astype(float)
        agg["Over 4.5 cart."] = ("over45", "mean")
        table = m.groupby("referee").agg(**agg).reset_index().rename(columns={"referee": "Arbitro"})
        table = table.sort_values("Cartellini/partita", ascending=False).reset_index(drop=True)
        out.tables["referees"] = table

        league_avg = m["cards"].mean()
        qualified = table[table["Partite"] >= 3]
        if len(qualified) >= 4:
            std = qualified["Cartellini/partita"].std()
            for r in qualified.itertuples(index=False):
                z = (r[2] - league_avg) / std if std else 0
                if abs(z) >= ctx.settings.z_threshold:
                    level = "severo" if z > 0 else "permissivo"
                    out.insights.append(Insight(
                        self.name, "arbitro", "tendenza", team=None, metric="referee_cards", value=float(r[2]),
                        league=float(league_avg), z=float(z), n=int(r[1]), score=abs(z),
                        text=f"{r[0]}: arbitro {level}, {r[2]:.1f} cartellini/partita in {r[1]} gare "
                             f"(media lega {league_avg:.1f}).",
                    ))
        out.tables["upcoming"] = self._upcoming(ctx, table)
        return out

    @staticmethod
    def _upcoming(ctx: Context, table: pd.DataFrame) -> dict:
        """Arbitro designato per le prossime partite (se presente nei dati manuali)."""
        fx = ctx.data.fixtures
        if fx is None or fx.empty or "referee" not in fx.columns:
            return {}
        stats = table.set_index("Arbitro")
        res = {}
        for r in fx.dropna(subset=["referee"]).itertuples():
            if r.referee in stats.index:
                res[(r.home, r.away)] = (r.referee, stats.loc[r.referee].to_dict())
            else:
                res[(r.home, r.away)] = (r.referee, None)
        return res
