"""Agente Arbitro: tendenze dei direttori di gara (cartellini, falli, sbilanciamento casa/ospite)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Agent, AgentOutput, Context, Insight


def referee_matches(ctx: Context) -> pd.DataFrame:
    """Partite con arbitro noto, stagione in corso + precedente (più campione per arbitro)."""
    frames = []
    for d, label in ((ctx.data, "corrente"), (ctx.data.previous, "precedente")):
        if d is None:
            continue
        m = d.matches
        if "referee" not in m.columns or "home_yellows" not in m.columns:
            continue
        m = m.dropna(subset=["referee"]).copy()
        m["stagione"] = label
        frames.append(m)
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True)
    reds_h = m["home_reds"] if "home_reds" in m.columns else 0
    reds_a = m["away_reds"] if "away_reds" in m.columns else 0
    m["home_cards"] = m["home_yellows"] + reds_h
    m["away_cards"] = m["away_yellows"] + reds_a
    m["cards"] = m["home_cards"] + m["away_cards"]
    if "home_fouls" in m.columns:
        m["fouls"] = m["home_fouls"] + m["away_fouls"]
    return m


def referee_factors(ctx: Context, k: float = 6.0) -> dict[str, float]:
    """Moltiplicatore cartellini per arbitro (1 = nella media), con shrinkage verso 1."""
    m = referee_matches(ctx)
    if m.empty:
        return {}
    avg = m["cards"].mean()
    g = m.groupby("referee")["cards"].agg(["sum", "size"])
    return {ref: float((r["sum"] + k * avg) / (r["size"] + k) / avg) for ref, r in g.iterrows()}


class RefereeAgent(Agent):
    name = "Arbitri"
    role = "Severità degli arbitri (stagione in corso + precedente): cartellini e falli a partita, casa/ospite."

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        m = referee_matches(ctx)
        if m.empty:
            out.notes.append(
                "Dati arbitri non disponibili (SofaScore non raggiungibile e data/manual/referees.csv vuoto)."
            )
            return out
        factors = referee_factors(ctx)
        agg = {
            "Partite": ("cards", "size"),
            "Cartellini/partita": ("cards", "mean"),
            "Casa": ("home_cards", "mean"),
            "Ospiti": ("away_cards", "mean"),
        }
        if "fouls" in m.columns:
            agg["Falli/partita"] = ("fouls", "mean")
        m["over45"] = (m["cards"] > 4.5).astype(float)
        agg["Over 4.5 cart."] = ("over45", "mean")
        table = m.groupby("referee").agg(**agg)
        table["Di cui stagione in corso"] = m[m.stagione == "corrente"].groupby("referee").size()
        table["Di cui stagione in corso"] = table["Di cui stagione in corso"].fillna(0).astype(int)
        table["Fattore cartellini"] = pd.Series(factors)
        career = ctx.data.referee_career
        if career is not None and not career.empty:
            c = career.reindex(table.index)
            table["Gialli/partita (carriera)"] = c["ref_career_yellows"] / c["ref_career_games"]
        table["Over 4.5 cart."] = (table["Over 4.5 cart."] * 100).round(0).astype(int).astype(str) + "%"
        for col in ("Cartellini/partita", "Casa", "Ospiti", "Falli/partita", "Fattore cartellini",
                    "Gialli/partita (carriera)"):
            if col in table.columns:
                table[col] = table[col].astype(float).round(2)
        table = table.sort_values("Fattore cartellini", ascending=False).reset_index()
        table = table.rename(columns={"referee": "Arbitro"})
        out.tables["referees"] = table

        league_avg = m["cards"].mean()
        for ref, f in factors.items():
            n = int((m["referee"] == ref).sum())
            if n >= 4 and abs(f - 1) >= 0.12:
                level = "severo" if f > 1 else "permissivo"
                avg = m.loc[m["referee"] == ref, "cards"].mean()
                out.insights.append(Insight(
                    self.name, "arbitro", "tendenza", team=None, metric="referee_cards", value=float(avg),
                    league=float(league_avg), n=n, score=abs(f - 1) * 5,
                    text=f"{ref}: arbitro {level}, {avg:.1f} cartellini/partita in {n} gare "
                         f"(media {league_avg:.1f}; fattore stimato {f:.2f}).",
                ))
        out.tables["upcoming"] = self._upcoming(ctx, table)
        return out

    @staticmethod
    def _upcoming(ctx: Context, table: pd.DataFrame) -> list[dict]:
        """Arbitro designato per le prossime partite."""
        fx = ctx.data.fixtures
        if fx is None or fx.empty or "referee" not in fx.columns:
            return []
        stats = table.set_index("Arbitro")
        res = []
        for r in fx.dropna(subset=["referee"]).itertuples():
            s = stats.loc[r.referee] if r.referee in stats.index else None
            res.append({
                "Data": pd.Timestamp(r.date).strftime("%d/%m"),
                "Partita": f"{r.home} – {r.away}",
                "Arbitro": r.referee,
                "Partite analizzate": int(s["Partite"]) if s is not None else 0,
                "Cartellini/partita": s["Cartellini/partita"] if s is not None else np.nan,
                "Fattore cartellini": s["Fattore cartellini"] if s is not None else np.nan,
            })
        return res
