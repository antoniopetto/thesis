"""Agente Statistico (anteprima partite) — PRIMO PASSO verso le distribuzioni (fase 2).

Modello volutamente semplice e trasparente (moltiplicativo forza attacco x debolezza difesa):
    atteso_casa = media_lega_casa * attacco_casa(squadra) * difesa_trasferta(avversario)
calcolato su gol (mescolando gol reali e xG quando disponibili), corner e cartellini, poi
trasformato in probabilità con Poisson / Binomiale Negativa. Serve come riferimento da
confrontare con le frequenze storiche e le quote, NON come segnale di gioco automatico.
"""
from __future__ import annotations

import numpy as np

from ..stats import distributions as dist
from ..stats.metrics import lookup
from .base import Agent, AgentOutput, Context


def _rate(ctx: Context, team: str, metric: str, venue: str) -> tuple[float, float] | None:
    """(valore shrunk, media lega) per squadra/metrica/campo."""
    r = lookup(ctx.profiles, team, metric, venue)
    if r is None or np.isnan(r.shrunk) or not r.league:
        return None
    return float(r.shrunk), float(r.league)


def expected(ctx: Context, home: str, away: str, for_metric: str, against_metric: str) -> tuple[float, float] | None:
    hf, ha = _rate(ctx, home, for_metric, "H"), _rate(ctx, home, against_metric, "H")
    af, aa = _rate(ctx, away, for_metric, "A"), _rate(ctx, away, against_metric, "A")
    if None in (hf, ha, af, aa):
        return None
    league_home, league_away = hf[1], af[1]  # media "fatti" in casa / in trasferta
    lam_home = league_home * (hf[0] / league_home) * (aa[0] / aa[1])
    lam_away = league_away * (af[0] / league_away) * (ha[0] / ha[1])
    return lam_home, lam_away


class PreviewAgent(Agent):
    name = "Statistico (anteprime)"
    role = "Valori attesi e probabilità di base per le prossime partite (fase 2, sperimentale)."

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        fx = ctx.data.fixtures
        if fx is None or fx.empty:
            out.notes.append("Nessuna partita in calendario trovata (fixtures.csv).")
            return out
        tm = ctx.data.team_matches
        teams = set(ctx.data.teams)
        disp = {}
        for base in ("corners_for", "cards_for"):
            if ctx.data.has(base):
                disp[base] = {v: dist.fit_dispersion(tm.loc[tm.venue == v, base]) for v in ("H", "A")}
        ref_stats = self._referee_factor(ctx)

        previews = []
        for f in fx.itertuples():
            if f.home not in teams or f.away not in teams:
                continue
            p = {"home": f.home, "away": f.away, "date": f.date}
            goals = expected(ctx, f.home, f.away, "goals_for", "goals_against")
            if goals and ctx.data.has("xg_for"):
                xg = expected(ctx, f.home, f.away, "xg_for", "xg_against")
                if xg:
                    goals = ((goals[0] + xg[0]) / 2, (goals[1] + xg[1]) / 2)
            if goals:
                mat = dist.score_matrix(*goals)
                p["goals"] = {"exp": goals, "markets": dist.match_markets(mat)}
            corners = expected(ctx, f.home, f.away, "corners_for", "corners_against")
            if corners:
                d = disp.get("corners_for", {})
                total = dist.sum_pmf(dist.negbin_pmf(corners[0], d.get("H")), dist.negbin_pmf(corners[1], d.get("A")))
                p["corners"] = {"exp": corners, "over": {l: dist.over_prob(total, l) for l in (8.5, 9.5, 10.5)}}
            cards = expected(ctx, f.home, f.away, "cards_for", "cards_against")
            if cards:
                ref = getattr(f, "referee", None)
                factor = ref_stats.get(ref, 1.0) if isinstance(ref, str) else 1.0
                cards = (cards[0] * factor, cards[1] * factor)
                d = disp.get("cards_for", {})
                total = dist.sum_pmf(dist.negbin_pmf(cards[0], d.get("H")), dist.negbin_pmf(cards[1], d.get("A")))
                p["cards"] = {"exp": cards, "over": {l: dist.over_prob(total, l) for l in (3.5, 4.5, 5.5)},
                              "referee": ref if isinstance(ref, str) else None, "ref_factor": factor}
            odds = {k: getattr(f, k, None) for k in ("odds_home", "odds_draw", "odds_away", "odds_over25")}
            p["odds"] = {k: float(v) for k, v in odds.items() if v is not None and not np.isnan(v)}
            previews.append(p)
        out.tables["previews"] = previews
        return out

    @staticmethod
    def _referee_factor(ctx: Context, k: float = 5.0) -> dict[str, float]:
        """Moltiplicatore cartellini per arbitro, con shrinkage verso 1."""
        m = ctx.data.matches
        if "referee" not in m.columns or "home_yellows" not in m.columns:
            return {}
        m = m.dropna(subset=["referee"])
        if m.empty:
            return {}
        cards = m["home_yellows"] + m["away_yellows"] + m.get("home_reds", 0) + m.get("away_reds", 0)
        avg = cards.mean()
        g = cards.groupby(m["referee"]).agg(["sum", "size"])
        return {ref: float((r["sum"] + k * avg) / (r["size"] + k) / avg) for ref, r in g.iterrows()}
