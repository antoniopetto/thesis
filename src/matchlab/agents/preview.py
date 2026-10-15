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
from .referee import referee_factors


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
            out.notes.append("Nessuna partita della prossima giornata ancora pubblicata su football-data (fixtures.csv): "
                           "di solito compare 2-3 giorni prima; in alternativa usa --fixtures-csv.")
            return out
        tm = ctx.data.team_matches
        teams = set(ctx.data.teams)
        disp = {}
        for base in ("corners_for", "cards_for"):
            if ctx.data.has(base):
                disp[base] = {v: dist.fit_dispersion(tm.loc[tm.venue == v, base]) for v in ("H", "A")}
        ref_stats = referee_factors(ctx)

        previews = []
        for f in fx.itertuples():
            if f.home not in teams or f.away not in teams:
                continue
            book = _book_odds(f)
            p = {"home": f.home, "away": f.away, "date": f.date, "rows": []}
            goals = expected(ctx, f.home, f.away, "goals_for", "goals_against")
            if goals and ctx.data.has("xg_for"):
                xg = expected(ctx, f.home, f.away, "xg_for", "xg_against")
                if xg:
                    goals = ((goals[0] + xg[0]) / 2, (goals[1] + xg[1]) / 2)
            if goals:
                mat = dist.score_matrix(*goals)
                m = dist.match_markets(mat)
                p["goals"] = {"exp": goals, "top_scores": m["top_scores"]}
                for key, label in (("1", "1"), ("X", "X"), ("2", "2")):
                    p["rows"].append(_row(label, m[key], book.get("1x2"), key))
                p["rows"].append(_row("Gol (entrambe segnano)", m["gol"], book.get("btts"), "Yes"))
                total = dist.total_goals_pmf(mat)
                for line in _lines((1.5, 2.5, 3.5), book.get("goals"), 1.5, 4.5):
                    p["rows"].append(_row(f"Over {line} gol", dist.over_prob(total, line),
                                          (book.get("goals") or {}).get(line), "Over"))
            corners = expected(ctx, f.home, f.away, "corners_for", "corners_against")
            if corners:
                d = disp.get("corners_for", {})
                total = dist.sum_pmf(dist.negbin_pmf(corners[0], d.get("H")), dist.negbin_pmf(corners[1], d.get("A")))
                p["corners"] = {"exp": corners}
                for line in _lines((8.5, 9.5, 10.5), book.get("corners"), 6.5, 13.5):
                    p["rows"].append(_row(f"Over {line} corner", dist.over_prob(total, line),
                                          (book.get("corners") or {}).get(line), "Over"))
            cards = expected(ctx, f.home, f.away, "cards_for", "cards_against")
            if cards:
                ref = getattr(f, "referee", None)
                ref = ref if isinstance(ref, str) else None
                factor = ref_stats.get(ref, 1.0) if ref else 1.0
                cards = (cards[0] * factor, cards[1] * factor)
                d = disp.get("cards_for", {})
                total = dist.sum_pmf(dist.negbin_pmf(cards[0], d.get("H")), dist.negbin_pmf(cards[1], d.get("A")))
                p["cards"] = {"exp": cards, "referee": ref, "ref_factor": factor}
                for line in _lines((3.5, 4.5, 5.5), book.get("cards"), 2.5, 7.5):
                    p["rows"].append(_row(f"Over {line} cartellini", dist.over_prob(total, line),
                                          (book.get("cards") or {}).get(line), "Over"))
            p["has_book"] = bool(book)
            previews.append(p)
        out.tables["previews"] = previews
        if previews and not any(p["has_book"] for p in previews):
            out.notes.append("Quote non disponibili per la prossima giornata: confronto col mercato assente.")
        return out


# Differenza minima (punti percentuali) tra modello e mercato per segnalare un esito.
EDGE_FLAG = 0.05


def _row(label: str, model_p: float, choices: dict | None, key: str) -> dict:
    """Riga anteprima: probabilità del modello vs probabilità del bookmaker senza margine."""
    book_odds = book_p = None
    if choices and choices.get(key):
        book_odds = choices[key]
        inv = {k: 1 / v for k, v in choices.items() if v}
        book_p = inv[key] / sum(inv.values()) if inv else None
    edge = model_p - book_p if book_p is not None else None
    return {"label": label, "model_p": model_p, "fair_odds": dist.fair_odds(model_p), "book_odds": book_odds,
            "book_p": book_p, "edge": edge, "flag": edge is not None and edge >= EDGE_FLAG}


def _lines(default, book_lines: dict | None, lo: float, hi: float) -> list[float]:
    extra = [l for l in (book_lines or {}) if lo <= l <= hi]
    return sorted(set(default) | set(extra))


def _book_odds(f) -> dict:
    """Quote SofaScore se presenti, altrimenti medie football-data (1X2 e Over/Under 2.5)."""
    book = getattr(f, "book_odds", None)
    if isinstance(book, dict) and book:
        return book
    def num(k):
        v = getattr(f, k, None)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
    out = {}
    if all(num(k) for k in ("odds_home", "odds_draw", "odds_away")):
        out["1x2"] = {"1": num("odds_home"), "X": num("odds_draw"), "2": num("odds_away")}
    if num("odds_over25") and num("odds_under25"):
        out["goals"] = {2.5: {"Over": num("odds_over25"), "Under": num("odds_under25")}}
    return out
