"""Agente Statistico (anteprime partite) — fase 2.

Usa `matchlab.stats.model.MatchModel` (forza attacco × debolezza difesa, Poisson /
Dixon-Coles per i gol, binomiale negativa per corner e cartellini, correzione arbitro)
con i parametri tarati dal backtest (data/model_params.json), e confronta le
probabilità con quelle del bookmaker senza margine.
"""
from __future__ import annotations

import numpy as np

from ..stats import distributions as dist
from ..stats.model import MatchModel, ModelParams
from ..stats.context import team_tags
from .absences import team_absences
from .opponents import matchup_lines
from .base import Agent, AgentOutput, Context
from .referee import referee_matches


class PreviewAgent(Agent):
    name = "Statistico (anteprime)"
    role = "Valori attesi e probabilità per la prossima giornata, a confronto con le quote (fase 2)."

    def __init__(self, params: ModelParams | None = None):
        self.params = params

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        fx = ctx.data.fixtures
        if fx is None or fx.empty:
            out.notes.append("Nessuna partita della prossima giornata trovata (football-data / SofaScore).")
            return out
        teams = set(ctx.data.teams)
        prev = ctx.data.previous.team_matches if ctx.data.previous is not None else None
        model = MatchModel(ctx.data.team_matches, self.params or ModelParams.load(), prev, referee_matches(ctx))
        out.tables["params"] = model.p.to_dict()

        previews = []
        for f in fx.itertuples():
            if f.home not in teams or f.away not in teams:
                continue
            book = _book_odds(f)
            p = {"home": f.home, "away": f.away, "date": f.date, "rows": []}
            ref = getattr(f, "referee", None)
            ref = ref if isinstance(ref, str) else None
            missing = getattr(f, "missing", None)
            missing = missing if isinstance(missing, dict) else {}
            abs_h, mult_h = team_absences(ctx.data.players, f.home, missing.get("home", []))
            abs_a, mult_a = team_absences(ctx.data.players, f.away, missing.get("away", []))
            p["absences"] = {"home": abs_h, "away": abs_a, "mult": (mult_h, mult_a),
                             "confirmed": missing.get("confirmed", False)}
            pred = model.predict(f.home, f.away, ref, goal_mult=(mult_h, mult_a))
            if "goals" in pred:
                mat = pred["goals"]["matrix"]
                m = dist.match_markets(mat)
                p["goals"] = {"exp": pred["goals"]["exp"], "top_scores": m["top_scores"]}
                for key, label in (("1", "1"), ("X", "X"), ("2", "2")):
                    p["rows"].append(_row(label, m[key], book.get("1x2"), key))
                p["rows"].append(_row("Gol (entrambe segnano)", m["gol"], book.get("btts"), "Yes"))
                total = pred["goals"]["total"]
                for line in _lines((1.5, 2.5, 3.5), book.get("goals"), 1.5, 4.5):
                    p["rows"].append(_row(f"Over {line} gol", dist.over_prob(total, line),
                                          (book.get("goals") or {}).get(line), "Over"))
            if "corners" in pred:
                total = pred["corners"]["total"]
                p["corners"] = {"exp": pred["corners"]["exp"]}
                for line in _lines((8.5, 9.5, 10.5), book.get("corners"), 6.5, 13.5):
                    p["rows"].append(_row(f"Over {line} corner", dist.over_prob(total, line),
                                          (book.get("corners") or {}).get(line), "Over"))
            if "cards" in pred:
                total = pred["cards"]["total"]
                p["cards"] = {"exp": pred["cards"]["exp"], "referee": ref, "ref_factor": pred["cards"]["ref_factor"]}
                for line in _lines((3.5, 4.5, 5.5), book.get("cards"), 2.5, 7.5):
                    p["rows"].append(_row(f"Over {line} cartellini", dist.over_prob(total, line),
                                          (book.get("cards") or {}).get(line), "Over"))
            p["has_book"] = bool(book)
            p["matchup"] = {
                "home": {"tags": team_tags(ctx.opponents.classes, f.away) if ctx.opponents else [],
                         "lines": matchup_lines(ctx.opponents, f.home, f.away)},
                "away": {"tags": team_tags(ctx.opponents.classes, f.home) if ctx.opponents else [],
                         "lines": matchup_lines(ctx.opponents, f.away, f.home)},
            }
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
