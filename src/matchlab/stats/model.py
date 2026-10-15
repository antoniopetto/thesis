"""Modello di previsione partita (fase 2): forza attacco × debolezza difesa.

Per ogni grandezza (gol, corner, cartellini):
    atteso_casa = livello_lega_casa × attacco_casa^e × difesa_ospite^e
dove attacco/difesa sono rapporti rispetto alla media del campionato, calcolati con
shrinkage verso un punto di partenza (media lega + parte dello scarto della stagione
scorsa), e il livello di lega è "ancorato" alla stagione precedente finché la stagione
in corso ha poche partite (evita di inseguire un inizio di campionato anomalo).

Tutti i parametri sono in `ModelParams` e vengono tarati dal backtest
(`matchlab.stats.backtest`), che salva i valori migliori in data/model_params.json.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from . import distributions as dist

PARAMS_PATH = PROJECT_ROOT / "data" / "model_params.json"

# grandezza -> (colonna "fatti", colonna "subiti")
KINDS = {
    "goals": ("goals_for", "goals_against"),
    "xg": ("xg_for", "xg_against"),
    "corners": ("corners_for", "corners_against"),
    "cards": ("cards_for", "cards_against"),
}


@dataclass
class ModelParams:
    shrinkage: float = 4.0  # partite "virtuali" al punto di partenza
    prior_weight: float = 0.6  # quota dello scarto della stagione scorsa ereditata
    league_anchor: float = 60.0  # partite "virtuali" della stagione scorsa nel livello di lega
    xg_weight: float = 0.5  # peso degli xG rispetto ai gol reali nella forza della squadra
    strength_exp: dict = field(default_factory=lambda: {"goals": 1.0, "corners": 1.0, "cards": 1.0})
    rho: float = -0.05  # correzione Dixon-Coles per i punteggi bassi
    ref_shrink: float = 6.0  # partite "virtuali" nella media per il fattore arbitro

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> "ModelParams":
        try:
            raw = json.loads(Path(path).read_text())
            return cls(**raw.get("params", raw))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return cls()

    def to_dict(self) -> dict:
        return asdict(self)


def _venue_sums(tm: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Somme e conteggi per (squadra, campo) e medie di lega per campo."""
    g = tm.groupby(["team", "venue"])[cols]
    return g.sum(min_count=1), g.count()


class MatchModel:
    def __init__(
        self,
        tm: pd.DataFrame,
        params: ModelParams | None = None,
        prev_tm: pd.DataFrame | None = None,
        referee_matches: pd.DataFrame | None = None,
    ):
        self.p = params or ModelParams()
        self.cols = [c for pair in KINDS.values() for c in pair if c in tm.columns and tm[c].notna().any()]
        self.has_xg = {"xg_for", "xg_against"} <= set(self.cols)
        sums, counts = _venue_sums(tm, self.cols)
        league = tm.groupby("venue")[self.cols].mean()
        n_rows = tm.groupby("venue")[self.cols].count()

        prev_team = prev_league = None
        if prev_tm is not None and not prev_tm.empty:
            pcols = [c for c in self.cols if c in prev_tm.columns]
            prev_team = prev_tm.groupby(["team", "venue"])[pcols].mean()
            prev_league = prev_tm.groupby("venue")[pcols].mean()

        # Livello di lega ancorato alla stagione precedente.
        level = league.copy()
        if prev_league is not None:
            for c in prev_league.columns:
                w_cur = n_rows[c]
                level[c] = (w_cur * league[c] + self.p.league_anchor * prev_league[c]) / (w_cur + self.p.league_anchor)

        # Punto di partenza per squadra: media lega corrente + quota dello scarto dell'anno scorso.
        center = pd.DataFrame(index=sums.index, columns=self.cols, dtype=float)
        venues = sums.index.get_level_values("venue")
        for c in self.cols:
            base = league[c].reindex(venues).to_numpy()
            if prev_team is not None and c in prev_team.columns:
                prev_dev = (prev_team[c] - prev_league[c].reindex(prev_team.index.get_level_values("venue")).to_numpy())
                dev = prev_dev.reindex(sums.index).fillna(0).to_numpy()
            else:
                dev = 0.0
            center[c] = base + self.p.prior_weight * dev
        k = self.p.shrinkage
        self.rates = (sums.fillna(0) + k * center) / (counts + k)
        self.league = league
        self.level = level

        self.dispersion = {}
        pooled = tm if prev_tm is None else pd.concat([prev_tm, tm], ignore_index=True)
        for kind in ("corners", "cards"):
            col = KINDS[kind][0]
            if col in pooled.columns:
                self.dispersion[kind] = {v: dist.fit_dispersion(pooled.loc[pooled.venue == v, col]) for v in ("H", "A")}

        self.ref_factor: dict[str, float] = {}
        if referee_matches is not None and not referee_matches.empty and "cards" in referee_matches.columns:
            avg = referee_matches["cards"].mean()
            g = referee_matches.groupby("referee")["cards"].agg(["sum", "size"])
            kr = self.p.ref_shrink
            self.ref_factor = {r: float((x["sum"] + kr * avg) / (x["size"] + kr) / avg) for r, x in g.iterrows()}

    def knows(self, team: str) -> bool:
        return (team, "H") in self.rates.index and (team, "A") in self.rates.index

    def _ratio(self, team: str, venue: str, col: str) -> float:
        return float(self.rates.at[(team, venue), col] / self.league.at[venue, col])

    def _strength(self, team: str, venue: str, kind: str, side: int) -> float:
        """Rapporto rispetto alla media (side 0 = fatti, 1 = subiti); per i gol mescola gol ed xG."""
        r = self._ratio(team, venue, KINDS[kind][side])
        if kind == "goals" and self.has_xg and self.p.xg_weight:
            r = (1 - self.p.xg_weight) * r + self.p.xg_weight * self._ratio(team, venue, KINDS["xg"][side])
        return r

    def expected(self, home: str, away: str, kind: str) -> tuple[float, float] | None:
        f, _ = KINDS[kind]
        if f not in self.cols or not (self.knows(home) and self.knows(away)):
            return None
        e = self.p.strength_exp.get(kind, 1.0)
        lam_h = self.level.at["H", f] * (self._strength(home, "H", kind, 0) * self._strength(away, "A", kind, 1)) ** e
        lam_a = self.level.at["A", f] * (self._strength(away, "A", kind, 0) * self._strength(home, "H", kind, 1)) ** e
        return float(lam_h), float(lam_a)

    def predict(
        self, home: str, away: str, referee: str | None = None, goal_mult: tuple[float, float] = (1.0, 1.0)
    ) -> dict:
        """`goal_mult`: correzione dei gol attesi (es. assenze importanti in attacco)."""
        out: dict = {}
        goals = self.expected(home, away, "goals")
        if goals:
            goals = (goals[0] * goal_mult[0], goals[1] * goal_mult[1])
            mat = dist.score_matrix(*goals, rho=self.p.rho)
            out["goals"] = {"exp": goals, "matrix": mat, "total": dist.total_goals_pmf(mat)}
        for kind in ("corners", "cards"):
            lam = self.expected(home, away, kind)
            if not lam:
                continue
            factor = 1.0
            if kind == "cards" and referee:
                factor = self.ref_factor.get(referee, 1.0)
                lam = (lam[0] * factor, lam[1] * factor)
            d = self.dispersion.get(kind, {})
            total = dist.sum_pmf(dist.negbin_pmf(lam[0], d.get("H")), dist.negbin_pmf(lam[1], d.get("A")))
            out[kind] = {"exp": lam, "total": total, "ref_factor": factor}
        return out
