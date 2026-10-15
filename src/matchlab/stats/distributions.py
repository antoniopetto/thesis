"""Distribuzioni di probabilità per gol, corner e cartellini.

Fase 2 del progetto: da qui si passa da "tendenze" a probabilità vere e proprie.
Per ora: Poisson (gol, con correzione Dixon-Coles opzionale per i punteggi bassi)
e Binomiale Negativa (corner/cartellini, che hanno più varianza di una Poisson).
"""
from __future__ import annotations

import math

import numpy as np


def poisson_pmf(lam: float, kmax: int = 15) -> np.ndarray:
    k = np.arange(kmax + 1)
    logp = k * math.log(max(lam, 1e-12)) - lam - np.array([math.lgamma(i + 1) for i in k])
    p = np.exp(logp)
    return p / p.sum()


def negbin_pmf(mean: float, dispersion: float | None, kmax: int = 30) -> np.ndarray:
    """Binomiale negativa parametrizzata con media e `dispersion` r (var = mean + mean²/r).

    dispersion None/inf -> ricade sulla Poisson.
    """
    if dispersion is None or not np.isfinite(dispersion) or dispersion <= 0:
        return poisson_pmf(mean, kmax)
    r, mu = dispersion, max(mean, 1e-9)
    p = r / (r + mu)
    k = np.arange(kmax + 1)
    logpmf = (
        np.array([math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1) for i in k])
        + r * math.log(p)
        + k * math.log(1 - p)
    )
    out = np.exp(logpmf)
    return out / out.sum()


def fit_dispersion(values) -> float | None:
    """Stima r col metodo dei momenti. None se i dati non sono sovradispersi."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return None
    mu, var = x.mean(), x.var(ddof=1)
    return mu**2 / (var - mu) if var > mu else None


def over_prob(pmf: np.ndarray, line: float) -> float:
    """P(X > line), es. line=2.5 -> Over 2.5."""
    return float(pmf[np.arange(len(pmf)) > line].sum())


def sum_pmf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distribuzione della somma di due variabili indipendenti (es. corner casa + ospite)."""
    return np.convolve(a, b)


def score_matrix(lam_home: float, lam_away: float, rho: float = -0.05, kmax: int = 10) -> np.ndarray:
    """Matrice P(gol casa = i, gol ospite = j). rho = correzione Dixon-Coles (0 = Poisson indipendente)."""
    m = np.outer(poisson_pmf(lam_home, kmax), poisson_pmf(lam_away, kmax))
    if rho:
        m[0, 0] *= 1 - lam_home * lam_away * rho
        m[0, 1] *= 1 + lam_home * rho
        m[1, 0] *= 1 + lam_away * rho
        m[1, 1] *= 1 - rho
        m = np.clip(m, 0, None)
    return m / m.sum()


def match_markets(matrix: np.ndarray, lines=(1.5, 2.5, 3.5)) -> dict:
    n = matrix.shape[0]
    i, j = np.indices(matrix.shape)
    total = i + j
    out = {
        "1": float(matrix[i > j].sum()),
        "X": float(matrix[i == j].sum()),
        "2": float(matrix[i < j].sum()),
        "gol": float(matrix[(i > 0) & (j > 0)].sum()),
    }
    for line in lines:
        out[f"over_{line}"] = float(matrix[total > line].sum())
    flat = [((a, b), matrix[a, b]) for a in range(n) for b in range(n)]
    out["top_scores"] = [(f"{a}-{b}", float(p)) for (a, b), p in sorted(flat, key=lambda t: -t[1])[:5]]
    return out


def fair_odds(p: float) -> float | None:
    return round(1 / p, 2) if p > 0 else None
