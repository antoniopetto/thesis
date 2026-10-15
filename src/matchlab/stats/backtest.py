"""Backtest "walk-forward" e taratura del modello di previsione.

Per ogni giornata di una stagione passata il modello vede SOLO le partite giocate prima
(più la stagione precedente), prevede le partite di quel giorno e il risultato viene
confrontato con:
- il mercato (quote medie di chiusura football-data, senza margine) per 1X2 e Over/Under 2.5;
- una previsione "ingenua" (frequenze del campionato), per capire se il modello aggiunge qualcosa.

La taratura sceglie i parametri su una stagione e li verifica su quella successiva,
così i numeri finali sono "fuori campione" (non ottimizzati sui dati su cui si misurano).
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.dataset import LeagueData
from . import distributions as dist
from .model import PARAMS_PATH, MatchModel, ModelParams

EPS = 1e-9
MIN_GAMES = 2  # partite minime giocate da ciascuna squadra nella stagione per prevedere


def _referee_matches(matches: pd.DataFrame) -> pd.DataFrame:
    if "referee" not in matches.columns or "home_yellows" not in matches.columns:
        return pd.DataFrame()
    m = matches.dropna(subset=["referee"]).copy()
    m["cards"] = m["home_yellows"] + m["away_yellows"] + m.get("home_reds", 0) + m.get("away_reds", 0)
    return m[["date", "referee", "cards"]]


def _fair(*odds) -> list[float] | None:
    if any(o is None or not np.isfinite(o) or o <= 1 for o in odds):
        return None
    inv = [1 / o for o in odds]
    return [x / sum(inv) for x in inv]


def walk_forward(data: LeagueData, params: ModelParams) -> pd.DataFrame:
    """Una riga per partita prevista con probabilità del modello, del mercato e l'esito reale."""
    return walk_forward_variants(data, [params])[0]


def walk_forward_variants(data: LeagueData, variants: list[ModelParams]) -> list[pd.DataFrame]:
    """Come walk_forward, per più varianti che condividono i parametri di profilo.

    shrinkage / prior_weight / league_anchor / ref_shrink devono essere uguali in tutte le
    varianti (servono a costruire il modello); il resto (xg_weight, esponenti, rho) può cambiare.
    """
    params = variants[0]
    tm, matches = data.team_matches, data.matches.reset_index(drop=True)
    prev_tm = data.previous.team_matches if data.previous is not None else None
    ref_all = pd.concat(
        [_referee_matches(data.previous.matches) if data.previous is not None else pd.DataFrame(),
         _referee_matches(matches)], ignore_index=True,
    )
    rows: list[list[dict]] = [[] for _ in variants]
    for day in sorted(matches["date"].dt.normalize().unique()):
        train = tm[tm["date"] < day]
        if train.empty:
            continue
        games = train.groupby("team").size()
        refs = ref_all[ref_all["date"] < day] if not ref_all.empty else None
        model = MatchModel(train, params, prev_tm, refs)
        for m in matches[matches["date"].dt.normalize() == day].itertuples():
            if games.get(m.home, 0) < MIN_GAMES or games.get(m.away, 0) < MIN_GAMES:
                continue
            ref = getattr(m, "referee", None)
            for vi, variant in enumerate(variants):
                model.p = variant
                r = _evaluate_match(model, m, day, ref if isinstance(ref, str) else None)
                if r is not None:
                    rows[vi].append(r)
    return [pd.DataFrame(r) for r in rows]


def _evaluate_match(model: MatchModel, m, day, ref: str | None) -> dict | None:
    pred = model.predict(m.home, m.away, ref)
    if "goals" not in pred:
        return None
    hg, ag = int(m.home_goals), int(m.away_goals)
    mat = pred["goals"]["matrix"]
    i, j = np.indices(mat.shape)
    r = {
        "date": day, "home": m.home, "away": m.away, "hg": hg, "ag": ag,
        "lam_h": pred["goals"]["exp"][0], "lam_a": pred["goals"]["exp"][1],
        "p1": float(mat[i > j].sum()), "px": float(mat[i == j].sum()), "p2": float(mat[i < j].sum()),
        "p_over25": dist.over_prob(pred["goals"]["total"], 2.5),
        "p_btts": float(mat[1:, 1:].sum()),
        "ll_score": math.log(max(mat[min(hg, mat.shape[0] - 1), min(ag, mat.shape[1] - 1)], EPS)),
        "result": "1" if hg > ag else ("X" if hg == ag else "2"),
        "over25": hg + ag > 2.5, "btts": hg > 0 and ag > 0,
    }
    for prefix, cols in (("open", ("odds_home", "odds_draw", "odds_away")),
                         ("close", ("close_odds_home", "close_odds_draw", "close_odds_away"))):
        vals = [getattr(m, c, None) for c in cols]
        fair = _fair(*vals)
        if fair:
            r[f"{prefix}_p1"], r[f"{prefix}_px"], r[f"{prefix}_p2"] = fair
            r[f"{prefix}_o1"], r[f"{prefix}_ox"], r[f"{prefix}_o2"] = vals
    for prefix, cols in (("open", ("odds_over25", "odds_under25")),
                         ("close", ("close_odds_over25", "close_odds_under25"))):
        vals = [getattr(m, c, None) for c in cols]
        fair = _fair(*vals)
        if fair:
            r[f"{prefix}_pover"] = fair[0]
            r[f"{prefix}_oover"], r[f"{prefix}_ounder"] = vals
    for kind, line, hcol, acol in (("corners", 9.5, "home_corners", "away_corners"),
                                   ("cards", 4.5, None, None)):
        if kind not in pred:
            continue
        if kind == "corners":
            total = getattr(m, hcol) + getattr(m, acol)
        else:
            total = m.home_yellows + m.away_yellows + getattr(m, "home_reds", 0) + getattr(m, "away_reds", 0)
        if pd.isna(total):
            continue
        pmf = pred[kind]["total"]
        total = int(total)
        r[f"{kind}_total"] = total
        r[f"{kind}_exp"] = sum(pred[kind]["exp"])
        r[f"ll_{kind}"] = math.log(max(pmf[min(total, len(pmf) - 1)], EPS))
        r[f"p_{kind}_over"] = dist.over_prob(pmf, line)
    return r


# --- metriche -----------------------------------------------------------------

def _logloss_1x2(df: pd.DataFrame, prefix: str) -> float:
    cols = {"1": f"{prefix}p1", "X": f"{prefix}px", "2": f"{prefix}p2"}
    p = np.array([df.loc[i, cols[r]] for i, r in zip(df.index, df["result"])], dtype=float)
    return float(-np.log(np.clip(p, EPS, 1)).mean())


def _logloss_bin(p: pd.Series, y: pd.Series) -> float:
    p = np.clip(p.astype(float).to_numpy(), EPS, 1 - EPS)
    y = y.astype(float).to_numpy()
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _naive_1x2(df: pd.DataFrame) -> pd.DataFrame:
    """Frequenze 1/X/2 delle partite precedenti (stessa stagione), come previsione ingenua."""
    out = df.copy()
    freq = pd.get_dummies(df["result"]).reindex(columns=["1", "X", "2"], fill_value=0).astype(float)
    cum = freq.expanding().mean().shift(1).fillna(1 / 3)
    out["naive_p1"], out["naive_px"], out["naive_p2"] = cum["1"], cum["X"], cum["2"]
    for col in ("over25", "corners_over", "cards_over"):
        src = {"over25": "over25", "corners_over": "corners_total", "cards_over": "cards_total"}[col]
        if src not in df.columns:
            continue
        y = df[src] > (9.5 if col == "corners_over" else 4.5) if col != "over25" else df[src]
        out[f"naive_{col}"] = y.astype(float).expanding().mean().shift(1).fillna(0.5)
    return out


def evaluate(df: pd.DataFrame) -> dict:
    df = _naive_1x2(df.sort_values("date").reset_index(drop=True))
    res: dict = {"partite": len(df)}
    res["1x2"] = {"modello": _logloss_1x2(df, ""), "ingenuo": _logloss_1x2(df, "naive_")}
    mk = df.dropna(subset=["close_p1"]) if "close_p1" in df.columns else df.iloc[0:0]
    if len(mk):
        res["1x2"]["mercato_chiusura"] = _logloss_1x2(mk, "close_")
        res["1x2"]["modello_stesse_partite"] = _logloss_1x2(mk, "")
    res["over25"] = {"modello": _logloss_bin(df["p_over25"], df["over25"]),
                     "ingenuo": _logloss_bin(df["naive_over25"], df["over25"])}
    if "close_pover" in df.columns:
        mo = df.dropna(subset=["close_pover"])
        res["over25"]["mercato_chiusura"] = _logloss_bin(mo["close_pover"], mo["over25"])
        res["over25"]["modello_stesse_partite"] = _logloss_bin(mo["p_over25"], mo["over25"])
    res["gol_media_prevista"] = float((df["lam_h"] + df["lam_a"]).mean())
    res["gol_media_reale"] = float((df["hg"] + df["ag"]).mean())
    res["ll_risultato_esatto"] = float(df["ll_score"].mean())
    for kind in ("corners", "cards"):
        if f"ll_{kind}" in df.columns:
            d = df.dropna(subset=[f"ll_{kind}"])
            line = 9.5 if kind == "corners" else 4.5
            y = d[f"{kind}_total"] > line
            res[kind] = {
                "ll_totale": float(d[f"ll_{kind}"].mean()),
                f"over{line}_modello": _logloss_bin(d[f"p_{kind}_over"], y),
                f"over{line}_ingenuo": _logloss_bin(d[f"naive_{kind}_over"], y),
                "media_prevista": float(d[f"{kind}_exp"].mean()),
                "media_reale": float(d[f"{kind}_total"].mean()),
            }
    res["calibrazione_over25"] = calibration(df["p_over25"], df["over25"])
    res["calibrazione_1"] = calibration(df["p1"], df["result"] == "1")
    res["scommesse_simulate"] = simulate_bets(df)
    return res


def calibration(p: pd.Series, y: pd.Series, bins=(0, .3, .4, .5, .6, .7, 1.0)) -> list[dict]:
    cut = pd.cut(p, bins, include_lowest=True)
    g = pd.DataFrame({"p": p, "y": y.astype(float), "b": cut}).groupby("b", observed=True)
    return [{"fascia": str(b), "partite": int(len(x)), "prevista": float(x.p.mean()), "reale": float(x.y.mean())}
            for b, x in g if len(x)]


def simulate_bets(df: pd.DataFrame, edge: float = 0.05) -> dict:
    """Puntata fissa di 1 dove il modello vede almeno `edge` in più del mercato (quote medie di apertura)."""
    bets = []
    for r in df.itertuples():
        d = r._asdict()
        for key, outcome, ok in (("1", "p1", "o1"), ("X", "px", "ox"), ("2", "p2", "o2")):
            mp, odds = d.get(f"open_{outcome}"), d.get(f"open_{ok}")
            if mp is None or odds is None or pd.isna(mp):
                continue
            if d[outcome] - mp >= edge:
                bets.append(("1X2", odds, r.result == key))
        mp = d.get("open_pover")
        if mp is not None and not pd.isna(mp):
            if r.p_over25 - mp >= edge:
                bets.append(("Over 2.5", d["open_oover"], bool(r.over25)))
            elif (1 - r.p_over25) - (1 - mp) >= edge:
                bets.append(("Under 2.5", d["open_ounder"], not r.over25))
    out = {}
    for market in ("1X2", "Over 2.5", "Under 2.5", "tutte"):
        b = [x for x in bets if market == "tutte" or x[0] == market]
        if b:
            profit = sum((o - 1) if win else -1 for _, o, win in b)
            out[market] = {"giocate": len(b), "vinte": int(sum(w for *_, w in b)), "profitto": round(profit, 2),
                           "roi": round(profit / len(b), 4)}
    return out


# --- taratura -----------------------------------------------------------------

GRID_PROFILE = {"shrinkage": [3.0, 6.0, 10.0], "prior_weight": [0.4, 0.7, 0.9], "league_anchor": [30.0, 100.0]}
GRID_GOALS = {"xg_weight": [0.0, 0.5, 0.8], "goals_exp": [0.7, 0.85, 1.0]}
GRID_SECONDARY = {"exp": [0.5, 0.75, 1.0]}


def tune(train: LeagueData, log=print) -> tuple[ModelParams, pd.DataFrame]:
    """Ricerca a griglia sulla stagione `train`; obiettivo: verosimiglianza media (più alta = meglio)."""
    results = []
    combos = list(itertools.product(*GRID_PROFILE.values()))
    for i, (k, pw, anchor) in enumerate(combos, 1):
        keys, variants = [], []
        for xw, ge in itertools.product(*GRID_GOALS.values()):
            for se in GRID_SECONDARY["exp"]:
                keys.append((xw, ge, se))
                variants.append(ModelParams(shrinkage=k, prior_weight=pw, league_anchor=anchor, xg_weight=xw,
                                            strength_exp={"goals": ge, "corners": se, "cards": se}))
        for (xw, ge, se), df in zip(keys, walk_forward_variants(train, variants)):
            row = {"shrinkage": k, "prior_weight": pw, "league_anchor": anchor, "xg_weight": xw,
                   "goals_exp": ge, "secondary_exp": se, "ll_score": df["ll_score"].mean()}
            for kind in ("corners", "cards"):
                if f"ll_{kind}" in df.columns:
                    row[f"ll_{kind}"] = df[f"ll_{kind}"].mean()
            results.append(row)
        log(f"  taratura {i}/{len(combos)}")
    res = pd.DataFrame(results)
    best_goals = res.sort_values("ll_score", ascending=False).iloc[0]
    # Corner e cartellini: stessi parametri di profilo dei gol, esponente scelto per ciascuno.
    same = res[(res.shrinkage == best_goals.shrinkage) & (res.prior_weight == best_goals.prior_weight)
               & (res.league_anchor == best_goals.league_anchor) & (res.xg_weight == best_goals.xg_weight)
               & (res.goals_exp == best_goals.goals_exp)]
    exp = {"goals": float(best_goals.goals_exp)}
    for kind in ("corners", "cards"):
        col = f"ll_{kind}"
        exp[kind] = float(same.sort_values(col, ascending=False).iloc[0].secondary_exp) if col in same else 1.0
    best = ModelParams(shrinkage=float(best_goals.shrinkage), prior_weight=float(best_goals.prior_weight),
                       league_anchor=float(best_goals.league_anchor), xg_weight=float(best_goals.xg_weight),
                       strength_exp=exp)
    return best, res


def run(train: LeagueData, test: LeagueData, out_path: Path = PARAMS_PATH, log=print) -> dict:
    log(f"Taratura sulla stagione {train.season}/{train.season + 1}...")
    best, grid = tune(train, log=log)
    log(f"Parametri scelti: {best.to_dict()}")
    report = {
        "generato": datetime.now().isoformat(timespec="seconds"),
        "campionato": test.league.name,
        "stagione_taratura": train.season,
        "stagione_verifica": test.season,
        "params": best.to_dict(),
        "verifica_default": evaluate(walk_forward(test, ModelParams())),
        "verifica_tarato": evaluate(walk_forward(test, best)),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    return report


def summary(report: dict) -> str:
    """Riassunto leggibile del backtest."""
    lines = [f"Verifica su {report['campionato']} {report['stagione_verifica']}/{(report['stagione_verifica'] + 1) % 100:02d} "
             f"(parametri tarati su {report['stagione_taratura']}/{(report['stagione_taratura'] + 1) % 100:02d})"]
    for label, key in (("Modello iniziale", "verifica_default"), ("Modello tarato", "verifica_tarato")):
        r = report[key]
        lines.append(f"\n{label} — {r['partite']} partite")
        for mk in ("1x2", "over25"):
            v = r[mk]
            txt = f"  {mk.upper():7s} log-loss: modello {v['modello']:.4f}, ingenuo {v['ingenuo']:.4f}"
            if "mercato_chiusura" in v:
                txt += f", mercato {v['mercato_chiusura']:.4f} (modello sulle stesse partite {v['modello_stesse_partite']:.4f})"
            lines.append(txt)
        lines.append(f"  Gol medi a partita: previsti {r['gol_media_prevista']:.2f}, reali {r['gol_media_reale']:.2f}")
        for kind in ("corners", "cards"):
            if kind in r:
                v = r[kind]
                ks = [k for k in v if k.startswith("over")]
                lines.append(f"  {kind}: media prevista {v['media_prevista']:.2f}, reale {v['media_reale']:.2f}; "
                             f"{ks[0]} {v[ks[0]]:.4f} vs {ks[1]} {v[ks[1]]:.4f}")
        for mk, b in r["scommesse_simulate"].items():
            lines.append(f"  Scommesse simulate {mk}: {b['giocate']} giocate, {b['vinte']} vinte, ROI {b['roi'] * 100:+.1f}%")
    return "\n".join(lines)
