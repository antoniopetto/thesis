"""Dati sintetici nel formato delle fonti reali (solo per i test, nessun valore reale)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

TEAMS = ["Alfa", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]


def _round_robin(teams):
    n = len(teams)
    rot = list(teams)
    rounds = []
    for _ in range(n - 1):
        pairs = [(rot[i], rot[n - 1 - i]) for i in range(n // 2)]
        rounds.append([(b, a) if len(rounds) % 2 else (a, b) for a, b in pairs])
        rot = [rot[0]] + [rot[-1]] + rot[1:-1]
    return rounds


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    rng = np.random.default_rng(7)
    strength = {t: s for t, s in zip(TEAMS, np.linspace(1.6, 0.6, len(TEAMS)))}
    rows, us_hist = [], {t: [] for t in TEAMS}
    day = pd.Timestamp("2026-08-23")
    for rnd, games in enumerate(_round_robin(TEAMS)):
        for k, (h, a) in enumerate(games):
            date = day + pd.Timedelta(days=7 * rnd)
            lh, la = 1.2 * strength[h] / strength[a] ** 0.5, 0.9 * strength[a] / strength[h] ** 0.5
            hg, ag = rng.poisson(lh), rng.poisson(la)
            hc, ac = rng.poisson(5.5 * strength[h] ** 0.5), rng.poisson(4.5 * strength[a] ** 0.5)
            hy, ay = rng.poisson(2.0 + (1.5 if h == "Kappa" else 0)), rng.poisson(2.3 + (1.5 if a == "Kappa" else 0))
            rows.append({
                "Div": "I1", "Date": date.strftime("%d/%m/%Y"), "Time": "20:45", "HomeTeam": h, "AwayTeam": a,
                "FTHG": hg, "FTAG": ag, "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
                "HTHG": min(hg, rng.integers(0, hg + 1)), "HTAG": min(ag, rng.integers(0, ag + 1)),
                "HS": rng.poisson(13), "AS": rng.poisson(11), "HST": rng.poisson(5), "AST": rng.poisson(4),
                "HF": rng.poisson(12), "AF": rng.poisson(13), "HC": hc, "AC": ac, "HY": hy, "AY": ay,
                "HR": int(rng.random() < 0.05), "AR": int(rng.random() < 0.07),
                "AvgH": 2.1, "AvgD": 3.3, "AvgA": 3.6, "Avg>2.5": 1.9, "Avg<2.5": 1.9,
            })
            for team, opp, ha, gf, ga, lam in ((h, a, "h", hg, ag, lh), (a, h, "a", ag, hg, la)):
                us_hist[team].append({
                    "h_a": ha, "xG": lam, "xGA": 1.0, "npxG": lam * 0.9, "npxGA": 0.9,
                    "ppda": {"att": 200 + 30 * TEAMS.index(team), "def": 20}, "ppda_allowed": {"att": 220, "def": 20},
                    "deep": 6 * strength[team], "deep_allowed": 5, "scored": gf, "missed": ga, "xpts": 1.3,
                    "result": "w" if gf > ga else ("d" if gf == ga else "l"), "date": f"{date:%Y-%m-%d} 20:45:00",
                })
    played = pd.DataFrame(rows[: len(rows) - 5])  # ultima giornata = "da giocare"
    fixtures = pd.DataFrame(rows[len(rows) - 5:]).drop(columns=["FTHG", "FTAG", "FTR", "HTHG", "HTAG"])
    d = tmp_path_factory.mktemp("data")
    played.to_csv(d / "I1.csv", index=False)
    fixtures.to_csv(d / "fixtures.csv", index=False)

    teams = {str(i): {"id": str(i), "title": ("AC " + t if t == "Beta" else t), "history": us_hist[t]}
             for i, t in enumerate(TEAMS)}
    players = []
    for i, t in enumerate(TEAMS):
        for j in range(4):
            players.append({
                "id": str(i * 10 + j), "player_name": f"Giocatore {t} {j}", "games": "8", "time": str(720 - 100 * j),
                "goals": str(max(0, 5 - j)), "xG": str(4.0 - j * 0.8 + (3 if (i, j) == (0, 0) else 0)),
                "assists": "1", "xA": "1.2", "shots": str(20 - 3 * j), "key_passes": str(10 - j),
                "yellow_cards": str(j), "red_cards": "0", "position": "F M", "team_title": t,
                "npg": "3", "npxG": "3.0", "xGChain": "4", "xGBuildup": "2",
            })
    (d / "understat.json").write_text(json.dumps({"teams": teams, "players": players}))
    manual = d / "manual"
    manual.mkdir()
    (manual / "coaches.csv").write_text("team,coach\nAlfa,Mister Uno\n")
    (manual / "player_stats.csv").write_text(
        "player,team,dribbles_completed\nGiocatore Alfa 0,Alfa,30\nGiocatore Beta 1,Beta,2\n"
    )
    ref_rows = ["date,home,away,referee"]
    for r in rows:
        ref = "Rossi" if r["HomeTeam"] in TEAMS[:5] else "Bianchi"
        ref_rows.append(f"{r['Date']},{r['HomeTeam']},{r['AwayTeam']},{ref}")
    (manual / "referees.csv").write_text("\n".join(ref_rows) + "\n")
    return d
