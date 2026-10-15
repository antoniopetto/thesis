import json

import numpy as np
import pytest

from matchlab.agents.orchestrator import Orchestrator
from matchlab.config import LEAGUES, AnalysisSettings
from matchlab.data import dataset, understat
from matchlab.data.names import canonical
from matchlab.report.builder import write_reports
from matchlab.stats import distributions as dist


@pytest.fixture()
def data(synthetic, monkeypatch, tmp_path):
    raw = json.loads((synthetic / "understat.json").read_text())
    monkeypatch.setattr(understat, "fetch_league", lambda *a, **k: raw)
    return dataset.build(
        LEAGUES["serie_a"], 2026, cache_dir=tmp_path / "cache", manual_dir=synthetic / "manual",
        offline=True, matches_csv=synthetic / "I1.csv", fixtures_csv=synthetic / "fixtures.csv",
    )


def test_team_matches_shape(data):
    tm = data.team_matches
    assert len(tm) == 2 * len(data.matches)
    assert {"goals_for", "corners_against", "cards_for", "xg_for", "ppda", "referee"} <= set(tm.columns)
    # Understat "AC Beta" ricondotto a "Beta" e xG unito a ogni partita.
    assert tm["xg_for"].notna().all()
    assert tm.groupby("match_id")["goals_for"].sum().equals(tm.groupby("match_id")["goals_against"].sum())


def test_fixtures_and_extras(data):
    assert len(data.fixtures) == 5
    assert data.fixtures["referee"].notna().all()
    assert data.coaches == {"Alfa": "Mister Uno"}
    assert "dribbles_completed" in data.players.columns


def test_full_run_and_report(data, tmp_path):
    result = Orchestrator(settings=AnalysisSettings(min_matches=3)).run(data)
    for name, out in result.outputs.items():
        assert not any(n.startswith("Errore") for n in out.notes), (name, out.notes)
    assert set(result.summaries) == set(data.teams)
    # La squadra più forte deve avere punti di forza in attacco; Kappa cartellini alti.
    alfa = result.summaries["Alfa"]
    assert alfa.strengths
    kappa_texts = " ".join(i.text for i in result.team_insights("Kappa"))
    assert "artellini" in kappa_texts
    previews = result.outputs["Statistico (anteprime)"].tables["previews"]
    assert len(previews) == 5
    rows = {r["label"]: r for r in previews[0]["rows"]}
    assert abs(rows["1"]["model_p"] + rows["X"]["model_p"] + rows["2"]["model_p"] - 1) < 1e-9
    # Quote medie football-data del CSV sintetico: probabilità bookmaker senza margine.
    assert abs(rows["1"]["book_p"] + rows["X"]["book_p"] + rows["2"]["book_p"] - 1) < 1e-9
    assert rows["Over 2.5 gol"]["book_p"] == pytest.approx(0.5)
    paths = write_reports(result, tmp_path / "out")
    html = paths["html"].read_text()
    assert "Alfa" in html and "Punti di forza" in html and "Prossime partite" in html
    assert json.loads(paths["insights_json"].read_text())


def test_canonical_names():
    known = ["Inter", "Milan", "Parma", "Verona"]
    assert canonical("AC Milan", known) == "Milan"
    assert canonical("Parma Calcio 1913", known) == "Parma"
    assert canonical("Hellas Verona", known) == "Verona"


def test_distributions():
    p = dist.negbin_pmf(10, 8)
    assert abs(p.sum() - 1) < 1e-9
    assert abs((np.arange(len(p)) * p).sum() - 10) < 0.05
    assert dist.fit_dispersion([5] * 20) is None
    mat = dist.score_matrix(1.5, 1.1)
    assert abs(mat.sum() - 1) < 1e-9
    assert 0 < dist.over_prob(dist.poisson_pmf(2.6), 2.5) < 1


def test_prior_from_previous_season(data):
    from matchlab.stats.metrics import lookup, profile_table

    tm = data.team_matches
    prior = profile_table(tm).copy()
    # Stagione scorsa fittizia: "Iota" segnava 1 gol in più della media.
    mask = (prior.team == "Iota") & (prior.metric == "goals_for")
    prior.loc[mask, "value"] = prior.loc[mask, "league"] + 1.0
    base = lookup(profile_table(tm), "Iota", "goals_for")
    with_prior = lookup(profile_table(tm, prior=prior, prior_weight=0.5), "Iota", "goals_for")
    assert with_prior.shrunk > base.shrunk
    assert with_prior.value == base.value  # il dato grezzo della stagione non cambia
    assert with_prior.prev == pytest.approx(base.league + 1.0, rel=0.2)


def test_book_row_margin_removed():
    from matchlab.agents.preview import _row

    r = _row("1", 0.55, {"1": 2.0, "X": 3.4, "2": 3.6}, "1")
    inv = 1 / 2.0 + 1 / 3.4 + 1 / 3.6
    assert r["book_p"] == pytest.approx((1 / 2.0) / inv)
    assert r["flag"] == (0.55 - r["book_p"] >= 0.05)
