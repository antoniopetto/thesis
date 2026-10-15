"""Riga di comando.

Esempi:
    python -m matchlab report --league serie_a --season 2026
    python -m matchlab report --season 2025 --matches-csv I1.csv --no-understat
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from .agents.orchestrator import Orchestrator
from .config import DEFAULT_CACHE_DIR, DEFAULT_MANUAL_DIR, LEAGUES, PROJECT_ROOT, AnalysisSettings
from .data import dataset
from .data.cache import FetchError
from .report.builder import write_reports


def _current_season() -> int:
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="matchlab", description="Analisi multi-agente delle squadre di calcio")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="Genera il report squadre")
    rp.add_argument("--league", default="serie_a", choices=sorted(LEAGUES))
    rp.add_argument("--season", type=int, default=_current_season(), help="Anno di inizio (2026 = 2026/27)")
    rp.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports")
    rp.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    rp.add_argument("--manual-dir", type=Path, default=DEFAULT_MANUAL_DIR)
    rp.add_argument("--offline", action="store_true", help="Usa solo i dati già in cache")
    rp.add_argument("--matches-csv", type=Path, help="CSV football-data già scaricato")
    rp.add_argument("--fixtures-csv", type=Path, help="fixtures.csv già scaricato")
    rp.add_argument("--no-understat", action="store_true", help="Non usare Understat (niente xG/giocatori)")
    rp.add_argument("--min-matches", type=int, default=AnalysisSettings.min_matches)
    rp.add_argument("--z", type=float, default=AnalysisSettings.z_threshold, help="Soglia z-score per forza/debolezza")
    rp.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    league = LEAGUES[args.league]
    try:
        data = dataset.build(
            league, args.season, cache_dir=args.cache_dir, manual_dir=args.manual_dir, offline=args.offline,
            matches_csv=args.matches_csv, fixtures_csv=args.fixtures_csv, use_understat=not args.no_understat,
        )
    except FetchError as exc:
        print(f"Impossibile scaricare i dati: {exc}", file=sys.stderr)
        return 1
    if data.matches.empty:
        print("Nessuna partita giocata trovata per questa stagione.", file=sys.stderr)
        return 1
    settings = AnalysisSettings(min_matches=args.min_matches, z_threshold=args.z)
    result = Orchestrator(settings=settings).run(data)
    paths = write_reports(result, args.out)
    for w in data.warnings:
        print(f"Attenzione: {w}", file=sys.stderr)
    print(f"{league.name} {args.season}: {len(data.matches)} partite, {len(result.insights)} segnalazioni.")
    for kind, p in paths.items():
        print(f"  {kind:14s} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
