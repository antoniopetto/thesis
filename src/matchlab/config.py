"""Configurazione campionati e parametri di analisi."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DEFAULT_MANUAL_DIR = PROJECT_ROOT / "data" / "manual"


@dataclass(frozen=True)
class League:
    key: str
    name: str
    football_data_code: str  # codice divisione su football-data.co.uk (es. I1)
    understat_code: str | None  # nome campionato su understat.com


LEAGUES: dict[str, League] = {
    "serie_a": League("serie_a", "Serie A", "I1", "Serie_A"),
    "serie_b": League("serie_b", "Serie B", "I2", None),
    "premier_league": League("premier_league", "Premier League", "E0", "EPL"),
    "la_liga": League("la_liga", "La Liga", "SP1", "La_liga"),
    "bundesliga": League("bundesliga", "Bundesliga", "D1", "Bundesliga"),
    "ligue_1": League("ligue_1", "Ligue 1", "F1", "Ligue_1"),
}


def season_code(start_year: int) -> str:
    """2026 -> '2627' (formato football-data.co.uk)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year}/{(start_year + 1) % 100:02d}"


@dataclass
class AnalysisSettings:
    """Soglie usate dagli agenti per decidere cosa segnalare nel report."""

    # Partite "virtuali" di media campionato aggiunte a ogni squadra (shrinkage):
    # riduce il rumore quando il campione è piccolo.
    shrinkage_matches: float = 4.0
    # z-score (rispetto alle altre squadre) oltre il quale un dato è "forza/debolezza".
    z_threshold: float = 1.0
    # Minimo partite per segnalare una tendenza.
    min_matches: int = 4
    # Frequenze di mercato considerate notevoli (es. Over 2.5 nel 75% delle partite).
    market_high: float = 0.70
    market_low: float = 0.30
    # Partite per la "forma recente".
    form_window: int = 5
    # Minuti minimi per includere un giocatore nelle statistiche per 90'.
    player_min_minutes: int = 270
    # Quanti giocatori mostrare per squadra.
    players_per_team: int = 8
    extra: dict = field(default_factory=dict)
