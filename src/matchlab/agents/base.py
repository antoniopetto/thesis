"""Struttura comune degli agenti.

Ogni agente ha UN ruolo, legge il contesto condiviso (dati + profili statistici) e
restituisce `Insight` strutturati. In questa fase gli agenti sono deterministici
(regole + statistica); l'interfaccia è pensata per affiancare in seguito agenti LLM
(notizie, formazioni probabili) senza cambiare il resto della pipeline.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import pandas as pd

from ..config import AnalysisSettings
from ..data.dataset import LeagueData
from ..stats.metrics import METRICS_BY_KEY, VENUES, display_rank

KINDS = ("forza", "debolezza", "tendenza", "info")


@dataclass
class Insight:
    agent: str
    category: str
    kind: str  # forza | debolezza | tendenza | info
    text: str
    team: str | None = None  # None = insight di campionato (es. arbitri)
    metric: str | None = None
    venue: str = "all"
    value: float | None = None
    league: float | None = None
    z: float | None = None
    n: int | None = None
    score: float = 0.0  # rilevanza, per ordinare (più alto = più notevole)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentOutput:
    agent: str
    role: str
    insights: list[Insight] = field(default_factory=list)
    # Tabelle per il report: chiave -> DataFrame (di campionato) o dict team -> DataFrame.
    tables: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Context:
    data: LeagueData
    settings: AnalysisSettings
    profiles: pd.DataFrame  # output di stats.metrics.profile_table

    @property
    def league_name(self) -> str:
        return self.data.league.name


class Agent(ABC):
    name: str = "agente"
    role: str = ""

    @abstractmethod
    def run(self, ctx: Context) -> AgentOutput: ...

    # --- utilità condivise -------------------------------------------------

    def metric_insights(self, ctx: Context, categories: set[str]) -> list[Insight]:
        """Trasforma i profili statistici in punti di forza / debolezza / tendenze.

        Regola: si segnala ciò che si discosta dalle altre squadre di almeno
        `z_threshold` deviazioni standard (sul valore con shrinkage), con almeno
        `min_matches` partite. Casa/trasferta si segnalano solo se dicono qualcosa
        di diverso dal dato complessivo.
        """
        s = ctx.settings
        prof = ctx.profiles
        prof = prof[prof["metric"].map(lambda k: METRICS_BY_KEY[k].category in categories)]
        out: list[Insight] = []
        seen: set[tuple[str, str, str]] = set()
        for venue in ("all", "H", "A"):
            for row in prof[prof.venue == venue].itertuples():
                if row.n < s.min_matches or row.z is None or math.isnan(row.z) or abs(row.z) < s.z_threshold:
                    continue
                m = METRICS_BY_KEY[row.metric]
                if m.higher_is_better is None:
                    kind = "tendenza"
                elif (row.z > 0) == m.higher_is_better:
                    kind = "forza"
                else:
                    kind = "debolezza"
                if venue != "all" and (row.team, row.metric, kind) in seen:
                    continue
                seen.add((row.team, row.metric, kind))
                level = "molto alto" if row.z > 0 else "molto basso"
                where = "" if venue == "all" else f" ({VENUES[venue].lower()})"
                text = (
                    f"{m.label}{where}: {m.format(row.value)} — {level} "
                    f"(media lega {m.format(row.league)}, {display_rank(m, row.rank, row.n_teams)}; "
                    f"ultime {s.form_window}: {m.format(row.last_n)})"
                )
                out.append(
                    Insight(
                        agent=self.name, category=m.category, kind=kind, text=text, team=row.team,
                        metric=m.key, venue=venue, value=float(row.value), league=float(row.league),
                        z=float(row.z), n=int(row.n), score=abs(float(row.z)),
                    )
                )
        return out

    def team_metric_table(self, ctx: Context, team: str, categories: set[str]) -> pd.DataFrame:
        """Tabella metrica x (totale/casa/fuori) per il report della squadra."""
        prof = ctx.profiles[(ctx.profiles.team == team)]
        prof = prof[prof["metric"].map(lambda k: METRICS_BY_KEY[k].category in categories)]
        rows = []
        for key in dict.fromkeys(prof["metric"]):
            m = METRICS_BY_KEY[key]
            r = {"Metrica": m.label}
            for venue, label in VENUES.items():
                x = prof[(prof.metric == key) & (prof.venue == venue)]
                r[label] = m.format(x.iloc[0].value) if not x.empty else "–"
            tot = prof[(prof.metric == key) & (prof.venue == "all")]
            if not tot.empty:
                r["Media lega"] = m.format(tot.iloc[0].league)
                r["Posizione"] = display_rank(m, tot.iloc[0]["rank"], tot.iloc[0]["n_teams"])
                r[f"Ultime {ctx.settings.form_window}"] = m.format(tot.iloc[0].last_n)
            rows.append(r)
        return pd.DataFrame(rows)
