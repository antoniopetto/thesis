"""Orchestratore: prepara il contesto condiviso, esegue gli agenti e sintetizza per squadra."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..config import AnalysisSettings
from ..data.dataset import LeagueData
from ..stats.metrics import profile_table
from .base import Agent, AgentOutput, Context, Insight
from .markets import MarketsAgent
from .players import PlayersAgent
from .preview import PreviewAgent
from .referee import RefereeAgent
from .team_agents import AttackDefenseAgent, CornersAgent, DisciplineAgent, ResultsAgent, StyleAgent

log = logging.getLogger(__name__)


def default_agents() -> list[Agent]:
    return [
        ResultsAgent(),
        AttackDefenseAgent(),
        CornersAgent(),
        DisciplineAgent(),
        StyleAgent(),
        PlayersAgent(),
        RefereeAgent(),
        MarketsAgent(),
        PreviewAgent(),
    ]


@dataclass
class TeamSummary:
    team: str
    strengths: list[Insight] = field(default_factory=list)
    weaknesses: list[Insight] = field(default_factory=list)
    tendencies: list[Insight] = field(default_factory=list)


@dataclass
class AnalysisResult:
    context: Context
    outputs: dict[str, AgentOutput]
    summaries: dict[str, TeamSummary]

    @property
    def insights(self) -> list[Insight]:
        return [i for o in self.outputs.values() for i in o.insights]

    def team_insights(self, team: str) -> list[Insight]:
        return [i for i in self.insights if i.team == team]


class Orchestrator:
    def __init__(self, agents: list[Agent] | None = None, settings: AnalysisSettings | None = None):
        self.agents = agents or default_agents()
        self.settings = settings or AnalysisSettings()

    def run(self, data: LeagueData) -> AnalysisResult:
        st = self.settings
        prev_profiles = None
        if data.previous is not None:
            prev_profiles = profile_table(data.previous.team_matches, st.shrinkage_matches, st.form_window)
        profiles = profile_table(
            data.team_matches, st.shrinkage_matches, st.form_window, prior=prev_profiles, prior_weight=st.prior_weight
        )
        ctx = Context(data=data, settings=st, profiles=profiles, prev_profiles=prev_profiles)
        # Gli agenti leggono solo il contesto condiviso: possono girare in parallelo.
        with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
            results = list(pool.map(lambda a: self._safe_run(a, ctx), self.agents))
        outputs = {o.agent: o for o in results}
        return AnalysisResult(ctx, outputs, self.synthesize(ctx, outputs))

    @staticmethod
    def _safe_run(agent: Agent, ctx: Context) -> AgentOutput:
        try:
            return agent.run(ctx)
        except Exception as exc:  # un agente rotto non deve bloccare il report
            log.exception("Agente %s fallito", agent.name)
            return AgentOutput(agent.name, agent.role, notes=[f"Errore: {exc}"])

    @staticmethod
    def synthesize(ctx: Context, outputs: dict[str, AgentOutput], top: int = 6) -> dict[str, TeamSummary]:
        """Agente di sintesi: per ogni squadra i punti più rilevanti, ordinati per rilevanza."""
        summaries = {t: TeamSummary(t) for t in ctx.data.teams}
        for o in outputs.values():
            for i in o.insights:
                if i.team not in summaries:
                    continue
                s = summaries[i.team]
                {"forza": s.strengths, "debolezza": s.weaknesses}.get(i.kind, s.tendencies).append(i)
        for s in summaries.values():
            for lst in (s.strengths, s.weaknesses, s.tendencies):
                lst.sort(key=lambda i: -i.score)
            s.strengths, s.weaknesses = s.strengths[:top], s.weaknesses[:top]
            s.tendencies = s.tendencies[: top * 2]
        return summaries
