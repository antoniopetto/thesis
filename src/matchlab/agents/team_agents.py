"""Agenti di squadra: risultati/forma, attacco/difesa, corner, disciplina, stile dell'allenatore."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats.metrics import lookup
from .base import Agent, AgentOutput, Context, Insight

FORM_LETTER = {"W": "V", "D": "N", "L": "P"}


class ResultsAgent(Agent):
    name = "Risultati & forma"
    role = "Classifica, rendimento casa/trasferta, forma recente e confronto punti reali vs attesi."

    def run(self, ctx: Context) -> AgentOutput:
        tm = ctx.data.team_matches
        s = ctx.settings
        out = AgentOutput(self.name, self.role)
        table = (
            tm.groupby("team")
            .agg(
                G=("result", "size"), V=("result", lambda r: (r == "W").sum()),
                N=("result", lambda r: (r == "D").sum()), P=("result", lambda r: (r == "L").sum()),
                GF=("goals_for", "sum"), GS=("goals_against", "sum"), Pt=("points", "sum"),
            )
            .reset_index()
        )
        table["DR"] = table["GF"] - table["GS"]
        if ctx.data.has("xpts"):
            table["xPt"] = tm.groupby("team")["xpts"].sum().reindex(table["team"]).round(1).values
        table = table.sort_values(["Pt", "DR", "GF"], ascending=False).reset_index(drop=True)
        table.insert(0, "Pos", np.arange(1, len(table) + 1))
        out.tables["classifica"] = table.rename(columns={"team": "Squadra"})

        prev = self._previous_table(ctx)
        per_team = {}
        for team, g in tm.sort_values("date").groupby("team"):
            last = g.tail(s.form_window)
            form = "".join(FORM_LETTER[r] for r in last["result"])
            rec = {}
            for venue, label in (("H", "Casa"), ("A", "Trasferta")):
                gv = g[g.venue == venue]
                rec[label] = f"{(gv.result == 'W').sum()}V {(gv.result == 'D').sum()}N {(gv.result == 'L').sum()}P"
            per_team[team] = {"forma": form, **rec}
            if prev is not None:
                if team in prev.index:
                    pr = prev.loc[team]
                    per_team[team]["stagione_scorsa"] = (
                        f"{int(pr.Pos)}° con {int(pr.Pt)} punti ({pr.Pt / pr.G:.2f}/partita)"
                    )
                    if len(g) >= s.min_matches and abs(g["points"].mean() - pr.Pt / pr.G) >= 0.6:
                        now = g["points"].mean()
                        trend = "molto meglio" if now > pr.Pt / pr.G else "molto peggio"
                        out.insights.append(Insight(
                            self.name, "risultati", "tendenza", team=team, score=abs(now - pr.Pt / pr.G) * 2,
                            text=f"Sta facendo {trend} della stagione scorsa: {now:.2f} punti/partita "
                                 f"contro {pr.Pt / pr.G:.2f} (chiuse {int(pr.Pos)}°).",
                        ))
                else:
                    per_team[team]["stagione_scorsa"] = f"neopromossa (non era in {ctx.league_name})"

            ppg_all = g["points"].mean()
            ppg_last = last["points"].mean()
            if len(g) >= s.min_matches + 2 and abs(ppg_last - ppg_all) >= 0.8:
                kind = "forza" if ppg_last > ppg_all else "debolezza"
                trend = "in crescita" if ppg_last > ppg_all else "in calo"
                out.insights.append(Insight(
                    self.name, "risultati", kind, team=team, score=abs(ppg_last - ppg_all) * 2,
                    text=f"Forma {trend}: {ppg_last:.2f} punti/partita nelle ultime {len(last)} ({form}) "
                         f"contro {ppg_all:.2f} di media stagionale.",
                ))
            if ctx.data.has("xpts") and len(g) >= s.min_matches:
                diff = g["points"].sum() - g["xpts"].sum()
                if abs(diff) >= max(3.0, 0.35 * len(g)):
                    if diff > 0:
                        txt = (f"Ha {diff:.1f} punti in più di quelli attesi dagli xG: risultati migliori "
                               "delle prestazioni, possibile calo (attenzione a quote basse).")
                    else:
                        txt = (f"Ha {-diff:.1f} punti in meno di quelli attesi dagli xG: prestazioni migliori "
                               "dei risultati, possibile rimbalzo (possibile valore nelle quote).")
                    out.insights.append(Insight(self.name, "risultati", "tendenza", txt, team=team,
                                                value=float(diff), score=abs(diff) / 2))
        out.tables["per_team"] = per_team
        out.insights += self.metric_insights(ctx, {"risultati"})
        return out


    @staticmethod
    def _previous_table(ctx: Context) -> pd.DataFrame | None:
        prev = ctx.data.previous
        if prev is None:
            return None
        tm = prev.team_matches
        t = tm.groupby("team").agg(G=("points", "size"), Pt=("points", "sum"),
                                   DR=("goals_for", "sum"), GS=("goals_against", "sum"))
        t["DR"] = t["DR"] - t["GS"]
        t = t.sort_values(["Pt", "DR"], ascending=False)
        t["Pos"] = np.arange(1, len(t) + 1)
        return t


class AttackDefenseAgent(Agent):
    name = "Attacco & difesa"
    role = "Produzione offensiva e solidità difensiva: gol, tiri, xG, qualità delle occasioni, tempi di gioco."

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        out.insights = self.metric_insights(ctx, {"attacco", "difesa"})
        tm = ctx.data.team_matches
        s = ctx.settings
        out.tables["per_team"] = {
            t: self.team_metric_table(ctx, t, {"attacco", "difesa"}) for t in ctx.data.teams
        }
        if ctx.data.has("xg_for"):
            for team, g in tm.groupby("team"):
                g = g.dropna(subset=["xg_for"])
                if len(g) < s.min_matches:
                    continue
                fin = (g["goals_for"].sum() - g["xg_for"].sum()) / len(g)
                gk = (g["xg_against"].sum() - g["goals_against"].sum()) / len(g)
                if abs(fin) >= 0.35:
                    txt = (f"Finalizzazione {'sopra' if fin > 0 else 'sotto'} le attese: "
                           f"{fin:+.2f} gol/partita rispetto agli xG "
                           f"({'cinismo o fortuna, tende a rientrare' if fin > 0 else 'spreca molto, i gol potrebbero arrivare'}).")
                    out.insights.append(Insight(self.name, "attacco", "tendenza", txt, team=team,
                                                metric="finishing", value=float(fin), n=len(g), score=abs(fin) * 3))
                if abs(gk) >= 0.35:
                    txt = (f"Subisce {'meno' if gk > 0 else 'più'} gol di quanto concede in xG "
                           f"({gk:+.2f}/partita): {'portiere/difesa in stato di grazia' if gk > 0 else 'difesa punita oltre il dovuto'}.")
                    out.insights.append(Insight(self.name, "difesa", "tendenza", txt, team=team,
                                                metric="gk_overperf", value=float(gk), n=len(g), score=abs(gk) * 3))
        # Squilibrio casa/trasferta in attacco
        for team in ctx.data.teams:
            h, a = lookup(ctx.profiles, team, "goals_for", "H"), lookup(ctx.profiles, team, "goals_for", "A")
            if h is None or a is None or min(h.n, a.n) < 2:
                continue
            if h.value - a.value >= 1.0 or a.value - h.value >= 0.8:
                better = "in casa" if h.value > a.value else "in trasferta"
                out.insights.append(Insight(
                    self.name, "attacco", "tendenza", team=team, score=abs(h.value - a.value),
                    text=f"Molto più pericolosa {better}: {h.value:.2f} gol/partita in casa vs {a.value:.2f} fuori.",
                ))
        return out


class CornersAgent(Agent):
    name = "Calci d'angolo"
    role = "Corner battuti e concessi, casa/trasferta, totale corner nelle partite della squadra."

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        if not ctx.data.has("corners_for"):
            out.notes.append("Dati corner non disponibili.")
            return out
        out.insights = self.metric_insights(ctx, {"corner"})
        out.tables["per_team"] = {t: self.team_metric_table(ctx, t, {"corner"}) for t in ctx.data.teams}
        return out


class DisciplineAgent(Agent):
    name = "Disciplina"
    role = "Falli commessi/subiti e cartellini, per capire squadre e partite 'calde'."

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        if not ctx.data.has("cards_for"):
            out.notes.append("Dati cartellini non disponibili.")
            return out
        out.insights = self.metric_insights(ctx, {"disciplina"})
        out.tables["per_team"] = {t: self.team_metric_table(ctx, t, {"disciplina"}) for t in ctx.data.teams}
        return out


def _tercile(values: pd.Series, team: str) -> int | None:
    """0 = terzo più basso, 2 = terzo più alto."""
    if team not in values.index or values.isna().all():
        return None
    pct = values.rank(pct=True)[team]
    return 0 if pct <= 1 / 3 else (2 if pct > 2 / 3 else 1)


class StyleAgent(Agent):
    name = "Allenatore & stile"
    role = "Stile di gioco dedotto dai dati (pressing, volume e qualità dei tiri, partite aperte o chiuse)."

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        prof = ctx.profiles[ctx.profiles.venue == "all"].pivot(index="team", columns="metric", values="shrunk")
        out.insights = self.metric_insights(ctx, {"stile"})
        rules = [  # (metrica, segno, etichette per terzile basso/medio/alto)
            ("ppda", -1, ("baricentro basso / pressing blando", "pressing medio", "pressing alto")),
            ("shots_for", 1, ("pochi tiri", "volume di tiro nella media", "tanto volume di tiro")),
            ("xg_per_shot", 1, ("molti tiri da fuori/bassa qualità", "qualità tiro nella media",
                                "cerca occasioni pulite (alta qualità tiro)")),
            ("goals_total", 1, ("partite bloccate (pochi gol)", "partite equilibrate", "partite aperte (tanti gol)")),
            ("fouls_for", 1, ("pochi falli", None, "gioco spezzettato/aggressivo (molti falli)")),
        ]
        styles = {}
        for team in ctx.data.teams:
            tags = []
            for metric, sign, labels in rules:
                if metric not in prof:
                    continue
                t = _tercile(sign * prof[metric], team)
                if t is not None and labels[t]:
                    tags.append(labels[t])
            styles[team] = {"allenatore": ctx.data.coaches.get(team), "profilo": tags}
        out.tables["styles"] = styles
        if not ctx.data.coaches:
            out.notes.append("Nomi allenatori non configurati: compila data/manual/coaches.csv.")
        return out
