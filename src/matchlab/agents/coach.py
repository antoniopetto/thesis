"""Agente Allenatore: chi guida la squadra, da quando, con che moduli e con che stile di gioco.

Lo stile si misura sulle partite dirette dall'allenatore attuale in questa squadra
(stagione in corso + precedente), con le statistiche di gioco SofaScore e il PPDA Understat,
e si confronta con le altre squadre del campionato. Se l'allenatore è nuovo, si confronta
la squadra prima/dopo e si mostra come giocava la sua squadra precedente.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Agent, AgentOutput, Context, Insight


@dataclass(frozen=True)
class StyleMetric:
    key: str
    label: str
    fmt: str = "{:.1f}"
    pct: bool = False


STYLE_METRICS = [
    StyleMetric("possession", "Possesso palla %", "{:.0f}"),
    StyleMetric("passes", "Passaggi a partita", "{:.0f}"),
    StyleMetric("pass_acc", "Precisione passaggi", pct=True),
    StyleMetric("long_share", "Lanci lunghi riusciti (% dei passaggi)", pct=True),
    StyleMetric("crosses", "Cross riusciti a partita"),
    StyleMetric("final_third", "Ingressi nell'ultimo terzo"),
    StyleMetric("box_touches", "Tocchi in area avversaria"),
    StyleMetric("big_chances", "Grandi occasioni create"),
    StyleMetric("out_box_share", "Tiri da fuori area (% dei tiri)", pct=True),
    StyleMetric("ppda", "PPDA (più basso = pressa di più)"),
    StyleMetric("recoveries", "Palloni recuperati"),
    StyleMetric("def_actions", "Contrasti + intercetti"),
    StyleMetric("clearances", "Respinte difensive"),
    StyleMetric("xg_for", "xG a partita", "{:.2f}"),
    StyleMetric("xg_against", "xG concessi a partita", "{:.2f}"),
    StyleMetric("sh_share", "Gol segnati nel 2° tempo (% dei gol)", pct=True),
]

# (metrica, terzile basso, terzile alto): etichette dell'identikit.
STYLE_TAGS = [
    ("possession", "lascia il pallone all'avversario", "squadra di possesso"),
    ("long_share", "costruzione corta, palla a terra", "gioco diretto, molti lanci lunghi"),
    ("crosses", None, "attacca molto sulle fasce (tanti cross)"),
    ("box_touches", "entra poco in area", "occupa molto l'area avversaria"),
    ("out_box_share", "cerca il tiro da dentro l'area", "tira molto da fuori"),
    ("ppda", "pressing alto e aggressivo", "blocco basso, aspetta l'avversario"),
    ("clearances", None, "difesa spesso schiacciata (molte respinte)"),
    ("big_chances", "crea poche occasioni nitide", "crea molte occasioni nitide"),
]


def style_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Metriche di stile a partita (una riga per squadra-partita)."""
    out = pd.DataFrame(index=df.index)
    get = lambda c: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)  # noqa: E731
    out["possession"] = get("possession_for")
    out["passes"] = get("passes_for")
    out["pass_acc"] = get("acc_passes_for") / get("passes_for")
    out["long_share"] = get("long_balls_for") / get("acc_passes_for")
    out["crosses"] = get("crosses_for")
    out["final_third"] = get("final_third_for")
    out["box_touches"] = get("box_touches_for")
    out["big_chances"] = get("big_chances_for")
    out["out_box_share"] = get("shots_out_box_for") / (get("shots_in_box_for") + get("shots_out_box_for"))
    out["ppda"] = get("ppda")
    out["recoveries"] = get("recoveries_for")
    out["def_actions"] = get("tackles_for") + get("interceptions_for")
    out["clearances"] = get("clearances_for")
    out["xg_for"] = get("xg_for")
    out["xg_against"] = get("xg_against")
    out["_sh_goals"] = get("sh_goals_for")
    out["_goals"] = get("goals_for")
    return out


def summarize(rows: pd.DataFrame) -> pd.Series:
    s = rows.drop(columns=["_sh_goals", "_goals"]).mean()
    s["sh_share"] = rows["_sh_goals"].sum() / rows["_goals"].sum() if rows["_goals"].sum() else np.nan
    return s


def current_coach(tm: pd.DataFrame, team: str) -> tuple[str | None, str | None]:
    g = tm[(tm["team"] == team)].sort_values("date")
    if "coach" not in g.columns or g["coach"].dropna().empty:
        return None, None
    last = g.dropna(subset=["coach"]).iloc[-1]
    return last["coach"], last.get("coach_id")


class CoachAgent(Agent):
    name = "Allenatore & stile"
    role = "Allenatore, moduli, identikit di gioco rispetto al campionato, cambi di panchina."

    def run(self, ctx: Context) -> AgentOutput:
        out = AgentOutput(self.name, self.role)
        data = ctx.data
        cur = data.team_matches.assign(stagione="corrente")
        prev = data.previous.team_matches.assign(stagione="precedente") if data.previous is not None else None
        pooled = pd.concat([cur, prev], ignore_index=True) if prev is not None else cur
        if "coach" not in pooled.columns or pooled["coach"].isna().all():
            out.notes.append("Allenatori per partita non disponibili (SofaScore): stile calcolato su tutta la stagione.")
            pooled["coach"] = None
        pooled = pooled.sort_values("date")
        style = style_rows(pooled)
        pooled = pd.concat([pooled, style.add_prefix("st_")], axis=1)

        # Profilo di ogni squadra con il suo allenatore attuale (per i confronti di lega).
        profiles, info = {}, {}
        for team in data.teams:
            coach, coach_id = current_coach(cur, team)
            # Il nome manuale (data/manual/coaches.csv) serve solo se SofaScore non lo fornisce.
            coach = coach or data.coaches.get(team)
            g = pooled[pooled["team"] == team]
            under = g[g["coach"] == coach] if coach and g["coach"].notna().any() else g[g["stagione"] == "corrente"]
            # Mandato: partite consecutive più recenti con questo allenatore.
            since, prev_coach = None, None
            if coach and g["coach"].notna().any():
                changed = g[g["coach"] != coach]
                last_other = changed["date"].max() if not changed.empty else None
                tenure = g[g["date"] > last_other] if last_other is not None else g
                under = tenure
                since = tenure["date"].min()
                if not changed.empty:
                    prev_coach = changed.sort_values("date").iloc[-1]["coach"]
            profiles[team] = summarize(under[[f"st_{c}" for c in style.columns]].rename(columns=lambda c: c[3:]))
            elsewhere = None
            if coach and prev is not None and "coach" in prev.columns:
                other = prev[(prev["coach"] == coach) & (prev["team"] != team)]
                if not other.empty:
                    elsewhere = other["team"].mode().iloc[0]
            formations = under["formation"].dropna().value_counts() if "formation" in under.columns else pd.Series()
            info[team] = {"coach": coach, "coach_id": coach_id, "since": since, "n": len(under),
                          "prev_coach": prev_coach, "elsewhere": elsewhere, "formations": formations,
                          "rows": under}

        league = pd.DataFrame(profiles).T
        styles, tables = {}, {}
        for team in data.teams:
            i = info[team]
            p = league.loc[team]
            pct = league.rank(pct=True).loc[team]
            rows = []
            for m in STYLE_METRICS:
                if m.key not in league.columns or pd.isna(p.get(m.key)):
                    continue
                rank = int(league[m.key].rank(ascending=False, method="min")[team])
                fmt = (lambda v: f"{v * 100:.0f}%") if m.pct else (lambda v, f=m.fmt: f.format(v))
                rows.append({"Indicatore": m.label, "Valore": fmt(p[m.key]), "Media lega": fmt(league[m.key].mean()),
                             "Posizione": f"{rank}° più alto su {league[m.key].notna().sum()}"})
            tables[team] = pd.DataFrame(rows)
            tags = []
            for key, low, high in STYLE_TAGS:
                if key in pct and not pd.isna(pct[key]):
                    if pct[key] <= 1 / 3 and low:
                        tags.append(low)
                    elif pct[key] > 2 / 3 and high:
                        tags.append(high)
            if "sh_share" in pct and not pd.isna(p.get("sh_share")):
                if p["sh_share"] >= 0.65:
                    tags.append("cresce nel secondo tempo")
                elif p["sh_share"] <= 0.4:
                    tags.append("parte forte, segna soprattutto nel primo tempo")
            fm = i["formations"]
            formations_txt = ", ".join(f"{f} ({c / fm.sum() * 100:.0f}%)" for f, c in fm.head(3).items()) if len(fm) else None
            profile = data.coach_info.get(team, {}) if hasattr(data, "coach_info") else {}
            styles[team] = {
                "allenatore": i["coach"], "profilo": tags, "dal": i["since"], "partite": i["n"],
                "moduli": formations_txt, "precedente": i["prev_coach"], "altrove": i["elsewhere"],
                "modulo_preferito": profile.get("preferred_formation"),
                "carriera": profile,
            }
            if i["coach"]:
                txt = f"{i['coach']}: " + (", ".join(tags) if tags else "stile vicino alla media del campionato")
                if formations_txt:
                    txt += f". Moduli: {formations_txt}"
                out.insights.append(Insight(self.name, "stile", "info", txt + ".", team=team, score=0.1))
            # Cambio di allenatore: prima/dopo.
            if i["prev_coach"]:
                g = pooled[pooled["team"] == team]
                before = g[(g["coach"] == i["prev_coach"]) & (g["date"] < (i["since"] or g["date"].max()))]
                after = i["rows"]
                if len(before) >= 4 and len(after) >= 3:
                    cmp = []
                    for key, label in (("points", "punti/partita"), ("xg_for", "xG"), ("xg_against", "xG concessi"),
                                       ("possession_for", "possesso"), ("ppda", "PPDA")):
                        if key in g.columns and before[key].notna().any() and after[key].notna().any():
                            cmp.append({"Indicatore": label, f"Con {i['prev_coach']}": round(before[key].mean(), 2),
                                        f"Con {i['coach']}": round(after[key].mean(), 2)})
                    styles[team]["cambio"] = pd.DataFrame(cmp)
            if i["elsewhere"] and prev is not None:
                other = prev[(prev["coach"] == i["coach"]) & (prev["team"] == i["elsewhere"])]
                o = summarize(style_rows(other))
                styles[team]["altrove_stile"] = (
                    f"L'anno scorso al {i['elsewhere']} ({len(other)} partite): possesso {o['possession']:.0f}%, "
                    f"PPDA {o['ppda']:.1f}, xG {o['xg_for']:.2f} fatti / {o['xg_against']:.2f} concessi a partita."
                    if not pd.isna(o.get("possession")) else None
                )
        out.tables["styles"] = styles
        out.tables["per_team"] = tables
        return out
