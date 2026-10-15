"""Generazione del report (HTML navigabile + Markdown + export JSON/CSV degli insight)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..agents.orchestrator import AnalysisResult
from ..config import season_label

TEMPLATES = Path(__file__).parent / "templates"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _html_table(df: pd.DataFrame | None, classes: str = "tbl") -> str:
    if df is None or df.empty:
        return ""
    return df.to_html(index=False, border=0, classes=classes, na_rep="–", escape=True)


def _team_view(result: AnalysisResult, team: str) -> dict:
    o = result.outputs
    get = lambda agent, key="per_team": (o[agent].tables.get(key) or {}) if agent in o else {}  # noqa: E731
    standings = o["Risultati & forma"].tables.get("classifica") if "Risultati & forma" in o else None
    pos = None
    if standings is not None:
        row = standings[standings["Squadra"] == team]
        pos = row.iloc[0].to_dict() if not row.empty else None
    s = result.summaries[team]
    return {
        "name": team,
        "slug": slug(team),
        "standing": pos,
        "form": get("Risultati & forma").get(team, {}),
        "style": get("Allenatore & stile", "styles").get(team, {}),
        "strengths": s.strengths,
        "weaknesses": s.weaknesses,
        "tendencies": s.tendencies,
        "tables": [
            ("Attacco e difesa", _html_table(get("Attacco & difesa").get(team))),
            ("Calci d'angolo", _html_table(get("Calci d'angolo").get(team))),
            ("Disciplina", _html_table(get("Disciplina").get(team))),
            ("Frequenze esiti (schedina)", _html_table(get("Mercati").get(team))),
            ("Giocatori chiave (per 90')", _html_table(get("Scout giocatori").get(team))),
        ],
    }


def _previews(result: AnalysisResult) -> list[dict]:
    prev = result.outputs.get("Statistico (anteprime)")
    items = []
    for p in (prev.tables.get("previews", []) if prev else []):
        item = {"home": p["home"], "away": p["away"], "date": pd.Timestamp(p["date"]).strftime("%d/%m/%Y"),
                "home_slug": slug(p["home"]), "away_slug": slug(p["away"]), "rows": p["rows"],
                "has_book": p.get("has_book")}
        if p.get("goals"):
            g = p["goals"]
            item["xg"] = f"{g['exp'][0]:.2f} – {g['exp'][1]:.2f}"
            item["scores"] = ", ".join(f"{s} ({pr * 100:.0f}%)" for s, pr in g["top_scores"][:3])
        if p.get("corners"):
            c = p["corners"]["exp"]
            item["corners"] = f"{c[0]:.1f} – {c[1]:.1f}"
        if p.get("cards"):
            c = p["cards"]
            item["cards"] = f"{c['exp'][0]:.1f} – {c['exp'][1]:.1f}"
            item["referee"] = c.get("referee")
            item["ref_factor"] = c.get("ref_factor")
        item["flags"] = [r for r in p["rows"] if r["flag"]]
        items.append(item)
    return items


def build_context(result: AnalysisResult) -> dict:
    data = result.context.data
    o = result.outputs
    ref = o.get("Arbitri")
    return {
        "league": data.league.name,
        "season": season_label(data.season),
        "generated": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "n_matches": len(data.matches),
        "last_date": data.matches["date"].max().strftime("%d/%m/%Y") if len(data.matches) else "–",
        "sources": data.sources,
        "warnings": data.warnings + [n for out in o.values() for n in out.notes],
        "agents": [(out.agent, out.role, len(out.insights)) for out in o.values()],
        "standings": _html_table(o["Risultati & forma"].tables.get("classifica"), "tbl lead2") if "Risultati & forma" in o else "",
        "referees": _html_table(ref.tables.get("referees"), "tbl lead1") if ref else "",
        "referees_next": _html_table(pd.DataFrame(ref.tables.get("upcoming") or []), "tbl lead3") if ref else "",
        "league_insights": sorted((i for i in result.insights if i.team is None), key=lambda i: -i.score),
        "teams": [_team_view(result, t) for t in data.teams],
        "previews": _previews(result),
    }


def render_html(result: AnalysisResult) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    env.filters["pct"] = lambda v: "–" if v is None else f"{v * 100:.0f}%"
    return env.get_template("report.html.j2").render(**build_context(result))


def render_markdown(result: AnalysisResult) -> str:
    ctx = build_context(result)
    lines = [f"# Report squadre — {ctx['league']} {ctx['season']}", "",
             f"Partite analizzate: {ctx['n_matches']} (fino al {ctx['last_date']}). Fonti: {', '.join(ctx['sources'])}.", ""]
    for w in ctx["warnings"]:
        lines.append(f"> ⚠️ {w}")
    for t in ctx["teams"]:
        lines += ["", f"## {t['name']}", ""]
        if t["form"]:
            lines.append(f"Forma: **{t['form'].get('forma', '')}** · Casa {t['form'].get('Casa', '')} · "
                         f"Trasferta {t['form'].get('Trasferta', '')}"
                         + (f" · Stagione scorsa: {t['form']['stagione_scorsa']}" if t['form'].get('stagione_scorsa') else ""))
        style = t["style"]
        if style:
            coach = f"{style.get('allenatore')} — " if style.get("allenatore") else ""
            lines.append(f"Stile: {coach}{', '.join(style.get('profilo', []))}")
        for title, items in (("Punti di forza", t["strengths"]), ("Punti deboli", t["weaknesses"]),
                             ("Tendenze utili", t["tendencies"])):
            if items:
                lines += ["", f"**{title}**", ""] + [f"- {i.text}" for i in items]
    return "\n".join(lines) + "\n"


def write_reports(result: AnalysisResult, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = result.context.data
    stem = f"{data.league.key}_{data.season}"
    paths = {
        "html": out_dir / f"report_{stem}.html",
        "markdown": out_dir / f"report_{stem}.md",
        "insights_json": out_dir / f"insights_{stem}.json",
        "profiles_csv": out_dir / f"profiles_{stem}.csv",
    }
    paths["html"].write_text(render_html(result), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(result), encoding="utf-8")
    paths["insights_json"].write_text(
        json.dumps([i.to_dict() for i in result.insights], ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    result.context.profiles.to_csv(paths["profiles_csv"], index=False)
    return paths
