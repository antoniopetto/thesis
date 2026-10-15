"""Normalizzazione dei nomi squadra tra fonti diverse (football-data, Understat, CSV manuali)."""
from __future__ import annotations

import difflib
import re
import unicodedata

# Alias noti -> nome canonico (quello di football-data.co.uk).
ALIASES = {
    "ac milan": "Milan",
    "inter milan": "Inter",
    "internazionale": "Inter",
    "fc internazionale": "Inter",
    "as roma": "Roma",
    "ss lazio": "Lazio",
    "ssc napoli": "Napoli",
    "hellas verona": "Verona",
    "parma calcio 1913": "Parma",
    "atalanta bc": "Atalanta",
    "juventus fc": "Juventus",
    "us lecce": "Lecce",
    "us cremonese": "Cremonese",
    "us sassuolo": "Sassuolo",
    "como 1907": "Como",
    "pisa sporting club": "Pisa",
    "cagliari calcio": "Cagliari",
    "genoa cfc": "Genoa",
    "torino fc": "Torino",
    "udinese calcio": "Udinese",
    "acf fiorentina": "Fiorentina",
    "bologna fc": "Bologna",
    "spezia calcio": "Spezia",
    "venezia fc": "Venezia",
    "empoli fc": "Empoli",
    "frosinone calcio": "Frosinone",
    "monza": "Monza",
    "ac monza": "Monza",
    "salernitana": "Salernitana",
    "palermo fc": "Palermo",
}


def _simplify(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def canonical(name: str, known: list[str] | None = None) -> str:
    """Restituisce il nome canonico; se `known` è dato prova a ricondurre il nome a uno di quelli."""
    simple = _simplify(name)
    if simple in ALIASES:
        return ALIASES[simple]
    if not known:
        return name.strip()
    by_simple = {_simplify(k): k for k in known}
    if simple in by_simple:
        return by_simple[simple]
    # Uno contenuto nell'altro ("parma calcio" vs "parma").
    for s, k in by_simple.items():
        if s and (s in simple.split() or simple in s.split() or s in simple or simple in s):
            return k
    match = difflib.get_close_matches(simple, list(by_simple), n=1, cutoff=0.6)
    return by_simple[match[0]] if match else name.strip()
