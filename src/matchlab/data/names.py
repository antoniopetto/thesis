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


def _player_key(name: str) -> str:
    return _simplify(name)


def match_players(
    left: list[str],
    right: list[str],
    cutoff: float = 0.82,
    left_minutes: dict[str, float] | None = None,
    right_minutes: dict[str, float] | None = None,
) -> dict[str, str]:
    """Abbina nomi giocatore tra due fonti (stessa squadra).

    1) nome identico (senza accenti); 2) parole del nome in comune (es. "Bremer" /
    "Gleison Bremer"), a parità si sceglie il giocatore con minuti più simili;
    3) somiglianza del testo.
    """
    lm, rm = left_minutes or {}, right_minutes or {}
    out: dict[str, str] = {}
    rmap = {_player_key(r): r for r in right}
    used: set[str] = set()
    pending = []
    for name in left:
        k = _player_key(name)
        if k in rmap and rmap[k] not in used:
            out[name] = rmap[k]
            used.add(rmap[k])
        else:
            pending.append(name)

    def tokens(key: str) -> set[str]:
        return {t for t in key.split() if len(t) >= 4}

    still = []
    for name in pending:
        tk = tokens(_player_key(name))
        cands = [r for rk, r in rmap.items() if r not in used and tk & tokens(rk)]
        if not cands:
            still.append(name)
            continue
        if len(cands) > 1 and name in lm:
            cands.sort(key=lambda r: abs(rm.get(r, 1e9) - lm[name]))
            # Due candidati con minuti quasi uguali: troppo ambiguo, meglio non abbinare.
            if abs(rm.get(cands[0], 1e9) - lm[name]) == abs(rm.get(cands[1], 1e9) - lm[name]):
                still.append(name)
                continue
        elif len(cands) > 1:
            still.append(name)
            continue
        out[name] = cands[0]
        used.add(cands[0])
    for name in still:
        free = {rk: r for rk, r in rmap.items() if r not in used}
        m = difflib.get_close_matches(_player_key(name), list(free), n=1, cutoff=cutoff)
        if m:
            out[name] = free[m[0]]
            used.add(free[m[0]])
    return out
