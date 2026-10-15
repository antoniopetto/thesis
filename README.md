# matchlab — analisi multi-agente delle squadre di calcio

Fase 1 del progetto: per ogni squadra un **report** con punti di forza, punti deboli e
tendenze utili (gol, corner, cartellini, giocatori), da usare per scegliere cosa giocare
su Sisal e chi schierare su Sorare. La struttura per le **distribuzioni di probabilità**
(fase 2) è già presente, ma va considerata sperimentale.

## Uso rapido

```bash
pip install -e .
python -m matchlab report --league serie_a --season 2026      # 2026 = stagione 2026/27
```

Il report viene scritto in `reports/`:

| File | Contenuto |
|---|---|
| `report_<lega>_<anno>.html` | Report navigabile: classifica, arbitri, prossime partite, una scheda per squadra |
| `report_<lega>_<anno>.md` | Lo stesso in Markdown (solo forza/debolezze/tendenze) |
| `insights_<lega>_<anno>.json` | Tutte le segnalazioni degli agenti, strutturate (input per la fase 2) |
| `profiles_<lega>_<anno>.csv` | Tutte le metriche per squadra × casa/fuori, con media lega, z-score, posizione |

Taratura del modello di previsione (da rifare ogni tanto, ad es. a fine stagione):

```bash
python -m matchlab backtest --train 2024 --test 2025   # tara sul 2024/25, verifica sul 2025/26
```

Salva i parametri migliori e i risultati della verifica in `data/model_params.json`, che il report
usa automaticamente (sezione "Affidabilità del modello" nelle anteprime).

Opzioni utili: `--offline` (usa la cache), `--matches-csv`/`--fixtures-csv` (file già scaricati),
`--no-understat`, `--min-matches`, `--z` (sensibilità delle segnalazioni).

## Fonti dati

| Fonte | Cosa dà | Note |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | risultati, tiri, corner, falli, cartellini, quote medie, calendario | gratuita; per la Serie A **manca l'arbitro** |
| [understat.com](https://understat.com) | xG/xGA, PPDA (pressing), deep, xPts, giocatori (xG, xA, tiri, passaggi chiave, cartellini) | gratuita, solo top-5 campionati |
| [SofaScore](https://www.sofascore.com) (API pubblica non ufficiale) | arbitro di ogni partita (anche designazioni della prossima giornata), statistiche di carriera arbitri, calendario e quote pre-partita (1X2, Gol/NoGol, Over/Under gol, corner, cartellini), indisponibili (infortuni, squalifiche, in dubbio), statistiche giocatori (dribbling, contrasti, intercetti, falli fatti/subiti, cross, grandi occasioni create, voto medio) | il primo download della stagione precedente richiede alcuni minuti, poi resta in cache |
| `data/manual/*.csv` | allenatori, designazioni arbitrali, statistiche giocatori extra (dribbling, contrasti, falli...) | da compilare o esportare (SofaScore, FBref, Opta se disponibile) |

I download vengono salvati in `data/cache/` (12 ore), così si può rigenerare il report offline.

## Architettura

```
fonti dati ──► dataset (una riga per squadra per partita) ──► profili statistici (shrinkage, z-score)
                                                                   │
                       ┌────────────────── Orchestratore ──────────┴───────────────┐
                       ▼            ▼          ▼          ▼         ▼        ▼      ▼
                  Risultati   Attacco &   Corner   Disciplina  Allenatore  Scout  Arbitri  Mercati  Statistico
                  & forma     difesa                           & stile   giocatori                (anteprime)
                       └──────────────────────────► Sintesi per squadra ──► Report HTML/MD/JSON
```

| Agente | Ruolo |
|---|---|
| Risultati & forma | classifica, casa/trasferta, forma recente, punti reali vs attesi (xPts) |
| Attacco & difesa | gol, tiri, xG, qualità dei tiri, gol per tempo, finalizzazione sopra/sotto gli xG |
| Calci d'angolo | corner battuti/concessi/totali |
| Disciplina | falli e cartellini fatti/subiti |
| Allenatore & stile | profilo di gioco dai dati (pressing, volume/qualità tiri, partite aperte/chiuse) |
| Scout giocatori | statistiche per 90' e percentili di lega (tiri, xG, xA, gialli, dribbling...) |
| Arbitri | severità per arbitro su stagione in corso + precedente, "fattore cartellini", designazioni della prossima giornata |
| Mercati | frequenze storiche: 1X2, Over/Under, Gol/NoGol, corner, cartellini |
| Statistico (anteprime) | **fase 2**: gol/corner/cartellini attesi e probabilità per la prossima giornata, a confronto con le quote senza margine; gol attesi corretti per gli assenti in base al loro peso offensivo (quota di xG+xA) |

**Stagione precedente.** Viene caricata automaticamente: compare come colonna "Stag. scorsa" e fa da
punto di partenza delle medie (media di lega + 60% dello scarto della squadra l'anno prima; per le
neopromosse solo la media di lega).

**Regole di segnalazione.** Una metrica diventa "punto di forza/debolezza" quando la squadra si
discosta di almeno `z` deviazioni standard (default 1) dalle altre, calcolando le medie con
*shrinkage*: a ogni squadra si aggiungono 4 partite "di media campionato", così con poche
giornate i valori estremi non vengono sopravvalutati. Una frequenza di mercato è "tendenza"
se è ≥70% o ≤30% e lontana almeno 15 punti dalla media di lega.

**Aggiungere un agente.** Si estende `matchlab.agents.base.Agent`, si implementa `run(ctx)` che
restituisce `AgentOutput` con `Insight` strutturati, e lo si aggiunge a `default_agents()`.
Così si potranno inserire anche agenti LLM (notizie, formazioni probabili, infortuni).

## Modello di previsione e backtest

`matchlab/stats/model.py` contiene un unico modello usato sia dal report sia dal backtest:
attesi = livello di lega × (attacco squadra × difesa avversaria)^esponente, con forze calcolate
con shrinkage verso la stagione precedente; Poisson con correzione Dixon-Coles per i gol,
binomiale negativa per corner e cartellini, fattore arbitro per i cartellini.

`matchlab/stats/backtest.py` lo verifica **giornata per giornata senza usare dati futuri** e lo
confronta con le quote di chiusura (senza margine) e con una previsione ingenua. I parametri sono
scelti su una stagione e misurati sulla successiva, per evitare risultati "gonfiati".

## Prossimi passi

1. Agenti LLM per notizie e formazioni probabili (dati non numerici).
2. Effetto delle assenze anche in difesa (portiere, centrali) e formazioni ufficiali un'ora prima.
3. Agente Trader (Sisal: valore atteso, gestione bankroll) e Manager Sorare (punteggio atteso, formazione ottima).

## Test

```bash
pip install -e '.[dev]'
pytest
```
I test usano dati sintetici nel formato delle fonti reali.
