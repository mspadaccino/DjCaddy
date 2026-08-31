"""Costruzione delle righe cue (frasi e regioni vocali) e loro mappatura sui
marcatori di djay Pro.

Ogni sezione di frase e ogni regione vocale diventano DUE punti indipendenti
(start/end) nella tabella: è la forma più comoda da correggere a mano.
Al momento di scrivere in djay Pro, però, i due banchi disponibili hanno
capienza e semantica diverse (vedi `plan_djay_markers`). Condiviso fra l'app
a singolo brano e la pagina batch, per non duplicare la logica.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import (SECTION_COLORS, VOCAL_END, VOCAL_MARKER_COLORS,
                     VOCAL_START)

PHRASE_START = "phrase_start"
VOCAL_START_KIND = "vocal_start"
VOCAL_END_KIND = "vocal_end"
VOCAL_KINDS = (VOCAL_START_KIND, VOCAL_END_KIND)

# djay Pro: 8 pad hot-cue e 8 slot loop, due banchi indipendenti.
DJAY_SLOTS = 8


def build_cue_rows(sections, vocal_regions, bpm: float | None = None) -> list[dict]:
    """Costruisce le righe cue di default per un brano.

    `sections`: iterabile di dict (o oggetti Section) con start/end/label.
    `vocal_regions`: iterabile di coppie (start, end).

    Una frase è UNA riga sola, con la sua fine come informazione a corredo:
    le sezioni sono contigue, quindi una riga di fine cadrebbe sullo stesso
    istante dell'inizio della successiva e in djay Pro sarebbe un doppione
    (vedi `plan_djay_markers`). Le regioni vocali restano invece due righe,
    perché entrambi gli estremi servono davvero a costruire il loop.

    Ogni riga: id, kind, label, start (secondi), end (solo le frasi), beats
    (solo le frasi: lunghezza in battiti, per riferimento).
    """
    beat_seconds = (60.0 / bpm) if bpm else None
    rows: list[dict] = []
    for i, s in enumerate(sections):
        label = s["label"] if isinstance(s, dict) else s.label
        start = float(s["start"] if isinstance(s, dict) else s.start)
        end = float(s["end"] if isinstance(s, dict) else s.end)
        beats = round((end - start) / beat_seconds, 1) if beat_seconds else None
        rows.append({"id": f"sec{i}", "kind": PHRASE_START, "label": label,
                    "start": start, "end": end, "beats": beats})

    for j, (vs, ve) in enumerate(vocal_regions):
        rows.append({"id": f"vs{j}", "kind": VOCAL_START_KIND, "label": VOCAL_START,
                    "start": float(vs), "end": None, "beats": None})
        rows.append({"id": f"ve{j}", "kind": VOCAL_END_KIND, "label": VOCAL_END,
                    "start": float(ve), "end": None, "beats": None})
    return rows


def is_vocal_row(row: dict) -> bool:
    return row["kind"] in VOCAL_KINDS


def phrase_ends(kinds: dict, starts: dict, analysis_end: dict) -> dict:
    """Fine di ogni frase: l'inizio della frase successiva, perché le sezioni
    sono contigue — così la colonna resta coerente anche dopo che l'utente ha
    spostato un inizio. Per l'ultima frase resta la fine rilevata
    dall'analisi (`analysis_end`, per id). Le righe vocali non hanno una fine
    propria: la loro sta nella riga "Vocal end" gemella.

    In core perché le tabelle cue sono DUE — la pagina Streamlit e quella
    Qt — e la regola della fine deve restare una sola.
    """
    order = sorted((rid for rid, k in kinds.items() if k == PHRASE_START),
                   key=lambda rid: starts[rid])
    return {
        rid: (starts[order[i + 1]] if i + 1 < len(order)
              else analysis_end.get(rid))
        for i, rid in enumerate(order)
    }


def marker_color(kind: str, tag: str) -> str:
    """Il colore di un marcatore, per waveform ed export (stessi ovunque)."""
    if kind in VOCAL_KINDS:
        return VOCAL_MARKER_COLORS.get(tag, "#ffffff")
    base = tag.rsplit(" ", 1)[0] if tag.endswith((" start", " end")) else tag
    return SECTION_COLORS.get(base, "#ffffff")


@dataclass
class DjayPlan:
    """Come le righe della tabella finiscono nei due banchi di djay Pro.

    `cues`: [(row_id, pad, start)] — le frasi, un pad ciascuna.
    `loops`: [(region_id, slot, start, end)] — le regioni vocali, un solo
    slot per regione (il banco loop tiene inizio E fine insieme).
    `slot_label`: {row_id: "Cue 3" / "Loop 2" / ""} per mostrare in tabella
    dove finisce ogni riga ("" = non ci sta).
    `dropped`: id delle righe scartate perché oltre l'ottavo slot.
    `unpaired`: id delle righe vocali rimaste senza gemella (una regione
    vocale ha bisogno di inizio E fine per diventare un loop).
    """
    cues: list[tuple[str, int, float]]
    loops: list[tuple[str, int, float, float]]
    slot_label: dict[str, str]
    dropped: list[str]
    unpaired: list[str]


def plan_djay_markers(rows) -> DjayPlan:
    """Assegna gli slot di djay Pro alle righe, in ordine cronologico.

    Diventano hot cue (pad 1-8, la posizione decide il colore) solo gli
    INIZI di frase: le sezioni sono contigue, quindi la fine di una frase
    cade sullo stesso istante dell'inizio della successiva e scriverle
    entrambe metterebbe due cue sovrapposti, sprecando metà dei pad.
    (L'unica fine che non coincide con nulla è quella dell'ultima frase,
    cioè la fine del brano: come cue non serve.)

    Le regioni vocali diventano invece loop salvati (slot 1-8) e lì servono
    ENTRAMBI gli estremi — ma stanno in un solo slot, non in due pad. I due
    banchi sono indipendenti, quindi 8 frasi e 8 regioni vocali convivono.

    Oltre l'ottavo slot di ciascun banco le righe vengono scartate (le più
    tarde nel brano): djay Pro non ha altri posti dove metterle.

    `rows`: iterabile di dict con id, kind, start.
    """
    rows = list(rows)
    by_id = {r["id"]: r for r in rows}

    cues, loops, slot_label, unpaired = [], [], {}, []

    phrases = sorted((r for r in rows if r["kind"] == PHRASE_START),
                     key=lambda r: r["start"])
    for pad, r in enumerate(phrases[:DJAY_SLOTS]):
        cues.append((r["id"], pad, r["start"]))
        slot_label[r["id"]] = f"Cue {pad + 1}"

    # Le regioni vocali arrivano come coppie di righe "vs{j}"/"ve{j}": una
    # regione diventa un loop solo se ha ancora entrambe le sue righe.
    regions = []
    for r in rows:
        if r["kind"] != VOCAL_START_KIND:
            continue
        end_row = by_id.get("ve" + r["id"][2:])
        if end_row is None or end_row["start"] <= r["start"]:
            unpaired.append(r["id"])
            continue
        regions.append((r, end_row))
    for r in rows:
        if r["kind"] == VOCAL_END_KIND and "vs" + r["id"][2:] not in by_id:
            unpaired.append(r["id"])

    regions.sort(key=lambda pair: pair[0]["start"])
    for slot, (start_row, end_row) in enumerate(regions[:DJAY_SLOTS]):
        loops.append((start_row["id"], slot, start_row["start"], end_row["start"]))
        slot_label[start_row["id"]] = slot_label[end_row["id"]] = f"Loop {slot + 1}"

    for r in rows:
        slot_label.setdefault(r["id"], "")
    # "Scartata" vale solo per una riga che uno slot lo voleva: le fini di
    # frase restano senza slot per scelta, non perché non ci stanno.
    wanted = {r["id"] for r in phrases} | {r["id"] for r in rows if is_vocal_row(r)}
    dropped = [r["id"] for r in rows
               if r["id"] in wanted and not slot_label[r["id"]]
               and r["id"] not in unpaired]
    return DjayPlan(cues, loops, slot_label, dropped, unpaired)
