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

# rekordbox: 8 pad hot-cue (A..H) e basta — ma i memory cue non hanno
# numero, quindi quello che non entra nei pad non si perde, ci va accanto.
RB_HOT_CUES = 8


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


@dataclass
class RekordboxMarker:
    """Un marcatore da scrivere in rekordbox.

    `pad` 1-8 è un hot cue (i pad A..H); `pad` None è un memory cue, che
    non ha numero e di cui rekordbox non ha un tetto. `end` distingue un
    cue da un loop: se c'è, il marcatore è un loop da `start` a `end`.
    """
    row_id: str
    start: float
    end: float | None
    label: str
    pad: int | None


@dataclass
class RekordboxPlan:
    """Come le righe della tabella finiscono nella libreria di rekordbox.

    `slot_label`: {row_id: "Hot cue A" / "Memory cue" / "Loop"} per la
    colonna della tabella. `unpaired`: righe vocali rimaste senza gemella.
    Non c'è una lista di scartati come per djay: in rekordbox non si scarta
    niente, oltre l'ottavo pad si continua in memory cue.
    """
    markers: list[RekordboxMarker]
    slot_label: dict[str, str]
    unpaired: list[str]


def plan_rekordbox_markers(rows) -> RekordboxPlan:
    """Assegna i marcatori di rekordbox alle righe, in ordine cronologico.

    Gli INIZI di frase diventano cue: i primi otto sui pad A..H, gli altri
    memory cue. Le fini di frase no, per la stessa ragione di djay: le
    sezioni sono contigue, quindi la fine di una cade sull'inizio della
    successiva e sarebbe un doppione.

    Le regioni vocali diventano loop di memoria — servono entrambi gli
    estremi, e uno solo non è una regione. Loop e cue NON si contendono i
    pad qui, perché i loop nascono già memory: in rekordbox i pad sono otto
    in tutto, e spenderli in loop vorrebbe dire non avere più dove mettere
    le frasi.

    A differenza di `plan_djay_markers` non scarta niente: è il vantaggio
    dei memory cue, che non hanno un numero da esaurire.

    `rows`: iterabile di dict con id, kind, start, label.
    """
    rows = list(rows)
    by_id = {r["id"]: r for r in rows}
    markers: list[RekordboxMarker] = []
    slot_label: dict[str, str] = {}
    unpaired: list[str] = []

    phrases = sorted((r for r in rows if r["kind"] == PHRASE_START),
                     key=lambda r: r["start"])
    for i, r in enumerate(phrases):
        pad = i + 1 if i < RB_HOT_CUES else None
        markers.append(RekordboxMarker(r["id"], r["start"], None,
                                       r.get("label") or r.get("tag") or "",
                                       pad))
        slot_label[r["id"]] = (f"Hot cue {chr(ord('A') + i)}" if pad
                               else "Memory cue")

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
    for start_row, end_row in regions:
        markers.append(RekordboxMarker(
            start_row["id"], start_row["start"], end_row["start"],
            start_row.get("label") or start_row.get("tag") or "", None))
        slot_label[start_row["id"]] = slot_label[end_row["id"]] = "Loop"

    for r in rows:
        slot_label.setdefault(r["id"], "")
    markers.sort(key=lambda m: m.start)
    return RekordboxPlan(markers, slot_label, unpaired)
