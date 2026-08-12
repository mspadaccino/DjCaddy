"""Scrittura di hot cue nel database live di djay Pro (Mac, versione App Store).

Reverse engineering fatto per diff mirati (before/after, un solo cambiamento
alla volta) sul file reale dell'utente, con autorizzazione esplicita. Il
database (`~/Music/djay/djay Media Library.djayMediaLibrary/MediaLibrary.db`)
è YapDatabase (SQLite: tabella `database2`, colonne collection/key/data). Ogni
traccia ha una riga in `mediaItemUserData` (collection) il cui campo `data` è
un blob binario proprietario (firma `TSAF`) contenente, fra l'altro, l'array
`cuePoints`.

SCOPERTA CHIAVE (confermata su un test reale con tutti e 8 i pad, tempi
incrociati uno per uno con l'utente): il colore/pad di un hot cue in djay
Pro è determinato dalla POSIZIONE nell'array cuePoints (0-indicizzata): il
1° elemento è il pad 1/rosso, il 2° il pad 2/arancio, ... fino all'8°
pad 8/viola (vedi PAD_COLORS). djay Pro pre-alloca posizioni "placeholder"
(time=-1.0/endTime=-1.0) per i pad non ancora usati, così che le posizioni
successive restino agli indici giusti; scrive sempre `number` = posizione+1.

FORMATO TSAF
------------
Il blob NON è una sequenza di byte rattoppabile: è un formato serializzato
con stato globale. Le tre proprietà che lo rendono tale (ognuna verificata
byte per byte: `_encode` rigenera da zero tutti e 8 i campioni reali
catturati da djay Pro, identici al bit):

1. TABELLA DELLE STRINGHE. Ogni stringa letterale (`\\x08<testo>\\x00`)
   entra in una tabella nell'ordine in cui compare; ogni occorrenza
   successiva della stessa stringa è scritta come riferimento
   `\\x05<indice>` (indice 0-based nella tabella). Inserire o togliere una
   sola stringa fa scalare TUTTI gli indici successivi: i riferimenti nella
   coda del blob (nomi dei campi `playCount`, `timestampIdentifier`, ...)
   punterebbero a stringhe sbagliate.

1-bis. TABELLA DEGLI OGGETTI. Allo stesso modo ogni oggetto, collezione o
   blob binario riceve un indice nell'ordine in cui COMINCIA, e `\\x02
   <indice>` è un riferimento a un oggetto già serializzato. Il campo
   `timestampIdentifier` è proprio questo: un puntatore all'oggetto
   `ADCAudioAlignmentFingerprint`. Vale la stessa trappola delle stringhe,
   con conseguenze peggiori: aggiungere dei cue sposta l'impronta audio più
   avanti nella numerazione, e un riferimento non aggiornato finisce su un
   `ADCCuePoint`. djay Pro allora chiama un metodo dell'impronta su un cue
   point e si rifiuta di caricare il brano
   ("-[ADCCuePoint fingerprintFloatsArray]: unrecognized selector").
   È anche la spiegazione del fatto che `timestampIdentifier` "cambia a
   ogni modifica dei cue" (osservato 4 -> 8 -> 10 su tre catture reali):
   non è un checksum, è un puntatore che djay rinumera.

2. ALLINEAMENTO A 4 BYTE. Ogni scalare multi-byte (float32, uint32, e la
   lunghezza/il contenuto dei blob binari) è allineato a 4 byte rispetto
   all'INIZIO del blob, con byte 0x00 di riempimento. Spostare qualunque
   cosa di un numero di byte non multiplo di 4 invalida il riempimento di
   tutto ciò che segue.

3. CONTATORI IN TESTA. Offset 8 (uint64 LE) = numero di oggetti
   serializzati (oggetti `\\x2b`, collezioni, blob binari); offset 16
   (uint32 LE) = numero di stringhe distinte nella tabella del punto 1.
   Sono le dimensioni con cui il decoder pre-alloca le due tabelle.

Un blob che viola anche solo una di queste tre regole non è leggibile: il
decoder di djay Pro va fuori sincrono e l'app rigenera il record da zero,
scartando i cue scritti dall'esterno. Per questo qui si fa sempre
parse completo -> modifica dell'albero -> riserializzazione completa, mai
una patch di byte in-place.

Tag incontrati nei dati reali (`_Parser`): 0x08 stringa letterale,
0x05 riferimento a stringa, 0x2b oggetto (nome classe + coppie
valore/nome, chiuso da 0x00), 0x0a/0x0b/0x1a collezioni (conteggio uint32
+ elementi), 0x13 float32, 0x0f intero su 1 byte, 0x02 intero su 1 byte
(usato per `timestampIdentifier`), 0x15 blob binario, 0x2c-0x2f valori
interi brevi codificati nel tag stesso (djay usa 0x2d per `number` = 1).

Struttura di un cue: oggetto di classe `ADCCuePoint` con i campi, in
quest'ordine, `time` (float32), `endTime` (float32, -1.0 = non impostato),
opzionalmente `comment` (stringa vuota), `number` (intero). Il campo
`comment` NON distingue un cue reale da un placeholder — nei dati reali
djay Pro scrive cue veri sia con sia senza (vedi `zztop_8cues.bin`, pad
5-8 senza commento, resi correttamente nella UI, e `userdata_after.bin`,
3 cue veri tutti senza commento). L'unico discriminante è `time`: -1.0
significa slot vuoto. Qui i cue vengono scritti senza `comment`, come
nell'esempio reale più recente.

Il campo `loop` (controllo Loop in cima al mixer) NON risulta scritto in
questo blob affatto (verificato: un loop reale impostato dall'utente non
ha lasciato traccia) — quindi endTime resta sempre -1.0 per tutto ciò che
scrive Wavecut.

Non tocca MAI il database live direttamente senza: (1) un backup fresco del
bundle .djayMediaLibrary, (2) una verifica che rilegge il nuovo blob e
conferma che tutto il resto dell'albero è invariato e i cue hanno i valori
attesi.
"""

from __future__ import annotations

import shutil
import sqlite3
import struct
import subprocess
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"TSAF"
_HEADER_SIZE = 20
_CUE_CLASS = b"ADCCuePoint"
_CUE_FIELD = b"cuePoints"
_LOOP_CLASS = b"ADCLoopRegion"
_LOOP_FIELD = b"loopRegions"
_CLOUD_KEYS_FIELD = b"userChangedCloudKeys"
_ARRAY_TAG = 0x0A  # tag di collezione usato da djay Pro per cuePoints/loopRegions

# Ordine dei campi nella radice osservato sulle tracce reali: un campo nuovo
# va inserito subito dopo l'ultimo che lo precede in questo elenco.
_FIELD_ORDER = [
    b"uuid", _CLOUD_KEYS_FIELD, b"titleIDs", b"startPoint", _CUE_FIELD, _LOOP_FIELD,
]
_MARKER_FIELDS = (_CUE_FIELD, _LOOP_FIELD)

DEFAULT_LIBRARY_PATH = (
    Path.home() / "Music" / "djay" / "djay Media Library.djayMediaLibrary"
)


# Palette fissa dei pad hot-cue di djay Pro (confermata dall'utente sulla
# UI reale): l'indice nella lista è la posizione 0-based nell'array
# cuePoints, cioè pad on-screen "N" (1-8) -> posizione N-1.
PAD_COLORS = [
    "red",     # pad 1
    "orange",  # pad 2
    "blue",    # pad 3
    "amber",   # pad 4
    "green",   # pad 5
    "magenta", # pad 6
    "teal",    # pad 7
    "purple",  # pad 8
]


@dataclass
class CuePoint:
    time: float
    pad: int                 # posizione 0-7 nell'array cuePoints: determina il colore, vedi PAD_COLORS
    end_time: float = -1.0   # -1.0 = non impostato


@dataclass
class LoopRegion:
    """Un loop salvato (pagina "Loops" dei pad, banco separato dagli hot cue).

    ATTENZIONE: l'array `loopRegions` funziona al contrario di `cuePoints` —
    è denso (niente placeholder) ed è il campo `number` a dire in quale
    slot va il loop, non la posizione. Qui `slot` è 0-based, `number` = slot+1.
    """
    start: float
    end: float
    slot: int                # 0-7


@dataclass
class WriteResult:
    ok: bool
    message: str
    track_uuid: str | None = None
    backup_path: Path | None = None
    cues_before: list[CuePoint] = field(default_factory=list)
    cues_after: list[CuePoint] = field(default_factory=list)
    loops_before: list[LoopRegion] = field(default_factory=list)
    loops_after: list[LoopRegion] = field(default_factory=list)


class DjayWriteError(Exception):
    """Errore che deve fermare l'operazione senza scrivere nulla."""


# --------------------------------------------------------------------------
# Formato TSAF: albero, parser, encoder
# --------------------------------------------------------------------------
#
# I nodi dell'albero sono tuple, con il primo elemento a fare da tag:
#   ("str", bytes)              stringa (i riferimenti sono già risolti qui)
#   ("f32", float)              float32
#   ("int", tag: int, val: int) intero: `tag` conserva la codifica originale
#                               (0x0f/0x02 = valore nel byte successivo,
#                                0x2c-0x2f = valore codificato nel tag)
#   ("data", bytes)             blob binario
#   ("coll", tag: int, [nodi])  collezione
#   ("obj", classe, [(nome, valore)])   oggetto con nome di classe
#   ("struct", [(nome, valore)])        struttura anonima a campi contati
#   ("objref", nodo)            riferimento a un oggetto già serializzato:
#                               tiene il NODO, non l'indice, così il numero
#                               viene ricalcolato in scrittura

# Valori senza payload: il tag stesso è il valore (0x0e compare come
# `useStraightBeats`, quindi è un booleano; 0x2c-0x2f sono interi brevi,
# djay usa 0x2d per `number` = 1). Qui il tag viene conservato tale e quale.
_INLINE_VALUE_TAGS = (0x0E, 0x2C, 0x2D, 0x2E, 0x2F)
_COLLECTION_TAGS = (0x0A, 0x0B, 0x0C, 0x1A)


class _Parser:
    def __init__(self, blob: bytes) -> None:
        self.b = blob
        self.p = _HEADER_SIZE
        self.strings: list[bytes] = []
        self.objects: list = []   # nell'ordine in cui cominciano, per \x02

    def _fail(self, what: str) -> None:
        raise DjayWriteError(
            f"Blob djay Pro non riconosciuto ({what} a offset {self.p}): "
            "formato diverso da quello verificato, mi fermo."
        )

    def _align(self) -> None:
        while self.p % 4:
            if self.b[self.p] != 0:
                self._fail("riempimento di allineamento non nullo")
            self.p += 1

    def _u32(self) -> int:
        self._align()
        if self.p + 4 > len(self.b):
            self._fail("uint32 troncato")
        v = struct.unpack("<I", self.b[self.p:self.p + 4])[0]
        self.p += 4
        return v

    def value(self):
        if self.p >= len(self.b):
            self._fail("fine del blob inattesa")
        tag = self.b[self.p]

        if tag == 0x08:  # stringa letterale
            self.p += 1
            end = self.b.find(b"\x00", self.p)
            if end == -1:
                self._fail("stringa non terminata")
            s = self.b[self.p:end]
            self.p = end + 1
            self.strings.append(s)
            return ("str", s)

        if tag == 0x05:  # riferimento a stringa già vista
            self.p += 2
            idx = self.b[self.p - 1]
            if idx >= len(self.strings):
                self._fail("riferimento a stringa fuori tabella")
            return ("str", self.strings[idx])

        if tag == 0x13:  # float32
            self.p += 1
            self._align()
            if self.p + 4 > len(self.b):
                self._fail("float32 troncato")
            v = struct.unpack("<f", self.b[self.p:self.p + 4])[0]
            self.p += 4
            return ("f32", v)

        if tag == 0x0F:  # intero, valore nel byte successivo
            self.p += 2
            return ("int", tag, self.b[self.p - 1])

        if tag == 0x02:  # riferimento a un oggetto già serializzato
            self.p += 2
            idx = self.b[self.p - 1]
            if idx >= len(self.objects) or self.objects[idx] is None:
                # Un riferimento in avanti, o a un oggetto che ci contiene,
                # non si saprebbe riscrivere: meglio fermarsi che indovinare.
                self._fail("riferimento a un oggetto non ancora serializzato")
            return ("objref", self.objects[idx])

        if tag in _INLINE_VALUE_TAGS:  # valore codificato nel tag stesso
            self.p += 1
            return ("int", tag, 0)

        # Da qui in poi sono tutti "oggetti": prendono un indice nella
        # tabella PRIMA di leggere il contenuto (quindi un contenitore ha
        # sempre un indice più basso di ciò che contiene).
        if tag == 0x15:  # blob binario
            self.p += 1
            slot = self._reserve()
            n = self._u32()
            if self.p + n > len(self.b):
                self._fail("blob binario troncato")
            d = self.b[self.p:self.p + n]
            self.p += n
            self._align()
            return self._store(slot, ("data", d))

        if tag in _COLLECTION_TAGS:
            self.p += 1
            slot = self._reserve()
            n = self._u32()
            return self._store(slot, ("coll", tag, [self.value() for _ in range(n)]))

        if tag == 0x18:  # struttura anonima: N coppie valore/nome, senza chiusura
            self.p += 1
            slot = self._reserve()
            n = self._u32()
            return self._store(slot, ("struct", [self._field() for _ in range(n)]))

        if tag == 0x2B:  # oggetto: nome classe, poi coppie valore/nome
            self.p += 1
            slot = self._reserve()
            cls = self.value()
            fields = []
            while True:
                if self.p >= len(self.b):
                    self._fail("oggetto non chiuso")
                if self.b[self.p] == 0x00:
                    self.p += 1
                    break
                fields.append(self._field())
            return self._store(slot, ("obj", cls, fields))

        self._fail(f"tag sconosciuto 0x{tag:02x}")

    def _reserve(self) -> int:
        self.objects.append(None)
        return len(self.objects) - 1

    def _store(self, slot: int, node):
        self.objects[slot] = node
        return node

    def _field(self) -> tuple[bytes, tuple]:
        """Un campo: prima il valore, poi il nome."""
        val = self.value()
        name = self.value()
        if name[0] != "str":
            self._fail("nome di campo non testuale")
        return (name[1], val)


class _Encoder:
    """Riserializza un albero rigenerando da zero tabella delle stringhe,
    allineamenti e contatori di testa."""

    def __init__(self, header_prefix: bytes) -> None:
        self.out = bytearray(header_prefix[:8].ljust(_HEADER_SIZE, b"\x00"))
        self.index: dict[bytes, int] = {}
        self.objects = 0
        self.obj_index: dict[int, int] = {}   # id(nodo) -> indice assegnato

    def _register(self, node) -> None:
        """Assegna a `node` il prossimo indice della tabella degli oggetti.
        Va chiamato PRIMA di scriverne il contenuto, come fa il parser."""
        self.obj_index[id(node)] = self.objects
        self.objects += 1

    def _align(self) -> None:
        while len(self.out) % 4:
            self.out.append(0)

    def _u32(self, v: int) -> None:
        self._align()
        self.out += struct.pack("<I", v)

    def value(self, node) -> None:
        kind = node[0]
        if kind == "str":
            s = node[1]
            if s in self.index:
                self.out += b"\x05" + bytes([self.index[s]])
            else:
                if len(self.index) > 0xFF:
                    raise DjayWriteError(
                        "Troppe stringhe distinte nel blob per il formato dei "
                        "riferimenti a 1 byte: mi fermo."
                    )
                self.index[s] = len(self.index)
                self.out += b"\x08" + s + b"\x00"
        elif kind == "f32":
            self.out += b"\x13"
            self._align()
            self.out += struct.pack("<f", node[1])
        elif kind == "int":
            tag, val = node[1], node[2]
            self.out += bytes([tag]) if tag in _INLINE_VALUE_TAGS else bytes([tag, val])
        elif kind == "objref":
            target = self.obj_index.get(id(node[1]))
            if target is None:
                raise DjayWriteError(
                    "Riferimento a un oggetto non ancora scritto: mi fermo "
                    "invece di produrre un puntatore sbagliato."
                )
            self.out += b"\x02" + bytes([target])
        elif kind == "data":
            self._register(node)
            self.out += b"\x15"
            self._u32(len(node[1]))
            self.out += node[1]
            self._align()
        elif kind == "coll":
            self._register(node)
            self.out += bytes([node[1]])
            self._u32(len(node[2]))
            for item in node[2]:
                self.value(item)
        elif kind == "struct":
            self._register(node)
            self.out += b"\x18"
            self._u32(len(node[1]))
            self._fields(node[1])
        elif kind == "obj":
            self._register(node)
            self.out += b"\x2b"
            self.value(node[1])
            self._fields(node[2])
            self.out += b"\x00"
        else:  # pragma: no cover - l'albero lo produciamo solo noi
            raise DjayWriteError(f"Nodo di tipo sconosciuto: {kind!r}")

    def _fields(self, fields) -> None:
        for name, val in fields:
            self.value(val)
            self.value(("str", name))

    def finish(self) -> bytes:
        self.out[8:16] = struct.pack("<Q", self.objects)
        self.out[16:20] = struct.pack("<I", len(self.index))
        return bytes(self.out)


def _parse(blob: bytes):
    """Ritorna la radice dell'albero. Pretende che il blob sia consumato
    per intero: se avanza anche un solo byte il modello non descrive questo
    file e non ci si può fidare di riscriverlo."""
    if not blob.startswith(MAGIC):
        raise DjayWriteError("Il blob non ha la firma TSAF attesa: formato inatteso.")
    parser = _Parser(blob)
    root = parser.value()
    if parser.p != len(blob):
        raise DjayWriteError(
            f"Blob djay Pro non riconosciuto: {len(blob) - parser.p} byte "
            "residui dopo la radice, mi fermo."
        )
    if root[0] != "obj":
        raise DjayWriteError("La radice del blob non è un oggetto: formato inatteso.")
    return root


def _encode(blob: bytes, root) -> bytes:
    enc = _Encoder(blob[:8])
    enc.value(root)
    return enc.finish()


def _get_field(root, name: bytes):
    for field_name, val in root[2]:
        if field_name == name:
            return val
    return None


# --------------------------------------------------------------------------
# Lettura dei cue point
# --------------------------------------------------------------------------

def _slots_from_root(root) -> list[tuple[float, float]]:
    cue_field = _get_field(root, _CUE_FIELD)
    if cue_field is None:
        return []
    if cue_field[0] != "coll":
        raise DjayWriteError("Il campo cuePoints non è una collezione: mi fermo.")
    slots = []
    for entry in cue_field[2]:
        if entry[0] != "obj":
            raise DjayWriteError("Elemento di cuePoints non è un oggetto: mi fermo.")
        t = _get_field(entry, b"time")
        if t is None or t[0] != "f32":
            raise DjayWriteError("Cue point senza un campo 'time' valido: mi fermo.")
        # `endTime` manca del tutto sui cue arrivati da un import (es. XML
        # rekordbox): lì vale comunque "non impostato".
        et = _get_field(entry, b"endTime")
        if et is not None and et[0] != "f32":
            raise DjayWriteError("Cue point con 'endTime' non numerico: mi fermo.")
        slots.append((t[1], -1.0 if et is None else et[1]))
    return slots


def _read_all_slots(blob: bytes) -> list[tuple[float, float]]:
    """Legge tutti gli slot dell'array cuePoints, INCLUSI i placeholder
    (time=-1.0): la posizione (indice 0-based) determina il pad/colore in
    djay Pro (vedi PAD_COLORS). Lista vuota se l'array non esiste affatto."""
    return _slots_from_root(_parse(blob))


def _int_value(node) -> int:
    """Valore di un nodo intero, tenendo conto del tag breve 0x2d = 1."""
    if node[0] != "int":
        raise DjayWriteError("Atteso un campo intero: mi fermo.")
    return 1 if node[1] == 0x2D else node[2]


def _loops_from_root(root) -> list[LoopRegion]:
    loop_field = _get_field(root, _LOOP_FIELD)
    if loop_field is None:
        return []
    if loop_field[0] != "coll":
        raise DjayWriteError("Il campo loopRegions non è una collezione: mi fermo.")
    loops = []
    for entry in loop_field[2]:
        if entry[0] != "obj":
            raise DjayWriteError("Elemento di loopRegions non è un oggetto: mi fermo.")
        start = _get_field(entry, b"startTime")
        end = _get_field(entry, b"endTime")
        number = _get_field(entry, b"number")
        if start is None or start[0] != "f32" or end is None or end[0] != "f32":
            raise DjayWriteError("Loop senza startTime/endTime validi: mi fermo.")
        if number is None:
            raise DjayWriteError("Loop senza il campo 'number': mi fermo.")
        loops.append(LoopRegion(start=start[1], end=end[1], slot=_int_value(number) - 1))
    return loops


def read_loop_regions(blob: bytes) -> list[LoopRegion]:
    """Legge i loop salvati. A differenza dei cue point l'array è denso e lo
    slot è dato dal campo `number` (qui riportato come `slot`, 0-based)."""
    return _loops_from_root(_parse(blob))


def read_cue_points(blob: bytes) -> list[CuePoint]:
    """Estrae i cue REALI (non placeholder) da un blob mediaItemUserData.
    `pad` è la posizione nell'array (0-7): è quella che determina il
    colore in djay Pro, vedi PAD_COLORS."""
    return [
        CuePoint(time=t, pad=pos, end_time=et)
        for pos, (t, et) in enumerate(_read_all_slots(blob))
        if t != -1.0
    ]


# --------------------------------------------------------------------------
# Scrittura dei cue point
# --------------------------------------------------------------------------

def _make_cue_entry(position: int, time: float, end_time: float):
    """Costruisce un oggetto ADCCuePoint come lo scrive djay Pro: campi
    nell'ordine time, endTime, number, con `number` = posizione+1 (0x2d è
    la codifica breve che djay usa per il valore 1)."""
    number = position + 1
    number_node = ("int", 0x2D, 0) if number == 1 else ("int", 0x0F, number)
    return (
        "obj",
        ("str", _CUE_CLASS),
        [
            (b"time", ("f32", time)),
            (b"endTime", ("f32", end_time)),
            (b"number", number_node),
        ],
    )


def _make_loop_entry(loop: LoopRegion):
    """Costruisce un ADCLoopRegion come lo scrive djay Pro: startTime,
    endTime, number — dove `number` (slot+1) è ciò che sceglie lo slot."""
    number = loop.slot + 1
    number_node = ("int", 0x2D, 0) if number == 1 else ("int", 0x0F, number)
    return (
        "obj",
        ("str", _LOOP_CLASS),
        [
            (b"startTime", ("f32", loop.start)),
            (b"endTime", ("f32", loop.end)),
            (b"number", number_node),
        ],
    )


def _declare_cloud_key(root, name: bytes) -> None:
    """Aggiunge `name` all'elenco `userChangedCloudKeys` se manca: djay Pro
    ce lo mette quando l'utente crea il primo cue/loop di una traccia."""
    keys = _get_field(root, _CLOUD_KEYS_FIELD)
    if keys is None or keys[0] != "coll":
        raise DjayWriteError(
            f"Campo '{_CLOUD_KEYS_FIELD.decode()}' assente o di tipo inatteso: "
            "struttura del blob diversa da quella verificata, mi fermo."
        )
    if ("str", name) not in keys[2]:
        keys[2].append(("str", name))


def _set_marker_field(root, name: bytes, entries) -> None:
    """Sostituisce il campo `name`, o lo crea nella posizione in cui lo mette
    djay Pro (vedi _FIELD_ORDER), dichiarandolo in userChangedCloudKeys."""
    new_field = ("coll", _ARRAY_TAG, entries)
    for i, (field_name, _) in enumerate(root[2]):
        if field_name == name:
            root[2][i] = (name, new_field)
            return
    precede = set(_FIELD_ORDER[:_FIELD_ORDER.index(name)])
    after = max(
        (i for i, (field_name, _) in enumerate(root[2]) if field_name in precede),
        default=-1,
    )
    root[2].insert(after + 1, (name, new_field))
    _declare_cloud_key(root, name)


def _check_slots(items, attr: str, what: str) -> None:
    seen = set()
    for item in items:
        slot = getattr(item, attr)
        if not (0 <= slot <= 7):
            raise DjayWriteError(f"{what} {slot + 1} fuori range (1-8).")
        if slot in seen:
            raise DjayWriteError(
                f"{what} {slot + 1} richiesto due volte: djay Pro ne può "
                "tenere uno solo per slot, scegline uno diverso."
            )
        seen.add(slot)


def _cue_entries(root, cues: list[CuePoint], overwrite: bool):
    """Array cuePoints: sparso, la POSIZIONE decide il pad, quindi le
    posizioni saltate vanno riempite con placeholder (time=-1.0)."""
    cue_field = _get_field(root, _CUE_FIELD)
    existing = [] if (overwrite or cue_field is None) else list(cue_field[2])
    existing_slots = [] if overwrite else _slots_from_root(root)

    updates = {c.pad: (c.time, c.end_time) for c in cues}
    entries, slots = [], []
    for pos in range(max([*updates.keys(), len(existing) - 1]) + 1):
        if pos in updates:
            t, et = updates[pos]
            entries.append(_make_cue_entry(pos, t, et))
        elif pos < len(existing):
            entries.append(existing[pos])  # slot non toccato: oggetto originale
            t, et = existing_slots[pos]
        else:
            t, et = -1.0, -1.0
            entries.append(_make_cue_entry(pos, t, et))
        slots.append((t, et))
    return entries, slots


def _loop_entries(root, loops: list[LoopRegion], overwrite: bool):
    """Array loopRegions: denso e ordinato per slot, con lo slot scritto nel
    campo `number` — niente placeholder, al contrario dei cue point."""
    kept = {} if overwrite else {lr.slot: lr for lr in _loops_from_root(root)}
    kept.update({lr.slot: lr for lr in loops})
    ordered = [kept[s] for s in sorted(kept)]
    return [_make_loop_entry(lr) for lr in ordered], ordered


def write_markers(
    blob: bytes,
    cues: list[CuePoint] = (),
    loops: list[LoopRegion] = (),
    overwrite: bool = False,
) -> bytes:
    """Scrive hot cue e/o loop salvati nel blob, in un colpo solo.

    `cues`: ognuno va al pad indicato da `.pad` (0-7 — è la posizione
    nell'array a determinare il colore, vedi PAD_COLORS). `loops`: ognuno
    va allo slot indicato da `.slot` (0-7) del banco "Loops", che in djay
    Pro è separato dagli hot cue.

    `overwrite=False` (default): preserva cue e loop già presenti negli
    slot non toccati, riusando gli oggetti originali senza modificarli.
    `overwrite=True`: ignora tutto quello che c'era prima. Funziona sia se
    i campi non esistono ancora sia se esistono già.
    """
    if not cues and not loops:
        raise DjayWriteError("Nessun cue o loop da scrivere.")
    _check_slots(cues, "pad", "Pad")
    _check_slots(loops, "slot", "Slot loop")
    for lr in loops:
        if lr.end <= lr.start:
            raise DjayWriteError(
                f"Loop allo slot {lr.slot + 1}: la fine ({lr.end:.2f}s) non è "
                f"dopo l'inizio ({lr.start:.2f}s)."
            )

    root = _parse(blob)
    slots = loops_after = None

    # Si riscrive solo il banco per cui abbiamo dei marcatori: `overwrite`
    # dice di non preservare quello che c'era lì dentro, non di svuotare
    # l'altro banco (un array vuoto non si è mai visto nei dati reali).
    if cues:
        entries, slots = _cue_entries(root, list(cues), overwrite)
        _set_marker_field(root, _CUE_FIELD, entries)
    if loops:
        entries, loops_after = _loop_entries(root, list(loops), overwrite)
        _set_marker_field(root, _LOOP_FIELD, entries)

    updated = _encode(blob, root)
    verify_write(blob, updated, slots, loops_after)
    return updated


def verify_write(
    original: bytes,
    updated: bytes,
    expected_slots: list[tuple[float, float]] | None = None,
    expected_loops: list[LoopRegion] | None = None,
) -> None:
    """Verifica di sicurezza: rilegge `updated` (il che di per sé prova che
    è un blob TSAF valido e consumabile per intero) e conferma che ogni
    cue/loop corrisponda a quanto atteso e che TUTTO il resto dell'albero —
    ogni altro campo, l'impronta audio, il timestampIdentifier — sia
    rimasto identico all'originale."""
    updated_root = _parse(updated)

    if expected_slots is not None:
        got_slots = _slots_from_root(updated_root)
        if len(got_slots) != len(expected_slots):
            raise DjayWriteError(
                f"Verifica fallita: attesi {len(expected_slots)} slot dopo la "
                f"scrittura, trovati {len(got_slots)}."
            )
        for i, (t, et) in enumerate(expected_slots):
            gt, get_ = got_slots[i]
            if abs(gt - t) > 1e-3 or abs(get_ - et) > 1e-3:
                raise DjayWriteError(
                    f"Verifica fallita: lo slot #{i} non corrisponde "
                    f"({(t, et)} -> {(gt, get_)})."
                )

    if expected_loops is not None:
        got_loops = _loops_from_root(updated_root)
        if len(got_loops) != len(expected_loops):
            raise DjayWriteError(
                f"Verifica fallita: attesi {len(expected_loops)} loop dopo la "
                f"scrittura, trovati {len(got_loops)}."
            )
        for want, got in zip(expected_loops, got_loops):
            if (want.slot != got.slot
                    or abs(want.start - got.start) > 1e-3
                    or abs(want.end - got.end) > 1e-3):
                raise DjayWriteError(
                    f"Verifica fallita: il loop allo slot {want.slot + 1} non "
                    f"corrisponde ({want} -> {got})."
                )

    if _without_markers(_parse(original)) != _without_markers(updated_root):
        raise DjayWriteError(
            "Verifica fallita: la scrittura ha modificato campi diversi dai "
            "cue point e dai loop."
        )


def _without_markers(root):
    """Campi della radice esclusi i marcatori: `cuePoints`, `loopRegions` e
    le loro dichiarazioni in `userChangedCloudKeys` (le sole cose che una
    scrittura di marcatori può legittimamente cambiare)."""
    out = []
    for name, val in root[2]:
        if name in _MARKER_FIELDS:
            continue
        if name == _CLOUD_KEYS_FIELD and val[0] == "coll":
            val = ("coll", val[1],
                   [k for k in val[2] if k[1] not in _MARKER_FIELDS])
        out.append((name, val))
    return out


# --------------------------------------------------------------------------
# Individuazione della traccia (path del file -> uuid interno di djay Pro)
# --------------------------------------------------------------------------

def _db_path(library_path: Path) -> Path:
    return library_path / "MediaLibrary.db"


def _open_readonly(db_file: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)


def find_track_uuid(db_file: Path, filepath: Path) -> str:
    """Trova l'uuid di djay Pro per il file dato.

    Cerca in localMediaItemLocations/globalMediaItemLocations un file:// URI
    che, decodificato, combacia col path assoluto del file. Se più uuid
    puntano allo stesso path (visto in pratica: entry duplicate/orfane),
    seleziona quello con una riga mediaItemUserData (la traccia deve essere
    stata almeno aperta/analizzata da djay Pro una volta, non serve che
    abbia già cue point: `write_cues` gestisce anche l'array assente).
    """
    target = str(filepath.resolve())
    conn = _open_readonly(db_file)
    try:
        candidates: list[str] = []
        for coll in ("localMediaItemLocations", "globalMediaItemLocations"):
            rows = conn.execute(
                "SELECT key, data FROM database2 WHERE collection=?", (coll,)
            ).fetchall()
            for key, data in rows:
                idx = data.find(b"file://")
                if idx == -1:
                    continue
                end = data.find(b"\x00", idx)
                uri = data[idx:end].decode("utf-8", errors="replace")
                if urllib.parse.unquote(uri[len("file://"):]) == target:
                    candidates.append(key)

        if not candidates:
            raise DjayWriteError(
                "Nessuna traccia trovata in djay Pro per questo file. "
                "Deve essere già presente nella tua libreria djay Pro."
            )

        with_data = []
        for uuid in candidates:
            row = conn.execute(
                "SELECT data FROM database2 WHERE collection='mediaItemUserData' AND key=?",
                (uuid,),
            ).fetchone()
            if row is not None:
                with_data.append(uuid)

        if len(with_data) == 1:
            return with_data[0]
        if len(with_data) == 0:
            raise DjayWriteError(
                "Trovata la traccia in djay Pro ma non è mai stata aperta/"
                "analizzata da djay Pro (nessun dato utente associato). "
                "Aprila almeno una volta in djay Pro (anche solo un doppio "
                "click) prima di usare questa funzione."
            )

        # Più candidati con dati: prova a disambiguare preferendo quello con
        # cue point già esistenti (è l'unico segnale extra disponibile).
        populated = []
        for uuid in with_data:
            row = conn.execute(
                "SELECT data FROM database2 WHERE collection='mediaItemUserData' AND key=?",
                (uuid,),
            ).fetchone()
            try:
                has_cues = bool(_read_all_slots(row[0]))
            except DjayWriteError:
                has_cues = False
            if has_cues:
                populated.append(uuid)

        if len(populated) == 1:
            return populated[0]
        raise DjayWriteError(
            f"Trovate {len(with_data)} tracce diverse in djay Pro con dati "
            "utente per lo stesso file (probabile duplicato nella libreria "
            "djay Pro): non posso scegliere in automatico. Consolida le "
            "voci duplicate in djay Pro prima di riprovare."
        )
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Backup e scrittura effettiva
# --------------------------------------------------------------------------

def is_djay_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-x", "djay Pro"], capture_output=True, text=True,
        )
        return out.returncode == 0
    except Exception:
        return False  # best-effort: se il check fallisce non blocchiamo da soli


def backup_library(library_path: Path = DEFAULT_LIBRARY_PATH) -> Path:
    """Copia l'intero bundle .djayMediaLibrary (db+wal+shm) prima di scrivere."""
    if not library_path.exists():
        raise DjayWriteError(f"Libreria djay Pro non trovata in {library_path}")
    dest_root = Path.home() / "Music" / "Wavecut" / "Backups"
    stamp = time.strftime("%Y-%m-%d-(%H-%M-%S)")
    dest = dest_root / f"{stamp}-djayPro-backup" / library_path.name
    dest.mkdir(parents=True, exist_ok=True)
    for f in library_path.glob("MediaLibrary.db*"):
        shutil.copy2(f, dest / f.name)
    return dest


def _require_library(library_path: Path) -> Path:
    db_file = _db_path(library_path)
    if not db_file.exists():
        raise DjayWriteError(
            f"Libreria djay Pro non trovata in {library_path}. "
            "Questa funzione è verificata solo per l'installazione Mac App Store."
        )
    return db_file


def read_djay_bpm(filepath: Path,
                  library_path: Path = DEFAULT_LIBRARY_PATH) -> float | None:
    """BPM che djay Pro ha già calcolato per questo file, se lo conosce.

    È il tempo della griglia che djay disegna in console, quindi è quello
    giusto su cui allineare i cue — e su due tracce reali è risultato
    sensibilmente diverso dalla stima del beat tracker (126.00 e 125.70
    contro 123.0 in entrambi i casi).

    Best-effort per costruzione: se la libreria non c'è, la traccia non è
    stata importata o non è ancora stata analizzata, ritorna None e chi
    chiama si arrangia con la propria stima. Non solleva mai.
    """
    try:
        db_file = _db_path(library_path)
        if not db_file.exists():
            return None
        uuid = find_track_uuid(db_file, filepath)
        conn = _open_readonly(db_file)
        try:
            row = conn.execute(
                "SELECT data FROM database2 WHERE collection='mediaItemAnalyzedData' "
                "AND key=?", (uuid,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        bpm = _get_field(_parse(row[0]), b"bpm")
        if bpm is None or bpm[0] != "f32" or not (0 < bpm[1] < 500):
            return None
        return float(bpm[1])
    except Exception:
        return None


def _load_user_data(db_file: Path, filepath: Path) -> tuple[str, bytes]:
    uuid = find_track_uuid(db_file, filepath)
    conn = _open_readonly(db_file)
    try:
        row = conn.execute(
            "SELECT data FROM database2 WHERE collection='mediaItemUserData' AND key=?",
            (uuid,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise DjayWriteError("Riga mediaItemUserData non trovata per questa traccia.")
    return uuid, row[0]


def _result(uuid: str, original: bytes, updated: bytes, **kw) -> WriteResult:
    return WriteResult(
        ok=True, track_uuid=uuid,
        cues_before=read_cue_points(original), cues_after=read_cue_points(updated),
        loops_before=read_loop_regions(original), loops_after=read_loop_regions(updated),
        **kw,
    )


def preview_write(
    filepath: Path, new_cues: list[CuePoint] = (),
    library_path: Path = DEFAULT_LIBRARY_PATH,
    overwrite: bool = False, new_loops: list[LoopRegion] = (),
) -> WriteResult:
    """Calcola cosa verrebbe scritto, verifica, ma NON scrive nulla.

    Ogni cue va al pad (posizione 0-7, determina il colore) indicato da
    `.pad`, ogni loop allo slot indicato da `.slot`. `overwrite=False`
    (default) preserva cue e loop già presenti negli slot non toccati;
    `overwrite=True` li ignora. Funziona sia se la traccia non ha mai avuto
    cue/loop sia se ne ha già.
    """
    db_file = _require_library(library_path)
    uuid, original = _load_user_data(db_file, filepath)
    updated = write_markers(original, new_cues, new_loops, overwrite=overwrite)
    return _result(uuid, original, updated,
                   message="Anteprima verificata: pronta per la scrittura.")


def write_new_cues(
    filepath: Path, new_cues: list[CuePoint] = (),
    library_path: Path = DEFAULT_LIBRARY_PATH,
    allow_while_running: bool = False,
    overwrite: bool = False,
    new_loops: list[LoopRegion] = (),
) -> WriteResult:
    """Scrive davvero i nuovi cue/loop nel database live di djay Pro.

    Sempre, in quest'ordine: verifica che djay Pro non sia in esecuzione
    (a meno di override esplicito), backup del bundle intero, calcolo del
    nuovo blob, verifica (`write_markers`), e SOLO SE tutto torna, UPDATE
    della riga nel database live.
    """
    if not allow_while_running and is_djay_running():
        raise DjayWriteError(
            "djay Pro risulta in esecuzione: chiudilo prima di scrivere, per "
            "evitare conflitti con l'auto-salvataggio dell'app."
        )

    db_file = _require_library(library_path)
    backup_path = backup_library(library_path)
    uuid, original = _load_user_data(db_file, filepath)
    # si ferma qui, senza toccare nulla, se qualcosa non torna
    updated = write_markers(original, new_cues, new_loops, overwrite=overwrite)

    write_conn = sqlite3.connect(str(db_file))
    try:
        write_conn.execute(
            "UPDATE database2 SET data=? WHERE collection='mediaItemUserData' AND key=?",
            (updated, uuid),
        )
        write_conn.commit()
        write_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        write_conn.close()

    verb = "sostituiti con" if overwrite else "aggiunti"
    written = ", ".join(
        p for p in (f"{len(new_cues)} hot cue" if new_cues else "",
                    f"{len(new_loops)} loop" if new_loops else "") if p
    )
    return _result(uuid, original, updated,
                   message=f"Marcatori {verb}: {written}.",
                   backup_path=backup_path)
