"""Ricerca dei duplicati in una libreria musicale, in tre livelli distinti.

L'idea è che "duplicato" non è una cosa sola, e trattarli allo stesso modo
porta a cancellare roba che serviva:

A. DUPLICATE_SAME_FOLDER  stessa cartella, stessa dimensione, stesso MD5.
   Duplicato certo: due file identici nella stessa cartella non hanno
   ragione di esistere. È l'unico livello su cui ha senso agire.

B. DUPLICATE_OTHER_FOLDER stessa dimensione e stesso MD5, cartelle diverse.
   Solo CANDIDATO: lo stesso brano in `80s/`, `DANCE RETRO/` e `Workout/`
   è probabilmente voluto, ed è così che è organizzata la libreria.

C. SIMILAR_NAME           nome molto simile ma contenuto diverso.
   Puramente informativo: possono essere edit, remix, rip diversi.

Prima di tutto questo però va tolto di mezzo un caso insidioso: i file
identici perché ugualmente ROTTI. Nella libreria reale ci sono 49 file da
zero byte — tutti con lo stesso MD5, quello del vuoto — e gruppi di stub da
2311 byte contenenti solo un tag ID3 e nessun audio. Sono brani DIVERSI che
si somigliano solo nel fallimento, e proporne la cancellazione sarebbe il
peggior errore possibile: si toglierebbe l'unica traccia di un brano che
manca. Finiscono quindi in `broken`, che non si può selezionare.

Niente in questo modulo cancella file. `build_quarantine_plan` prepara uno
spostamento in una cartella di quarantena, che è un'operazione separata ed
esplicita: la libreria resta verificabile in djay Pro finché non si svuota
la quarantena a mano, giorni dopo.
"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .folder_scan import QUARANTINE_DIRNAME, ScannedFile, check_readable, human_size

LEVEL_SAME_FOLDER = "DUPLICATE_SAME_FOLDER"
LEVEL_OTHER_FOLDER = "DUPLICATE_OTHER_FOLDER"
LEVEL_SIMILAR_NAME = "SIMILAR_NAME"


# Segni che un nome è la copia fatta dal sistema operativo, non l'originale.
_COPY_MARKERS = re.compile(r"(\(\d+\)$|\bcopy\b|\bcopia\b|\bconflicted\b)", re.IGNORECASE)
# Sequenze tipiche di una codifica sbagliata letta come UTF-8 (mojibake).
_MOJIBAKE = re.compile(r"[ÃÂ]\S|�")


@dataclass
class DuplicateGroup:
    """Un insieme di file che il livello considera legati fra loro.

    `keep` è solo una PROPOSTA (vedi `choose_keeper`), non una decisione:
    per i livelli B e C non va applicata affatto.
    """
    level: str
    size: int
    md5: str | None
    keep: Path
    duplicates: list[Path] = field(default_factory=list)

    @property
    def copies(self) -> int:
        return len(self.duplicates) + 1

    @property
    def folder(self) -> Path:
        return self.keep.parent

    @property
    def wasted_bytes(self) -> int:
        """Spazio che si libererebbe tenendo un solo file."""
        return self.size * len(self.duplicates)


@dataclass
class BrokenGroup:
    """File byte-identici che però non sono audio: stessa rottura, non stessa
    musica. Si riportano soltanto, senza mai proporne la rimozione."""
    size: int
    md5: str
    reason: str
    paths: list[Path] = field(default_factory=list)


@dataclass
class DuplicateReport:
    same_folder: list[DuplicateGroup] = field(default_factory=list)
    other_folder: list[DuplicateGroup] = field(default_factory=list)
    similar_name: list[DuplicateGroup] = field(default_factory=list)
    broken: list[BrokenGroup] = field(default_factory=list)
    hashed_files: int = 0
    unreadable: list[tuple[Path, str]] = field(default_factory=list)

    def all_groups(self) -> list[DuplicateGroup]:
        return [*self.same_folder, *self.other_folder, *self.similar_name]


# --------------------------------------------------------------------------
# Nomi: normalizzazione e scelta di quale file tenere
# --------------------------------------------------------------------------

def normalized_name(path: Path) -> str:
    """Nome ridotto all'osso, per confrontare titoli scritti in modi diversi.

    Toglie accenti e rende uguale qualunque punteggiatura, così
    "O'NEIL& Dj Quba - What Is Love" e "O_NEIL& Dj Quba - What Is Love"
    diventano la stessa stringa.
    """
    stem = unicodedata.normalize("NFKD", path.stem).casefold()
    stem = "".join(c for c in stem if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", stem).strip()


def name_quality(path: Path) -> tuple:
    """Chiave d'ordinamento: il più piccolo è il nome "più pulito".

    Serve a scegliere in modo deterministico fra file byte-identici, dove
    l'audio è lo stesso e conta solo come è scritto il nome. Preferisce
    l'apostrofo vero al trattino basso che spesso lo sostituisce, evita i
    nomi con segni di codifica sbagliata e quelli marcati come copia.
    L'ultimo criterio è alfabetico, così l'esito non dipende dall'ordine in
    cui il filesystem ha restituito i file.
    """
    name = path.name
    return (
        1 if _COPY_MARKERS.search(path.stem) else 0,
        1 if _MOJIBAKE.search(name) else 0,
        0 if unicodedata.is_normalized("NFC", name) else 1,
        name.count("_"),
        len(path.parts),          # a parità, il percorso meno annidato
        str(path),
    )


def choose_keeper(paths: list[Path]) -> tuple[Path, list[Path]]:
    """(file da tenere, file duplicati) secondo `name_quality`."""
    ordered = sorted(paths, key=name_quality)
    return ordered[0], ordered[1:]


# --------------------------------------------------------------------------
# Hash
# --------------------------------------------------------------------------

def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(files: list[ScannedFile], progress=None,
                    hasher=md5_file, verify_audio: bool = True) -> DuplicateReport:
    """Costruisce il report a tre livelli su una lista di file già scansionati.

    L'MD5 si calcola SOLO per i file la cui dimensione compare più di una
    volta: due file di dimensione diversa non possono essere identici, e su
    una libreria vera questo evita di leggere per intero decine di gigabyte.

    Con `verify_audio` ogni gruppo di file identici viene controllato una
    volta sola — sono uguali byte per byte, quindi guardarne uno basta — e
    se non è audio finisce in `broken` invece che fra i duplicati.
    """
    report = DuplicateReport()

    by_size: dict[int, list[ScannedFile]] = defaultdict(list)
    for f in files:
        by_size[f.size].append(f)
    candidates = [f for group in by_size.values() if len(group) > 1 for f in group]

    digests: dict[Path, str] = {}
    for i, f in enumerate(candidates, 1):
        try:
            digests[f.path] = hasher(f.path)
        except OSError as e:
            report.unreadable.append((f.path, str(e)))
        if progress is not None:
            progress(i, len(candidates))
    report.hashed_files = len(digests)

    # Insiemi di file byte-identici: stessa dimensione E stesso hash.
    identical: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for f in candidates:
        if f.path in digests:
            identical[(f.size, digests[f.path])].append(f.path)

    for (size, digest), paths in sorted(identical.items(), key=lambda kv: kv[0][1]):
        if len(paths) < 2:
            continue

        # I file del gruppo sono byte-identici, quindi basta guardarne UNO per
        # sapere se sono audio: se non lo sono, sono brani diversi accomunati
        # solo dall'essere rotti allo stesso modo (tipicamente zero byte, o
        # uno stub di soli tag). Non vanno mai proposti per la rimozione.
        if verify_audio:
            reason = check_readable(paths[0])
            if reason is not None:
                report.broken.append(BrokenGroup(
                    size=size, md5=digest, reason=reason, paths=sorted(paths)))
                continue

        by_folder: dict[Path, list[Path]] = defaultdict(list)
        for p in paths:
            by_folder[p.parent].append(p)

        # A: copie nella stessa cartella
        for folder in sorted(by_folder):
            same = by_folder[folder]
            if len(same) > 1:
                keep, dups = choose_keeper(same)
                report.same_folder.append(DuplicateGroup(
                    level=LEVEL_SAME_FOLDER, size=size, md5=digest,
                    keep=keep, duplicates=dups))

        # B: lo stesso file presente in cartelle diverse. Si tiene un
        # rappresentante per cartella: le copie interne le ha già viste A.
        if len(by_folder) > 1:
            representatives = [choose_keeper(v)[0] for v in by_folder.values()]
            keep, dups = choose_keeper(representatives)
            report.other_folder.append(DuplicateGroup(
                level=LEVEL_OTHER_FOLDER, size=size, md5=digest,
                keep=keep, duplicates=dups))

    # C: nomi che si somigliano ma contenuto diverso. Si scarta il gruppo solo
    # se i suoi file sono identici FRA LORO, perché in quel caso li ha già
    # visti A o B. Non basta che uno di essi sia copia di un terzo file
    # altrove: la somiglianza di nome dentro il gruppo resta un'informazione.
    by_name: dict[str, list[ScannedFile]] = defaultdict(list)
    for f in files:
        by_name[normalized_name(f.path)].append(f)
    for name in sorted(by_name):
        group = by_name[name]
        if not name or len(group) < 2:
            continue
        signatures = {(f.size, digests.get(f.path)) for f in group}
        if len(signatures) == 1 and next(iter(signatures))[1] is not None:
            continue
        keep, dups = choose_keeper([f.path for f in group])
        report.similar_name.append(DuplicateGroup(
            level=LEVEL_SIMILAR_NAME, size=max(f.size for f in group),
            md5=None, keep=keep, duplicates=dups))

    return report


# --------------------------------------------------------------------------
# Report CSV
# --------------------------------------------------------------------------

CSV_COLUMNS = ["action", "level", "folder", "keep", "delete",
               "size_bytes", "size", "md5", "copies"]


def write_csv(groups: list[DuplicateGroup], destination: Path) -> Path:
    """Una riga per ogni file duplicato. `action` è DELETE solo per i
    duplicati certi (stessa cartella); gli altri livelli restano REVIEW,
    perché non è detto che vadano tolti."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for g in groups:
            action = "DELETE" if g.level == LEVEL_SAME_FOLDER else "REVIEW"
            for dup in g.duplicates:
                writer.writerow([
                    action, g.level, str(g.folder), str(g.keep), str(dup),
                    g.size, human_size(g.size), g.md5 or "", g.copies,
                ])
    return destination


# --------------------------------------------------------------------------
# Quarantena (mai cancellazione)
# --------------------------------------------------------------------------

def duplicates_of(groups: list[DuplicateGroup]) -> list[Path]:
    """Tutti i file duplicati dei gruppi dati (esclusi i `keep`)."""
    return [dup for g in groups for dup in g.duplicates]


def build_quarantine_plan(paths, root: Path,
                          dirname: str = QUARANTINE_DIRNAME) -> list[tuple[Path, Path]]:
    """(origine, destinazione) per ogni file da mettere in quarantena.

    Prende i PERCORSI e non i gruppi, perché la scelta di cosa spostare la
    fa l'utente riga per riga: un gruppo con tre copie ne mette in
    quarantena due, ed è per questo che il numero di gruppi e quello dei
    file da spostare non coincidono.

    La struttura di cartelle originale viene ricalcata dentro la quarantena,
    così si vede da dove veniva ogni file e lo si può rimettere a posto. I
    file già in quarantena vengono ignorati, così ripetere l'operazione non
    fa danni.
    """
    quarantine = root / dirname
    plan: list[tuple[Path, Path]] = []
    taken: set[Path] = set()
    seen: set[Path] = set()
    for src in paths:
        # Lo stesso file puo' arrivare da due selezioni diverse: spostarlo
        # una volta sola, o il secondo tentativo fallirebbe perche' non e'
        # piu' dov'era.
        if src in seen or quarantine in src.parents:
            continue
        seen.add(src)
        try:
            relative = src.parent.relative_to(root)
        except ValueError:
            relative = Path(src.parent.name)
        dest = quarantine / relative / src.name
        # Due cartelle diverse possono contenere lo stesso nome file.
        stem, suffix, n = dest.stem, dest.suffix, 1
        while dest in taken or dest.exists():
            dest = dest.with_name(f"{stem}__{n}{suffix}")
            n += 1
        taken.add(dest)
        plan.append((src, dest))
    return plan


def apply_quarantine_plan(plan: list[tuple[Path, Path]],
                          dry_run: bool = True) -> tuple[int, list[tuple[Path, str]]]:
    """Sposta (non cancella) i file secondo il piano.

    Ritorna (quanti spostati, errori). Con `dry_run` non tocca nulla: è il
    default apposta, perché questa è l'unica funzione del modulo che scrive
    sul filesystem.
    """
    moved, errors = 0, []
    for src, dest in plan:
        try:
            if not src.exists():
                errors.append((src, "non esiste più"))
                continue
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            moved += 1
        except OSError as e:
            errors.append((src, str(e)))
    return moved, errors
