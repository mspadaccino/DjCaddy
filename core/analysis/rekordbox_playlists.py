"""Le playlist dello scaffale scritte DIRETTAMENTE nella libreria di
rekordbox — niente XML, niente import.

Il gemello di `rekordbox_write` per le scalette: stesso database
(`master.db`, via pyrekordbox), stesse regole — rekordbox chiuso, backup del
giorno, il brano cercato per percorso intero e mai indovinato — e in più
la forma che rekordbox dà alle playlist: una cartella «DjCaddy» sotto la
radice, dentro una playlist per nome, con i brani nell'ordine dello
scaffale. Una playlist che c'è già con quel nome si RIFÀ: si cancella e si
ricrea, che è l'unico modo di garantire che dentro ci sia quello che c'è
sullo scaffale e nell'ordine giusto. Le altre playlist della cartella, e
tutto il resto della libreria, non si toccano.

Un brano che rekordbox non ha resta fuori dalla sua playlist e viene
nominato: importarlo là è un gesto che spetta a chi ha la libreria in
mano, non a un programma che scrive dentro il suo database.

La parte che decide — cosa si troverebbe, cosa mancherebbe, cosa si
rifarebbe — è `plan`, pura: prende una funzione che cerca i brani e non sa
niente di database. Il resto è aprire, cercare e consegnare a pyrekordbox.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .rekordbox_write import (RekordboxWriteError, backup_database,
                              database_path, find_track,
                              is_rekordbox_running, open_database)

FOLDER = "DjCaddy"
ROOT = "root"


@dataclass
class PlaylistPlan:
    name: str
    found: list = field(default_factory=list)      # le righe di rekordbox
    missing: list[Path] = field(default_factory=list)
    replaces: bool = False                          # c'era già col suo nome


@dataclass
class ShelfWriteResult:
    ok: bool
    message: str
    playlists: list[PlaylistPlan] = field(default_factory=list)
    backup_path: Path | None = None

    @property
    def found(self) -> int:
        return sum(len(p.found) for p in self.playlists)

    @property
    def missing(self) -> list[Path]:
        return [m for p in self.playlists for m in p.missing]


def plan(playlists: list[tuple[str, list[Path]]],
         find: Callable[[Path], object | None],
         existing: set[str] = frozenset()) -> list[PlaylistPlan]:
    """Cosa succederebbe: per ogni scaletta i brani che rekordbox ha, quelli
    che non ha, e se una playlist con quel nome c'è già nella cartella."""
    out = []
    for name, paths in playlists:
        one = PlaylistPlan(name=name, replaces=name in existing)
        for path in paths:
            row = find(Path(path))
            (one.found if row is not None else one.missing).append(
                row if row is not None else Path(path))
        out.append(one)
    return out


def summary(plans: list[PlaylistPlan]) -> str:
    found = sum(len(p.found) for p in plans)
    missing = sum(len(p.missing) for p in plans)
    replaced = [p.name for p in plans if p.replaces]
    told = [f"{len(plans)} playlist(s) in a «{FOLDER}» folder, "
            f"{found} track(s) rekordbox knows."]
    if missing:
        told.append(f"{missing} track(s) are not in rekordbox's library and "
                    "stay out of their playlist — import them there first "
                    "if you want them in.")
    if replaced:
        told.append("Already in the folder and rebuilt as on the shelf: "
                    + ", ".join(replaced) + ".")
    return " ".join(told)


def _folder(db, create: bool):
    """La cartella «DjCaddy» sotto la radice, creata se manca e se si può."""
    from pyrekordbox.db6.tables import DjmdPlaylist

    found = db.get_playlist(Name=FOLDER, ParentID=ROOT, Attribute=1).all()
    if found:
        return found[0]
    return db.create_playlist_folder(FOLDER) if create else None


def _inside(db, folder) -> dict[str, object]:
    """Le playlist della cartella, per nome."""
    if folder is None:
        return {}
    return {p.Name: p for p in db.get_playlist(ParentID=folder.ID).all()}


def preview_shelf_write(playlists: list[tuple[str, list[Path]]],
                        db_path: Path | None = None) -> ShelfWriteResult:
    """Cosa cambierebbe, senza cambiare niente."""
    if not playlists:
        raise RekordboxWriteError("Nothing on the shelf to write.")
    db = open_database(db_path)
    try:
        existing = set(_inside(db, _folder(db, create=False)))
        plans = plan(playlists, lambda p: find_track(db, p), existing)
    finally:
        db.close()
    return ShelfWriteResult(ok=True, message=summary(plans), playlists=plans)


def write_shelf(playlists: list[tuple[str, list[Path]]],
                db_path: Path | None = None) -> ShelfWriteResult:
    """Scrive davvero: rekordbox chiuso, backup del giorno, poi la cartella,
    e dentro ogni playlist rifatta com'è sullo scaffale."""
    if not playlists:
        raise RekordboxWriteError("Nothing on the shelf to write.")
    if is_rekordbox_running():
        raise RekordboxWriteError(
            "rekordbox is running: quit it before writing, or its own save "
            "will overwrite this.")
    path = db_path or database_path()
    if path is None or not Path(path).is_file():
        raise RekordboxWriteError(
            "No rekordbox library was found on this computer.")
    backup = backup_database(Path(path))

    db = open_database(path)
    try:
        folder = _folder(db, create=True)
        inside = _inside(db, folder)
        plans = plan(playlists, lambda p: find_track(db, p), set(inside))
        for one in plans:
            if one.replaces:
                db.delete_playlist(inside[one.name])
            made = db.create_playlist(one.name, parent=folder)
            for row in one.found:
                db.add_to_playlist(made, row)
        db.commit()
    finally:
        db.close()
    return ShelfWriteResult(
        ok=True, playlists=plans, backup_path=backup,
        message=f"Written into rekordbox: {summary(plans)}")
