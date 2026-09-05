"""Lo scaffale: le playlist con un nome, una accanto all'altra.

La pagina Map lavora su UNA playlist — la linea sulla mappa, la lavagna,
il Chain Maker che ci appende, Radio Mix e Journey che ne leggono la coda.
Per costruire una serata servono però molte scalette (house_intro,
house_buildup, funky_climax…) e finché la playlist era una sola ogni
nuova scaletta voleva un file salvato e la precedente sgomberata. Lo
scaffale tiene le altre mentre una sta sul tavolo.

Una cartella di `.m3u8`, un file per playlist, il nome del file è il nome
della playlist: lo stesso formato che la pagina esporta e rilegge, così la
cartella si apre anche dal Finder e una scaletta si porta fuori copiando
un file. Sta in `~/Documents/DjCaddy/Playlists` (`user_files.user_dir`),
non nella cache: la mappa si può cancellare e rifare, le scalette sono
lavoro del DJ e vanno dove si vedono e si salvano.

`.active` ricorda quale playlist sta sul tavolo, per ritrovarla al
prossimo avvio. Un nome vale se è un nome di file: niente separatori di
cartella, niente punto davanti (sarebbe nascosto), non vuoto.
"""

from __future__ import annotations

from pathlib import Path

from core.analysis.dj_export import build_m3u8, read_m3u8
from core.analysis.user_files import user_dir

DEFAULT_NAME = "Playlist"
_SUFFIX = ".m3u8"
_ACTIVE = ".active"


def default_shelf_dir() -> Path:
    return user_dir() / "Playlists"


def valid_name(name: str) -> bool:
    name = name.strip()
    return bool(name) and not name.startswith(".") \
        and "/" not in name and "\\" not in name


class Shelf:
    """La cartella, letta e scritta. Non tiene niente in memoria: i nomi
    si rileggono dal disco a ogni domanda, che è quanto basta per una
    decina di file e non può andare fuori sincrono."""

    def __init__(self, folder: Path | str | None = None) -> None:
        self.folder = Path(folder) if folder else default_shelf_dir()

    def path(self, name: str) -> Path:
        return self.folder / f"{name}{_SUFFIX}"

    def names(self) -> list[str]:
        try:
            return sorted((p.stem for p in self.folder.glob(f"*{_SUFFIX}")),
                          key=str.casefold)
        except OSError:
            return []

    def read(self, name: str) -> list[str]:
        try:
            return read_m3u8(self.path(name).read_text("utf-8",
                                                       errors="replace"))
        except OSError:
            return []

    def write(self, name: str, paths: list[str]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path(name).write_text(
            build_m3u8([{"path": Path(p)} for p in paths]), "utf-8")

    def rename(self, old: str, new: str) -> None:
        follow = self._active_raw() == old
        self.path(old).rename(self.path(new))
        if follow:
            self.set_active(new)

    def delete(self, name: str) -> None:
        self.path(name).unlink(missing_ok=True)

    def free_name(self, wanted: str) -> str:
        """`wanted` se non è preso, altrimenti «wanted 2», «wanted 3»…"""
        taken = set(self.names())
        if wanted not in taken:
            return wanted
        n = 2
        while f"{wanted} {n}" in taken:
            n += 1
        return f"{wanted} {n}"

    # --- quale sta sul tavolo ---
    def _active_raw(self) -> str | None:
        try:
            return (self.folder / _ACTIVE).read_text("utf-8").strip()
        except OSError:
            return None

    def active(self) -> str | None:
        """Il nome scritto, se la sua playlist c'è ancora."""
        name = self._active_raw()
        return name if name in self.names() else None

    def set_active(self, name: str) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / _ACTIVE).write_text(name, "utf-8")
