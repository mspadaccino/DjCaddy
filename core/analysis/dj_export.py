"""Export dei cue verso il formato rekordbox XML.

rekordbox XML è il formato "hub" più diffuso fra i software DJ: rekordbox lo
importa nativamente, e i principali convertitori a pagamento (DJ Conversion
Utility, MIXO, Lexicon) lo accettano in ingresso per produrre Serato,
Traktor o **djay Pro**. Non esiste un modo verificabile per scrivere hot cue
direttamente nei tag Serato o nel database di djay Pro da qui (vedi
analysis/vocals.py per lo stesso principio applicato al rilevamento voce:
non si costruisce un formato che non si può verificare), quindi questo
modulo si concentra sull'unico formato che si può generare con fiducia.

Lo schema (attributi TRACK, POSITION_MARK) è documentato pubblicamente e
stabile da anni; qui ne viene generata una versione minima ma valida.
"""

from __future__ import annotations

import os
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from .models import SECTION_COLORS, Section

MAX_HOT_CUES = 8  # slot A..H in rekordbox/Serato: oltre, i cue diventano "memory cue"


def file_uri(path: Path) -> str:
    """URI file:// per l'attributo Location di rekordbox (percent-encoded)."""
    return "file://localhost" + quote(str(path.resolve()))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def section_cues(sections: list[Section]) -> list[dict]:
    """Converte le sezioni classificate in cue point per l'export.

    Ogni cue prende il nome dall'etichetta della sezione (con 🎤 se vocal) e il
    colore dalla stessa palette usata nel grafico, per coerenza visiva in
    rekordbox.
    """
    cues = []
    for s in sections:
        name = f"{'VOCAL ' if s.vocal else ''}{s.label}"
        cues.append({
            "name": name,
            "start": s.start,
            "color": SECTION_COLORS.get(s.label, "#ffffff"),
        })
    return cues


def build_rekordbox_xml(tracks: list[dict]) -> str:
    """Costruisce l'XML rekordbox (stringa) per una lista di tracce.

    Ogni elemento di `tracks` è un dict:
        {"path": Path, "name": str, "artist": str, "bpm": float | None,
         "duration": float | None, "cues": list[{"name","start","color"}]}

    Pura: nessun I/O, solo costruzione della stringa XML — testabile con dati
    sintetici.
    """
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="DjCaddy", Version="1.0", Company="DjCaddy")
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks)))

    track_keys = []
    for i, t in enumerate(tracks, start=1):
        path: Path = t["path"]
        duration = t.get("duration")
        bpm = t.get("bpm")
        attrs = {
            "TrackID": str(i),
            "Name": t.get("name") or path.stem,
            "Artist": t.get("artist") or "",
            "Album": "",
            "Genre": t.get("genre") or "",
            "Kind": f"{path.suffix.lstrip('.').upper()} File",
            "Size": "0",
            "TotalTime": f"{duration:.0f}" if duration else "0",
            "Location": file_uri(path),
        }
        if bpm:
            attrs["AverageBpm"] = f"{bpm:.2f}"
        track_el = ET.SubElement(collection, "TRACK", attrs)

        if bpm:
            ET.SubElement(track_el, "TEMPO", Inizio="0.000", Bpm=f"{bpm:.2f}",
                         Metro="4/4", Battito="1")

        for slot, cue in enumerate(t.get("cues", [])):
            r, g, b = _hex_to_rgb(cue.get("color", "#ffffff"))
            num = str(slot) if slot < MAX_HOT_CUES else "-1"  # oltre 8: memory cue
            ET.SubElement(track_el, "POSITION_MARK", {
                "Name": cue["name"], "Type": "0",
                "Start": f"{cue['start']:.3f}", "Num": num,
                "Red": str(r), "Green": str(g), "Blue": str(b),
            })
        track_keys.append(str(i))

    playlists = ET.SubElement(root, "PLAYLISTS")
    root_node = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="1")
    playlist_node = ET.SubElement(root_node, "NODE", Name="DjCaddy", Type="1",
                                  Entries=str(len(track_keys)))
    for key in track_keys:
        ET.SubElement(playlist_node, "TRACK", Key=key)

    return ET.tostring(root, encoding="UTF-8", xml_declaration=True).decode("utf-8")


def write_rekordbox_xml(tracks: list[dict], out_path: Path) -> None:
    """Scrive l'XML rekordbox su file (wrapper I/O attorno a build_rekordbox_xml)."""
    out_path.write_text(build_rekordbox_xml(tracks), encoding="utf-8")


def read_title_artist(filepath: Path) -> tuple[str, str]:
    """Legge titolo/artista dai tag (ID3 per mp3, Vorbis comment per flac).

    Best-effort: ripiega sul nome del file se i tag mancano o sono illeggibili.
    """
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(filepath, easy=True)
        if audio is not None:
            title = (audio.get("title") or [filepath.stem])[0]
            artist = (audio.get("artist") or [""])[0]
            return title, artist
    except Exception:
        pass
    return filepath.stem, ""


def build_m3u8(tracks: list[dict]) -> str:
    """Una playlist M3U8 con i brani nell'ordine dato.

    Percorsi assoluti: la playlist esce da qui per essere aperta da un altro
    programma sullo stesso Mac, non per essere spostata insieme ai file.
    """
    lines = ["#EXTM3U"]
    for t in tracks:
        path: Path = t["path"]
        duration = t.get("duration") or 0
        artist = t.get("artist") or ""
        name = t.get("name") or path.stem
        title = f"{artist} - {name}" if artist else name
        lines.append(f"#EXTINF:{duration:.0f},{title}")
        lines.append(str(path))
    return "\n".join(lines) + "\n"


def read_m3u8(text: str) -> list[str]:
    """I percorsi di una playlist M3U8, nell'ordine in cui stanno nel file.

    L'inverso di `build_m3u8`, e non solo per le playlist che escono da qui:
    l'M3U8 è il formato in cui ogni programma DJ sa salvare una scaletta,
    quindi questo è il modo di riprendere in mano un lavoro cominciato
    altrove.

    Delle righe che cominciano per `#` non serve niente: `#EXTINF` porta un
    titolo, ma il brano lo si ritrova per percorso, e il titolo scritto lì
    dentro può essere vecchio quanto la playlist. Alcuni programmi scrivono i
    percorsi come URI `file://`, con gli spazi percent-encoded: vanno riportati
    a percorsi, o non somiglierebbero a niente di quello che sta su disco.
    """
    paths = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("file://"):
            line = unquote(urlparse(line).path)
        paths.append(line)
    return paths


def _composed(text: str) -> str:
    """Il testo con gli accenti in un carattere solo (NFC), per confrontarlo."""
    return unicodedata.normalize("NFC", text)


def playlist_positions(paths, at_path: dict[str, int]) -> tuple[list[int], list[str]]:
    """Da una playlist letta da file alle posizioni sulla mappa.

    Il percorso scritto nel file e quello registrato sulla mappa possono non
    coincidere pur essendo lo stesso brano — un disco montato con un'altra
    lettera, la libreria spostata, una playlist salvata da un altro programma
    con percorsi relativi — quindi dopo il percorso si prova il nome del file.
    È il ripiego che salva il caso normale (la libreria è una sola, i nomi
    dentro sono unici) senza pretendere di indovinare: se due cartelle
    contengono lo stesso nome, vince la prima, e resta un brano da spostare a
    mano invece di una playlist che non si carica.

    Chi non si trova torna indietro per nome: sono i brani che sulla mappa non
    ci sono ancora, e la playlist non può indicarli perché una posizione che
    non esiste non è un brano.

    **Gli accenti si confrontano composti.** macOS scrive i nomi dei file
    decomposti — "Hervé" è "Herve" più il segno di accento, due caratteri —
    mentre chi riscrive la playlist di solito li ricompone: rekordbox lo fa.
    Sono la stessa parola sullo schermo e due stringhe diverse per il
    programma, quindi il percorso non combaciava e nemmeno il ripiego sul
    nome. Non è un caso di confine: 4.067 brani su 87.010 di questa libreria
    (il 4,7%) hanno un nome decomposto, e bastava un artista accentato perché
    una scaletta tornata da rekordbox arrivasse monca — con l'aggravante che
    il messaggio mandava ad aggiungere alla mappa una cartella che c'era già
    tutta. Si confronta allora una forma sola, e la mappa continua a
    conservare il percorso VERO, che è quello che poi riapre il file.
    """
    by_path: dict[str, int] = {}
    by_name: dict[str, int] = {}
    for path, i in at_path.items():
        by_path.setdefault(_composed(path), i)
        by_name.setdefault(_composed(os.path.basename(path)), i)

    found: list[int] = []
    missing: list[str] = []
    for path in paths:
        i = by_path.get(_composed(os.path.abspath(path)))
        if i is None:
            i = by_name.get(_composed(os.path.basename(path)))
        if i is None:
            missing.append(path)
        elif i not in found:
            found.append(i)
    return found, missing
