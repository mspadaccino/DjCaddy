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
    ET.SubElement(root, "PRODUCT", Name="Wavecut", Version="1.0", Company="Wavecut")
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
    playlist_node = ET.SubElement(root_node, "NODE", Name="Wavecut", Type="1",
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
