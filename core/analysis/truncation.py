"""Distinguere una canzone interrotta da un file corto voluto.

Il problema non si risolve con i metadati, ed e' stato verificato prima di
scrivere il resto: tagliando un mp3 a meta' download, l'intestazione
dichiara 17,4 s e i byte ne contengono 17,9 — coerenti — e `ffprobe` non
segnala nulla. Per il contenitore un download interrotto e un file corto
sono la stessa cosa.

Restano due domande, e servono entrambe:

1. COME FINISCE. Un file tagliato finisce a volume pieno, in mezzo alla
   musica; un file finito apposta si spegne. Misurato: canzone troncata
   1,25 · taglio netto a 30 s 1,16 · sample con dissolvenza 0,07.
2. DOVEVA ESSERE UNA CANZONE. Il primo segnale da solo non basta, perche'
   un estratto tagliato di netto finisce esattamente come un download
   interrotto. Ma i tag stanno in TESTA al file, quindi un download
   interrotto conserva artista, titolo e album della canzone intera: dice
   di essere un brano completo pur non durando niente.

L'incrocio dei due e' la risposta: finisce di netto E si dichiara una
canzone intera = quasi certamente troncato.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Oltre questo rapporto fra la coda e il corpo il file finisce a volume
# pieno. Sta fra 0,07 (dissolvenza) e 1,16 (taglio netto): mezzo e' largo.
ABRUPT_ENDING = 0.5

# Quanta parte finale si guarda. Meno di così pesca il singolo colpo di
# batteria finale; molto di più annacqua un taglio netto nella coda vera.
TAIL_SECONDS = 0.3


@dataclass
class ShortFileVerdict:
    path: Path
    seconds: float = 0.0
    ending: float | None = None        # coda / corpo, None se illeggibile
    artist: str | None = None
    title: str | None = None
    album: str | None = None
    error: str | None = None

    @property
    def ends_abruptly(self) -> bool:
        return self.ending is not None and self.ending >= ABRUPT_ENDING

    @property
    def claims_to_be_a_song(self) -> bool:
        """Si presenta come un brano completo: ha artista E titolo.

        L'album conta come rinforzo ma non si pretende: molti singoli non
        ce l'hanno, e chiederlo lascerebbe fuori troncature vere.
        """
        return bool(self.artist and self.title)

    @property
    def truncated(self) -> bool:
        return self.ends_abruptly and self.claims_to_be_a_song

    @property
    def verdict(self) -> str:
        if self.error:
            return "unreadable"
        if self.truncated:
            return "cut off"
        if self.ends_abruptly:
            return "ends sharp, untagged"
        return "finished"


def ending_ratio(path: Path, tail: float = TAIL_SECONDS) -> float | None:
    """Quanto suona la fine rispetto al corpo del file.

    Si rapporta alla MEDIANA e non al massimo: un solo picco altrove
    schiaccerebbe tutto il resto e farebbe sembrare spenta anche una coda a
    volume pieno.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    if len(y) < sr // 2:
        return None
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    quante = max(1, int(tail * sr / 512))
    corpo = float(np.median(rms))
    if corpo <= 0:
        return None
    return float(np.mean(rms[-quante:]) / corpo)


def _tags(path: Path) -> tuple[str | None, str | None, str | None]:
    import mutagen

    audio = mutagen.File(path, easy=True)
    if audio is None or audio.tags is None:
        return None, None, None
    primo = lambda k: (audio.tags.get(k) or [None])[0]
    return primo("artist"), primo("title"), primo("album")


def inspect(path: Path, seconds: float = 0.0) -> ShortFileVerdict:
    """Il verdetto su un file corto, con sotto le prove che lo sostengono."""
    v = ShortFileVerdict(path=Path(path), seconds=seconds)
    try:
        v.artist, v.title, v.album = _tags(v.path)
    except Exception as e:                                  # noqa: BLE001
        v.error = f"tag: {type(e).__name__}: {e}"
    try:
        v.ending = ending_ratio(v.path)
    except Exception as e:                                  # noqa: BLE001
        v.error = f"audio: {type(e).__name__}: {e}"
    return v
