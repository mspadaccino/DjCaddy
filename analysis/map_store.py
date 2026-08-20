"""Dove vive la mappa della libreria: metadati, embedding, coordinate.

Tre file in una cartella, e la ragione della divisione è come vengono
scritti:

- `tracks.jsonl`  — una riga JSON per brano, si APPENDE mentre il job gira;
- `embeddings.f32` — i vettori grezzi float32 uno dietro l'altro, si
  APPENDE anche lui (la riga n del primo file è il blocco n di questo);
- `coords.npy`    — le due colonne X/Y di UMAP, RISCRITTE per intero ogni
  volta che si riproietta, perché la proiezione è un fatto della libreria
  intera e non di un brano.

Appendere invece di riscrivere è la differenza fra un job interrompibile e
uno no: a 90.000 brani la matrice degli embedding è mezzo giga, e
risalvarla ogni cinquanta brani vorrebbe dire passare la notte a
riscrivere lo stesso mezzo giga.

**Niente FAISS.** La specifica lo propone per le query di vicinanza, ma a
questa scala non serve: 90.000 × 1280 float32 sono 460 MB, e trovare i
vicini di un brano è un prodotto matrice-vettore che numpy fa in qualche
decina di millisecondi. Un indice approssimato si aggiunge quando la
ricerca esatta non sta più nel tempo dell'interazione — qui ci sta.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .map_profile import EMBEDDING_DIM, TrackProfile

STORE_VERSION = 1


def default_store_dir() -> Path:
    return Path.home() / ".cache" / "dj-library-tools" / "map"


def _key(path) -> str:
    """Il percorso in forma canonica: è così che si riconosce un brano.

    `abspath` e non `resolve()`: il secondo interroga il disco per ogni link
    della catena, e su novantamila file su un disco esterno si sentirebbe.
    Quello che conta qui è un'altra differenza: indicare la stessa cartella
    come "test_mp3" o come "/Users/…/test_mp3" darebbe due chiavi diverse
    per lo stesso brano, e quindi due punti sulla mappa.
    """
    return os.path.abspath(str(path))


@dataclass
class MapStore:
    """La mappa su disco, aperta in lettura e scrittura."""

    directory: Path
    rows: list[dict]
    embeddings: np.ndarray            # (N, EMBEDDING_DIM) float32
    coords: np.ndarray | None = None  # (N, 2) float32, None finché non si proietta

    # --- file della cartella ---
    @property
    def rows_file(self) -> Path:
        return self.directory / "tracks.jsonl"

    @property
    def embeddings_file(self) -> Path:
        return self.directory / "embeddings.f32"

    @property
    def coords_file(self) -> Path:
        return self.directory / "coords.npy"

    @property
    def meta_file(self) -> Path:
        return self.directory / "meta.json"

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def placed(self) -> int:
        """Quanti brani hanno un posto sulla mappa. Sono i primi `placed`:
        chi è arrivato dopo l'ultima proiezione aspetta la prossima."""
        return 0 if self.coords is None else len(self.coords)

    @property
    def projected(self) -> bool:
        """Le coordinate ci sono e valgono per TUTTI i brani presenti."""
        return self.placed == len(self.rows) and bool(self.rows)

    # --- lettura ---
    @classmethod
    def load(cls, directory: Path | None = None) -> "MapStore":
        directory = Path(directory or default_store_dir())
        store = cls(directory=directory, rows=[],
                    embeddings=np.empty((0, EMBEDDING_DIM), dtype=np.float32))
        if not store.rows_file.exists():
            return store

        rows = []
        with store.rows_file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        break      # riga tronca: fine di ciò di cui fidarsi
        vectors = np.fromfile(store.embeddings_file, dtype=np.float32) \
            if store.embeddings_file.exists() else np.empty(0, dtype=np.float32)
        vectors = vectors[:len(vectors) // EMBEDDING_DIM * EMBEDDING_DIM]
        vectors = vectors.reshape(-1, EMBEDDING_DIM)

        # I due file si appendono uno dopo l'altro: se il job è stato ucciso
        # nel mezzo, uno dei due ha un brano in più. Si tiene la parte su cui
        # sono d'accordo — un brano perso si rifà in otto secondi, una mappa
        # disallineata mostrerebbe per sempre il vicino sbagliato.
        keep = min(len(rows), len(vectors))
        rows, vectors = rows[:keep], vectors[:keep]

        # Un brano rianalizzato viene APPESO, non riscritto: i file non si
        # riaprono mai in mezzo, è il prezzo di poter interrompere il lavoro.
        # Quindi lo stesso percorso può comparire due volte — succede quando
        # il file è cambiato dopo essere finito sulla mappa, per esempio
        # perché Tag analysis gli ha riscritto i tag. Vale l'ULTIMA riga: è
        # quella che descrive il file com'è adesso.
        latest = {}
        for i, row in enumerate(rows):
            latest[_key(row["path"])] = i
        kept = sorted(latest.values())
        compacted = len(kept) < len(rows)
        if compacted:
            rows = [rows[i] for i in kept]
            vectors = vectors[kept]

        store.rows, store.embeddings = rows, vectors

        # Le coordinate stanno in fila come le righe, quindi ne coprono un
        # PREFISSO: i brani arrivati dopo l'ultima proiezione non ne hanno, e
        # va benissimo — la mappa continua a mostrare quelli che ce l'hanno
        # mentre il job aggiunge gli altri. Se però una riga è stata assorbita
        # come duplicato, la fila si è accorciata nel mezzo e la
        # corrispondenza non vale più: meglio nessuna coordinata che le
        # coordinate di un altro brano.
        if store.coords_file.exists() and not compacted:
            coords = np.load(store.coords_file)
            if len(coords) <= len(rows):
                store.coords = coords.astype(np.float32)
        return store

    # --- scrittura ---
    def append(self, profiles) -> int:
        """Aggiunge dei profili, su disco e in memoria. Ritorna quanti.

        I profili in errore non entrano: la mappa è fatta di brani che si
        sono lasciati analizzare, e chi non ce l'ha fatta lo racconta lo
        stato del job.
        """
        good = [p for p in profiles if p.error is None and p.embedding is not None]
        if not good:
            return 0
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.rows_file.open("a", encoding="utf-8") as text, \
                self.embeddings_file.open("ab") as binary:
            for profile in good:
                row = _row_of(profile)
                text.write(json.dumps(row, ensure_ascii=False) + "\n")
                binary.write(np.asarray(profile.embedding, dtype=np.float32).tobytes())
                self.rows.append(row)
        self.embeddings = np.vstack(
            [self.embeddings] + [np.asarray(p.embedding, dtype=np.float32)[None, :]
                                 for p in good])
        # Le coordinate valgono per i brani di prima: da qui in poi la mappa
        # è incompleta finché non la si riproietta.
        self.coords = None
        self.write_meta()
        return len(good)

    def remove(self, paths) -> int:
        """Toglie dei brani dalla mappa. Ritorna quanti ne ha tolti.

        È l'unica operazione che RISCRIVE i file invece di appenderli: una
        riga in mezzo non si toglie dalla coda. Si scrive di fianco e poi si
        rinomina, così un'interruzione lascia intera la mappa di prima.

        Le coordinate dei brani che restano non vengono buttate: la posizione
        di un brano è già stata calcolata e non dipende dai vicini che si
        tolgono dopo. Si tolgono solo le loro.
        """
        doomed = {_key(p) for p in paths}
        keep = [i for i, row in enumerate(self.rows)
                if _key(row["path"]) not in doomed]
        if len(keep) == len(self.rows):
            return 0

        removed = len(self.rows) - len(keep)
        rows = [self.rows[i] for i in keep]
        vectors = self.embeddings[keep] if len(self.embeddings) else self.embeddings
        placed = self.placed
        coords = (self.coords[[i for i in keep if i < placed]]
                  if self.coords is not None else None)

        self.directory.mkdir(parents=True, exist_ok=True)
        text_tmp = self.rows_file.with_suffix(".jsonl.tmp")
        text_tmp.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8")
        binary_tmp = self.embeddings_file.with_suffix(".f32.tmp")
        binary_tmp.write_bytes(np.asarray(vectors, dtype=np.float32).tobytes())
        text_tmp.replace(self.rows_file)
        binary_tmp.replace(self.embeddings_file)

        self.rows, self.embeddings, self.coords = rows, vectors, coords
        if coords is None:
            self.coords_file.unlink(missing_ok=True)
        else:
            np.save(self.coords_file, coords)
        self.write_meta()
        return removed

    def relocate(self, old_prefix, new_prefix) -> tuple[int, int]:
        """La libreria ha cambiato posto: aggiorna i percorsi.

        Ritorna (quante righe spostate, quante di quelle non si trovano al
        nuovo indirizzo) — il secondo numero dice subito se il percorso
        nuovo è quello sbagliato, invece di scoprirlo rianalizzando tutto.

        Serve perché un brano sulla mappa è riconosciuto dal suo percorso
        assoluto, e un disco nuovo ne cambia la prima parte
        (`/Volumes/Crucial X9/…` diventa `/Volumes/Altro/…`). Senza questo,
        una libreria spostata è una libreria SCONOSCIUTA: le righe vecchie
        restano a indicare file che non esistono più e ogni brano viene
        rianalizzato da capo.

        Al nuovo indirizzo si riprende anche la data di modifica, se la
        DIMENSIONE combacia. Copiare senza conservare le date è normale
        (`cp` lo fa), e senza questo accorgimento ogni file risulterebbe
        cambiato e verrebbe rianalizzato — cioè il contrario di quello che
        questa funzione serve a evitare. Se invece la dimensione è diversa
        il file non è più lo stesso, e allora è giusto che torni in coda.

        Embedding e coordinate non si toccano: è lo stesso brano, è la stessa
        analisi, è cambiato soltanto dove sta. Solo `tracks.jsonl` viene
        riscritto — di fianco e poi rinominato, come sempre.
        """
        old, new = _key(old_prefix), _key(new_prefix)
        moved = missing = 0
        for row in self.rows:
            path = _key(row["path"])
            # Il confronto è sul confine di cartella: senza, "/Volumes/X9"
            # prenderebbe dentro anche "/Volumes/X9 Backup".
            if path != old and not path.startswith(old + os.sep):
                continue
            row["path"] = new + path[len(old):]
            row["folder"] = os.path.dirname(row["path"])
            moved += 1
            try:
                stat = os.stat(row["path"])
            except OSError:
                missing += 1
                continue
            if stat.st_size == row.get("size"):
                row["mtime"] = stat.st_mtime
        if not moved:
            return 0, 0

        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.rows_file.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows),
            encoding="utf-8")
        tmp.replace(self.rows_file)
        self.write_meta()
        return moved, missing

    def set_coords(self, coords) -> None:
        coords = np.asarray(coords, dtype=np.float32)
        if len(coords) != len(self.rows):
            raise ValueError(
                f"{len(coords)} coordinate per {len(self.rows)} brani")
        self.coords = coords
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.coords_file, coords)
        self.write_meta()

    def write_meta(self) -> None:
        self.meta_file.write_text(json.dumps({
            "version": STORE_VERSION,
            "dim": EMBEDDING_DIM,
            "tracks": len(self.rows),
            "projected": bool(self.projected),
        }, indent=1))

    # --- interrogazione ---
    def index_of(self, path) -> int | None:
        target = _key(path)
        for i, row in enumerate(self.rows):
            if _key(row["path"]) == target:
                return i
        return None

    def known_paths(self) -> dict[str, tuple[float, int]]:
        return {_key(row["path"]): (row.get("mtime", 0.0), row.get("size", 0))
                for row in self.rows}

    def pending(self, paths) -> list[Path]:
        """Quali dei percorsi dati non sono ancora nella mappa.

        Un file già presente ma cambiato (mtime o dimensione diversi) torna
        in coda: è un altro audio, anche se il nome è lo stesso.
        """
        known = self.known_paths()
        todo = []
        for path in paths:
            path = Path(path)
            entry = known.get(_key(path))
            if entry is None:
                todo.append(path)
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if (stat.st_mtime, stat.st_size) != entry:
                todo.append(path)
        return todo

    def similar(self, index: int, k: int = 20,
                limit: int | None = None) -> list[tuple[int, float]]:
        """I brani acusticamente più vicini, per coseno sugli embedding.

        Questa è la vicinanza VERA, nelle 1280 dimensioni: la mappa 2D è una
        sua ombra, comoda da guardare ma schiacciata. Serve a chiedere "cosa
        suona come questo" senza passare per il disegno.
        """
        if not len(self.embeddings):
            return []
        # `limit` tiene la risposta dentro ai brani che la pagina conosce:
        # quelli già sulla mappa. Un vicino senza coordinate non si potrebbe
        # né mostrare né raggiungere.
        matrix = self.embeddings[:limit] if limit else self.embeddings
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = matrix / np.maximum(norms, 1e-9)
        scores = normalized @ normalized[index]
        scores[index] = -np.inf
        best = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        best = best[np.argsort(-scores[best])]
        return [(int(i), float(scores[i])) for i in best]


def _row_of(profile: TrackProfile) -> dict:
    row = profile.to_row()
    # Assoluto anche su disco: la riga serve poi a riaprire il file (lettore,
    # export), e un percorso relativo vale solo finché non cambia la cartella
    # da cui è partito il programma.
    row["path"] = _key(profile.path)
    row["folder"] = os.path.dirname(row["path"])
    try:
        stat = Path(profile.path).stat()
        row["mtime"], row["size"] = stat.st_mtime, stat.st_size
    except OSError:
        row["mtime"], row["size"] = 0.0, 0
    return row
