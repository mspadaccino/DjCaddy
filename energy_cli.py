#!/usr/bin/env python3
"""Entry point 5 — la prova dell'energia su un campione della libreria.

Prima di aggiungere tre campi a ottantasettemila righe conviene guardare
cosa dicono su duecento brani che si conoscono a orecchio. Questo script
sceglie il campione DALLA MAPPA — stratificato per genere e per colore del
mood, così copre la libreria invece di un angolo — misura i tre ingredienti
e stampa la classifica.

    poetry run python energy_cli.py --sample 200

    # su file scelti a mano, anche senza mappa
    poetry run python energy_cli.py --files "a.flac" "b.flac"

    # vedere che campione uscirebbe, senza toccare l'audio
    poetry run python energy_cli.py --sample 200 --dry-run

Non scrive niente nella mappa: produce un CSV da guardare. La scala 1-10 è
calcolata SUL CAMPIONE, quindi si giudica l'ORDINE e la separazione fra i
brani, non il voto assoluto — quello avrà senso solo sulla libreria intera.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

from analysis import energy, mood_scale
from analysis.map_profile import (ProfileSettings, default_workers,
                                  gain_for_target, rhythm_offset)
from analysis.map_store import MapStore, default_store_dir

ANALYSIS_RATE = 44100

# La sezione aurea: avanzando di questa frazione a ogni passo si copre
# l'intervallo 0-1 senza mai ricadere vicino a dove si è già stati.
_GOLDEN = 0.6180339887498949


def _human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def progress_line(done: int, total: int, path, started: float) -> str:
    """La riga di avanzamento del backfill.

    Funzione a se' e non una chiusura dentro `main` perche' e' l'unico pezzo
    del backfill che gira solo dopo il primo brano misurato: dentro `main`
    non la vedeva nessun test, ed e' bastato un nome non definito per far
    morire il job dopo la prima misura, con la mappa gia' aperta e niente
    scritto.
    """
    each = (time.time() - started) / max(1, done)
    return (f"  {done:,}/{total:,} · {each:.2f}s a brano · "
            f"~{_human(each * (total - done))} alla fine · "
            f"{Path(path).name[:34]:34s}")


# --------------------------------------------------------------------------
# Il campione
# --------------------------------------------------------------------------

# Sotto questa durata non e' un brano: e' un drop, un sample, uno stacco da
# toolbox. Non si toglie dalla mappa — resta cercabile — ma non partecipa a
# definire la scala dell'energia, dove occuperebbe i decili bassi al posto
# della musica calma vera. La durata e' un fatto, mentre l'etichetta di genere
# sulla coda tira a indovinare: sul campione un "Non-Music - Political" era
# Mary Wells.
MIN_SECONDS = 60.0


def playable(rows: list[dict], min_seconds: float = MIN_SECONDS) -> list[dict]:
    """Le sole righe abbastanza lunghe da essere un brano.

    Una durata assente o a zero non si giudica e passa: e' un dato mancante,
    non un brano corto, e se la finestra e' davvero vuota se ne accorge
    `energy.usable`.
    """
    return [r for r in rows
            if not (r.get("duration") or 0.0) or r["duration"] >= min_seconds]


def stratified(rows: list[dict], count: int, seed: int = 0,
               min_seconds: float = MIN_SECONDS) -> list[dict]:
    """Un campione che copre generi e mood invece di un angolo della mappa.

    A giro si prende un brano per genere, finché non se ne hanno abbastanza:
    così i generi rari entrano comunque, che è il punto — una scala tarata
    solo sulla house non saprebbe dove mettere un breakbeat.

    Dentro un genere i brani si ordinano per colore del mood e la posizione
    da cui pescare avanza di un giro d'oro a ogni pesca — 0,618 di frazione,
    che è la sequenza che copre l'intervallo lasciando meno buchi possibile.

    Il passo fisso che c'era prima sembrava equivalente e non lo era: questa
    libreria ha PIÙ di duecento generi, quindi un campione da duecento ne
    prende uno per genere, e uno per genere col passo fisso vuol dire sempre
    il primo — cioè il brano più SCURO di ogni genere, duecento volte. Il
    campione va invece steso anche sul mood: l'energia si giudica soprattutto
    dove somiglia al mood, ed è lì che si vede se le due misure si ripetono.
    """
    by_genre: dict[str, list[dict]] = {}
    for row in playable(rows, min_seconds):
        by_genre.setdefault(row.get("top_genre") or "—", []).append(row)

    ordered: dict[str, list[dict]] = {}
    for genre, group in by_genre.items():
        # `valence` è None per i brani senza mood: vanno in fondo, non a zero,
        # che li metterebbe in mezzo fra i bui e i chiari senza motivo.
        ordered[genre] = sorted(
            group, key=lambda r: (mood_scale.valence(r.get("moods")) is None,
                                  mood_scale.valence(r.get("moods")) or 0.0,
                                  r["path"]))

    genres = sorted(ordered)
    random.Random(seed).shuffle(genres)     # nessun genere sempre primo

    picked, used, turn = [], {g: set() for g in genres}, 0
    while len(picked) < count:
        moved = False
        for genre in genres:
            group = ordered[genre]
            if len(used[genre]) >= len(group) or len(picked) >= count:
                continue
            at = int(((turn * _GOLDEN) % 1.0) * len(group))
            while at in used[genre]:         # il primo posto libero da lì
                at = (at + 1) % len(group)
            used[genre].add(at)
            picked.append(group[at])
            turn += 1
            moved = True
        if not moved:
            break                            # finiti i brani, non il conteggio
    return picked[:count]


# --------------------------------------------------------------------------
# La misura
# --------------------------------------------------------------------------

def rhythm_window(path: Path, duration: float, settings: ProfileSettings):
    """I soli secondi che servono, decodificati e basta.

    `EasyLoader` apre il file al punto giusto invece di decodificarlo tutto:
    su un brano di sette minuti è la differenza fra secondi e frazioni di
    secondo, ed è ciò che rende il backfill una serata invece che una
    settimana. Se la versione di Essentia in uso non lo espone si ripiega su
    `MonoLoader`, che dà lo stesso audio pagandolo di più.
    """
    from essentia.standard import EasyLoader, MonoLoader

    start = rhythm_offset(duration, settings)
    try:
        return EasyLoader(filename=str(path), sampleRate=ANALYSIS_RATE,
                          startTime=start,
                          endTime=start + settings.rhythm_seconds,
                          replayGain=0.0)()
    except Exception:
        audio = MonoLoader(filename=str(path), sampleRate=ANALYSIS_RATE,
                           resampleQuality=1)()
        first = int(start * ANALYSIS_RATE)
        return audio[first:first + int(settings.rhythm_seconds * ANALYSIS_RATE)]


def probe(row: dict, settings: ProfileSettings) -> dict:
    """I tre ingredienti di un brano, dalla sua riga sulla mappa.

    BPM e loudness NON si ricalcolano: sono già nella riga. È il motivo per
    cui questa passata costa una frazione dell'analisi — niente modello,
    niente tempo, niente tonalità, solo trenta secondi di audio e una FFT.
    """
    from essentia.standard import OnsetRate

    path = Path(row["path"])
    audio = rhythm_window(path, float(row.get("duration") or 0.0), settings)
    if audio is None or not len(audio):
        return dict.fromkeys(energy.INGREDIENTS)

    # Allo stesso livello a cui il resto della pipeline porta i brani. Per
    # due misure su tre è cosmetico — un rapporto di bande e un centroide
    # non cambiano se moltiplichi il segnale — ed è esattamente il motivo
    # per cui la loudness non può rientrare da questa porta.
    audio = (np.asarray(audio, dtype=np.float32)
             * gain_for_target(row.get("lufs"))).astype(np.float32)

    rate = float(OnsetRate()(audio)[1])
    return energy.measure(audio, ANALYSIS_RATE, rate, row.get("bpm"))


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Il backfill: i quattro campi sulle righe che non ce l'hanno
# --------------------------------------------------------------------------

_SETTINGS: ProfileSettings | None = None


def _worker_start(settings: ProfileSettings) -> None:
    global _SETTINGS
    _SETTINGS = settings


def _worker(row: dict) -> tuple:
    try:
        return row["path"], probe(row, _SETTINGS), None
    except Exception as exc:                      # un brano rotto non ferma il job
        return row["path"], None, f"{type(exc).__name__}: {exc}"[:90]


def missing(rows: list[dict]) -> list[dict]:
    """Le righe che i quattro campi non ce l'hanno ancora.

    Si guarda se la CHIAVE c'è, non se il valore è pieno: una finestra muta
    dà quattro `None` legittimi, e cercarli per valore rimetterebbe quel brano
    in coda a ogni giro per sempre.
    """
    return [r for r in rows if not all(n in r for n in energy.INGREDIENTS)]


def backfill(store: MapStore, settings: ProfileSettings, workers: int,
             flush_every: int = 500, on_progress=None) -> tuple[int, list]:
    """Misura e scrive i quattro campi, riscrivendo `tracks.jsonl` per intero.

    Si riscrive invece di appendere perché qui non si aggiungono BRANI, si
    aggiunge un CAMPO a righe che ci sono già: l'ordine non cambia, quindi
    embedding e coordinate restano allineati e intatti — niente mezzo giga di
    vettori duplicati e niente proiezione da rifare. Vedi `MapStore.rewrite`.

    Si salva ogni `flush_every` brani: due ore di lavoro non devono dipendere
    dal fatto che nessuno chiuda il coperchio del portatile.

    **Quanti worker.** Il default e' meta' dei core, ma qui non e' la CPU a
    decidere: si legge trenta secondi di audio a brano da ottantasettemila
    file sparsi, e se la libreria sta su un disco esterno il collo di
    bottiglia e' il volume. Misurato su una libreria vera, con cinque worker
    ogni brano costava 3,3 secondi di CPU contro gli 0,56 della stessa
    funzione girata da sola: cinque lettori concorrenti fanno saltare la
    coda del disco invece di leggere. Due e' risultato il numero migliore —
    mentre uno aspetta il disco l'altro calcola, e dal terzo in poi si
    perde piu' di quanto si guadagni. Con la libreria su disco interno il
    discorso cambia e conviene alzarlo.
    """
    from concurrent.futures import ProcessPoolExecutor

    todo = missing(store.rows)
    if not todo:
        return 0, []
    at_path = {r["path"]: r for r in store.rows}
    done, failed = 0, []

    def keep(path: str, values: dict | None) -> None:
        row = at_path[path]
        for name in energy.INGREDIENTS:
            value = (values or {}).get(name)
            row[name] = round(value, 4) if value is not None else None

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_start,
                             initargs=(settings,)) as pool:
        for path, values, error in pool.map(_worker, todo, chunksize=8):
            keep(path, values)
            if error:
                failed.append((Path(path).name, error))
            done += 1
            if done % flush_every == 0:
                store.rewrite()
            if on_progress:
                on_progress(done, len(todo), path)
    store.rewrite()
    return done, failed


def _table(rows: list[dict], measures: list[dict]) -> list[dict]:
    columns = {name: [m.get(name) if m.get(name) is not None else np.nan
                      for m in measures] for name in energy.INGREDIENTS}
    level = energy.levels(*(columns[n] for n in energy.INGREDIENTS))
    out = []
    for row, measure, value in zip(rows, measures, level):
        out.append({
            "energy": "" if not np.isfinite(value) else int(value),
            "file": Path(row["path"]).name,
            "genre": row.get("top_genre") or "",
            "moods": row.get("moods") or "",
            "bpm": row.get("bpm") or "",
            "groove": row.get("danceability") if row.get("danceability") is not None else "",
            "lufs": row.get("lufs") if row.get("lufs") is not None else "",
            **{n: "" if measure.get(n) is None else round(measure[n], 3)
               for n in energy.INGREDIENTS},
            "path": row["path"],
        })
    return sorted(out, key=lambda r: (r["energy"] == "", -(r["energy"] or 0)))


def write_playlist(table: list[dict], path: Path) -> int:
    """La stessa tabella come playlist, dal più calmo al più energico.

    Perche' una scala di energia si giudica ASCOLTANDOLA di fila: letta in
    tabella si vede se un brano sta troppo in alto, ma sentita in rampa si
    sente se la rampa sale davvero — che e' la domanda vera, visto che a
    questo asse serve reggere la costruzione di un set.

    I brani che non si e' riusciti a misurare vanno in coda con un punto
    interrogativo invece di sparire: sentirli e' il modo piu' rapido di
    capire se il file e' rotto o se la finestra era davvero muta.
    """
    order = sorted(table, key=lambda r: (r["energy"] == "", r["energy"] or 0))
    with path.open("w", encoding="utf-8") as fh:
        fh.write("#EXTM3U\n")
        for row in order:
            level = row["energy"] if row["energy"] != "" else "?"
            fh.write(f"#EXTINF:-1,{level} · {row['file']}\n{row['path']}\n")
    return len(order)


def _from_csv(path: Path) -> list[dict]:
    """La tabella di una prova gia' fatta, senza riaprire un solo file audio.

    Serve a rileggere un campione di ieri — o a farne la playlist — senza
    ripagare i due minuti di misura.
    """
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["energy"] = int(row["energy"]) if row["energy"] else ""
    return sorted(rows, key=lambda r: (r["energy"] == "", -(r["energy"] or 0)))


def _print_table(table: list[dict]) -> None:
    print(f"\n{'en':>3s}  {'BPM':>5s} {'grv':>4s}  {'dens':>5s} {'bass':>5s} "
          f"{'brt':>5s} {'puls':>5s}  {'genere':26s} {'file'}")
    for r in table:
        print(f"{str(r['energy']):>3s}  {str(r['bpm'])[:5]:>5s} "
              f"{str(r['groove'])[:4]:>4s}  "
              f"{str(r['energy_density'])[:5]:>5s} "
              f"{str(r['energy_bass'])[:5]:>5s} "
              f"{str(r['energy_bright'])[:5]:>5s} "
              f"{str(r['energy_pulse'])[:5]:>5s}  "
              f"{(r['genre'] or '—')[:26]:26s} {r['file'][:44]}")


def _correlations(table: list[dict]) -> None:
    """I due test di ammissione, sul campione.

    Un asse che ridice il BPM non vale la manopola per sceglierlo; un asse
    che ridice la loudness vuol dire che l'abbiamo ricostruita per vie
    traverse, e tutto il ragionamento per cui non l'abbiamo usata era vano.
    """
    have = [r for r in table if r["energy"] != ""]
    if len(have) < 20:
        print("\n  Troppi pochi brani misurati per le correlazioni.")
        return
    value = np.array([r["energy"] for r in have], dtype=float)
    print("\n  Correlazione dell'energia con quello che c'è già:")
    for name, key, verdict in (("BPM", "bpm", "ridice il tempo"),
                               ("groove", "groove", "ridice il groove"),
                               ("loudness", "lufs", "è loudness travestita")):
        other = np.array([r[key] if r[key] != "" else np.nan for r in have],
                         dtype=float)
        ok = np.isfinite(other)
        if ok.sum() < 20:
            continue
        r = float(np.corrcoef(value[ok], other[ok])[0, 1])
        flag = f"  <-- ATTENZIONE: {verdict}" if abs(r) > 0.7 else ""
        print(f"    {name:9s} r = {r:+.2f}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Misura i tre ingredienti dell'energia su un campione.")
    parser.add_argument("--sample", type=int, default=200,
                        help="Quanti brani pescare dalla mappa")
    parser.add_argument("--files", type=Path, nargs="+",
                        help="Brani scelti a mano invece del campione")
    parser.add_argument("--store", type=Path, default=default_store_dir())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-seconds", type=float, default=MIN_SECONDS,
                        help="Sotto questa durata non e' un brano (0 = tieni tutto)")
    parser.add_argument("--out", type=Path, default=Path("energy_sample.csv"))
    parser.add_argument("--from-csv", type=Path,
                        help="Rilegge una prova gia' fatta invece di misurare")
    parser.add_argument("--backfill", action="store_true",
                        help="Misura e scrive i quattro campi su TUTTA la mappa")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="Con la libreria su un disco esterno il numero "
                             "buono e' BASSO: misurato, 2 batte sia 5 che 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra il campione senza aprire l'audio")
    args = parser.parse_args()

    settings = ProfileSettings()
    if args.backfill:
        store = MapStore.load(args.store)
        if not len(store):
            parser.error(f"Nessuna mappa in {args.store}.")
        todo = missing(store.rows)
        print(f"Mappa: {len(store):,} brani · da misurare {len(todo):,} "
              f"· parallelismo {args.workers}")
        if not todo:
            print("  Tutti i brani hanno gia' i quattro campi.")
            return
        print("  Nessun modello, nessun embedding: solo trenta secondi di "
              "audio a brano.\n  Si puo' fermare — quello che e' scritto "
              "resta e la volta dopo riparte da li'.\n", flush=True)

        t0 = time.time()

        def report(done, total, path):
            sys.stdout.write("\r" + progress_line(done, total, path, t0))
            sys.stdout.flush()

        done, failed = backfill(store, settings, args.workers,
                                on_progress=report)
        print(f"\n\nScritti {done:,} brani in {_human(time.time() - t0)}.")
        if failed:
            print(f"  falliti {len(failed)} (restano senza energia):")
            for name, why in failed[:5]:
                print(f"    {name[:50]:50s} {why}")
        print("\n  Embedding e coordinate non sono stati toccati: "
              "niente da riproiettare.")
        return

    if args.from_csv:
        table = _from_csv(args.from_csv)
        _print_table(table)
        _correlations(table)
        playlist = args.from_csv.with_suffix(".m3u")
        print(f"\n  Playlist di {write_playlist(table, playlist)} brani in "
              f"{playlist}, dal piu' calmo al piu' energico.")
        return

    if args.files:
        rows = [{"path": str(p.resolve()), "duration": 0.0, "bpm": None,
                 "lufs": None, "top_genre": "", "moods": ""} for p in args.files]
    else:
        store = MapStore.load(args.store)
        if not len(store):
            parser.error(f"Nessuna mappa in {args.store}: usa --files.")
        rows = stratified(store.rows, args.sample, args.seed, args.min_seconds)
        genres = len({r.get("top_genre") for r in rows})
        short = len(store) - len(playable(store.rows, args.min_seconds))
        print(f"Mappa: {len(store):,} brani · campione {len(rows)} "
              f"su {genres} generi")
        if short:
            print(f"  esclusi {short:,} sotto {args.min_seconds:.0f}s "
                  f"({short / len(store):.1%} della libreria): non sono brani")

    if args.dry_run:
        for row in rows:
            print(f"  {(row.get('top_genre') or '—')[:28]:28s} "
                  f"{(row.get('moods') or '—')[:34]:34s} "
                  f"{Path(row['path']).name[:50]}")
        return

    print(f"Misuro {len(rows)} brani (30 s ciascuno, niente modello)…\n",
          flush=True)
    measures, failed, t0 = [], [], time.time()
    for i, row in enumerate(rows, start=1):
        try:
            measures.append(probe(row, settings))
        except Exception as exc:                      # un brano rotto non ferma la prova
            measures.append(dict.fromkeys(energy.INGREDIENTS))
            failed.append((Path(row["path"]).name, str(exc)[:60]))
        sys.stdout.write(f"\r  {i}/{len(rows)} · "
                         f"{(time.time() - t0) / i:.2f}s a brano")
        sys.stdout.flush()

    table = _table(rows, measures)
    each = (time.time() - t0) / max(1, len(rows))
    empty = sum(1 for m in measures if all(v is None for v in m.values()))
    print(f"\n\nFatto in {time.time() - t0:.0f}s · {each:.2f}s a brano "
          f"· sugli 87.000 sarebbero ~{each * 87000 / 3600:.1f} h a un worker")
    if empty:
        print(f"  finestre mute o illeggibili, non misurate: {empty}")
    if failed:
        print(f"  falliti {len(failed)}: " +
              ", ".join(f"{n} ({e})" for n, e in failed[:3]))

    _print_table(table)
    _correlations(table)

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    playlist = args.out.with_suffix(".m3u")
    count = write_playlist(table, playlist)
    print(f"\n  Tabella completa in {args.out} (col percorso di ogni brano).")
    print(f"  Playlist di {count} brani in {playlist}, dal piu' calmo al piu' "
          f"energico:\n    open {playlist}")


if __name__ == "__main__":
    main()
