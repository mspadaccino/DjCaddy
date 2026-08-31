#!/usr/bin/env python3
"""Entry point 4 — la mappa della libreria come job lungo, fuori dall'app.

Profila una cartella (embedding + etichette + tempo + tonalità) e scrive
nella mappa, poi eventualmente ricalcola la proiezione UMAP.

    poetry run python map_cli.py CARTELLA [opzioni]

Esempi:
    # tutto quello che manca, e alla fine riproietta
    poetry run python map_cli.py "/percorso/della/libreria" --project

    # una prova da cinquanta brani prima di lanciarlo sul serio
    poetry run python map_cli.py ~/Music --limit 50

    # solo la proiezione, su una mappa già costruita
    poetry run python map_cli.py --project-only

Si può fermare quando si vuole: quello che è nella mappa ci resta, e la
volta dopo riparte da lì.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from core.analysis import energy, mood_scale
from core.analysis.essentia_tags import available, missing_models
from core.analysis.map_job import DEFAULT_MAP_STATE_FILE, run_job
from core.analysis.map_profile import ProfileSettings, default_workers
from core.analysis.map_projection import ProjectionSettings, project
from core.analysis.map_store import MapStore, default_store_dir


def _human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _report(state) -> None:
    # Il ritmo si mostra accanto all'attesa perché è quello che la produce:
    # senza, un'attesa che si allunga non si sa se venga da una macchina che
    # ha rallentato o da un conteggio sballato.
    ritmo = f" · {state.seconds_each:.2f}s a brano" if state.done > 2 else ""
    fine = f" · ~{_human(state.eta_seconds)} alla fine" if state.done > 2 else ""
    sys.stdout.write(
        f"\r  {state.done:,}/{state.total:,} · sulla mappa {state.written:,} "
        f"· falliti {state.failed:,}{ritmo}{fine} · {state.current[:40]:40s}")
    sys.stdout.flush()


def reproject(store_dir: Path, settings: ProjectionSettings) -> None:
    store = MapStore.load(store_dir)
    if not len(store):
        print("Mappa vuota: niente da proiettare.")
        return
    print(f"Proietto {len(store):,} brani con UMAP…", flush=True)
    t0 = time.time()
    store.set_coords(project(store.embeddings, settings))
    print(f"Fatto in {_human(time.time() - t0)}.")


def fields(store: MapStore) -> None:
    """Cosa la mappa ha davvero addosso, campo per campo.

    Nessun modello, nessun audio: legge `tracks.jsonl` e conta. Serve a
    sapere se un backfill è arrivato in fondo, e a guardare l'unica cosa che
    dice se il grafico a quadranti ha senso — se energia e valence dicono
    due cose diverse o la stessa cosa detta due volte. Se fossero la stessa,
    i brani starebbero su una diagonale e due quadranti su quattro sarebbero
    vuoti: un asse solo travestito da due.
    """
    rows = store.rows
    total = len(rows)
    print(f"Righe: {total:,}  ·  piazzate sulla mappa: {store.placed:,}")

    print("\nCampi:")
    for name in (*energy.INGREDIENTS, "valence", "mood_evidence", "mood_conf"):
        have = sum(1 for row in rows if row.get(name) is not None)
        short = total - have
        print(f"  {name:<16} {have:>8,} / {total:,}"
              + (f"   ← ne mancano {short:,}" if short else ""))

    drive = energy.from_rows(rows)
    colour = np.asarray(mood_scale.from_rows(rows), dtype=float)
    both = np.isfinite(drive) & np.isfinite(colour)
    if both.sum() < 100:
        print("\nTroppo pochi brani con tutte e due le misure per dire altro.")
        return

    # Sui RANGHI e non sui numeri grezzi: è come si leggono sugli assi, ed è
    # l'unico modo di mettere a confronto un rapporto di bande e una valence.
    across = energy.ranks(np.where(both, colour, np.nan))[both]
    up = energy.ranks(np.where(both, drive, np.nan))[both]
    print(f"\nEnergia e valence, su {int(both.sum()):,} brani:")
    print(f"  correlazione fra i due ranghi: {np.corrcoef(across, up)[0, 1]:+.3f}")
    print("  i quattro quadranti:")
    for pushing, name in ((True, "spinti"), (False, "calmi")):
        half = (up >= 0.5) == pushing
        print(f"    {name:<7} {int((half & (across < 0.5)).sum()):>8,} bui"
              f"  {int((half & (across >= 0.5)).sum()):>8,} chiari")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Costruisce la mappa acustica della libreria.")
    parser.add_argument("folder", type=Path, nargs="?",
                        help="Cartella da profilare (ricorsiva)")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--limit", type=int, default=0,
                        help="Fermati dopo N brani (0 = tutti)")
    parser.add_argument("--store", type=Path, default=default_store_dir(),
                        help="Cartella della mappa")
    parser.add_argument("--project", action="store_true",
                        help="Ricalcola la proiezione UMAP a fine job")
    parser.add_argument("--project-only", action="store_true",
                        help="Solo la proiezione, senza analizzare niente")
    parser.add_argument("--relocate", nargs=2, metavar=("VECCHIO", "NUOVO"),
                        help="La libreria ha cambiato posto: aggiorna i "
                             "percorsi sulla mappa invece di rianalizzarla")
    parser.add_argument("--neighbors", type=int, default=ProjectionSettings.n_neighbors)
    parser.add_argument("--min-dist", type=float, default=ProjectionSettings.min_dist)
    parser.add_argument("--genre-threshold", type=float,
                        default=ProfileSettings.genre_threshold)
    parser.add_argument("--mood-threshold", type=float,
                        default=ProfileSettings.mood_threshold)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_MAP_STATE_FILE)
    parser.add_argument("--fields", action="store_true",
                        help="cosa la mappa ha addosso campo per campo, e se "
                             "energia e valence dicono due cose diverse")
    args = parser.parse_args()

    if args.fields:
        fields(MapStore.load(args.store))
        return

    projection = ProjectionSettings(n_neighbors=args.neighbors,
                                    min_dist=args.min_dist)
    if args.relocate:
        vecchio, nuovo = args.relocate
        store = MapStore.load(args.store)
        moved, missing = store.relocate(vecchio, nuovo)
        print(f"Spostate {moved:,} righe su {len(store):,}.")
        if missing:
            print(f"  ATTENZIONE: {missing:,} di quelle non si trovano sotto "
                  f"{nuovo}.\n  Se sono tutte, il percorso nuovo non è quello "
                  "giusto: rilancia con quello vero (i percorsi sono già "
                  "stati riscritti, quindi il VECCHIO adesso è quello che hai "
                  "appena messo).")
        elif moved:
            print("  Tutte ritrovate al nuovo indirizzo: la prossima analisi "
                  "le salta.")
        return

    if args.project_only:
        reproject(args.store, projection)
        return

    if args.folder is None or not args.folder.is_dir():
        parser.error("serve una cartella (o --project-only)")
    if not available():
        parser.error("essentia non è importabile in questo ambiente")
    if missing_models():
        parser.error(f"Modelli mancanti: {', '.join(missing_models())}")

    print(f"Cartella: {args.folder}")
    print(f"Mappa: {args.store} · parallelismo {args.workers} "
          f"· stato in {args.state_file}")
    print("Costruisco la coda…", flush=True)

    t0 = time.time()
    state = run_job(
        args.folder,
        ProfileSettings(genre_threshold=args.genre_threshold,
                        mood_threshold=args.mood_threshold),
        workers=args.workers, store_dir=args.store,
        state_file=args.state_file, limit=args.limit, on_progress=_report)

    print(f"\n\nFinito in {_human(time.time() - t0)}.")
    print(f"  sulla mappa {state.written:,} · falliti {state.failed:,} "
          f"su {state.total:,}")
    if state.errors:
        print(f"\n  primi errori ({len(state.errors)} tenuti):")
        for e in state.errors[:5]:
            print(f"    {Path(e['file']).name[:60]} — {e['error'][:70]}")

    if args.project:
        # La pagina legge lo stato su file per sapere se il job sta ancora
        # lavorando (`state.running`), e ricarica la mappa non appena non lo
        # è più. La riproiezione è lo stesso job, non un lavoro a parte: se
        # lo stato dicesse già "finito" da qui, la pagina si ricaricherebbe
        # subito, prima che `coords.npy` abbia le posizioni dei brani appena
        # aggiunti — e siccome quel ricaricamento è un colpo solo, nessuno
        # gliene manda un secondo quando la proiezione finisce davvero.
        state.finished_at = None
        state.current = "proiezione UMAP…"
        state.save(args.state_file)
        reproject(args.store, projection)
        state.finished_at = time.time()
        state.current = ""
        state.save(args.state_file)
    elif state.written:
        print("\n  La mappa ha brani nuovi senza coordinate: "
              "`--project-only` (o il pulsante nella pagina) le calcola.")
    sys.exit(1 if state.failed and not state.written else 0)


if __name__ == "__main__":
    main()
