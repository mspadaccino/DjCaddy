"""Come cambierebbe la valence con la calibrazione per etichetta. Prova, non scrive.

La lettura attuale confronta le attivazioni CRUDE delle etichette scure e
chiare (`mood_scale.valence_of`), e il prior del modello vive etichetta per
etichetta: *Happy* porta un fondo alto su qualunque brano, *Sad* uno basso,
e nella formula pesano come se parlassero la stessa unità. La calibrazione
misura ogni etichetta contro la SUA distribuzione sulla libreria — il rango
percentile di `Sad` fra tutti i `Sad`, come `energy` già fa coi suoi
ingredienti — e poi fa la differenza fra i due lati.

Questo comando mette le due letture fianco a fianco su tutta la mappa e
tira fuori i brani che si spostano, dal più spostato: la lista da ascoltare
per decidere. Come `zoo_cli`, non scrive niente: né nella mappa, né nei tag.
Le attivazioni si rifanno dagli embedding su disco, l'audio non si tocca.

Il confronto è fra RANGHI, perché è così che la valence si legge ovunque:
uno shift di +0,30 vuol dire "salito di tre decili verso il chiaro".

    poetry run python mood_shift_cli.py                    # tutto, in minuti
    poetry run python mood_shift_cli.py --min-shift 0.2    # solo i più mossi

Da cancellare quando la decisione è presa: se la calibrazione convince, la
sua casa vera è `mood_scale`, con i suoi test; se non convince, via anche lui.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from core.analysis import energy, mood_scale
from core.analysis.map_store import MapStore, default_store_dir
from mood_cli import _head, scored


def activations_matrix(embeddings, predict) -> np.ndarray:
    """Le 56 attivazioni di ogni vettore, un lotto alla volta."""
    pieces, done = [], 0
    for chunk in scored(embeddings, predict):
        pieces.append(chunk)
        done += len(chunk)
        print(f"\r  attivazioni: {done:,}/{len(embeddings):,}", end="", flush=True)
    print()
    return np.vstack(pieces)


def current_reading(acts: np.ndarray, labels: list[str]) -> np.ndarray:
    """La valence come la calcola oggi la produzione, riga per riga.

    Passa dalla STESSA funzione della mappa (`valence_of`), non da una copia
    vettorizzata: quello che si confronta deve essere quello che c'è.
    """
    out = np.full(len(acts), np.nan)
    for i, row in enumerate(acts):
        value = mood_scale.valence_of(dict(zip(labels, row)))
        if value is not None:
            out[i] = value
    return out


def calibrated_reading(acts: np.ndarray, labels: list[str]) -> np.ndarray:
    """La valence calibrata: ogni etichetta al suo rango di libreria.

    `energy.ranks` colonna per colonna — pari merito gestiti, la stessa
    funzione con cui la libreria già trasforma unità incompatibili in una
    scala sola. Poi media dei ranghi chiari meno media dei ranghi scuri:
    da −1 a +1, centrata attorno allo zero per costruzione. Le neutre
    restano fuori, come in `valence_of`.
    """
    bright = [i for i, l in enumerate(labels) if l in mood_scale.BRIGHT]
    dark = [i for i, l in enumerate(labels) if l in mood_scale.DARK]
    bright_ranks = np.column_stack([energy.ranks(acts[:, i]) for i in bright])
    dark_ranks = np.column_stack([energy.ranks(acts[:, i]) for i in dark])
    return bright_ranks.mean(axis=1) - dark_ranks.mean(axis=1)


def label_priors(acts: np.ndarray, labels: list[str]) -> list[tuple[float, str, str]]:
    """L'attivazione media di ogni etichetta colorata: il prior, nudo."""
    rows = []
    for i, label in enumerate(labels):
        side = ("chiara" if label in mood_scale.BRIGHT
                else "scura" if label in mood_scale.DARK else None)
        if side:
            rows.append((float(acts[:, i].mean()), label, side))
    return sorted(rows, reverse=True)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confronta la valence attuale con quella calibrata per "
                    "etichetta. Non scrive niente: né nella mappa, né nei tag.")
    parser.add_argument("--min-shift", type=float, default=0.10,
                        help="spostamento minimo di rango perché un brano "
                             "entri nel CSV (default 0.10, un decile)")
    parser.add_argument("--out", type=Path, default=Path("mood_shift.csv"))
    parser.add_argument("--store", default=None)
    args = parser.parse_args()

    store = MapStore.load(args.store or default_store_dir())
    n = min(len(store.rows), len(store.embeddings))
    if n < len(store.rows) or n < len(store.embeddings):
        print(f"  righe e vettori non pari ({len(store.rows):,} contro "
              f"{len(store.embeddings):,}): mi fermo ai primi {n:,}")
    rows = store.rows[:n]
    print(f"Mappa: {n:,} brani")

    predict, labels = _head()
    acts = activations_matrix(store.embeddings[:n], predict)

    old_raw = current_reading(acts, labels)
    new_raw = calibrated_reading(acts, labels)

    # Controllo: la lettura attuale rifatta qui deve combaciare con quella
    # scritta sulle righe dal backfill. Se non combacia, sto confrontando la
    # calibrazione con qualcos'altro — modelli cambiati, store sbagliato — e
    # il CSV direbbe bugie.
    stored = np.array([r.get("valence") if r.get("valence") is not None
                       else np.nan for r in rows], dtype=float)
    match = _corr(old_raw, stored)
    print(f"  controllo: la valence rifatta combacia con la riga "
          f"(r={match:.4f})" + ("" if match > 0.999 else "  ← NON COMBACIA"))

    print("\nIl prior per etichetta (attivazione media sulla libreria):")
    for mean, label, side in label_priors(acts, labels):
        print(f"  {label:13s} {side:7s} {mean:.4f}")

    old_rank = energy.ranks(old_raw)
    new_rank = energy.ranks(new_raw)
    delta = new_rank - old_rank
    crosses = (old_rank - 0.5) * (new_rank - 0.5) < 0

    print("\nVecchia contro nuova:")
    print(f"  correlazione dei ranghi: {_corr(old_rank, new_rank):+.3f}")
    ok = np.isfinite(delta)
    for soglia in (0.10, 0.20, 0.30):
        print(f"  |shift| ≥ {soglia:.2f}: {np.mean(np.abs(delta[ok]) >= soglia):6.1%}")
    print(f"  cambiano lato del mezzo: {np.mean(crosses[ok]):.1%}")
    print(f"  sotto zero, in crudo: prima {np.mean(old_raw[np.isfinite(old_raw)] < 0):.1%}, "
          f"dopo {np.mean(new_raw < 0):.1%}")

    chosen = [i for i in np.argsort(-np.abs(np.where(ok, delta, 0)))
              if ok[i] and abs(delta[i]) >= args.min_shift]
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["shift", "verso", "cambia_lato", "rank_prima",
                         "rank_dopo", "val_prima", "val_dopo", "name",
                         "moods", "mood_conf", "bpm", "path"])
        for i in chosen:
            writer.writerow([
                f"{delta[i]:+.3f}",
                "chiaro" if delta[i] > 0 else "scuro",
                "sì" if crosses[i] else "",
                f"{old_rank[i]:.3f}", f"{new_rank[i]:.3f}",
                f"{old_raw[i]:.3f}" if np.isfinite(old_raw[i]) else "",
                f"{new_raw[i]:.3f}",
                rows[i].get("name", ""), rows[i].get("moods", ""),
                rows[i].get("mood_conf", ""), rows[i].get("bpm"),
                rows[i]["path"]])
    print(f"\n{len(chosen):,} brani in {args.out}, dal più spostato "
          f"(|shift| ≥ {args.min_shift}).")


if __name__ == "__main__":
    main()
