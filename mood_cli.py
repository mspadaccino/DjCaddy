"""Il mood come numero, sui brani che sono già sulla mappa.

`energy_cli` deve riaprire ogni file e riascoltarne trenta secondi: sono
ottantasettemila letture da disco e si contano in ore. Qui no. La testa che
legge il mood — `mtg_jamendo_moodtheme-discogs-effnet` — non guarda l'audio:
guarda l'embedding da 1280 numeri, e quello è già su disco, in
`embeddings.f32`, salvato la prima volta che il brano è stato analizzato.
Rifarle passare tutte dentro una rete di due strati è un prodotto fra
matrici: si conta in minuti.

**Cosa cambia rispetto a quello che c'è già.** Sulla riga il mood c'è come
elenco di parole — al massimo quattro, quelle sopra 0,05 — e `mood_scale`
ne ricava una direzione usando l'ORDINE al posto della confidenza. Le
attivazioni vere non sono mai state scritte. Da qui si riottengono tutte e
cinquantasei e si scrivono i tre campi che ne derivano: `valence`,
`mood_evidence`, `mood_conf`. Vedi `analysis.mood_scale.valence_of`.

**Un'avvertenza che vale più del resto.** L'embedding su disco è la MEDIA
delle fettine, mentre le etichette salvate a suo tempo erano la media delle
PREVISIONI fettina per fettina. Non sono la stessa cosa: la testa non è
lineare, quindi media-poi-testa e testa-poi-media danno numeri vicini ma non
uguali. Quanto vicini è una domanda a cui si risponde misurando, ed è cosa
fa `--check`: riprevede un campione e conta quante volte la prima etichetta
resta la stessa. Prima di riscrivere ottantasettemila righe conviene
guardare quel numero.

**Non girare insieme a `energy_cli --backfill`.** Riscrivono tutti e due
`tracks.jsonl` per intero: l'ultimo che salva cancella il lavoro dell'altro.

    python mood_cli.py --check 2000        # quanto regge la riprevisione
    python mood_cli.py --backfill          # scrive i tre campi su tutto
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from analysis import mood_scale
from analysis.essentia_tags import MODEL_DIR, MOOD_METADATA, MOOD_MODEL, _labels, format_mood_tag
from analysis.map_job import awake
from analysis.map_profile import MOOD_FIELDS, ProfileSettings, mood_numbers, select_labels
from analysis.map_store import MapStore, default_store_dir

# Quante righe per volta entrano nella testa. Il modello lavora a lotti, e
# ottantasettemila vettori da 1280 float in una volta sola sono quattrocento
# megabyte di attivazioni in uscita: si spezza per non tenerli tutti.
BATCH = 4096

# Ogni quanti brani si salva. Come nel backfill dell'energia: qui il lavoro
# dura minuti e non ore, ma il file è lo stesso e il motivo pure.
FLUSH_EVERY = 20000

FIELDS = MOOD_FIELDS

# Qual è il campo che dice "questa riga è fatta". NON `valence`: quello resta
# `None` sui brani a cui il modello non legge nessun colore — che è una
# risposta, non un buco — e chiedere di lui rimetterebbe quei brani nella
# lista da fare a ogni giro, per sempre. `mood_evidence` un numero ce l'ha
# sempre, zero compreso.
MARKER = "mood_evidence"


def missing(rows: list[dict]) -> list[int]:
    """Le posizioni delle righe a cui il mood come numero manca ancora."""
    return [i for i, row in enumerate(rows) if row.get(MARKER) is None]


def _head():
    """La testa del mood, caricata da sola: senza l'embedder, che qui non
    serve — l'embedding è già stato calcolato e sta su disco."""
    import essentia
    essentia.log.warningActive = False
    from essentia.standard import TensorflowPredict2D

    model = TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / MOOD_MODEL),
        input="model/Placeholder", output="model/Sigmoid")
    labels = [format_mood_tag(m) for m in _labels(MODEL_DIR / MOOD_METADATA)]
    return model, labels


def scored(vectors, predict, batch: int = BATCH):
    """Le attivazioni di tutti i vettori, un lotto alla volta.

    `predict` è la testa. Passarla come argomento invece di caricarla qui
    dentro è quello che rende la funzione provabile senza Essentia — che su
    questa macchina non c'è, e su qualunque macchina è mezzo gigabyte.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    for start in range(0, len(vectors), batch):
        chunk = vectors[start:start + batch]
        yield np.asarray(predict(chunk), dtype=float)


# I tre campi che una riga di attivazioni produce: la stessa funzione che
# `map_profile.analyze` chiama sui brani nuovi. E le riceve dallo stesso
# posto — la testa applicata alla media dei vettori — quindi il numero non
# dipende da quando un brano è stato analizzato.
written = mood_numbers


def agreement(scores, labels: list[str], stored: str,
              settings: ProfileSettings) -> tuple[bool, float]:
    """Quanto la riprevisione somiglia alle etichette che stanno sulla riga.

    Torna se la PRIMA etichetta è rimasta la stessa, e quanto si
    sovrappongono i due insiemi (Jaccard, da 0 a 1). La prima conta più
    dell'insieme: è quella che decide come il brano si legge in tabella.
    """
    fresh = [name for name, _ in select_labels(
        scores, labels, settings.mood_threshold, settings.max_moods)]
    old = mood_scale.split(stored)
    if not old or not fresh:
        return (not old and not fresh), 1.0 if not old and not fresh else 0.0
    union = set(fresh) | set(old)
    return fresh[0] == old[0], len(set(fresh) & set(old)) / len(union)


def check(store: MapStore, sample: int, settings: ProfileSettings) -> dict:
    """Riprevede un campione e conta quanto tiene, senza scrivere niente.

    **Cosa risponde davvero.** La domanda aperta è una sola: la testa
    applicata alla media dei vettori legge lo stesso brano che leggeva la
    media delle previsioni fettina per fettina? Delle attivazioni di allora
    non è rimasto niente su disco — solo le PAROLE che ne erano uscite — e
    quelle parole sono quindi l'unico paragone possibile.

    Per questo il numero da guardare è `top label kept`: da tutte e due le
    parti si applica la stessa soglia e la stessa regola di scelta, quindi
    l'unica cosa che cambia fra le due è il pooling, ed è proprio quello che
    si vuole misurare.

    `agrees with the old reading` NON risponde a quella domanda e non va
    letto come se lo facesse: confronta la valence nuova con quella vecchia
    per rango, quindi dentro ci sono DUE cambiamenti insieme — il pooling e
    il passaggio dai ranghi ai pesi veri. Il secondo è il miglioramento che
    si sta cercando, quindi un numero minore di 1 lì dentro è atteso e non
    vuol dire niente di male.
    """
    rows = store.rows
    total = min(sample, len(store.embeddings), len(rows))
    if not total:
        return {"tracks": 0}     # e senza caricare mezzo giga di modello
    predict, labels = _head()
    step = max(1, len(rows) // total)
    picked = list(range(0, len(rows), step))[:total]

    first = fresh_colour = 0
    overlap, colours = [], []
    for offset, scores in enumerate(
            row for chunk in scored(store.embeddings[picked], predict)
            for row in chunk):
        stored = rows[picked[offset]].get("moods", "")
        same, share = agreement(scores, labels, stored, settings)
        first += bool(same)
        overlap.append(share)
        old = mood_scale.valence(stored)
        new = mood_scale.valence_of(dict(zip(labels, map(float, scores))))
        if new is not None and old is None:
            fresh_colour += 1
        elif old is not None and new is not None:
            colours.append((old, new))

    out = {"tracks": len(overlap),
           "top label kept": first / len(overlap),
           "labels overlap": float(np.mean(overlap)),
           # Brani che una freccia non ce l'avevano: nessuna etichetta
           # passava la soglia, e le prove di colore sotto di essa non le
           # guardava nessuno.
           "newly measured": fresh_colour / len(overlap)}
    if len(colours) > 1:
        old, new = np.array(colours).T
        out["agrees with the old reading"] = float(np.corrcoef(old, new)[0, 1])
    return out


def backfill(store: MapStore, settings: ProfileSettings,
             flush_every: int = FLUSH_EVERY, on_progress=None) -> int:
    """Scrive i tre campi su tutte le righe che non li hanno.

    Riscrive `tracks.jsonl` per intero invece di appendere, per lo stesso
    motivo del backfill dell'energia: qui non si aggiungono brani, si
    aggiunge un campo a righe che ci sono già. Vedi `MapStore.rewrite`.
    """
    todo = missing(store.rows)
    if not todo:
        return 0
    if len(store.embeddings) < len(store.rows):
        raise ValueError(
            f"{len(store.rows)} righe ma {len(store.embeddings)} vettori: "
            "la mappa non è allineata e riscriverla la romperebbe")
    predict, labels = _head()
    done = 0
    with awake():
        for offset, scores in enumerate(
                row for chunk in scored(store.embeddings[todo], predict)
                for row in chunk):
            store.rows[todo[offset]].update(written(scores, labels, settings))
            done += 1
            if done % flush_every == 0:
                store.rewrite()
            if on_progress:
                on_progress(done, len(todo))
    store.rewrite()
    return done


def progress_line(done: int, total: int, started: float, now: float) -> str:
    """La riga che si riscrive sopra sé stessa mentre il lavoro va avanti."""
    share = done / total if total else 1.0
    spent = now - started
    left = spent / share - spent if share > 0 else 0.0
    return (f"  {done:,}/{total:,} ({share:6.1%})  "
            f"{_human(spent)} spesi, ~{_human(left)} rimasti")


def _human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Il mood come numero (valence) sui brani della mappa.")
    parser.add_argument("--check", type=int, metavar="N",
                        help="riprevede N brani e dice quanto regge, "
                             "senza scrivere niente")
    parser.add_argument("--backfill", action="store_true",
                        help="scrive valence, mood_evidence e mood_conf")
    parser.add_argument("--store", default=None,
                        help="la cartella della mappa (default: quella solita)")
    args = parser.parse_args()

    store = MapStore.load(args.store or default_store_dir())
    settings = ProfileSettings()
    print(f"Mappa: {len(store):,} brani, {len(store.embeddings):,} vettori")

    if args.check:
        for name, value in check(store, args.check, settings).items():
            print(f"  {name}: {value:.3f}" if isinstance(value, float)
                  else f"  {name}: {value:,}")
        return

    if not args.backfill:
        todo = missing(store.rows)
        print(f"  senza il mood come numero: {len(todo):,}")
        print("  --check N per provarlo, --backfill per scriverlo")
        return

    started = time.time()

    def report(done: int, total: int) -> None:
        if done % 1000 == 0 or done == total:
            sys.stdout.write("\r" + progress_line(done, total, started,
                                                  time.time()) + "   ")
            sys.stdout.flush()

    done = backfill(store, settings, on_progress=report)
    print(f"\nScritti {done:,} brani in {_human(time.time() - started)}.")


if __name__ == "__main__":
    main()
