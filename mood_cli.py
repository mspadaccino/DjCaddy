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
import csv
import sys
import time
from pathlib import Path

import numpy as np

from core.analysis import energy, mood_scale
from core.analysis.essentia_tags import MODEL_DIR, MOOD_METADATA, MOOD_MODEL, _labels, format_mood_tag
from core.analysis.map_job import awake
from core.analysis.map_profile import MOOD_FIELDS, ProfileSettings, mood_numbers, select_labels
from core.analysis.map_store import MapStore, default_store_dir

# Quante righe per volta entrano nella testa. Il modello lavora a lotti, e
# ottantasettemila vettori da 1280 float in una volta sola sono quattrocento
# megabyte di attivazioni in uscita: si spezza per non tenerli tutti.
BATCH = 4096

# Ogni quanti brani si salva. Come nel backfill dell'energia: qui il lavoro
# dura minuti e non ore, ma il file è lo stesso e il motivo pure.
FLUSH_EVERY = 20000

FIELDS = MOOD_FIELDS

# Le letture fra cui si sceglie, per `--check`: nome, soglia, se bilanciare
# i due lati sul numero di etichette che hanno. Vedi `mood_scale.SIDES`.
CANDIDATES = (("sums", 0.0, False),
              ("balanced", 0.0, True),
              ("floor 0.01", 0.01, True),
              ("floor 0.02", 0.02, True),
              ("floor 0.05", 0.05, True))

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

    **La domanda del pooling.** La testa applicata alla media dei vettori
    legge lo stesso brano che leggeva la media delle previsioni fettina per
    fettina? Delle attivazioni di allora non è rimasto niente su disco —
    solo le PAROLE che ne erano uscite — e quelle parole sono quindi l'unico
    paragone possibile. `top label kept` è il numero che risponde: da tutte
    e due le parti si applica la stessa soglia e la stessa regola di scelta,
    quindi l'unica cosa che cambia fra le due è il pooling.

    **Le altre due domande.** Fra la valence vecchia e la nuova ci sono in
    realtà TRE cambiamenti sovrapposti, e guardarli insieme non dice quale
    ha mosso cosa. Si separano così, misurando una cosa alla volta:

        vecchia  = ranghi (1, ½, ⅓), neutre nel denominatore
        di mezzo = pesi veri,        neutre nel denominatore
        nuova    = pesi veri,        neutre fuori

    `weights vs ranks` è il primo passo, `dropping the neutrals` il secondo.
    Un valore basso non è di per sé un errore — sono i miglioramenti che si
    stavano cercando — ma dice DOVE la libreria si è riordinata, e quello va
    saputo prima di riscrivere ottantasettemila righe.

    **E come si distribuisce.** Un asse non serve a niente se la libreria ci
    sta tutta da una parte. Per ognuna delle letture candidate si scrivono
    i nove decili e la frazione di brani sotto lo zero: se `below zero` è
    vicina a 0,50 lo zero è davvero il mezzo, e la croce dei quadranti può
    starci; se è vicina a 0 o a 1, lo zero non è il mezzo di niente e o si
    cambia lettura o si centra la croce sulla mediana.

    Le candidate sono tre, e sono lo stesso rimedio per lo stesso male preso
    in tre dosi: `sums` è la lettura senza rimedio (somme crude), `balanced`
    divide ogni lato per quante etichette ha, `balanced + floor` in più fa
    contare solo le attivazioni sopra una soglia. Il male è che le liste non
    sono grandi uguali — 13 chiare, 8 buie — e il fondo della sigmoide entra
    13 volte da una parte e 8 dall'altra.
    """
    rows = store.rows
    total = min(sample, len(store.embeddings), len(rows))
    if not total:
        return {"tracks": 0}     # e senza caricare mezzo giga di modello
    predict, labels = _head()
    step = max(1, len(rows) // total)
    picked = list(range(0, len(rows), step))[:total]

    first = fresh_colour = 0
    overlap, old_way, middle, new_way = [], [], [], []
    coloured = []
    for offset, scores in enumerate(
            row for chunk in scored(store.embeddings[picked], predict)
            for row in chunk):
        stored = rows[picked[offset]].get("moods", "")
        same, share = agreement(scores, labels, stored, settings)
        first += bool(same)
        overlap.append(share)

        whole = dict(zip(labels, map(float, scores)))
        coloured.append(whole)
        old = mood_scale.valence(stored)
        both = mood_scale.valence_of(whole, dilute=True)
        new = mood_scale.valence_of(whole)
        if new is not None and not old:
            # Le parole non davano nessun verso — o perché non ce n'erano, o
            # perché erano tutte neutre — e le prove sotto soglia uno ce
            # l'hanno dato.
            fresh_colour += 1
        if None not in (old, both, new):
            old_way.append(old)
            middle.append(both)
            new_way.append(new)

    out = {"tracks": len(overlap),
           "top label kept": first / len(overlap),
           "labels overlap": float(np.mean(overlap)),
           "newly measured": fresh_colour / len(overlap)}
    if len(old_way) > 1:
        out["weights vs ranks"] = float(np.corrcoef(old_way, middle)[0, 1])
        out["dropping the neutrals"] = float(np.corrcoef(middle, new_way)[0, 1])
        out["old vs new, both changes"] = float(
            np.corrcoef(old_way, new_way)[0, 1])
    # Le prove deboli dicono la stessa cosa di quelle forti, o sono rumore?
    # Si chiede ai brani che hanno tutte e due: la valence sulle sole
    # attivazioni sopra la soglia delle parole contro quella sulle sole
    # attivazioni sotto.
    #
    # **E il numero da solo non si legge.** La separazione si porta dietro un
    # bias meccanico: l'etichetta che vince viene promossa sopra soglia, e
    # quello che resta sotto e' fatto soprattutto del lato che ha perso.
    # Misurata su attivazioni completamente casuali questa correlazione esce
    # attorno a -0,5 — senza nessuna musica dentro. Un -0,4 sui dati veri
    # non vuol dire quindi "le prove deboli sbagliano": vuol dire che vanno
    # d'accordo con le forti PIU' di quanto il caso spiegherebbe.
    #
    # Il paragone si costruisce rimescolando: le stesse 56 attivazioni dello
    # stesso brano, riassegnate a caso alle 56 etichette. La distribuzione
    # dei numeri resta identica, il legame fra numeri e parole sparisce, e
    # quello che ne esce e' esattamente il bias meccanico da solo.
    shuffle = np.random.default_rng(0)
    pairs, control = [], []
    for whole in coloured:
        for target, activations in (
                (pairs, whole),
                (control, dict(zip(whole, shuffle.permutation(
                    np.fromiter(whole.values(), dtype=float)))))):
            above = mood_scale.valence_of(activations,
                                          floor=settings.mood_threshold)
            below = mood_scale.valence_of(activations,
                                          ceiling=settings.mood_threshold)
            if above is not None and below is not None:
                target.append((above, below))
    if len(pairs) > 1:
        real = float(np.corrcoef(*np.array(pairs).T)[0, 1])
        out["faint vs strong"] = real
        out["...measured on"] = len(pairs) / len(coloured)
        if len(control) > 1:
            chance = float(np.corrcoef(*np.array(control).T)[0, 1])
            out["...on shuffled labels"] = chance
            out["...so the faint evidence is worth"] = real - chance

    for name, floor, balanced in CANDIDATES:
        read = [mood_scale.valence_of(whole, floor=floor, balanced=balanced)
                for whole in coloured]
        values = np.asarray([v for v in read if v is not None])
        if not len(values):
            continue
        out[f"{name} · below zero"] = float(np.mean(values < 0))
        # Quanto costa la soglia: i brani che restano senza nessuna lettura.
        out[f"{name} · no reading"] = 1 - len(values) / len(read)
        out[f"{name} · deciles"] = [round(float(v), 2) for v in
                                    np.quantile(values, np.arange(0.1, 1.0, 0.1))]
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


def faint_only(whole, settings: ProfileSettings) -> bool:
    """Se il colore di questo brano viene SOLO da prove sotto soglia.

    Sono i brani su cui si sta discutendo: nessuna delle parole che il
    modello ha scritto sulla riga è buia o chiara, e la direzione gliela dà
    unicamente quello che sta sotto la soglia. Se quelle attivazioni sono
    segnale, questi brani sono un guadagno; se sono rumore, a questi brani
    stiamo dando un verso inventato — ed è l'unico gruppo su cui nessuna
    misura interna può decidere, perché di forte non hanno niente con cui
    confrontarsi. Restano le orecchie.
    """
    return (mood_scale.valence_of(whole, floor=settings.mood_threshold) is None
            and mood_scale.valence_of(whole) is not None)


def faint_sample(store: MapStore, count: int,
                 settings: ProfileSettings) -> list[dict]:
    """Un campione dei brani il cui colore sta tutto sotto soglia, dal più
    buio al più chiaro, per poterli ascoltare in fila.

    Presi a passo costante lungo la scala e non a caso: quello che si vuole
    sentire non è "sono giusti" ma "è un ORDINE" — se scendendo la lista i
    brani si scuriscono, le prove deboli sono segnale, e se sembrano
    mescolati sono rumore.
    """
    predict, labels = _head()
    found = []
    for offset, scores in enumerate(
            row for chunk in scored(store.embeddings, predict) for row in chunk):
        whole = dict(zip(labels, map(float, scores)))
        if faint_only(whole, settings):
            found.append((mood_scale.valence_of(whole), offset))
    found.sort()
    if not found:
        return []
    step = max(1, len(found) // count)
    return [{"valence": round(value, 4),
             "name": store.rows[i].get("name", ""),
             "moods": store.rows[i].get("moods", ""),
             "bpm": store.rows[i].get("bpm"),
             "path": store.rows[i]["path"]}
            for value, i in found[::step][:count]]


def spread_sample(store: MapStore, count: int) -> list[dict]:
    """Un campione lungo TUTTA la scala della valence, dal buio al chiaro.

    Niente modello e niente audio: a backfill fatto il numero sta già sulla
    riga. Si prende a passo costante sul RANGO e non sul valore, perché è il
    rango che si legge sugli assi — un passo costante sul numero grezzo
    prenderebbe venti brani ammassati dove la libreria è fitta.

    Serve alla stessa cosa a cui è servito il campione dell'energia: non a
    chiedere "questo brano è giusto", che su una scala relativa non vuol dire
    niente, ma a chiedere se scendendo la lista i brani si SCHIARISCONO.
    """
    ranked = energy.ranks(mood_scale.from_rows(store.rows))
    known = [(float(r), i) for i, r in enumerate(ranked) if np.isfinite(r)]
    known.sort()
    if not known:
        return []
    step = max(1, len(known) // count)
    return [{"rank": round(rank, 3),
             "valence": store.rows[i].get("valence"),
             "evidence": store.rows[i].get("mood_evidence"),
             "name": store.rows[i].get("name", ""),
             "moods": store.rows[i].get("moods", ""),
             "bpm": store.rows[i].get("bpm"),
             "path": store.rows[i]["path"]}
            for rank, i in known[::step][:count]]


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
    parser.add_argument("--sample", type=int, metavar="N",
                        help="N brani lungo tutta la scala della valence, "
                             "dal piu' buio al piu' chiaro, da ascoltare")
    parser.add_argument("--faint-sample", type=int, metavar="N",
                        help="N brani il cui colore sta tutto sotto soglia, "
                             "dal piu' buio al piu' chiaro, da ascoltare")
    parser.add_argument("--out", type=Path, default=Path("mood_faint.csv"))
    parser.add_argument("--store", default=None,
                        help="la cartella della mappa (default: quella solita)")
    args = parser.parse_args()

    store = MapStore.load(args.store or default_store_dir())
    settings = ProfileSettings()
    print(f"Mappa: {len(store):,} brani, {len(store.embeddings):,} vettori")

    if args.check:
        for name, value in check(store, args.check, settings).items():
            if isinstance(value, float):
                print(f"  {name}: {value:.3f}")
            elif isinstance(value, list):
                print(f"  {name}: {' '.join(f'{v:+.2f}' for v in value)}")
            else:
                print(f"  {name}: {value:,}")
        return

    if args.sample or args.faint_sample:
        table = (spread_sample(store, args.sample) if args.sample
                 else faint_sample(store, args.faint_sample, settings))
        if not table:
            print("  nessun brano da campionare")
            return
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"  {len(table)} brani in {args.out}, dal piu' buio al piu' chiaro")
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
