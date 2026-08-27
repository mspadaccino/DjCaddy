"""Le altre quattro teste del model zoo, provate senza toccare niente.

Essentia pubblica, costruiti sullo STESSO embedding Discogs-EffNet che
abbiamo già su disco, quattro classificatori binari: `mood_aggressive`,
`mood_relaxed`, `mood_party` e `danceability`. Aggiungerli costerebbe il
caricamento di quattro teste piccole — niente decodifica audio, niente
embedding da rifare — ma "costa poco" non è una ragione per aggiungere un
numero: un indicatore che non dice niente di nuovo è una colonna in più da
leggere e una decisione in più da prendere, per sempre.

Questo comando serve a decidere PRIMA, e non scrive da nessuna parte: né
nei tag dei file, né nella mappa. Legge gli embedding, li fa passare per le
quattro teste, e risponde a due domande.

**Si muovono?** Una testa che risponde 0,9 su tutto non è un indicatore, è
una costante: si vede dai decili e dalla quota sopra 0,5.

**Dicono qualcosa di nuovo?** Si correlano con quello che c'è già —
energia, valence, groove, BPM — e fra loro. Correlazione alta vuol dire che
il numero nuovo ripete uno vecchio con un altro nome. È lo stesso esame che
hanno passato l'energia (indipendente dalla valence: +0,005) e la valence.

Un caso da guardare a parte: la testa `danceability` ha lo stesso nome del
campo che la mappa già porta, e le due cose NON sono la stessa. Il nostro
`danceability` è la regolarità degli attacchi, misurata sul segnale; questa
è una rete che ha imparato da orecchie umane cos'è ballabile. Se vanno
d'accordo, una delle due è di troppo. Se non vanno d'accordo, è la più
interessante delle quattro.

    python zoo_cli.py                       # 2.000 brani, il rapporto
    python zoo_cli.py --sample 25 --by party   # da ascoltare, in ordine
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis import energy, mood_scale
from analysis.essentia_tags import MODEL_DIR
from analysis.map_store import MapStore, default_store_dir

# Le quattro teste: nome corto, e il file (senza estensione) da cui vengono
# il grafo e l'elenco delle classi.
HEADS = {
    "aggressive": "mood_aggressive-discogs-effnet-1",
    "relaxed": "mood_relaxed-discogs-effnet-1",
    "party": "mood_party-discogs-effnet-1",
    "danceable": "danceability-discogs-effnet-1",
}

# Quello che la mappa ha già, e con cui le teste nuove devono confrontarsi:
# se una di loro ripete una di queste, non serve.
KNOWN = ("energy", "valence", "groove", "BPM")

BATCH = 4096
SAMPLE = 2000

# Da dove si scaricano. Il model zoo tiene ogni testa in una cartella che
# porta il suo nome, senza il suffisso dell'embedding: `mood_party-discogs-
# effnet-1.pb` sta sotto `mood_party/`. Se un giorno cambia, l'errore dice
# quale indirizzo ha provato e la pagina da cui prendere quello giusto.
ZOO = "https://essentia.upf.edu/models/classification-heads"
ZOO_PAGE = "https://essentia.upf.edu/models.html"

# Da dove leggere i nomi dei nodi, se il JSON del modello li porta. Non è
# scontato che li porti, e indovinarli a mano è il modo classico di
# scoprire alla terza riga che il grafo si chiamava in un altro modo.
FALLBACK_IN = "model/Placeholder"
FALLBACK_OUT = "model/Softmax"


def positive_class(classes: list[str]) -> int:
    """Quale delle due classi è quella che ci interessa.

    Sono classificatori binari e l'ordine non è garantito: si prende quella
    che NON è la negazione dell'altra. Il prefisso cambia da modello a
    modello — `non_aggressive` da una parte, `not_danceable` dall'altra — e
    fidarsi dell'ordine avrebbe voluto dire, sul modello sbagliato, un
    numero perfettamente plausibile e rovesciato.
    """
    positives = [i for i, name in enumerate(classes)
                 if not name.lower().startswith(("non", "not"))]
    return positives[0] if len(positives) == 1 else 0


def nodes(metadata: dict) -> tuple[str, str]:
    """I nomi dei nodi di ingresso e uscita, dal JSON del modello."""
    schema = metadata.get("schema") or {}
    ins = schema.get("inputs") or []
    outs = schema.get("outputs") or []
    return (ins[0].get("name", FALLBACK_IN) if ins else FALLBACK_IN,
            outs[0].get("name", FALLBACK_OUT) if outs else FALLBACK_OUT)


def source_of(filename: str) -> str:
    """Da dove si scarica un file di modello, dal suo nome."""
    return f"{ZOO}/{filename.split('-discogs-effnet')[0]}/{filename}"


def fetch(filename: str, model_dir: Path) -> str | None:
    """Scarica un file nella cartella dei modelli. Torna l'errore, o `None`.

    Prima in un file temporaneo e poi al suo posto: una connessione che cade
    a metà lascerebbe altrimenti un `.pb` troncato con il nome giusto, che
    al caricamento dà un errore su cui nessuno pensa di ridare un occhio
    alla rete.
    """
    import urllib.error
    import urllib.request

    target = model_dir / filename
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(source_of(filename), timeout=120) as fh:
            partial.write_bytes(fh.read())
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        return str(error)
    partial.replace(target)
    return None


def missing(model_dir: Path = MODEL_DIR) -> list[str]:
    """Quali file mancano per provare le quattro teste."""
    return [f"{stem}{suffix}" for stem in HEADS.values()
            for suffix in (".pb", ".json")
            if not (model_dir / f"{stem}{suffix}").exists()]


def _load(model_dir: Path = MODEL_DIR):
    """Le quattro teste caricate, ognuna con l'indice della classe buona."""
    import essentia
    essentia.log.warningActive = False
    from essentia.standard import TensorflowPredict2D

    loaded = {}
    for name, stem in HEADS.items():
        metadata = json.loads((model_dir / f"{stem}.json").read_text())
        source, sink = nodes(metadata)
        loaded[name] = (
            TensorflowPredict2D(graphFilename=str(model_dir / f"{stem}.pb"),
                                input=source, output=sink),
            positive_class(metadata["classes"]))
    return loaded


def scored(vectors, heads, batch: int = BATCH) -> dict[str, np.ndarray]:
    """Le quattro risposte per ogni vettore, da 0 a 1.

    `heads` arriva da fuori — nome: (testa, indice della classe buona) — ed
    è quello che rende la funzione provabile senza Essentia, che su questa
    macchina non c'è e altrove è mezzo gigabyte.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    out = {}
    for name, (predict, wanted) in heads.items():
        pieces = [np.asarray(predict(vectors[at:at + batch]), dtype=float)
                  for at in range(0, len(vectors), batch)]
        out[name] = (np.concatenate(pieces)[:, wanted] if pieces
                     else np.empty(0))
    return out


def already_known(rows) -> dict[str, np.ndarray]:
    """Le misure che la mappa ha già, sulle stesse righe.

    Energia e valence come RANGHI, che è come si leggono ovunque; groove e
    BPM come stanno, perché una correlazione non cambia per un rango.
    """
    numeric = lambda name: np.asarray(  # noqa: E731
        [row.get(name) if row.get(name) is not None else np.nan
         for row in rows], dtype=float)
    return {"energy": energy.from_rows(rows),
            "valence": energy.ranks(mood_scale.from_rows(rows)),
            "groove": numeric("danceability"),
            "BPM": numeric("bpm")}


def spread(values: np.ndarray) -> dict:
    """Se una testa si muove o risponde sempre la stessa cosa.

    `distinct` è la parte che serve davvero, e mancava. I decili scritti a
    due cifre non distinguono "tutti a 1,00" da "tutti fra 0,996 e 1,000":
    escono identici. E la differenza decide tutto, perché ovunque leggiamo
    queste misure le leggiamo come RANGO — e il rango salva il secondo caso
    e non il primo. Un decimo di libreria appiattito su un valore solo resta
    appiattito comunque lo si normalizzi; un decimo distribuito su valori
    vicini si apre da sé.

    Si contano i valori ESATTI, senza arrotondare: qualunque soglia si
    scegliesse sarebbe arbitraria, e sotto di essa avrebbe cancellato
    proprio le differenze che il rango sa usare. Un pari merito nel float è
    un pari merito davvero.

    Vale letto INSIEME alle correlazioni: valori tutti diversi possono
    esserlo per rumore della softmax, e il rumore non si correla con
    niente. Distinti e correlati vuol dire ordine vero.
    """
    known = values[np.isfinite(values)]
    if not len(known):
        return {}
    return {"above 0.5": float(np.mean(known >= 0.5)),
            "distinct": len(np.unique(known)) / len(known),
            "deciles": [round(float(v), 3)
                        for v in np.quantile(known, np.arange(0.1, 1.0, 0.1))]}


def against(values: np.ndarray, others: dict[str, np.ndarray]) -> dict[str, float]:
    """Quanto una testa ripete ognuna delle misure che ci sono già.

    Una misura che non varia sul campione non entra: contro una costante la
    correlazione non è bassa, non esiste — e scriverla come `nan` in mezzo a
    dei numeri veri invita a leggerla come "zero, quindi indipendente", che
    è una conclusione dove non c'è niente da concludere.
    """
    out = {}
    for name, other in others.items():
        both = np.isfinite(values) & np.isfinite(other)
        if both.sum() > 1 and values[both].std() > 0 and other[both].std() > 0:
            out[name] = float(np.corrcoef(values[both], other[both])[0, 1])
    return out


def picked(rows, sample: int) -> list[int]:
    """Le posizioni da guardare, a passo costante su tutta la libreria."""
    total = min(sample, len(rows))
    if not total:
        return []
    return list(range(0, len(rows), max(1, len(rows) // total)))[:total]


def report(store: MapStore, sample: int, heads) -> dict:
    """Il rapporto: come si muovono le quattro, e cosa ripetono."""
    at = picked(store.rows, sample)
    if not at:
        return {}
    fresh = scored(store.embeddings[at], heads)
    rows = [store.rows[i] for i in at]
    others = already_known(rows)
    return {name: {**spread(values),
                   "repeats": against(values, {**others,
                                               **{k: v for k, v in fresh.items()
                                                  if k != name}})}
            for name, values in fresh.items()}


def listing(store: MapStore, sample: int, heads, by: str,
            top: bool = False) -> list[dict]:
    """Un campione ordinato su una delle quattro, da ascoltare in fila.

    `top` prende i piu' alti invece di distribuire su tutta la scala, e su
    una misura che dice sì al 3% dei brani e' l'unico campione che serve:
    distribuendo, ventiquattro righe su venticinque sarebbero brani che la
    testa non ha scelto, e la domanda — "quelli che indica sono davvero
    quelli?" — non riceverebbe risposta.
    """
    at = picked(store.rows, max(sample * 40, sample))
    fresh = scored(store.embeddings[at], heads)
    order = np.argsort(fresh[by])
    if top:
        order = order[::-1][:sample]
        step = 1
    else:
        step = max(1, len(order) // sample)
    return [{by: round(float(fresh[by][k]), 3),
             **{name: round(float(values[k]), 3)
                for name, values in fresh.items() if name != by},
             "name": store.rows[at[k]].get("name", ""),
             "moods": store.rows[at[k]].get("moods", ""),
             "bpm": store.rows[at[k]].get("bpm"),
             "path": store.rows[at[k]]["path"]}
            for k in order[::step][:sample]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prova le altre quattro teste del model zoo. Non scrive "
                    "niente: né nei tag, né nella mappa.")
    parser.add_argument("--sample", type=int, default=SAMPLE,
                        help=f"quanti brani guardare (default {SAMPLE})")
    parser.add_argument("--listen", type=int, metavar="N",
                        help="scrive N brani in ordine su una delle quattro, "
                             "da ascoltare")
    parser.add_argument("--by", default="party", choices=list(HEADS),
                        help="su quale delle quattro ordinare, con --listen")
    parser.add_argument("--top", action="store_true",
                        help="con --listen: i piu' alti invece di un campione "
                             "steso su tutta la scala")
    parser.add_argument("--out", type=Path, default=Path("zoo_sample.csv"))
    parser.add_argument("--store", default=None)
    parser.add_argument("--models", type=Path, default=MODEL_DIR)
    parser.add_argument("--download", action="store_true",
                        help="scarica le quattro teste che mancano")
    args = parser.parse_args()

    absent = missing(args.models)
    if absent and not args.download:
        print(f"Mancano {len(absent)} file in {args.models}:")
        for name in absent:
            print(f"  {name}")
        print("\nScaricali con  --download  (sono piccoli, meno di un mega "
              f"l'uno), oppure a mano da {ZOO_PAGE}")
        return
    if absent:
        args.models.mkdir(parents=True, exist_ok=True)
        for name in absent:
            print(f"  scarico {name}…", flush=True)
            error = fetch(name, args.models)
            if error:
                print(f"    non ci sono riuscito: {error}")
                print(f"    provato: {source_of(name)}")
                print(f"    se l'indirizzo è cambiato, il file sta in "
                      f"{ZOO_PAGE} e va messo in {args.models}")
                return
        print(f"  fatto: {len(absent)} file in {args.models}\n")

    store = MapStore.load(args.store or default_store_dir())
    print(f"Mappa: {len(store):,} brani, {len(store.embeddings):,} vettori")
    heads = _load(args.models)

    if args.listen:
        table = listing(store, args.listen, heads, args.by, top=args.top)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"  {len(table)} brani in {args.out}, ordinati su '{args.by}'")
        return

    for name, facts in report(store, args.sample, heads).items():
        print(f"\n{name}")
        print(f"  sopra 0,5: {facts['above 0.5']:.1%}"
              f"   valori distinti: {facts['distinct']:.1%}")
        print("  decili:   " + " ".join(f"{v:.3f}" for v in facts["deciles"]))
        ripete = sorted(facts["repeats"].items(), key=lambda kv: -abs(kv[1]))
        print("  ripete:   " + "  ".join(f"{k} {v:+.2f}" for k, v in ripete))


if __name__ == "__main__":
    main()
