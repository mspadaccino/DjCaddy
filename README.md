# dj-library-tools

Strumento locale (macOS) per analizzare una libreria di mp3/flac e preparare il
lavoro di DJing su **djay Pro**:

- classificazione automatica per **genere/vibe**,
- organizzazione in cartelle `Genere/Vibe`,
- identificazione di **phrase boundary** suggerite per il posizionamento
  manuale degli hot cue.

> **Nota sui cue point.** djay Pro conserva cue point e loop nel proprio
> database interno, non nei tag dei file, e non espone un'importazione di cue
> esterni. Lo strumento quindi **non scrive hot cue dentro djay Pro**: produce
> timestamp *suggeriti* che confermi a orecchio nell'app di revisione e poi
> posizioni a mano in djay Pro.

## Architettura

Un **motore di analisi condiviso** (`analysis/`, modulo Python puro) importato
sia dal CLI batch sia dall'app Streamlit — nessuna logica duplicata.

| Modulo | Responsabilità |
| --- | --- |
| `analysis/tags.py` | lettura tag genere via mutagen (ID3 per mp3, Vorbis comment per flac) |
| `analysis/audio_features.py` | caricamento audio (librosa) + BPM e RMS in un'unica passata |
| `analysis/vibe.py` | bucket di tempo + energia a percentili (two-pass) → vibe |
| `analysis/structure.py` | segmentazione strutturale (Foote novelty su self-similarity) → phrase boundary |
| `analysis/sections.py` | classificazione delle sezioni (Intro/Build-up/Drop/Breakdown/Outro) da arco di energia e presenza di basso |
| `analysis/vocals.py` | rilevamento voce via source separation (Demucs): regioni cantate + flag 🎤 per sezione |
| `analysis/waveform.py` | waveform colorata per bande di frequenza (stile djay Pro) |
| `analysis/cache.py` | cache per-file (chiave = path, valida per mtime+size) |
| `analysis/engine.py` | orchestrazione: two-pass, cache, piano di organizzazione |
| `cli.py` | entry point 1 — CLI batch |
| `app.py` | entry point 2 — app Streamlit di revisione |

**Caricamento audio.** Ogni file viene caricato **una sola volta** (22050 Hz,
mono); BPM/RMS sono calcolati sui primi 60 s di quel segnale, la segmentazione
sull'intero brano.

**Vibe.** BPM → bucket di tempo (`Warm-Up`/`Groove`/`Peak-Time`/
`High-Energy-Tempo`); RMS → percentili 33/66 relativi alla libreria →
`Low`/`Mid`/`High`. Vibe finale es. `Peak-Time-High`. Bucket e percentili si
regolano in `analysis/vibe.py`.

## Setup

Gli **mp3** richiedono **ffmpeg** a livello di sistema (librosa li decodifica
via audioread); i **flac** vengono letti nativamente da soundfile e non ne
hanno bisogno:

```bash
brew install ffmpeg
```

Dipendenze Python con Poetry (Python ^3.11):

```bash
poetry install
```

## Uso

### CLI batch

```bash
# solo report a video
poetry run python cli.py ~/Music/dj

# report su file + dry-run dell'organizzazione (copia, non sposta)
poetry run python cli.py ~/Music/dj --dest ~/Music/master --report report.csv --dry-run

# organizza davvero in Genere/Vibe (senza sovrascrivere file esistenti)
poetry run python cli.py ~/Music/dj --dest ~/Music/master
```

Il report (CSV o JSON) contiene per ogni traccia: path, genere, BPM, vibe e i
timestamp delle phrase boundary suggerite. La cache evita di rianalizzare i
file già processati: usa `--no-cache` per forzare la rianalisi.

### App Streamlit (revisione)

```bash
poetry run streamlit run app.py
```

Indica il **percorso** della cartella (i file sono già su disco, niente
upload), lancia l'analisi (stesso motore e cache del CLI), poi per ogni traccia
rivedi la **forma d'onda colorata per bande di frequenza** (stile djay Pro:
rosso = bassi, verde = medi, blu = alti) con i **tag di sezione** sovrapposti
(Intro/Build-up/Drop/Breakdown/Outro). Per ogni tag uno **slider** sposta
l'inizio della sezione e un menù ne cambia l'etichetta; il grafico e il report
scaricabile si aggiornano. Ascolti dall'inizio di ogni sezione per confermare a
orecchio.

> La classificazione delle sezioni è **euristica** (regole su energia e basso,
> soglie in `analysis/sections.py`): pensata come punto di partenza da correggere
> a orecchio, non come verità. Le sezioni ambigue sono marcate `Groove`. Le
> sezioni consecutive dello stesso tipo vengono **collassate**: ogni tag segna un
> **cambio di phrase**, per anticipare in djay Pro cosa sta per arrivare.

Il **rilevamento voce** usa Demucs (source separation) per isolare lo stem
vocale: le **regioni cantate** appaiono come bande rosa sul grafico (la parte da
non sovrapporre ad altre voci in mixaggio) e le sezioni con voce ricevono il flag
🎤. È accurato ma **pesante**: scarica un modello alla prima esecuzione e gira
una rete neurale su ogni brano. È opzionale — con `--no-vocals` (CLI) o togliendo
la spunta nell'app lo salti; se Demucs non è installato il flag resta manuale.

## Test

```bash
poetry run pytest
```

I test coprono la logica pura (bucket/percentili, cache, novelty). L'analisi
audio end-to-end va provata su un **piccolo sottoinsieme** di file prima di
girare sull'intera libreria.
