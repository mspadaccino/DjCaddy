# Piano: Wavecut Desktop (Qt6) in parallel run con Streamlit

Stato: **approvato in linea di principio, da implementare a fasi**. Ogni fase è
pensata per una sessione di lavoro a sé, con criteri di verifica espliciti:
una sessione futura può prendere in mano una fase leggendo solo questo
documento e il codice.

## Obiettivo

Una app desktop tradizionale (PySide6/Qt6) per macOS e Win11, più fluida e
reattiva di Streamlit, che conserva **tutto**: lo stile dei grafici, la mappa
con selezione a lasso/rettangolo e clic, la lavagna della playlist col
trascinamento, il player, l'accesso agli stessi file JSON/store. In più:
tabelle native con riordino righe per trascinamento. Deploy standalone
(installazione semplice su macchine esterne, senza Python).

La app Streamlit **non si cancella**: le due app convivono su tre cartelle —
una per Streamlit, una per Qt, una per la logica condivisa — e leggono gli
stessi dati, per confrontare funzionalità e prestazioni (parallel run).

## Punto di partenza (rilevato il 30/08/2026)

- `analysis/` (~9.000 righe, 30 moduli) è già logica pura senza Streamlit:
  engine, map store, proiezione UMAP, tag Essentia, mixing, playlist, export.
  Coperta da ~30 file di test.
- `views/` (~6.000 righe) è lo strato Streamlit, ma contiene anche logica
  riusabile: costruzione delle figure Plotly (mappa, quad chart), palette e
  colonne (`track_columns.py`), payload della lavagna e della ruota Camelot.
- Due frontend HTML custom: `views/graph_board_frontend/index.html` (806
  righe: lavagna con drag&drop delle schede e cestino) e
  `views/camelot_wheel_frontend/index.html` (147 righe). Parlano col Python
  tramite l'API componenti di Streamlit (`postMessage`).
- Il job di costruzione mappa gira già come **processo separato**
  (`map_cli.py` lanciato con `sys.executable`, stato su file, pausa/riprendi):
  architettura riusabile identica da Qt.
- Sul branch `qt_test` esiste un prototipo PySide6 (commit `2dd9ff2`):
  QWebEngineView + `plotly.io.to_html`, `PandasModel`/QTableView,
  QMediaPlayer. Conferma la fattibilità dello stack; si riparte pulito ma si
  riciclano quelle idee (e il pin `pyside6 ^6.11`).

## Decisioni tecnologiche (con tradeoff)

### Grafici: Plotly dentro QWebEngineView, con ponte QWebChannel — SCELTO

La mappa e i grafici restano **le stesse figure Plotly di oggi**, costruite
da funzioni condivise e renderizzate in un `QWebEngineView` (Chromium
embedded, lo stesso motore in cui girano ora nel browser). Un ponte
QWebChannel porta gli eventi JS (`plotly_click`, `plotly_selected` per lasso
e rettangolo, `plotly_deselect`) a segnali Qt con gli indici dei punti.

- Pro: stile identico per costruzione; lasso/rettangolo/clic gratis; zero
  reimplementazione del rendering; WebGL regge i 120k punti come oggi.
- Contro: QtWebEngine pesa ~200MB nel bundle; il ponte JS è un pezzo nuovo.
- Alternativa scartata: **pyqtgraph** (più veloce, nativo GL) — perde lo
  stile e obbliga a rifare lasso, legenda, shading dei capitoli. Resta il
  fallback se lo spike di Fase 2 rivelasse lentezza (non atteso: il motore è
  lo stesso del browser, ma senza il rerun totale di Streamlit attorno).
- `plotly.min.js` va **incluso in locale** (niente CDN): l'app deve
  funzionare offline e nel bundle.

### Lavagna e ruota Camelot: riuso degli HTML esistenti — SCELTO

I due frontend HTML si spostano in `core/viz/frontend/` e si caricano in un
QWebEngineView con uno **shim** che imita l'API componenti di Streamlit
(stesso payload JSON in ingresso, `setComponentValue` → QWebChannel in
uscita). Il drag&drop delle schede, i colori, il cestino restano identici
senza riscrivere 800 righe di JS. Streamlit continua a caricarli dal nuovo
percorso col suo adapter attuale.

### Tabelle: QTableView nativo — SCELTO

`PandasModel` + `QTableView` con delegate per le "pastiglie" colorate
(tonalità Camelot, genere, energia — stesse palette di `core/viz`),
ordinamento per colonna, selezione multipla, **riordino righe per
trascinamento** (InternalMove), celle editabili dove oggi c'è
`st.data_editor`, menu contestuale (play, seed, aggiungi a playlist, mostra
nel Finder/Esplora). È qui che Qt dà la flessibilità chiesta: Streamlit non
sa riordinare righe col mouse, QTableView sì, e virtualizza 90k righe senza
serializzare nulla.

### Audio: QMediaPlayer nativo, player IDENTICO a oggi — SCELTO

**Requisito esplicito**: il riproduttore resta com'è ora — anteprima della
waveform a barre, porzione già riprodotta colorata, clic sulla waveform per
navigare lungo il brano, tempo corrente/durata. Cambia solo il motore sotto.

Player nel dock inferiore (equivalente di `st.bottom`), persistente su tutte
le pagine. QMediaPlayer legge mp3/flac direttamente da file (AVFoundation su
macOS, Media Foundation su Windows): spariscono i data-URI base64 e i
transcode ffmpeg fatti oggi per il browser. La waveform (dock e Wave
analysis) diventa un widget QPainter che disegna **gli stessi peaks**
(`analysis.waveform`) con le stesse barre e gli stessi colori del canvas
attuale in `components.py`, sincronizzato con `positionChanged`,
click-to-seek, e in Wave analysis i marker delle frasi/hot cue.

Criterio di accettazione: confronto fianco a fianco col dock Streamlit —
stesso brano, stessa forma d'onda, stessa risposta al clic.

### Stato: AppState con segnali, al posto di session_state

Un oggetto `AppState` (QObject) tiene seed, selezione, playlist, filtri,
now-playing; i widget si collegano ai segnali. È il motivo strutturale per
cui Qt sarà più reattivo: un clic sulla mappa aggiorna i due widget
interessati **in-process**, invece di rieseguire l'intero script e
riserializzare tabelle e figure sul websocket come fa Streamlit.

### Job lunghi: stessa architettura a processo

La costruzione mappa resta `map_cli.py` in subprocess con stato su file; Qt
fa polling con QTimer invece del rerun. Scansioni e duplicati vanno in
QThreadPool con segnali di progresso. Attenzione: in `core` ci sono chiamate
**solo-macOS** (`caffeinate`, `osascript`/Terminal.app, `ps`, Finder) da
mettere dietro un check di piattaforma con equivalenti o no-op su Windows.

### Packaging: PyInstaller, due profili — SCELTO

Build per piattaforma sulla piattaforma stessa (no cross-compile): `.app` +
DMG su macOS, `.exe` + installer (Inno Setup) su Win11.

Tradeoff dimensioni, da decidere in Fase 5 ma con raccomandazione già ora:

- **Bundle "DJ/viewer"** (raccomandato come artefatto di default): mappa,
  playlist, lavagna, wave review, folder analysis. Esclude demucs (torch,
  ~2GB) ed essentia-tensorflow, che sono già oggi opzionali a runtime
  (`available()`); pesa nell'ordine di 700MB–1GB (WebEngine + numba/librosa).
- **Install completa** (analisi, tagging, costruzione mappa): via poetry sul
  Mac di lavoro, come oggi. Eventualmente un bundle "full" solo macOS.

## Vincolo noto: Essentia non esiste per Windows

`essentia-tensorflow` pubblica wheel solo Linux/macOS. Su Win11 quindi:
Tag analysis e la **costruzione** della mappa (embedding) non girano in
nativo; **consumare** una mappa già costruita (store JSON), le playlist, la
lavagna, il player e Folder analysis funzionano. Il piano lo assume: Windows
è piattaforma di *consumo*, il Mac resta la macchina di *analisi*. Se un
domani servisse analisi su Windows: WSL o un servizio remoto — fuori scope.

## Struttura target del repository

```
Wavecut/
├── core/                      # logica condivisa dalle due app
│   ├── analysis/              # ← analysis/ attuale, spostata intera (import: core.analysis)
│   └── viz/                   # estratto da views/: figure Plotly, palette,
│       │                      #   colonne, payload lavagna/ruota — funzioni pure
│       └── frontend/          # graph_board e camelot_wheel (HTML riusati da entrambe)
├── streamlit_app/
│   ├── app.py                 # ← app.py attuale
│   └── views/                 # ← views/ attuale, assottigliate in Fase 1
├── qt_app/
│   ├── main.py
│   ├── state.py               # AppState (segnali)
│   ├── theme.py               # QSS scuro, stesso linguaggio visivo di oggi
│   ├── pages/                 # map, wave, tag, folder
│   └── widgets/               # plotly_view, track_table, board_view,
│                              #   wheel_view, player_dock, waveform
├── cli.py, map_cli.py, …      # restano in root; il percorso di map_cli
│                              #   centralizzato in core.analysis.map_job
├── tests/                     # invariati di posto, import aggiornati
├── docs/piano-qt.md           # questo documento
└── pyproject.toml             # gruppi poetry: base, streamlit, qt, essentia, dev
```

Regola d'oro di `core/viz`: le funzioni ricevono dataframe e stato, e
restituiscono **dati** (Figure Plotly, DataFrame, payload dict) — mai un
`st.*`, mai un widget Qt. Sono il contratto che garantisce "stesso grafico
nelle due app".

Naming: `core.analysis` comporta un sed meccanico di ~50 file (`from
analysis` → `from core.analysis`), gettato in sicurezza dai test. Il lancio
del job non ne risente (usa percorsi file, non `-m`).

## Fasi

### Fase 0 — Ristrutturazione del repo (parallel-run scaffold)

Nessuna funzionalità nuova: solo spostamenti.

1. Branch `qt6` da `main` (il vecchio `qt_test` resta come riferimento).
2. `git mv analysis core/analysis`; `git mv views streamlit_app/views`;
   `git mv app.py streamlit_app/app.py`.
3. Sed degli import (`analysis.` → `core.analysis.`, `views.` →
   `streamlit_app.views.`) in views, tests, CLI; aggiornare i percorsi in
   `st.Page(...)` e il percorso di `map_cli.py` in `map_analysis.py`
   (centralizzarlo in `core.analysis.map_job`).
4. `pyproject.toml`: gruppo `streamlit` (streamlit), gruppo `qt` (pyside6
   ^6.11), `essentia` e `dev` come oggi; il resto in base.
5. Verifica: `pytest` verde; `poetry run streamlit run streamlit_app/app.py`
   con lo store reale — le quattro sezioni si aprono e la mappa risponde;
   smoke di un CLI.

### Fase 1 — Estrazione di `core/viz` (la fase delicata)

Estrarre da `views/` la logica di presentazione condivisa, senza cambiare
comportamento:

- da `track_columns.py`: palette, ordine di lettura, regole colore (pure) →
  `core/viz`; i wrapper `st.column_config` restano in `streamlit_app`;
- da `map_analysis.py`: costruzione della figura mappa (punti, legenda per
  genere, cerchio del seme, cerchio della selezione playlist, shading dei
  capitoli) e del quad chart; logica dei filtri;
- da `graph_board.py`: payload JSON della lavagna e della ruota; tabelle del
  Chain Maker (pandas puro);
- spostare i due frontend HTML in `core/viz/frontend/` con l'adapter
  Streamlit aggiornato al nuovo percorso.

Verifica: **snapshot test** — prima del refactor si salvano i JSON delle
figure prodotte su un dataset fisso; dopo, le funzioni estratte devono
produrre JSON identici. Più `pytest` verde e confronto visivo su Streamlit.

### Fase 2 — Fondamenta Qt + spike di rischio

- Skeleton `qt_app`: MainWindow, 4 tab, tema QSS scuro, `AppState`, dock
  player persistente con anteprima waveform e click-to-seek (parità visiva
  col dock attuale — vedi criterio nella sezione Audio).
- `PlotlyView`: pagina HTML locale con plotly.js incluso, `set_figure()`,
  aggiornamenti via `Plotly.react`, ponte QWebChannel → segnali
  `point_clicked`, `points_selected(lasso/box)`, `deselected`.
- `TrackTable`: PandasModel, delegate pastiglie, sort, riordino drag&drop.
- **SPIKE go/no-go**: la mappa vera (store reale, decine di migliaia di
  punti) dentro PlotlyView — pan, zoom, lasso fluidi? Se no (non atteso),
  si rivaluta pyqtgraph prima di investire nella Fase 3.
- Verifica: tab demo dove il lasso sulla mappa reale stampa i path
  selezionati e il doppio clic su una riga di tabella suona il brano;
  pytest-qt su PandasModel e delegate.

**Esito (30/08/2026): GO.** Misurato sullo store reale (87.026 brani, tutto
in `qt_app/`, si lancia con `poetry run python -m qt_app.main`):

- primo disegno: 0,6 s di `Plotly.react`; aggiornamento con gli anelli di
  selezione: 0,2 s. La parte cara è ricostruire figura+JSON in Python a ogni
  gesto (~1,4 s a mappa piena, ~15 MB): per la Fase 3 conviene aggiornare i
  soli tracciati di contorno (seme, anelli, percorso) invece di rimandare
  tutta la nuvola. Pan/zoom/lasso da provare a mano, ma il motore è lo
  stesso del browser di oggi;
- lo shim della lavagna è stato provato **col frontend vero** (rischio
  chiuso): payload accettato e impaginato, `setComponentValue` torna come
  segnale Qt. Unica cura: gli args del componente sono il payload PIÙ
  `selected`/`chapters`/`dark`, e il fondo pagina va detto a QWebEnginePage
  (`setBackgroundColor`), perché qui il widget può essere più alto del
  disegno;
- il dock suona mp3 veri dal volume, onda a 800 barre da
  `core.analysis.waveform.envelope` (estratto apposta: stessi peaks per i
  due dock), click-to-seek verificato (60 s chiesti → 60,7 s);
- `plotly.min.js` si carica dal pacchetto Python via `baseUrl` file:// con
  `LocalContentCanAccessFileUrls` — niente copie, e nel bundle il pacchetto
  c'è comunque.

### Fase 3 — Pagina Map completa

Parità funzionale con la pagina Streamlit, spuntando una checklist 1:1:

- clic → seed; lasso/rettangolo → selezione; clic su vuoto → deselezione
  (con le stesse regole playlist→seed dei commit recenti);
- playlist: da selezione, da M3U8, magic sort, cerchio playlist sulla mappa;
- Chapter Builder con shading persistente;
- Chain Maker (tabelle native) + lavagna (BoardView con shim) + ruota
  Camelot; ricerca live; export M3U8/Rekordbox (già in core);
- costruzione mappa: lancio `map_cli`, polling QTimer, pausa/riprendi, log
  (finestra interna al posto di Terminal.app), guard di piattaforma per
  `caffeinate`/`osascript`.

### Fase 4 — Wave, Tag e Folder analysis

- Wave: WaveformWidget nativo (peaks + marker frasi + hot cue, click-to-seek,
  sync col player), tabella editabile al posto di `st.data_editor`.
- Tag: coda di lavoro dai file, job con progresso, breakdown; su Windows la
  pagina spiega che serve il Mac (vincolo Essentia).
- Folder: scansione in thread, duplicati, piano di quarantena.

### Fase 5 — Packaging

- Spec PyInstaller: dati inclusi (plotly.min.js, frontend HTML, eventuali
  modelli), esclusioni (torch, tensorflow nel profilo viewer), icona.
- macOS: `.app` + DMG (firma/notarizzazione se serve distribuire fuori);
  Win11: `.exe` + Inno Setup.
- Verifica: bundle installato su una macchina/utente pulito, senza Python:
  apre lo store, mappa e playlist funzionano, il player suona.

### Fase 6 — Parallel run e confronto

Le due app leggono gli stessi store; i job che scrivono si lanciano da una
sola app per volta (il lock/pid di `map_job` già esiste). Misure da
raccogliere: tempo di risposta a un lasso sulla mappa, apertura della
pagina Map, scroll di una tabella da 90k righe, avvio riproduzione. Esito:
decidere se e quando Streamlit va in pensione (non in questo piano).

## Come procedere: sessioni, agenti, modelli

**Una sessione nuova per fase**, col prompt: *"Leggi docs/piano-qt.md ed
esegui la Fase N"*. Il documento è scritto per bastare da solo; la sessione
fresca costa molto meno di una conversazione infinita e non trascina
contesto stantio. Niente sciami di agenti in parallelo: le fasi 0→3 sono
sequenziali per costruzione, e un agente in più su una fase sequenziale è
solo contesto pagato due volte. L'unica parallelizzazione sensata arriva in
Fase 4, dove Wave e Tag+Folder sono indipendenti (volendo, due sessioni o
due worktree in contemporanea).

**Modelli diversi per risparmiare: sì, dove i test fanno da rete.** La
regola: il modello economico va bene dove il criterio di verifica è
meccanico e stringente (pytest, snapshot, "il bundle parte"); il modello
forte serve dove il criterio è un giudizio (parità visiva, codice nuovo
senza riferimento, decisioni di spike). Il modello si sceglie all'avvio
della sessione (selettore del modello, o `claude --model claude-sonnet-5`
da terminale).

| Sessione | Fase | Modello | Perché | Esito atteso |
|---|---|---|---|---|
| 1 | Fase 0 | Sonnet | git mv + sed, gettato da pytest | repo ristrutturato, pytest verde, Streamlit invariato |
| 2 | Fase 1 | Opus/Fable | chirurgia su `map_analysis.py` (2.800 righe) | `core/viz` estratto, snapshot test verdi |
| 3 | Fase 2 | Opus/Fable | ponte QWebChannel nuovo + giudizio sullo spike | skeleton Qt + spike mappa superato |
| 4–5 | Fase 3 | Opus/Fable | la pagina più ricca, parità da giudicare | pagina Map a parità funzionale |
| 6 | Fase 4 (Wave) | Opus | widget waveform con parità visiva | wave review nativa, player identico |
| 7 | Fase 4 (Tag+Folder) | Sonnet | tabelle + job, pattern già stabiliti | quattro pagine complete |
| 8 | Fase 5 | Sonnet | iterativo, verificato dal bundle che parte | DMG + installer Windows |
| 9 | Fase 6 | Sonnet | misurazioni | numeri del confronto |

Le fasi con Sonnet si possono anche delegare a un agente lanciato da una
sessione di supervisione (Opus/Fable) che poi verifica i gate — utile se si
vuole restare in un'unica conversazione — ma il default resta la sessione
dedicata: più semplice, interattiva, e con checkpoint naturali.

## Rischi principali

| Rischio | Mitigazione |
|---|---|
| Perf mappa in QtWebEngine sotto le attese | spike in Fase 2 prima di investire; fallback pyqtgraph |
| Shim Streamlit→QWebChannel per la lavagna più ostico del previsto | provarlo in Fase 2 col frontend vero; piano B: riscrivere la lavagna come widget nativo (perde il riuso, non lo stile) |
| Essentia assente su Windows | Windows = consumo; analisi sul Mac; messaggi chiari in UI |
| Bundle enorme | profilo "viewer" senza torch/tensorflow; "full" solo macOS |
| Regressioni durante l'estrazione di Fase 1 | snapshot test dei JSON delle figure, prima di toccare |
| Chiamate solo-macOS in core (`caffeinate`, `osascript`, `ps`) | guard di piattaforma in Fase 3, no-op o equivalenti su Windows |

## Note operative

- Il prototipo su `qt_test` non si porta avanti come branch: si riparte da
  `main` con branch `qt6`; il commit `2dd9ff2` resta consultabile.
- Su `main` è rimasta una `QtApp/__pycache__/` orfana (residuo del cambio
  branch): da cancellare nella Fase 0.
- Python in uso: 3.14 (pin pyside6 `python <3.15` dal prototipo è corretto).
