"""L'energia come tre misure e una scala sola.

Il mood dice di che colore è un brano, il groove quanto è regolare il suo
ritmo. Nessuno dei due dice quanto SPINGE — e per costruire un set è la
domanda che si fa più spesso: da qui in poi si sale o si scende.

**Perché tre misure e non una.** L'energia percepita non ha un correlato
unico nel segnale. Ha tre componenti che si muovono in modo indipendente:

- `density` — quanti attacchi ci sono per battito. Un kick da solo e un kick
  con hats, shaker e percussioni hanno lo stesso tempo e lo stesso groove:
  cambia quanto è fitto il tessuto ritmico.
- `bass` — quanta potenza sta sotto i 200 Hz. È la parte fisica: un
  breakdown senza cassa ha energia percepita bassa anche se il resto suona.
  La soglia è la stessa che `sections.py` usa per dire "il basso c'è".
- `bright` — dove sta il baricentro dello spettro. È l'unica che separa una
  deep roller da una peak-time a PARITÀ di tempo, di groove e di basso: la
  prima è chiusa e scura, la seconda ha hats aperti e lead.
- `pulse` — quanto il basso batte IN TEMPO, cioè quanta della sua energia è
  periodica sul battito. Le altre tre misurano quanto un brano è AFFOLLATO;
  questa misura se SPINGE, che non è la stessa cosa. Una cassa dritta pulsa
  su ogni battito; un 808 sincopato con gli hats a riempire i buchi ha la
  stessa densità e non muove niente.

Nessuna da sola mette i brani nell'ordine giusto. La deep roller ha PIÙ
basso della peak-time; a leggere solo `bass` le si invertirebbe. E il
giudizio a orecchio su quattrocento brani ha mostrato che le prime tre non
bastano: hip hop, ragga e pop a 85-95 BPM stanno nel decile alto di tutte e
tre — sono produzioni affollate — e nessun peso poteva portarli dove un DJ
li sente senza trascinare giù anche l'acid house.

**Perché il rango e non il valore.** Le tre misure hanno unità
incompatibili — attacchi per battito, un rapporto, degli hertz — e sommarle
vorrebbe dire far decidere tutto agli hertz, che sono numeri mille volte
più grandi. Si passa allora per il rango percentile: non "4,2 attacchi per
battito" ma "più fitto dell'81% della libreria". Dopo quel passaggio le tre
misure dicono la stessa cosa e si possono mediare.

C'è un secondo motivo, ed è quello che conta di più. L'asse energia era già
stato provato sulle etichette di mood e scartato (vedi `mood_scale`): il
decile inferiore restava incollato allo zero e la curva sapeva solo salire.
Con il rango quel fallimento è impossibile per costruzione — se ordino la
libreria dal meno al più energico, il 10% più basso esiste per definizione.

**Cosa si salva su disco.** Le tre misure grezze, non il valore da 1 a 10.
Il voto dipende dalla libreria intera: congelarlo nella riga vorrebbe dire
che il giorno che aggiungi cinquemila brani tutti i voti vecchi diventano
sbagliati. Tenendo i grezzi la scala si ritara da sola quando la mappa
cresce, e i pesi si cambiano senza rianalizzare niente — è lo stesso
two-pass che `vibe` già dichiara.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# I nomi con cui le tre misure vivono nella riga della mappa.
INGREDIENTS = ("energy_density", "energy_bass", "energy_bright", "energy_pulse")

# Quanto pesa ognuna nella media. Uguali: non ci sono dati per giustificare
# altro, e un peso inventato è peggio di un peso neutro. Si taranno
# ascoltando, come le soglie di `sections`.
WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Sotto questo tempo non si conta: si raddoppia. Una libreria da DJ porta i
# BPM dai tag, e i tag scrivono il mezzo tempo — un hip hop segnato 60 è un
# 120 contato a metà, e diviso per 60 la sua densità viene il doppio del vero.
#
# Solo verso l'alto, e questa asimmetria viene da un giudizio a orecchio su
# duecento brani. Piegare anche verso il basso — l'ottava 70-140 che questo
# codice ha avuto per un giorno — dimezzava il denominatore dei brani veloci e
# quindi ne RADDOPPIAVA la densità: i due errori peggiori del campione erano
# una bossa nova a 151,6 (data 9, vale 2) e un forró a 149 (data 9, vale 3),
# cioè i due brani più veloci di tutti. Sopra i 140 il tag di solito ha
# ragione: drum'n'bass, hardcore, samba e forró sono veloci davvero, e
# piegarli è inventare un tempo che nessuno sente.
TEMPO_FLOOR = 70.0

# Sotto questo livello la finestra non è musica: è un decode fallito o un file
# vuoto. Misurare lì dentro dà numeri che sembrano validi e non lo sono.
SILENCE_RMS = 1e-4        # -80 dBFS, molto sotto qualunque registrazione vera

LOW_HZ = 200.0    # "sotto" è la stessa soglia con cui sections.py sente la cassa
MIN_HZ = 20.0     # sotto non c'è musica: c'è la continua e il rumore di trasporto

# Finestra e passo dell'analisi spettrale: gli stessi di sections.py, così
# le due parti del programma guardano lo spettro con lo stesso occhio.
_N_FFT = 2048
_HOP = 512
_BLOCK = 256      # quanti frame per volta, per non tenere 40 MB per worker


# --------------------------------------------------------------------------
# Le tre misure, da audio già caricato
# --------------------------------------------------------------------------

def _scan(audio, sr: float):
    """Una passata sola sui frame: lo spettro medio E l'inviluppo del basso.

    Le due cose escono dalla stessa FFT e calcolarle separatamente
    raddoppiava il costo del backfill — misurato, mezzo secondo a brano che
    su ottantasettemila diventa una notte in piu'.
    """
    x = np.asarray(audio, dtype=np.float32).ravel()
    if x.size < _N_FFT:
        return None, None, None
    frames = sliding_window_view(x, _N_FFT)[::_HOP]
    if not len(frames):
        return None, None, None

    window = np.hanning(_N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(_N_FFT, 1 / sr)
    band = (freqs >= MIN_HZ) & (freqs < LOW_HZ)
    total = np.zeros(_N_FFT // 2 + 1, dtype=np.float64)
    envelope = np.empty(len(frames))
    for start in range(0, len(frames), _BLOCK):
        chunk = frames[start:start + _BLOCK]
        # Via la continua, frame per frame. Non e' pignoleria: la finestra di
        # Hann sparpaglia il valore medio sui due bin accanto allo zero, e a
        # 44,1 kHz con 2048 punti il primo di quelli cade a 21,5 Hz, cioe'
        # DENTRO la banda bassa. Un file con un offset di registrazione
        # risulterebbe il piu' bassoso della libreria: misurato, il 40% della
        # sua potenza finiva sotto i 200 Hz senza che ci fosse una nota.
        chunk = (chunk - chunk.mean(axis=1, keepdims=True)) * window
        power = np.abs(np.fft.rfft(chunk, axis=1)) ** 2
        total += power.sum(axis=0)
        envelope[start:start + len(chunk)] = power[:, band].sum(axis=1)
    return freqs, total / len(frames), envelope


def _modulation(envelope, sr: float, tempo: float) -> float | None:
    """Quanto l'inviluppo va su e giu' alla frequenza del battito, in
    proporzione al suo livello medio. Vedi `pulse` per il perche'."""
    if envelope is None:
        return None
    per_second = sr / _HOP
    # Almeno quattro battiti di inviluppo, o la componente non si distingue
    # da un andamento qualunque.
    if len(envelope) / per_second * tempo / 60.0 < 4:
        return None
    taper = np.hanning(len(envelope))
    level = float(np.dot(envelope, taper) / taper.sum())
    if level <= 0:
        return None
    turns = 2 * np.pi * (tempo / 60.0) / per_second * np.arange(len(envelope))
    beat = np.abs(np.dot(envelope * taper, np.exp(-1j * turns)))
    return float(2 * beat / taper.sum() / level)


def spectrum(audio, sr: float):
    """Lo spettro di potenza medio della finestra, e le sue frequenze.

    `None, None` se il pezzo è più corto di una finestra di FFT: da meno di
    2048 campioni non esce uno spettro, e inventarne uno sarebbe peggio che
    dire "non lo so".

    Si media la potenza dei frame invece di trasformare tutto il pezzo in
    un colpo solo: una FFT lunga quanto trenta secondi darebbe una
    risoluzione in frequenza che non serve a nessuno e sarebbe dominata da
    qualunque transiente.
    """
    freqs, power, _ = _scan(audio, sr)
    return freqs, power


def bass_share(freqs, power) -> float | None:
    """La quota di potenza sotto i 200 Hz, da 0 a 1.

    Un rapporto e non un livello: così non dipende da quanto è forte il
    brano, che è esattamente la proprietà che la loudness NON ha e per cui
    la loudness non entra in questo conto.

    Sotto i 20 Hz si taglia: lì non c'è musica ma può esserci del rumore di
    trasporto, e finirebbe dritto nel numeratore facendo passare per bassoso
    un brano che ha solo un file registrato male. La continua vera e propria
    è già stata tolta in `spectrum`, dove il taglio da solo non la fermava.
    """
    if freqs is None:
        return None
    band = freqs >= MIN_HZ
    total = power[band].sum()
    if total <= 0:
        return None
    return float(power[band & (freqs < LOW_HZ)].sum() / total)


def brightness(freqs, power) -> float | None:
    """Il centroide spettrale in Hz: dove sta il baricentro del suono.

    Sull'ampiezza e non sulla potenza, che è la definizione consueta: la
    potenza pesa i picchi al quadrato e sposta il centroide su qualunque
    singola frequenza forte, mentre qui interessa dove sta il corpo del
    suono.
    """
    if freqs is None:
        return None
    band = freqs >= MIN_HZ
    magnitude = np.sqrt(power[band])
    total = magnitude.sum()
    if total <= 0:
        return None
    return float((freqs[band] * magnitude).sum() / total)


def fold_tempo(bpm: float | None) -> float | None:
    """Il tempo riportato sopra la soglia, raddoppiando finché serve.

    Un mezzo tempo scritto nel tag è lo stesso battito contato ogni due, e
    raddoppiarlo non perde niente. Un tempo alto invece si lascia stare: vedi
    `TEMPO_FLOOR` per il perché l'asimmetria non è una svista.
    """
    if bpm is None or not np.isfinite(bpm) or bpm <= 0:
        return None
    while bpm < TEMPO_FLOOR:
        bpm *= 2
    return float(bpm)


def per_beat(onset_rate: float | None, bpm: float | None) -> float | None:
    """Gli attacchi per battito, dal loro numero al secondo.

    Normalizzare sul tempo non è un dettaglio: otto attacchi al secondo
    sono 3,75 per battito a 128 BPM e 5,3 a 90: senza dividere, l'energia
    ridirebbe il BPM invece di aggiungere qualcosa.
    """
    tempo = fold_tempo(bpm)
    if onset_rate is None or tempo is None or not np.isfinite(onset_rate):
        return None
    return float(onset_rate * 60.0 / tempo)


def usable(audio) -> bool:
    """Se in questa finestra c'è abbastanza segnale da misurare qualcosa.

    Un decode fallito — su una libreria vera capita, ffmpeg lo dice e tira
    dritto — lascia una finestra quasi muta. Le tre misure ci girano sopra
    lo stesso e restituiscono numeri dall'aria sana: sul campione un brano
    letto male è uscito con basso 0,000 e centroide a 10 kHz, che nessuna
    registrazione ha. Meglio dire "non lo so".
    """
    x = np.asarray(audio, dtype=np.float64).ravel()
    if x.size < _N_FFT:
        return False
    return float(np.sqrt(np.mean(x * x))) > SILENCE_RMS


def pulse(audio, sr: float, bpm: float | None) -> float | None:
    """Quanto il basso batte in tempo: la profondità della sua pulsazione.

    Si prende l'inviluppo della sola banda bassa — quanta potenza sotto i
    200 Hz c'è istante per istante — e se ne misura la componente ALLA
    frequenza del battito, rapportata al suo livello medio. È l'indice di
    modulazione: quanto il fondo va su e giù a tempo, in proporzione a
    quanto fondo c'è.

    Una cassa dritta alterna colpo e silenzio su ogni battito e dà un valore
    alto; un 808 sincopato, che cade in punti diversi di ogni battuta, mette
    la sua energia ad altre frequenze e su questa quasi niente; un sub
    tenuto, che è bassissimo ma non batte, dà zero — ed è il caso per cui la
    prima versione di questa funzione, che correlava l'inviluppo con sé
    stesso spostato di un battito, dava 0,78: un inviluppo costante meno la
    sua media è rumore numerico, e la correlazione di quel rumore non
    significa niente. Rapportare alla media invece si annulla da sé.

    Il battito viene dal BPM piegato: un tag a mezzo tempo cercherebbe la
    modulazione a metà frequenza, dove una cassa dritta ha molta meno
    energia che sul battito vero.

    Misura la pulsazione, NON la quantità: un brano con pochissimo fondo, se
    quel poco cade in tempo, esce alto lo stesso. Non è un difetto da
    correggere qui — quanto fondo ci sia lo dice già `bass_share`, e le due
    entrano nella media una di fianco all'altra. Separarle è il motivo per
    cui un ragga fitto ma senza cassa dritta e una house con la cassa dritta
    non si confondono più: il primo perde su `pulse`, la seconda vince su
    entrambe.
    """
    tempo = fold_tempo(bpm)
    if tempo is None:
        return None
    return _modulation(_scan(audio, sr)[2], sr, tempo)


def measure(audio, sr: float, onset_rate: float | None,
            bpm: float | None) -> dict:
    """Le tre misure di un brano, dalla sua finestra ritmica.

    `onset_rate` arriva da fuori perché nella mappa è già calcolato e
    buttato via: `OnsetRate` restituisce gli onset E il loro numero al
    secondo, e `map_profile` prende solo i primi. Ricalcolarlo qui
    vorrebbe dire pagare due volte la stessa cosa.
    """
    if not usable(audio):
        return dict.fromkeys(INGREDIENTS)
    freqs, power, envelope = _scan(audio, sr)
    tempo = fold_tempo(bpm)
    return {"energy_density": per_beat(onset_rate, bpm),
            "energy_bass": bass_share(freqs, power),
            "energy_bright": brightness(freqs, power),
            "energy_pulse": _modulation(envelope, sr, tempo) if tempo else None}


# --------------------------------------------------------------------------
# Da tre misure a una scala, sulla libreria intera
# --------------------------------------------------------------------------

def ranks(values) -> np.ndarray:
    """Il rango percentile di ogni valore, da 0 a 1. `nan` resta `nan`.

    I pari merito prendono il rango medio del loro gruppo: senza, due brani
    identici finirebbero uno sopra l'altro per l'ordine in cui stanno nel
    file, che non è un'informazione.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n == 0:
        return out
    if n == 1:
        out[ok] = 0.5      # da solo un brano non è né in alto né in basso
        return out
    _, inverse, counts = np.unique(v[ok], return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out[ok] = (starts + (counts - 1) / 2.0)[inverse] / (n - 1)
    return out


def mix(*columns, weights=WEIGHTS) -> np.ndarray:
    """La media pesata dei tre ranghi, da 0 a 1. `nan` se non c'è niente.

    Una misura che manca esce dal conto invece di valere zero: un brano
    senza BPM non ha `density`, e contarla nulla lo spedirebbe in fondo
    alla scala per un dato assente invece che per come suona.
    """
    stacked = np.vstack([ranks(c) for c in columns])
    weights = np.asarray(weights, dtype=float)[:len(columns)]
    present = np.isfinite(stacked)
    weight = np.where(present, np.asarray(weights, dtype=float)[:, None], 0.0)
    total = weight.sum(axis=0)
    weighted = np.where(present, stacked, 0.0) * weight
    return np.divide(weighted.sum(axis=0), total,
                     out=np.full(stacked.shape[1], np.nan), where=total > 0)


def spread(*columns, weights=WEIGHTS) -> np.ndarray:
    """L'energia da 0 (in basso) a 1 (in cima), distribuita davvero.

    È `mix` passata un'altra volta per il rango, e il secondo giro non è
    una ripetizione: la media di tre ranghi NON è distribuita in modo
    uniforme, si stringe attorno a 0,5 come ogni media di variabili
    indipendenti. Senza questo passaggio la libreria starebbe quasi tutta
    fra 0,35 e 0,65 e la lavagna disegnerebbe una riga piatta — cioè
    esattamente il difetto per cui l'asse energia dal mood era stato
    scartato, ricreato per un'altra strada.
    """
    return ranks(mix(*columns, weights=weights))


def levels(*columns, weights=WEIGHTS) -> np.ndarray:
    """L'energia come la scrive DJoid: un intero da 1 a 10. `nan` se manca.

    Sono decili: il 10 è "il decimo più energico di CIÒ CHE HAI", non un
    livello assoluto. È la stessa lettura relativa del groove e del mood,
    che si tarano anche loro sui decili di questa libreria, ed è l'unica
    onesta — "energia 8" non vuol dire niente finché non si dice rispetto
    a cosa.
    """
    value = spread(*columns, weights=weights)
    out = np.full(value.shape, np.nan)
    ok = np.isfinite(value)
    out[ok] = np.clip(1 + np.floor(value[ok] * 10), 1, 10)
    return out
