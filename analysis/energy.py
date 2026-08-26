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
- `bright` — dove sta il baricentro dello spettro. È l'unica delle tre che
  separa una deep roller da una peak-time a PARITÀ di tempo, di groove e
  di basso: la prima è chiusa e scura, la seconda ha hats aperti e lead.

Nessuna delle tre da sola mette i brani nell'ordine giusto. La deep roller
ha PIÙ basso della peak-time; a leggere solo `bass` le si invertirebbe.

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
INGREDIENTS = ("energy_density", "energy_bass", "energy_bright")

# Quanto pesa ognuna nella media. Uguali: non ci sono dati per giustificare
# altro, e un peso inventato è peggio di un peso neutro. Si taranno
# ascoltando, come le soglie di `sections`.
WEIGHTS = (1.0, 1.0, 1.0)

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
    x = np.asarray(audio, dtype=np.float32).ravel()
    if x.size < _N_FFT:
        return None, None
    frames = sliding_window_view(x, _N_FFT)[::_HOP]
    if not len(frames):
        return None, None

    window = np.hanning(_N_FFT).astype(np.float32)
    total = np.zeros(_N_FFT // 2 + 1, dtype=np.float64)
    for start in range(0, len(frames), _BLOCK):
        chunk = frames[start:start + _BLOCK]
        # Via la continua, frame per frame. Non e' pignoleria: la finestra di
        # Hann sparpaglia il valore medio sui due bin accanto allo zero, e a
        # 44,1 kHz con 2048 punti il primo di quelli cade a 21,5 Hz, cioe'
        # DENTRO la banda bassa. Un file con un offset di registrazione
        # risulterebbe il piu' bassoso della libreria: misurato, il 40% della
        # sua potenza finiva sotto i 200 Hz senza che ci fosse una nota.
        chunk = (chunk - chunk.mean(axis=1, keepdims=True)) * window
        total += (np.abs(np.fft.rfft(chunk, axis=1)) ** 2).sum(axis=0)
    return np.fft.rfftfreq(_N_FFT, 1 / sr), total / len(frames)


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


def per_beat(onset_rate: float | None, bpm: float | None) -> float | None:
    """Gli attacchi per battito, dal loro numero al secondo.

    Normalizzare sul tempo non è un dettaglio: otto attacchi al secondo
    sono 3,75 per battito a 128 BPM e 5,3 a 90: senza dividere, l'energia
    ridirebbe il BPM invece di aggiungere qualcosa.
    """
    if onset_rate is None or bpm is None or bpm <= 0:
        return None
    if not np.isfinite(onset_rate) or not np.isfinite(bpm):
        return None
    return float(onset_rate * 60.0 / bpm)


def measure(audio, sr: float, onset_rate: float | None,
            bpm: float | None) -> dict:
    """Le tre misure di un brano, dalla sua finestra ritmica.

    `onset_rate` arriva da fuori perché nella mappa è già calcolato e
    buttato via: `OnsetRate` restituisce gli onset E il loro numero al
    secondo, e `map_profile` prende solo i primi. Ricalcolarlo qui
    vorrebbe dire pagare due volte la stessa cosa.
    """
    freqs, power = spectrum(audio, sr)
    return {"energy_density": per_beat(onset_rate, bpm),
            "energy_bass": bass_share(freqs, power),
            "energy_bright": brightness(freqs, power)}


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


def mix(density, bass, bright, weights=WEIGHTS) -> np.ndarray:
    """La media pesata dei tre ranghi, da 0 a 1. `nan` se non c'è niente.

    Una misura che manca esce dal conto invece di valere zero: un brano
    senza BPM non ha `density`, e contarla nulla lo spedirebbe in fondo
    alla scala per un dato assente invece che per come suona.
    """
    stacked = np.vstack([ranks(density), ranks(bass), ranks(bright)])
    present = np.isfinite(stacked)
    weight = np.where(present, np.asarray(weights, dtype=float)[:, None], 0.0)
    total = weight.sum(axis=0)
    weighted = np.where(present, stacked, 0.0) * weight
    return np.divide(weighted.sum(axis=0), total,
                     out=np.full(stacked.shape[1], np.nan), where=total > 0)


def spread(density, bass, bright, weights=WEIGHTS) -> np.ndarray:
    """L'energia da 0 (in basso) a 1 (in cima), distribuita davvero.

    È `mix` passata un'altra volta per il rango, e il secondo giro non è
    una ripetizione: la media di tre ranghi NON è distribuita in modo
    uniforme, si stringe attorno a 0,5 come ogni media di variabili
    indipendenti. Senza questo passaggio la libreria starebbe quasi tutta
    fra 0,35 e 0,65 e la lavagna disegnerebbe una riga piatta — cioè
    esattamente il difetto per cui l'asse energia dal mood era stato
    scartato, ricreato per un'altra strada.
    """
    return ranks(mix(density, bass, bright, weights))


def levels(density, bass, bright, weights=WEIGHTS) -> np.ndarray:
    """L'energia come la scrive DJoid: un intero da 1 a 10. `nan` se manca.

    Sono decili: il 10 è "il decimo più energico di CIÒ CHE HAI", non un
    livello assoluto. È la stessa lettura relativa del groove e del mood,
    che si tarano anche loro sui decili di questa libreria, ed è l'unica
    onesta — "energia 8" non vuol dire niente finché non si dice rispetto
    a cosa.
    """
    value = spread(density, bass, bright, weights)
    out = np.full(value.shape, np.nan)
    ok = np.isfinite(value)
    out[ok] = np.clip(1 + np.floor(value[ok] * 10), 1, 10)
    return out
