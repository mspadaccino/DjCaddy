"""Il grafo costruito a mano: due brani, e poi uno alla volta.

La mappa risponde alla domanda "cosa c'è vicino"; questo modulo risponde a
quella dopo, che è "cosa ci metto DIETRO". La differenza è che qui il
percorso non si disegna in un gesto solo — si cresce un brano per volta,
scegliendo ogni passo fra una rosa di adiacenti, e a ogni scelta il grafo
propone la rosa successiva. Il DJ resta quello che decide: la macchina
restringe il campo, non sceglie.

**I nodi sono percorsi di file, non posizioni nella libreria.** Vale qui la
stessa ragione per cui la sessione della mappa tiene i percorsi: una
posizione vale finché la mappa non cambia, e basta che un job aggiunga brani
e rifaccia la proiezione perché la 200 sia un altro brano. Un grafo che
indica le tracce sbagliate è peggio di un grafo perduto.

Il modulo è puro di proposito: niente Streamlit, niente DataFrame. Prende e
restituisce tipi elementari, così si può provarlo senza avviare l'app — e
così la lavagna, che è l'unico pezzo che sa di pixel, resta sottile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from analysis.mixing import TransitionCost, nearest


@dataclass
class GraphPlaylist:
    """Brani su una lavagna, collegati da linee.

    `places` tiene dove sta ogni brano (coordinate della lavagna, non della
    mappa: qui il posto lo decide la mano, non l'embedding). `links` tiene le
    coppie collegate, non orientate e senza ripetizioni. `order` ricorda in
    che ordine sono arrivati, che serve a dare un capo al percorso quando il
    grafo non ne ha uno evidente.
    """

    places: dict[str, tuple[float, float]] = field(default_factory=dict)
    links: list[tuple[str, str]] = field(default_factory=list)
    order: list[str] = field(default_factory=list)

    # -- lettura -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.places)

    def __contains__(self, track: str) -> bool:
        return track in self.places

    @property
    def tracks(self) -> list[str]:
        """I brani sulla lavagna, nell'ordine in cui ci sono arrivati."""
        return list(self.order)

    def neighbours(self, track: str) -> list[str]:
        """I brani collegati a questo, nell'ordine in cui sono arrivati."""
        touching = {b if a == track else a
                    for a, b in self.links if track in (a, b)}
        return [t for t in self.order if t in touching]

    def degree(self, track: str) -> int:
        return len(self.neighbours(track))

    def ends(self) -> list[str]:
        """I capi liberi: i brani con un solo collegamento.

        Sono i punti da cui ha senso continuare a crescere il percorso, e
        quelli da cui conviene leggerlo.
        """
        return [t for t in self.order if self.degree(t) == 1]

    # -- scrittura ---------------------------------------------------------

    def start(self, *tracks: str, spread: float = 0.22) -> "GraphPlaylist":
        """Ricomincia da capo con uno o più brani, in fila e collegati.

        Basta un brano. Per un po' ne servivano due, con la scusa che è la
        coppia a dire in che direzione si sta andando — ma non era vero di
        questo codice: `suggestions` chiede la rosa a UN brano solo, e il
        secondo non entrava nel conto. Chiedere due cose per usarne una è far
        pagare all'utente una regola che non esiste.

        Più di uno si accetta perché la selezione arriva anche da fuori — dal
        gesto sulla mappa, che di brani ne prende quanti gliene si indicano —
        e vanno in fila da sinistra a destra, collegati in sequenza: l'ordine
        in cui sono stati scelti è l'unico che si conosca.
        """
        chosen = list(dict.fromkeys(tracks))
        if not chosen:
            raise ValueError("serve almeno un brano di partenza")
        if len(chosen) != len(tracks):
            raise ValueError("i brani di partenza devono essere diversi")
        if len(chosen) == 1:
            self.places = {chosen[0]: (0.5, 0.5)}
        else:
            step = 2 * spread / (len(chosen) - 1)
            self.places = {track: (0.5 - spread + step * n, 0.5)
                           for n, track in enumerate(chosen)}
        self.links = list(zip(chosen, chosen[1:]))
        self.order = chosen
        return self

    def add(self, source: str, track: str,
            place: tuple[float, float] | None = None) -> "GraphPlaylist":
        """Attacca `track` al brano `source`, da cui è stato scelto.

        Il collegamento non è un dettaglio grafico: dice DA DOVE viene il
        suggerimento, ed è quello che rende il grafo leggibile a distanza di
        giorni. Un brano già sulla lavagna non viene duplicato — si aggiunge
        solo il collegamento mancante.
        """
        if source not in self.places:
            raise KeyError(f"il brano di partenza non è sulla lavagna: {source}")
        if track == source:
            raise ValueError("un brano non si collega a sé stesso")
        if track not in self.places:
            self.places[track] = place or _beside(
                self.places[source], len(self.order),
                tuple(self.places.values()))
            self.order.append(track)
        self.connect(source, track)
        return self

    def connect(self, a: str, b: str) -> "GraphPlaylist":
        """Collega due brani già sulla lavagna, se non lo sono già."""
        if a not in self.places or b not in self.places:
            raise KeyError("si collegano solo brani già sulla lavagna")
        if a != b and not self.linked(a, b):
            self.links.append((a, b))
        return self

    def linked(self, a: str, b: str) -> bool:
        return (a, b) in self.links or (b, a) in self.links

    def move(self, track: str, x: float, y: float) -> "GraphPlaylist":
        """Sposta un brano. È l'unica cosa che la lavagna fa da sola."""
        if track in self.places:
            self.places[track] = (float(x), float(y))
        return self

    def straighten(self, per_row: int = 6) -> "GraphPlaylist":
        """Rimette i brani in fila nell'ordine in cui si leggono.

        Trascinando si finisce con una lavagna che dice il vero sui
        collegamenti e il falso sulla sequenza: due brani vicini d'occhio
        possono stare a due rami di distanza. Questo la riallinea all'ordine
        di `walk`, che è quello con cui la scaletta uscirà davvero.

        Le righe si alternano di verso, come si scrive un solco: così il
        brano che chiude una riga resta accanto a quello che apre la
        successiva, invece di attraversare tutta la lavagna per raggiungerlo.
        """
        walk = self.walk()
        if not walk:
            return self
        rows = max(1, -(-len(walk) // per_row))
        for n, track in enumerate(walk):
            row, seat = divmod(n, per_row)
            wide = min(per_row, len(walk) - row * per_row)
            if row % 2:
                seat = wide - 1 - seat
            x = 0.5 if wide == 1 else 0.08 + 0.84 * seat / (wide - 1)
            y = 0.5 if rows == 1 else 0.15 + 0.7 * row / (rows - 1)
            self.places[track] = (x, y)
        return self

    def remove(self, track: str) -> "GraphPlaylist":
        """Toglie un brano e ricuce il percorso.

        Se stava in mezzo a due — il caso normale in una catena — i suoi due
        vicini restano collegati fra loro: togliere un brano da una scaletta
        non deve spezzarla in due. Se invece era uno snodo con tre o più
        diramazioni, ricucire vorrebbe dire inventare collegamenti che nessuno
        ha scelto, e allora i rami restano separati.
        """
        if track not in self.places:
            return self
        touching = self.neighbours(track)
        self.places.pop(track)
        self.order.remove(track)
        self.links = [(a, b) for a, b in self.links if track not in (a, b)]
        if len(touching) == 2:
            self.connect(*touching)
        return self

    # -- uscita ------------------------------------------------------------

    def walk(self) -> list[str]:
        """Il grafo letto come una scaletta, dall'inizio alla fine.

        Si parte da un capo libero (o dal primo brano messo, se capi non ce
        ne sono perché il grafo si chiude ad anello) e si percorre in
        profondità: su una catena — la forma che viene fuori scegliendo un
        brano dopo l'altro — è semplicemente l'ordine dei brani. Su un grafo
        con diramazioni si scende un ramo fino in fondo prima di tornare
        indietro a prendere il successivo, che è il modo in cui quel ramo è
        stato pensato.
        """
        if not self.order:
            return []
        free = self.ends()
        start = free[0] if free else self.order[0]
        seen: list[str] = []
        stack = [start]
        while stack:
            track = stack.pop()
            if track in seen:
                continue
            seen.append(track)
            # In pila al contrario, così a uscire per primo è il vicino
            # arrivato per primo: il ramo più vecchio si legge prima.
            stack.extend(reversed([t for t in self.neighbours(track)
                                   if t not in seen]))
        return seen

    # -- stato -------------------------------------------------------------

    def to_state(self) -> dict:
        """Il grafo come tipi elementari: sessione Streamlit e JSON."""
        return {
            "places": {t: [x, y] for t, (x, y) in self.places.items()},
            "links": [[a, b] for a, b in self.links],
            "order": list(self.order),
        }

    @classmethod
    def from_state(cls, state: dict | None) -> "GraphPlaylist":
        """Rilegge quello che `to_state` ha scritto. Da `None` un grafo vuoto.

        I brani senza posto o fuori da `order` vengono scartati, e i
        collegamenti che pendono nel vuoto con loro: uno stato mezzo rotto
        deve dare una lavagna povera, non una schermata di errore.
        """
        if not state:
            return cls()
        raw = state.get("places") or {}
        places = {t: (float(p[0]), float(p[1])) for t, p in raw.items()
                  if isinstance(p, (list, tuple)) and len(p) == 2}
        order = [t for t in (state.get("order") or []) if t in places]
        # Un brano che ha un posto ma nessuno lo elenca esiste comunque: in
        # coda, che è dove sarebbe arrivato.
        order += [t for t in places if t not in order]
        links = []
        for pair in state.get("links") or []:
            if len(pair) == 2 and pair[0] in places and pair[1] in places:
                a, b = str(pair[0]), str(pair[1])
                if a != b and (a, b) not in links and (b, a) not in links:
                    links.append((a, b))
        return cls(places=places, links=links, order=order)


# Quanto occupa una scheda sulla lavagna, in coordinate normalizzate. Le due
# misure sono molto diverse perché la lavagna è larga circa il doppio di
# quanto è alta, e la scheda è più alta che larga: in verticale "una scheda"
# vale tre volte quello che vale in orizzontale. Un raggio unico per le due
# direzioni — che è quello che c'era — le faceva sovrapporre sempre, perché
# bastava a scostarle di fianco ma era metà di quanto serve sopra e sotto.
CARD_SPAN = (0.11, 0.33)

# Quanti posti provare prima di arrendersi e posare comunque. Tre giri di
# otto: oltre, il raggio è cresciuto tanto da uscire dalla lavagna e si
# starebbe solo scegliendo quale bordo affollare.
_PLACE_TRIES = 24


def _on_board(x: float, y: float) -> tuple[float, float]:
    """Il posto più vicino in cui la scheda ci sta tutta.

    Le coordinate sono il CENTRO della scheda, quindi i bordi vanno tenuti a
    mezza scheda di distanza: schiacciare la y a zero, come si faceva, mette
    metà scheda fuori dal bordo di sopra.
    """
    half = (CARD_SPAN[0] / 2, CARD_SPAN[1] / 2)
    return (min(1 - half[0], max(half[0], x)),
            min(1 - half[1], max(half[1], y)))


def _beside(place: tuple[float, float], seed: int,
            taken: tuple[tuple[float, float], ...] = ()) -> tuple[float, float]:
    """Un posto libero accanto a `place` per un brano appena scelto.

    Non è una disposizione: è un punto di partenza decente, che il DJ
    sposterà. Ma deve essere davvero libero, e scostarsi dalla sorgente non
    basta a garantirlo: la catena torna su sé stessa, e il posto "accanto"
    può essere occupato da un brano scelto tre passi fa. Quindi si prova un
    giro di posti e si prende il primo che non tocca nessuno, allargando il
    raggio a ogni giro completo.

    Il giro è un'ellisse e non un cerchio: deve stare largo quanto la scheda,
    e la scheda non è quadrata.

    L'angolo viene dal numero d'ordine e non dal caso: rifare gli stessi
    passi rifà lo stesso disegno, e una lavagna che si rimescola da sé a ogni
    rerun è inutilizzabile.
    """
    from math import cos, sin

    spot = place
    for attempt in range(_PLACE_TRIES):
        step = seed + attempt
        angle = step * 2.399963  # angolo aureo: riempie senza allineare
        grow = 1.0 + 0.12 * (step % 4) + 0.5 * (attempt // 8)
        spot = _on_board(place[0] + CARD_SPAN[0] * grow * cos(angle),
                         place[1] + CARD_SPAN[1] * grow * sin(angle))
        if all(abs(spot[0] - x) >= CARD_SPAN[0]
               or abs(spot[1] - y) >= CARD_SPAN[1] for x, y in taken):
            return spot
    # Attorno alla sorgente non c'è più posto. Si cerca allora sulla lavagna
    # intera, a griglia, il primo posto che non tocca nessuno: sta lontano da
    # chi l'ha chiamato, ma il collegamento dice comunque da dove viene, e una
    # scheda lontana si vede — una sovrapposta no.
    free = _free_cell(taken)
    if free is not None:
        return free
    # Lavagna piena davvero: si posa comunque, e a rimettere ordine ci pensa
    # `straighten` — meglio una scheda sovrapposta che una scelta rifiutata.
    return spot


def _free_cell(taken) -> tuple[float, float] | None:
    """Il primo posto della griglia che non tocca nessuna scheda, o `None`."""
    first = _on_board(0.0, 0.0)
    last = _on_board(1.0, 1.0)
    cols = int((last[0] - first[0]) / CARD_SPAN[0]) + 1
    rows = int((last[1] - first[1]) / CARD_SPAN[1]) + 1
    for row in range(rows):
        for col in range(cols):
            spot = (first[0] + col * CARD_SPAN[0],
                    first[1] + row * CARD_SPAN[1])
            if all(abs(spot[0] - x) >= CARD_SPAN[0]
                   or abs(spot[1] - y) >= CARD_SPAN[1] for x, y in taken):
                return spot
    return None


def suggestions(cost: TransitionCost, seed: int, taken, k: int = 8,
                pool=None, key_of=None) -> list[tuple[int, float, list[int]]]:
    """La rosa di brani da cui scegliere il prossimo, escluso il già preso.

    È `nearest` con una regola in più: quello che sta già sulla lavagna non
    si ripropone. Va escluso DOPO aver cercato e non prima, o si chiederebbe
    `k` candidati e se ne otterrebbero meno — per questo si pesca largo e si
    taglia dopo.

    Ogni voce è `(brano, costo, copie)`. Senza `key_of` ogni brano fa voce a
    sé e `copie` contiene lui soltanto.

    Con `key_of` — una funzione che dice, dato un brano, quale musica è — le
    copie dello stesso pezzo diventano UNA voce. Servono due cose che una
    senza l'altra non bastano: due copie hanno gli stessi BPM e la stessa
    tonalità, quindi esattamente lo stesso costo, e senza raccoglierle
    occupano la rosa in fila l'una all'altra; e se una è già sulla lavagna
    vanno escluse tutte, o si costruisce un set con lo stesso brano due
    volte.

    Quale delle copie usare non si decide qui. Le copie restano tutte nella
    voce, e si sceglie al momento di prenderla: differiscono per cartella,
    per bitrate, per come è scritto il nome, e scartarle adesso vorrebbe dire
    decidere al posto di chi le ha messe lì.
    """
    taken = {int(t) for t in taken}
    if k <= 0:
        return []
    # Quanto largo: abbastanza da sopravvivere a una lavagna che ha già
    # mangiato i primi vicini, senza chiedere l'intera libreria. Raggruppando
    # serve più largo ancora, perché le copie che si fondono in una voce non
    # ne aprono una nuova.
    reach = (2 * k if key_of else k) + len(taken)
    blocked = {key_of(t) for t in taken} if key_of else set()

    found: list[tuple[int, float, list[int]]] = []
    voices: dict = {}
    for track, value in nearest(cost, seed, reach, pool):
        if track in taken:
            continue
        if key_of is None:
            found.append((track, value, [track]))
            if len(found) == k:
                break
            continue
        name = key_of(track)
        if name in blocked:
            continue
        if name in voices:
            # Una copia in più di una voce già aperta: si accoda anche a rosa
            # piena, perché è un modo di prendere quella voce, non una voce.
            voices[name][2].append(track)
        elif len(found) < k:
            voices[name] = (track, value, [track])
            found.append(voices[name])
    return found
