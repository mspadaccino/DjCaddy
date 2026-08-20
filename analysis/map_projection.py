"""Da 1280 dimensioni a due: la proiezione che rende la libreria una mappa.

UMAP e non t-SNE perché conserva insieme le due cose che servono qui: i
brani quasi identici finiscono uno sull'altro (struttura locale) E i
generi lontani restano lontani (struttura globale). Con t-SNE i gruppi
sono buoni ma la distanza fra un gruppo e l'altro non vuol dire niente, e
su una mappa che si attraversa disegnando una linea quella distanza è
esattamente ciò che si sta usando.

Prima di UMAP si passa da una PCA a 64 dimensioni. Non cambia la mappa —
la PCA tiene la quasi totalità della varianza di questi embedding — ma
toglie a UMAP il 95% del lavoro di ricerca dei vicini, che è quasi tutto
il suo tempo su decine di migliaia di brani.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PCA_DIM = 64


@dataclass
class ProjectionSettings:
    # Quanti vicini definiscono il "locale". Basso = tanti gruppetti
    # riconoscibili, alto = un continente unico con transizioni morbide. 15 è
    # il valore in cui i micro-generi si vedono ma la mappa resta percorribile.
    n_neighbors: int = 15
    # Quanto stretti si possono impacchettare i punti. Basso = grumi densi e
    # separati, che su una mappa da attraversare col mouse sono più facili da
    # prendere di una nuvola uniforme.
    min_dist: float = 0.1
    # Coseno e non euclidea: negli embedding di una rete conta la DIREZIONE
    # del vettore, non la sua lunghezza (che segue il volume del brano).
    metric: str = "cosine"
    seed: int = 42


def available() -> bool:
    try:
        import umap  # noqa: F401
    except Exception:
        return False
    return True


def project(embeddings, settings: ProjectionSettings | None = None) -> np.ndarray:
    """Le coordinate (N, 2) dei brani sulla mappa."""
    settings = settings or ProjectionSettings()
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(matrix) < 3:
        # Con due punti non c'è varietà da preservare: si mettono in fila e
        # si va avanti, invece di far fallire UMAP su un caso di prova.
        return np.column_stack([np.arange(len(matrix), dtype=np.float32),
                                np.zeros(len(matrix), dtype=np.float32)])

    import umap
    from sklearn.decomposition import PCA

    reduced = matrix
    if matrix.shape[1] > PCA_DIM:
        reduced = PCA(n_components=min(PCA_DIM, len(matrix) - 1),
                      random_state=settings.seed).fit_transform(matrix)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(settings.n_neighbors, len(matrix) - 1),
        min_dist=settings.min_dist,
        metric=settings.metric,
        random_state=settings.seed,
    )
    return reducer.fit_transform(reduced).astype(np.float32)
