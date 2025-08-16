__all__ = ["extract_kernels_from_data", "RocketExtracted2"]
import multiprocessing
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from numba import prange
from sklearn.cluster import KMeans
from sktime.transformations.base import BaseTransformer
from sktime.transformations.panel.rocket import Rocket


def extract_kernels_from_data(
    X_train,
    window_size=50,
    n_kernels=100,
    method="per_segment",
    n_neighbors=20,
    random=False,
):
    X_train = np.squeeze(X_train)

    if X_train.ndim != 1:
        X_train = X_train[0]

    if len(X_train) < window_size:
        return None, None
    else:
        if random:
            max_start_idx = len(X_train) - window_size
            if max_start_idx < 1:
                raise ValueError(
                    "La serie temporale è troppo corta per estrarre finestre."
                )

            indices = np.random.choice(
                np.arange(max_start_idx), size=100_000, replace=False
            )
            segments = np.array(
                [X_train[idx : idx + window_size] for idx in indices], dtype=np.float64
            )
        else:
            segments = np.array(
                [
                    X_train[i : i + window_size]
                    for i in range(len(X_train) - window_size + 1)
                ],
                dtype=np.float64,
            )

    means = segments.mean(axis=1, keepdims=True)
    stds = segments.std(axis=1, keepdims=True)
    stds[stds < 1e-8] = 1e-8
    segments_norm = (segments - means) / stds

    if method == "kmeans":
        if len(segments_norm) < n_kernels:
            n_kernels = len(segments_norm)
        # Clustering K-Means sui segmenti filtrati
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_kernels, random_state=42, n_init=10)
        kmeans.fit(segments_norm)
        kernels = kmeans.cluster_centers_  # shape: (n_kernels, window_size)
    elif method == "per_segment":
        kernels = segments_norm
        n_kernels = kernels.shape[0]
    else:
        raise ValueError(f"Metodo sconosciuto: {method}")

    return kernels


def extract_segments_from_data(data, step_duration=1, segment_duration=100):
    if len(data) < segment_duration:
        return None
    segments = []
    index = 0
    while index + segment_duration < len(data):
        segments.append(data[index : index + segment_duration])
        index += step_duration
    return segments


class ShapeletMiner:
    """
    Riscrivere la classe RocketExtracted, a questa nell'inizializzazione viene passato una lista di pool di anomalie,
    queste però vengono passate subito a una funzione interna che per ciascun pool di kernel calcola il suo score rispetto agli altri pool nel seguente modo:
    lo score di un kernel è dato dalla differenza tra il massimo valore di ppv ottenuto convolvendo il kernel su tutti i segmenti nominali (da cui dovrebbero essere escluse le rare anomalies)
    e la media di tutti i massimi valori ppv ottenuti dalla convoluzione del kernel sui pool.
    I kernel finali saranno scelti in base allo score e al ranking all'interno del pool a cui appartengono.
    """

    def __init__(
        self,
        anomaly_pools,
        nominal_segments,
        num_kernels=10,
        k_min_length=30,
        k_length=40,
        segment_duration=50,
        step_duration=1,
        n_jobs=-1,
    ):

        self.nominal_segments = nominal_segments
        self.k_length = k_length
        self.num_kernels = num_kernels
        self.segment_duration = segment_duration
        self.step_duration = step_duration
        self.random = random
        self.n_jobs = n_jobs
        self.k_min_length = k_min_length
        all_kernels = []
        all_scores = []

        if not anomaly_pools:
            self.kernels = []
            for _ in range(self.num_kernels):
                # 1) scegli una lunghezza casuale
                L = random.randint(self.k_min_length, self.k_length)
                # 2) genera un vettore Dirichlet simmetrico di dimensione L
                #    np.ones(L) = parametri α_i = 1 per ogni componente
                k = np.random.default_rng().dirichlet(np.ones(L)).astype(np.float32)
                self.kernels.append(k)
            return

        for i, pool in enumerate(anomaly_pools):
            other_pools = anomaly_pools[:i] + anomaly_pools[i + 1 :]
            kernels, scores = self.score_anomaly_kernels(pool, other_pools)

            all_kernels.extend(kernels)
            all_scores.extend(scores)

        self.kernels = self.select_best_kernels_by_clustering(
            all_kernels, all_scores, num_kernels
        )

    def _fit(self, X, y=None):
        _, self.n_columns, n_timepoints = X.shape
        self._is_fitted = True

        return self

    def _transform(self, X, y=None):
        import multiprocessing

        from numba import (
            get_num_threads,
            set_num_threads,
        )
        from sktime.transformations.panel.rocket._rocket_numba import _apply_kernels

        from spaceai.segmentators.cython_functions import _apply_kernels2

        """
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )"""
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)

        raw_features = _apply_kernels2(
            X.astype(np.float32), self.process_kernels(self.kernels)
        )
        set_num_threads(prev_threads)
        t = pd.DataFrame(raw_features)

        return t

    def select_best_kernels_by_clustering(self, kernels, scores, num_kernels):
        if not scores:
            return []

        # Calcola il 75° percentile degli score
        p75 = np.percentile(np.array(scores), 75)

        # Raggruppa i kernel per lunghezza (calcolata sul vettore flatten)
        groups = defaultdict(list)
        groups_scores = defaultdict(list)
        for k, score in zip(kernels, scores):
            flat = np.array(k).flatten()  # vettorizzazione del kernel
            length = flat.shape[0]
            # Includi solo se lo score è maggiore di 0 e supera il 75° percentile
            if score > 0 and score > p75:
                groups[length].append(flat)
                groups_scores[length].append(score)

        # Se nessun kernel sopravvive al filtro, usa TUTTI i kernel (anche con score negativi)
        if len(groups) == 0:
            # Raggruppa tutti i kernel per lunghezza (senza filtro)
            groups = defaultdict(list)
            groups_scores = defaultdict(list)
            for k, score in zip(kernels, scores):
                flat = np.array(k).flatten()
                length = flat.shape[0]
                groups[length].append(flat)
                groups_scores[length].append(score)

            # Fai un ordinamento semplice basato sui punteggi:
            # crea una lista di tuple (flattened_kernel, score), ordinale e prendi i primi num_kernels.
            all_kernels = [
                (np.array(k).flatten(), score) for k, score in zip(kernels, scores)
            ]
            all_kernels_sorted = sorted(all_kernels, key=lambda x: x[1], reverse=True)
            best_kernels = [item[0] for item in all_kernels_sorted]

            # Se ce ne sono più del necessario, troncali; altrimenti, replicali per raggiungere num_kernels
            if len(best_kernels) >= num_kernels:
                best_kernels = best_kernels[:num_kernels]
            else:
                extra_needed = num_kernels - len(best_kernels)
                best_kernels = best_kernels + best_kernels[:extra_needed]
            return best_kernels

        # Calcola la media degli score per ogni gruppo
        group_avg_scores = {}
        for length, score_list in groups_scores.items():
            group_avg_scores[length] = np.mean(score_list)

        # Lista di lunghezze (ogni lunghezza rappresenta un gruppo)
        unique_lengths = list(groups.keys())
        num_groups = len(unique_lengths)
        unique_lengths.sort(key=lambda length: group_avg_scores[length], reverse=True)

        # Se il numero di gruppi supera num_kernels, seleziona solo i gruppi migliori (con media score più alta)
        if num_groups > num_kernels:
            unique_lengths = unique_lengths[:num_kernels]

        # Inizializza l'allocazione: almeno 1 centroide per gruppo
        allocations = {length: 1 for length in unique_lengths}
        total_alloc = sum(allocations.values())

        # Ordina i gruppi in ordine decrescente in base alla media degli score
        sorted_lengths = sorted(
            unique_lengths, key=lambda l: group_avg_scores[l], reverse=True
        )

        # Distribuisci in maniera iterativa i cluster aggiuntivi
        while total_alloc < num_kernels:
            allocated_this_round = False
            for l in sorted_lengths:
                # Controlla se il gruppo può avere un ulteriore centroide (se non ha già tanti cluster quanti sample)
                if allocations[l] < len(groups[l]):
                    allocations[l] += 1
                    total_alloc += 1
                    allocated_this_round = True
                    if total_alloc >= num_kernels:
                        break
            if not allocated_this_round:
                break

        # Esegui clustering per ciascun gruppo con il numero di cluster assegnato,
        # selezionando per ciascun cluster il kernel con score massimo.
        centroids = []
        for l in allocations:
            kernels_list = groups[
                l
            ]  # lista dei kernel (flattened) del gruppo di lunghezza l
            group_scores = groups_scores[l]  # lista degli score corrispondenti
            k_clusters = allocations[l]
            data = np.vstack(kernels_list)

            if len(kernels_list) == k_clusters:
                # Se il numero di sample equivale al numero di cluster, usa i sample stessi
                centroids.extend(kernels_list)
            elif len(kernels_list) > k_clusters:
                # Esegui clustering su data
                kmeans = KMeans(
                    n_clusters=k_clusters, random_state=0, n_init=10
                ).fit(data)
                labels = kmeans.labels_
                # Per ciascun cluster, seleziona il kernel con score massimo
                for cluster_id in range(k_clusters):
                    # Trova gli indici dei sample appartenenti a questo cluster
                    indices = [
                        i for i, label in enumerate(labels) if label == cluster_id
                    ]
                    if indices:
                        # Seleziona l'indice con il punteggio massimo
                        best_index = max(indices, key=lambda i: group_scores[i])
                        centroids.append(kernels_list[best_index])
                    else:
                        # In caso non ci siano sample assegnati (caso raro)
                        pass
            else:
                # Caso in cui per un gruppo il numero di sample sia inferiore a quanto previsto
                centroids.extend(kernels_list)

        # Assicurati che ogni kernel sia flatten
        centroids = [np.array(kernel).flatten() for kernel in centroids]

        # Assicurati che il numero totale di centroidi sia esattamente num_kernels:
        if len(centroids) > num_kernels:
            centroids = centroids[:num_kernels]
        elif len(centroids) < num_kernels:
            extra_needed = num_kernels - len(centroids)
            centroids = centroids + centroids[:extra_needed]

        return centroids

    def score_anomaly_kernels(self, kernels_pool, test_pools):
        raw_kernels = []
        for k in range(self.k_min_length, self.k_length + 1):
            kernels = self.extract_kernels_from_data(kernels_pool, k)
            raw_kernels.extend(kernels)
        seen = set()
        unique_kernels = []
        for k in raw_kernels:
            k_tuple = tuple(np.array(k).flatten())
            if k_tuple not in seen:
                seen.add(k_tuple)
                unique_kernels.append(k)

        kernels = unique_kernels
        total_scores = np.zeros(len(kernels), dtype=np.float32)
        for test_pool in test_pools:
            segments = self.extract_segments_from_data(test_pool)
            scores = self.compute_kernel_scores(pool=segments, kernels=kernels)
            total_scores = np.maximum(total_scores, scores)
        nominal_scores = self.compute_kernel_scores(
            pool=self.nominal_segments, kernels=kernels
        )
        total_scores -= nominal_scores

        return kernels, total_scores

    def compute_kernel_scores(self, pool, kernels):
        from spaceai.segmentators.cython_functions import _apply_kernels2

        pool_array = np.array(pool, dtype=np.float32)
        if pool_array.ndim == 2:
            pool_array = pool_array[:, np.newaxis, :]
        elif pool_array.ndim != 3:
            raise ValueError(
                f"Formato del pool non valido: shape {pool_array.shape}. Atteso (n, c, l)."
            )
        raw_features = _apply_kernels2(pool_array, self.process_kernels(kernels))

        score1 = pd.DataFrame(raw_features).iloc[:, ::2].to_numpy()
        score2 = pd.DataFrame(raw_features).iloc[:, 1::2].to_numpy()
        score = np.maximum(score1, np.abs(score2))
        return score.max(axis=0)

    def process_kernels(self, kernels):
        weights_list = []
        lengths = []
        for i in range(len(kernels)):
            kernel = kernels[i]
            weights_list.append(kernel.astype(np.float32))
            lengths.append(len(kernel))
        weights = np.concatenate(weights_list)

        biases = np.zeros(len(kernels), dtype=np.float32)
        dilations = np.ones(len(kernels), dtype=np.int32)
        paddings = np.zeros(len(kernels), dtype=np.int32)
        num_channel_indices = np.ones(len(kernels), dtype=np.int32)
        channel_indices = np.zeros(len(kernels), dtype=np.int32)

        return (
            weights,
            lengths,
            biases,
            dilations,
            paddings,
            num_channel_indices,
            channel_indices,
        )

    def extract_kernels_from_data(self, X_train, length, method="per_segment"):
        X_train = np.squeeze(X_train)
        if X_train.ndim != 1:
            X_train = X_train[0]
        if len(X_train) < length:
            return None, None
        else:
            if self.random:
                max_start_idx = len(X_train) - length
                if max_start_idx < 1:
                    raise ValueError(
                        "La serie temporale è troppo corta per estrarre finestre."
                    )

                requested = 100_000
                # numero di posizioni possibili
                max_start_idx = len(X_train) - length

                # se la popolazione è più piccola del richiesto, prendila tutta
                num_samples = min(requested, max_start_idx)

                # ora posso chiedere un campione senza replacement
                indices = np.random.choice(
                    np.arange(max_start_idx), size=num_samples, replace=False
                )
                segments = np.array(
                    [X_train[idx : idx + length] for idx in indices], dtype=np.float64
                )
            else:
                segments = np.array(
                    [X_train[i : i + length] for i in range(len(X_train) - length + 1)],
                    dtype=np.float64,
                )

        means = segments.mean(axis=1, keepdims=True)
        stds = segments.std(axis=1, keepdims=True)
        stds[stds < 1e-8] = 1e-8
        segments_norm = (segments - means) / stds

        if method == "kmeans":
            if len(segments_norm) < n_kernels:
                n_kernels = len(segments_norm)
            # Clustering K-Means sui segmenti filtrati
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_kernels, random_state=42, n_init=10)
            kmeans.fit(segments_norm)
            kernels = kmeans.cluster_centers_  # shape: (n_kernels, window_size)
        elif method == "per_segment":
            kernels = segments_norm
            n_kernels = kernels.shape[0]
        else:
            raise ValueError(f"Metodo sconosciuto: {method}")

        return kernels

    def extract_segments_from_data(self, data):
        if len(data) < self.segment_duration:
            return None
        segments = []
        index = 0
        while index + self.segment_duration <= len(data):
            segments.append(data[index : index + self.segment_duration])
            index += self.step_duration
        return segments
