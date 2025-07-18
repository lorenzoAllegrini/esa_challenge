import json
import multiprocessing
import os
import random
from typing import (
    List,
    Tuple,
)

import numpy as np
import pandas as pd
from numba import prange
from sklearn.cluster import KMeans
from sktime.transformations.base import BaseTransformer
from sktime.transformations.panel.rocket import Rocket
from tslearn.clustering import TimeSeriesKMeans

from spaceai.data import ESA


class ShapeletMiner:

    def __init__(
        self,
        k_min_length: int,
        k_max_length: int,
        num_kernels: int,
        segment_duration: int,
        step_duration: int,
        n_jobs=-1,
        run_id: str = "esa_segments",
        exp_dir: str = "experiments",
        skip=True,
    ):
        self.run_id = run_id
        self.exp_dir = exp_dir
        self.k_min_length = k_min_length
        self.k_max_length = k_max_length
        self.num_kernels = num_kernels
        self.segment_duration = segment_duration
        self.step_duration = step_duration
        self.n_jobs = n_jobs
        self.kernels = None
        self.scores = None
        self.skip = skip

    def _random_kernels(self) -> list[np.ndarray]:
        """Return a list of random kernels used as fallback."""
        kernels: list[np.ndarray] = []
        rng = np.random.default_rng()
        for _ in range(self.num_kernels):
            L = rng.integers(self.k_min_length, self.k_max_length + 1)
            kernels.append(rng.dirichlet(np.ones(L)).astype(np.float32))
        return kernels

    def initialize_kernels(
        self, esa_channel: ESA, mask: Tuple[int, int], ensemble_id: str
    ):
        """Extract and store shapelet kernels from the given channel.

        Parameters
        ----------
        esa_channel : ESA
            Channel object containing the raw signal and anomaly intervals.
        mask : Tuple[int, int]
            Time range used to sample nominal segments when mining shapelets.
        ensemble_id : str
            Identifier of the current ensemble. Used to cache discovered
            shapelets on disk.

        Notes
        -----
        The mined kernels are saved in ``experiments/<run_id>/channel_segments``
        under ``shapelets.json`` and loaded back if available. When ``skip`` is
        True, the method only loads existing kernels without performing the
        expensive mining step.
        """
        channel_dir = os.path.join(
            self.exp_dir, self.run_id, "channel_segments", esa_channel.channel_id
        )
        os.makedirs(channel_dir, exist_ok=True)
        shapelets_path = os.path.join(channel_dir, "shapelets.json")

        # 1) Proviamo a caricare l'intero JSON (se esiste e non vuoto)
        all_payload = {}
        if os.path.exists(shapelets_path) and os.path.getsize(shapelets_path) > 0:
            try:
                with open(shapelets_path, "r") as f:
                    all_payload = json.load(f)
            except json.JSONDecodeError:
                pass

        # 2) Se la entry per questo ensemble_id esiste già, la carichiamo e usciamo
        if ensemble_id in all_payload:
            entry = all_payload[ensemble_id]
            self.kernels = [np.array(s, dtype=np.float32) for s in entry["shapelets"]]
            return

        if self.skip:
            self.kernels = self._random_kernels()
            return
        anomaly_pools, padded_intervals, nominal_segments = [], [], []

        data = esa_channel.data
        anomalies = esa_channel.anomalies
        for idx, (s, e) in enumerate(anomalies):
            if s < mask[0]:
                continue
            if e > mask[1]:
                break
            anom_start = anomalies[idx][0]
            anom_end = anomalies[idx][1]
            anom_len = anom_end - anom_start

            if anom_len < self.segment_duration:
                pad_right = (self.segment_duration - anom_len) // 2
                pad_left = (self.segment_duration - anom_len) - pad_right
                start = max(0, anom_start - pad_left)
                end = min(len(data), anom_end + pad_right)
            else:
                start = anom_start
                end = anom_start + self.segment_duration

            pool = data[start:end, 0]
            anomaly_pools.append(pool)
            padded_intervals.append(
                (s - 2 * self.segment_duration, e + 2 * self.segment_duration)
            )

        padded_intervals.sort()
        merged = []
        for s, e in padded_intervals:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)

        i = mask[0]
        while i + self.segment_duration < mask[1]:
            if not any(
                max(i, a) < min(i + self.segment_duration, b) for a, b in merged
            ):
                nominal_segments.append(data[i : i + self.segment_duration, 0])
                i += self.segment_duration
            else:
                j = next(
                    b
                    for a, b in merged
                    if max(i, a) < min(i + self.segment_duration, b)
                )
                i = j + 1

        if len(nominal_segments) > 1_000:
            nominal_segments = random.sample(nominal_segments, 1_000)

        if not anomaly_pools:
            self.kernels = []
            for _ in range(self.num_kernels):
                L = random.randint(self.k_min_length, self.k_max_length)
                k = np.random.default_rng().dirichlet(np.ones(L)).astype(np.float32)
                self.kernels.append(k)
            return

        all_kernels, all_scores = [], []

        for i, pool in enumerate(anomaly_pools):
            other_pools = anomaly_pools[:i] + anomaly_pools[i + 1 :]
            kernels, scores = self.score_anomaly_kernels(
                pool, other_pools, nominal_segments
            )

            all_kernels.extend(kernels)
            all_scores.extend(scores)

        self.kernels = self.select_best_kernels(all_kernels, all_scores)
        new_kernels = []
        rng = np.random.default_rng()
        for k in self.kernels:
            # k può essere list o np.ndarray; convertiamolo in array
            arr = np.array(k, dtype=np.float32).flatten()
            if np.all(arr == 0):
                # nessun peso: fallback a Dirichlet
                L = rng.integers(self.k_min_length, self.k_max_length + 1)
                fallback = rng.dirichlet(np.ones(L)).astype(np.float32)
                new_kernels.append(fallback)
            else:
                new_kernels.append(arr)
        self.kernels = new_kernels

        all_payload[ensemble_id] = {
            "mask": [int(mask[0]), int(mask[1])],
            "shapelets": [k.tolist() for k in self.kernels],
            # "scores": all_scores  # se vuoi salvare anche gli scores
        }
        with open(shapelets_path, "w") as f:
            json.dump(all_payload, f)

    def score_anomaly_kernels(self, kernels_pool, test_pools, nominal_segments):
        """Score candidate kernels extracted from anomaly pools.

        Parameters
        ----------
        kernels_pool : array-like
            1-D array of anomalous data used to mine raw candidate kernels.
        test_pools : list[array-like]
            Additional anomalous pools used to evaluate kernel response.
        nominal_segments : list[array-like]
            Segments of nominal data against which kernels are penalised.

        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            The unique candidate kernels and their maximum scores across the
            provided pools, adjusted by nominal segments.
        """
        raw_kernels = []
        for k in range(self.k_min_length, self.k_max_length + 1):
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
            pool=nominal_segments, kernels=kernels
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

    def select_best_kernels(
        self,
        kernels: list[np.ndarray],
        scores: np.ndarray,
        num_kernels: int | None = None,
    ) -> list[np.ndarray]:
        """Return the top scoring kernels while avoiding duplicates.

        Parameters
        ----------
        kernels : list[np.ndarray]
            Candidate kernels to filter.
        scores : np.ndarray
            Score associated with each kernel.
        num_kernels : int, optional
            Desired number of kernels (defaults to ``self.num_kernels``).

        Returns
        -------
        list[np.ndarray]
            A list containing at most ``num_kernels`` kernels sorted by score and
            with a minimum diversity in their lengths.
        """
        if num_kernels is None:
            num_kernels = self.num_kernels

        if not kernels:
            return []

        # 1) filtro al di sopra del 75° percentile
        p75 = np.percentile(scores, 75)
        filtered = [(k, s) for k, s in zip(kernels, scores) if s >= p75 and s > 0]

        # 2) se sono pochi, uso tutti
        if len(filtered) < num_kernels:
            filtered = list(zip(kernels, scores))

        # 3) ordino per score decrescente
        filtered.sort(key=lambda x: x[1], reverse=True)

        # 4) greedy pick: almeno un kernel per lunghezza
        chosen: List[np.ndarray] = []
        used_lengths = set()
        for k, _ in filtered:
            flat = k.flatten()
            L = flat.size
            if L not in used_lengths or len(chosen) < num_kernels * 0.5:
                chosen.append(flat)
                used_lengths.add(L)
            if len(chosen) >= num_kernels:
                break

        # 5) se mancano kernel, prendo gli “extras” senza duplicati
        if len(chosen) < num_kernels:
            # trasformo chosen in un set di tuple per membership test
            chosen_keys = {tuple(arr.tolist()) for arr in chosen}
            extras: List[np.ndarray] = []
            for k, _ in filtered:
                key = tuple(k.flatten().tolist())
                if key not in chosen_keys:
                    extras.append(np.array(key, dtype=np.float32))
                    chosen_keys.add(key)
                if len(chosen) + len(extras) >= num_kernels:
                    break
            chosen.extend(extras)

        # 6) ritorno esattamente num_kernels
        return [arr for arr in chosen[:num_kernels]]

    def compute_pool(self, data: np.ndarray, anomaly_indices: list, anomaly_index: int):
        anom_start = anomaly_indices[anomaly_index][0]
        anom_end = anomaly_indices[anomaly_index][1]
        anom_len = anom_end - anom_start

        if anom_len < self.segment_duration:
            pad_right = (self.segment_duration - anom_len) // 2
            pad_left = (self.segment_duration - anom_len) - pad_right
            start = max(0, anom_start - pad_left)
            end = min(len(data), anom_end + pad_right)
        else:
            start = anom_start
            end = anom_start + self.segment_duration

        return data[start:end, 0], start, end

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
            return None
        else:
            segments = np.array(
                [X_train[i : i + length] for i in range(len(X_train) - length + 1)],
                dtype=np.float64,
            )

        means = segments.mean(axis=1, keepdims=True)
        stds = segments.std(axis=1, keepdims=True)
        stds[stds < 1e-8] = 1e-8
        segments_norm = (segments - means) / stds

        kernels = segments_norm
        n_kernels = kernels.shape[0]

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
