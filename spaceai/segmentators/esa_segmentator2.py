import ast
import itertools
import math
import os
import random
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import more_itertools as mit
import numpy as np
import pandas as pd
from scipy.stats import (
    kurtosis,
    skew,
)
from sktime.transformations.panel.rocket import Rocket

from spaceai.data.esa import ESA
from spaceai.segmentators.cython_functions import (
    apply_transformations_to_channel_cython,
    calculate_slope,
    compute_spectral_centroid,
    moving_average_error,
    spearman_correlation,
    stft_spectral_std,
    _apply_kernels2,
)
from spaceai.segmentators.functions import (
    autoregressive_deviation,
    calculate_slope,
    diff_peaks,
    diff_var,
    mann_kendall_test_tau,
    moving_average_prediction_error,
    spearman_correlation,
    spectral_energy,
    stft_spectral_std,
)
from spaceai.segmentators.shapelet_miner import ShapeletMiner


class EsaDatasetSegmentator2:
    """Segment ESA channels applying statistical transforms and shapelets."""

    available_transformations = {
        "se": spectral_energy,
        "ar": autoregressive_deviation,
        "ma": moving_average_prediction_error,
        "sc": compute_spectral_centroid,
        "stft": stft_spectral_std,
        "slope": calculate_slope,
        "sp_correlation": spearman_correlation,
        "mk_tau": mann_kendall_test_tau,
        "mean": np.mean,
        "var": np.var,
        "std": np.std,
        "kurtosis": kurtosis,
        "skew": skew,
        "diff_peaks": diff_peaks,
        "diff_var": diff_var,
        "median": np.median,
        "min": np.min,
        "max": np.max,
    }
    available_poolings = {"max": np.max, "min": np.min, "mean": np.mean, "std": np.std}

    def __init__(
        self,
        transformations: List[str],
        run_id: str = "esa_segments",
        exp_dir: str = "experiments",
        shapelet_miner: Optional[ShapeletMiner] = None,
        segment_duration: int = 50,
        step_duration: int = 10,
        save_csv: bool = True,
        telecommands: bool = False,
        poolings: str = ["max", "min"],
        pooling_config: Optional[Dict[str, Dict[str, str]]] = {
            "event": {"max": "event"},
            "start": {"min": "start"},
            "end": {"max": "end"},
        },
        pooling_segment_len: int = 20,
        pooling_segment_stride: int = 20,
        use_shapelets: bool = True,
        step_difference_feature: bool = True,
    ) -> None:

        for transformation in transformations:
            if transformation not in self.available_transformations:
                raise RuntimeError(f"Transformation {transformation} not available")
        self.transformations = transformations
        self.run_id = run_id
        self.exp_dir = exp_dir
        self.segment_duration = segment_duration
        self.step_duration = step_duration
        self.save_csv = save_csv
        self.telecommands = telecommands
        self.poolings = poolings
        for pooling in poolings:
            if pooling not in self.available_poolings:
                raise RuntimeError(f"Pooling {pooling} not available")
        self.pooling_segment_len = pooling_segment_len
        self.pooling_segment_stride = pooling_segment_stride
        self.combos = None
        self.pooling_config = pooling_config or {}
        self.shapelet_miner = shapelet_miner
        self.use_shapelets = use_shapelets
        self.step_difference_feature = step_difference_feature

    def segment(
        self,
        esa_channel: ESA,
        masks: List[Tuple[int, int]],
        ensemble_id: str,
        train_phase: bool = False,
    ):
        """Generate feature segments from a channel.

        Parameters
        ----------
        esa_channel : ESA
            Channel to segment.
        masks : list[Tuple[int, int]]
            Intervals to exclude from the nominal pool when computing features.
        ensemble_id : str
            Name used for caching the produced DataFrame on disk.
        train_phase : bool, optional
            When ``True`` additional shapelet features are computed.

        Returns
        -------
        Tuple[pd.DataFrame, list[list[int]]]
            DataFrame of computed features and list of anomaly intervals.
        """
        masks = sorted(masks, key=lambda iv: iv[0])
        output_dir = os.path.join(
            self.exp_dir, self.run_id, "channel_segments", esa_channel.channel_id
        )
        os.makedirs(output_dir, exist_ok=True)

        # usa .parquet invece di .csv
        pq_name = f"{ensemble_id}.parquet"
        pq_path = os.path.join(output_dir, pq_name)
        # se il file Parquet esiste già, caricalo e torna i segmenti
        if os.path.exists(pq_path):
            df = pd.read_parquet(pq_path)

            segments = df.values.tolist()
            anomalies = self.get_event_intervals(segments, label=1)

        else:
            segments: List[List[float]] = apply_transformations_to_channel_cython(
                self,
                esa_channel.data,
                np.array(esa_channel.anomalies),
                np.array(masks),
                train=train_phase,
            )
            base_columns = ["event", "start", "end"] + self.transformations.copy()

            if self.step_difference_feature:
                mean_idx = base_columns.index("mean")
                max_idx = base_columns.index("max")
                min_idx = base_columns.index("min")

                # ------------------------------------------------------------
                # 2) calcolo delle differenze per n = 1 … 5
                # ------------------------------------------------------------
                for i, row in enumerate(segments):
                    for n in range(1, 6):
                        if i >= n:
                            diff_mean = row[mean_idx] - segments[i - n][mean_idx]
                            diff_max = row[max_idx] - segments[i - n][max_idx]
                            diff_min = row[min_idx] - segments[i - n][min_idx]
                        else:
                            diff_mean = diff_max = diff_min = 0.0

                        # aggiungi le tre nuove feature alla fine della riga
                        row.append(diff_mean)  # step difference (mean)
                        row.append(diff_max)  # maximum difference
                        row.append(diff_min)  # minimum difference

                # ------------------------------------------------------------
                # 3) aggiorna base_columns in modo coerente
                # ------------------------------------------------------------
                for n in range(1, 6):
                    base_columns.append(f"step_difference_{n}")  # mean
                    base_columns.append(f"maximum_difference_{n}")  # max
                    base_columns.append(f"minimum_difference_{n}")  # min

            if self.telecommands:
                base_columns += [
                    f"telecommand_{i}" for i in range(1, esa_channel.data.shape[1])
                ]
            if self.use_shapelets:
                for i in range(self.shapelet_miner.num_kernels):
                    base_columns += [
                        f"kernel_{i}_max_convolution",
                        f"kernel_{i}_min_convolution",
                    ]

            if self.poolings:
                segments, pooled_columns = self.pooling_segmentation(
                    segments, base_columns
                )
            else:
                pooled_columns = base_columns

            anomalies = self.get_event_intervals(segments, label=1)
            df = pd.DataFrame(segments, columns=pooled_columns)

            # Salvo sempre in Parquet
            df.to_parquet(pq_path, index=False)
            # print(f"[Segmentator] Saved {len(df)} segments → {pq_path}")

        # Rimuovo i campi interni 'event'
        df = df.drop(columns=df.filter(like="event").columns)
        return df, anomalies

    # ------------------------------------------------------------------
    #  New two-step preprocessing helpers
    # ------------------------------------------------------------------
    def segment_statistical(
        self,
        esa_channel: ESA,
        masks: List[Tuple[int, int]],
        ensemble_id: str,
        train_phase: bool = False,
    ):
        """Compute only statistical features for a channel."""
        masks = sorted(masks, key=lambda iv: iv[0])
        output_dir = os.path.join(
            self.exp_dir, self.run_id, "channel_segments", esa_channel.channel_id
        )
        os.makedirs(output_dir, exist_ok=True)

        pq_path = os.path.join(output_dir, f"{ensemble_id}.parquet")
        if os.path.exists(pq_path):
            df = pd.read_parquet(pq_path)
            segments = df.values.tolist()
            anomalies = self.get_event_intervals(segments, label=1)
        else:
            orig_shapelets = self.use_shapelets
            self.use_shapelets = False
            segments: List[List[float]] = apply_transformations_to_channel_cython(
                self,
                esa_channel.data,
                np.array(esa_channel.anomalies),
                np.array(masks),
                train=train_phase,
            )
           
            self.use_shapelets = orig_shapelets

            base_columns = ["event", "start", "end"] + self.transformations.copy()

            if self.step_difference_feature:
                mean_idx = base_columns.index("mean")
                max_idx = base_columns.index("max")
                min_idx = base_columns.index("min")
                for i, row in enumerate(segments):
                    for n in range(1, 6):
                        if i >= n:
                            diff_mean = row[mean_idx] - segments[i - n][mean_idx]
                            diff_max = row[max_idx] - segments[i - n][max_idx]
                            diff_min = row[min_idx] - segments[i - n][min_idx]
                        else:
                            diff_mean = diff_max = diff_min = 0.0
                        row.append(diff_mean)
                        row.append(diff_max)
                        row.append(diff_min)
                for n in range(1, 6):
                    base_columns.append(f"step_difference_{n}")
                    base_columns.append(f"maximum_difference_{n}")
                    base_columns.append(f"minimum_difference_{n}")

            if self.telecommands:
                base_columns += [
                    f"telecommand_{i}" for i in range(1, esa_channel.data.shape[1])
                ]


            anomalies = self.get_event_intervals(segments, label=1)
            df = pd.DataFrame(segments, columns=base_columns)
            df.to_parquet(pq_path, index=False)

        df = df.drop(columns=df.filter(like="event").columns)
        return df, anomalies

    def add_shapelet_features(
        self,
        df: pd.DataFrame,
        esa_channel: ESA,
        mask: Tuple[int, int],
        ensemble_id: str,
        initialize=True
    ) -> pd.DataFrame:
        """Append shapelet responses to ``df`` for the given ``mask``."""
        if df.empty:
            return df
        if initialize:
            self.shapelet_miner.initialize_kernels(
                esa_channel, mask=mask, ensemble_id=ensemble_id
            )

        windows = []
        for _, row in df.iterrows():
            start, end = int(row["start"]), int(row["end"])
            windows.append(esa_channel.data[start:end, 0].astype(np.float32))
        X = np.stack(windows)[:, np.newaxis, :]
        raw_features = _apply_kernels2(
            X, self.shapelet_miner.process_kernels(self.shapelet_miner.kernels)
        )

        df = df.reset_index(drop=True).copy()
        for i in range(self.shapelet_miner.num_kernels):
            df[f"kernel_{i}_max_convolution"] = raw_features[:, 2 * i]
            df[f"kernel_{i}_min_convolution"] = raw_features[:, 2 * i + 1]
        
        return df

    def pooling_segmentation(
        self, segments: List[List[float]], columns: List[str]
    ) -> Tuple[List[List[float]], List[str]]:
        """
        Applica rolling‐window pooling per ciascuna feature:
        - usa `self.pooling_config[feat]` se presente,
        - altrimenti usa tutti i self.poolings con nome "pooling_feat".
        Ritorna (dati_poolati, nomi_colonne_poolate).
        """
        data = np.array(segments)

        N, F = data.shape
        w, s = self.pooling_segment_len, self.pooling_segment_stride

        new_columns = []
        for feat in columns:
            if feat in self.pooling_config:
                for pooling, out_name in self.pooling_config[feat].items():
                    new_columns.append(out_name)
            else:
                for pooling in self.poolings:
                    new_columns.append(f"{pooling}_{feat}")

        new_segments = []
        for start in range(0, N - w + 1, s):
            window = data[start : start + w, :]  # (w, F)
            row = []
            if np.min(window[:, 0], axis=0) == -1:
                continue
            for j, feat in enumerate(columns):
                if feat in self.pooling_config:
                    for pooling, out_name in self.pooling_config[feat].items():
                        func = self.available_poolings[pooling]
                        row.append(func(window[:, j], axis=0))
                else:
                    for pooling in self.poolings:
                        func = self.available_poolings[pooling]
                        row.append(func(window[:, j], axis=0))
            new_segments.append(row)

        return new_segments, new_columns

    def get_event_intervals(self, segments: list, label: int) -> list:
        labels = np.array([int(seg[0]) for seg in segments])
        indices = np.where(labels == label)[0]
        if indices.size == 0:
            return []
        groups = [list(group) for group in mit.consecutive_groups(indices)]

        intervals = [[group[0], group[-1]] for group in groups]
        return intervals

    def segment_shapelets(
        self,
        df: pd.DataFrame,
        esa_channel: ESA,
        shapelet_mask: Tuple[int, int],
        ensemble_id: str,
        masks: Optional[List[Tuple[int, int]]] = None,
        mode: str = "exclude",
        initialize=False,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, List[List[int]]]:
        """Return a masked dataset enriched with shapelet features.

        Parameters
        ----------
        df : pd.DataFrame
            Pre-computed statistical features.
        labels : np.ndarray
            Binary labels for each row in ``df`` used to rebuild anomaly
            intervals after masking.
        esa_channel : ESA
            Channel object providing the raw data for shapelet responses.
        mask : Tuple[int, int]
            Interval used to mine shapelets.
        ensemble_id : str
            Identifier to cache shapelet kernels on disk.
        masks : list[Tuple[int, int]], optional
            Intervals to exclude or include depending on ``mode``.
        mode : {"exclude", "include"}
            How to interpret ``masks``. ``"exclude"`` removes all rows
            overlapping any mask while ``"include"`` keeps only the rows
            fully contained in the single provided mask.

        Returns
        -------
        Tuple[pd.DataFrame, list[list[int]]]
            The selected DataFrame extended with shapelet responses and the
            corresponding anomaly intervals.
        """

        mask_bool = np.ones(len(df), dtype=bool)
        if masks:
            if mode == "exclude":
                for ms in masks:
                    mask_bool &= ~((df["start"] < ms[1]) & (df["end"] > ms[0]))
            elif mode == "include":
                ms = masks[0]
                mask_bool &= (df["start"] >= ms[0]) & (df["end"] <= ms[1])
            else:
                raise ValueError("mode must be 'exclude' or 'include'")

        sub_df = df[mask_bool].reset_index(drop=True)
        if labels is not None:
            sub_labels = labels[mask_bool]
            segs = [[int(l)] for l in sub_labels]
            anoms = self.get_event_intervals(segs, 1)
        else: 
            anoms = None
        
        sub_df = self.add_shapelet_features(
            sub_df, esa_channel, mask=shapelet_mask, ensemble_id=ensemble_id, initialize=initialize
        )
        base_columns = sub_df.columns
        segments = sub_df.values.tolist()

        if self.poolings:
            segments, pooled_columns = self.pooling_segmentation(
                segments, base_columns
            )
        else:
            pooled_columns = base_columns

        sub_df = pd.DataFrame(segments, columns=pooled_columns)
        return sub_df, anoms
