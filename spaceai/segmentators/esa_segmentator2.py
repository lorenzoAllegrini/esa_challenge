
import os 
import sys 
import pandas as pd
import numpy as np
import ast 
import math
import random 
import itertools
from sktime.transformations.panel.rocket import Rocket
from spaceai.segmentators.shapelet_miner import ShapeletMiner
from spaceai.data.esa import ESA
from scipy.stats import kurtosis, skew

from spaceai.segmentators.cython_functions import compute_spectral_centroid, calculate_slope, spearman_correlation, apply_transformations_to_channel_cython, stft_spectral_std, moving_average_error

from spaceai.segmentators.functions import (
    spectral_energy,
    autoregressive_deviation,
    moving_average_prediction_error,
    stft_spectral_std,
    calculate_slope,
    spearman_correlation,
    mann_kendall_test_tau,
    diff_peaks,
    diff_var
)
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
import more_itertools as mit 

class EsaDatasetSegmentator2:

    available_transformations = {
        "se": spectral_energy,
        "ar": autoregressive_deviation,
        "ma": moving_average_prediction_error,
        "sc": compute_spectral_centroid,
        "stft": stft_spectral_std,
        "slope": calculate_slope,
        "sp_correlation": spearman_correlation,
        "mk_tau": mann_kendall_test_tau,
        "mean" : np.mean,
        "var" : np.var,
        "std" : np.std,
        "kurtosis" : kurtosis,
        "skew" : skew,
        "diff_peaks" : diff_peaks,
        "diff_var" : diff_var,
        "median": np.median,
        "min": np.min,
        "max": np.max
    }
    available_poolings = {
        "max": np.max,
        "min": np.min,
        "mean": np.mean,
        "std": np.std
    }

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
        pooling_config: Optional[Dict[str, Dict[str,str]]] = 
            {
            "event": {"max": "event"},
            "start": {"min": "start"},
            "end": {"max": "end"}
            },
        pooling_segment_len: int = 20,
        pooling_segment_stride: int = 20,
        use_shapelets: bool = True,
        step_difference_feature: bool = True
        
        
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

    def segment(self, esa_channel: ESA, masks: List[Tuple[int,int]], ensemble_id:str, train_phase=False):
        masks = sorted(masks, key=lambda iv: iv[0])
        output_dir = os.path.join(
            self.exp_dir, self.run_id, "channel_segments", esa_channel.channel_id
        )
        os.makedirs(output_dir, exist_ok=True)

        # usa .parquet invece di .csv
        pq_name  = f"{ensemble_id}.parquet"
        pq_path  = os.path.join(output_dir, pq_name)
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
                max_idx  = base_columns.index("max")
                min_idx  = base_columns.index("min")

                # ------------------------------------------------------------
                # 2) calcolo delle differenze per n = 1 … 5
                # ------------------------------------------------------------
                for i, row in enumerate(segments):
                    for n in range(1, 6):
                        if i >= n:
                            diff_mean = row[mean_idx] - segments[i - n][mean_idx]
                            diff_max  = row[max_idx]  - segments[i - n][max_idx]
                            diff_min  = row[min_idx]  - segments[i - n][min_idx]
                        else:
                            diff_mean = diff_max = diff_min = 0.0

                        # aggiungi le tre nuove feature alla fine della riga
                        row.append(diff_mean)   # step difference (mean)
                        row.append(diff_max)    # maximum difference
                        row.append(diff_min)    # minimum difference

                # ------------------------------------------------------------
                # 3) aggiorna base_columns in modo coerente
                # ------------------------------------------------------------
                for n in range(1, 6):
                    base_columns.append(f"step_difference_{n}")       # mean
                    base_columns.append(f"maximum_difference_{n}")    # max
                    base_columns.append(f"minimum_difference_{n}")    # min

            if self.telecommands:
                base_columns += [f"telecommand_{i}" for i in range(1, esa_channel.data.shape[1])]
            if self.use_shapelets:
                for i in range(self.shapelet_miner.num_kernels):
                    base_columns += [
                        f"kernel_{i}_max_convolution",
                        f"kernel_{i}_min_convolution"
                    ]

            if self.poolings:
                segments, pooled_columns = self.pooling_segmentation(segments, base_columns)
            else:
                pooled_columns = base_columns
                
            anomalies = self.get_event_intervals(segments, label=1)
            df = pd.DataFrame(segments, columns=pooled_columns)

            # Salvo sempre in Parquet
            df.to_parquet(pq_path, index=False)
            #print(f"[Segmentator] Saved {len(df)} segments → {pq_path}")

        # Rimuovo i campi interni 'event'
        df = df.drop(columns=df.filter(like="event").columns)
        return df, anomalies


    def pooling_segmentation(
        self,
        segments: List[List[float]],
        columns:  List[str]
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
            if np.min(window[:,0], axis=0) == -1:
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
    
    def get_event_intervals(self, segments: list, label:int) -> list:
        labels = np.array([int(seg[0]) for seg in segments])
        indices = np.where(labels == label)[0]
        if indices.size == 0:
            return []
        groups = [list(group) for group in mit.consecutive_groups(indices)]

        intervals = [[group[0], group[-1]] for group in groups]
        return intervals

    