import os 
import sys 
import pandas as pd
import numpy as np
import ast 
import math
import random 
import itertools
from sktime.transformations.panel.rocket import Rocket
from scipy.stats import kurtosis, skew

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

import os 
import sys 
import pandas as pd
import numpy as np
import ast 
import math
import random 
import itertools
from sktime.transformations.panel.rocket import Rocket
from spaceai.segmentators.rocket_transformer2 import RocketExtracted2
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

class EsaDatasetSegmentator:

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
    }

    def __init__(
        self,
        transformations: List[str],
        segment_duration: int = 100,
        step_duration: int = 1,
        save_csv: bool = False,
        telecommands: bool = True,
        extract_features: bool = True,
        run_id: str = "esa_segments",
        exp_dir: str = "experiments",
    ) -> None:
        
        for transformation in transformations:
            if transformation not in self.available_transformations:
                raise RuntimeError(f"Transformation {transformation} not available")
        self.transformations = transformations
        self.segment_duration = segment_duration
        self.step_duration = step_duration
        self.save_csv = save_csv
        self.telecommands = telecommands
        self.run_id = run_id
        self.exp_dir = exp_dir
        self.extract_features = extract_features


    def segment(self, esa_channel: ESA) :

        output_dir = os.path.join(self.exp_dir, self.run_id, "channel_segments")
        os.makedirs(output_dir, exist_ok=True)
        train_file_name = f"{esa_channel.channel_id}_segments_train_.csv"
        test_file_name = f"{esa_channel.channel_id}_segments_test_.csv"
        output_file = train_file_name if esa_channel.train else test_file_name
        csv_path = os.path.join(output_dir, output_file)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            segments = df.values.tolist()
            anomalies = self.get_event_intervals(segments=segments, label=1)
            rare_events = self.get_event_intervals(segments=segments, label=2)

        else:
            segments = self.create_segments_from_channel(
                self, esa_channel.data, np.array(esa_channel.anomalies), np.array(esa_channel.rare_events)
                )
            anomalies = self.get_event_intervals(segments=segments, label=1)
            rare_events = self.get_event_intervals(segments=segments, label=2)

            if self.save_csv:
                base_columns = self.transformations.copy()
                if self.telecommands:
                    base_columns.extend([f"telecommand_{i}" for i in range(1, esa_channel.data.shape[1])])
                all_columns = ["event"] + base_columns
                if self.poolings:
                    columns = []
                    for pooling in self.poolings:
                        for col in all_columns:
                            columns.append(f"{pooling}_{col}")
                else:
                    columns = all_columns

                df = pd.DataFrame(segments, columns=columns)
                df = self.zero_out_partial_combos_df(df)
                df.to_csv(csv_path, index=False)
        df = df.drop(columns=df.filter(like="event").columns)

        return df, anomalies, rare_events

        
    def create_segments_from_channel(
        self,
        data: np.ndarray
    ) -> List[List[float]]:
        segments = []             
        index = 0
        while index + self.segment_duration < len(data):     
            values = data[index:(index + self.segment_duration), 0]
            if self.extract_features:
                segment = [
                    self.available_transformations[transformation](values)
                    for transformation in self.transformations
                ]
                if self.telecommands:
                    for telecommand_idx in range(1, data.shape[1]):
                        segment.append(float(np.sum(data[index:(index + self.segment_duration), telecommand_idx])))
            else:
                segment = values
            segments.append(segment)
            index += self.step_duration
        return segments

        
    def get_event_intervals(self, segments: list, label:int) -> list:
        labels = np.array([int(seg[0]) for seg in segments])
        indices = np.where(labels == label)[0]
        if indices.size == 0:
            return []
        groups = [list(group) for group in mit.consecutive_groups(indices)]

        intervals = [[group[0], group[-1]] for group in groups]
        return intervals