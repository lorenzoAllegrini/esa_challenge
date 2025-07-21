# pylint: disable=missing-module-docstring, too-many-lines
import logging
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import (
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch

from spaceai.utils.tools import download_and_extract_zip

from .anomaly_dataset import AnomalyDataset

class AnnotationLabel(
    Enum
):  # pylint: disable=missing-class-docstring, too-few-public-methods
    """Enuemeration of annotation labels for ESA dataset."""

    NOMINAL = 0
    ANOMALY = 1
    RARE_EVENT = 2
    GAP = 3
    INVALID = 4


@dataclass
class ESAMission:  # pylint: disable=too-many-instance-attributes
    """ESA mission dataclass with metadata of a single mission."""

    index: int
    """The index of the mission."""
    url_source: str
    """The URL source of the mission data."""
    dirname: str
    """The directory name of the mission data."""
    train_test_split: pd.Timestamp
    """The split date between training and testing data."""
    start_date: pd.Timestamp
    """The start date of the mission."""
    end_date: pd.Timestamp
    """The end date of the mission."""
    resampling_rule: pd.Timedelta
    """The resampling rule for the data."""
    monotonic_channel_range: tuple[int, int]
    """The range of monotonic channels."""
    parameters: list[str]
    """The list of parameters."""
    telecommands: list[str]
    """The list of telecommands."""
    target_channels: list[str]
    """The list of target channels."""

    @property
    def inner_dirpath(self):  # pylint: disable=missing-function-docstring
        return os.path.join(self.dirname, self.dirname)

    @property
    def all_channels(self):  # pylint: disable=missing-function-docstring
        return self.parameters + self.telecommands


class ESAMissions(Enum):
    """ESA missions enumeration that contains metadata of mission1 and mission2."""

    MISSION_1: ESAMission = ESAMission(
        index=1,
        url_source="https://zenodo.org/records/12528696/files/ESA-Mission1.zip?download=1",
        dirname="ESA-Mission1",
        train_test_split=pd.to_datetime("2007-01-01"),
        start_date=pd.to_datetime("2000-01-01"),
        end_date=pd.to_datetime("2014-01-01"),
        resampling_rule=pd.Timedelta(seconds=30),
        monotonic_channel_range=(4, 11),
        parameters=[f"channel_{i + 1}" for i in range(76)],
        telecommands=[f"telecommand_{i + 1}" for i in range(698)],
        target_channels=[
            f"channel_{i}"
            for i in [*list(range(12, 53)), *list(range(57, 67)), *list(range(70, 77))]
        ],
    )
    MISSION_2: ESAMission = ESAMission(
        index=2,
        url_source="https://zenodo.org/records/12528696/files/ESA-Mission2.zip?download=1",
        dirname="ESA-Mission2",
        train_test_split=pd.to_datetime("2001-10-01"),
        start_date=pd.to_datetime("2000-01-01"),
        end_date=pd.to_datetime("2003-07-01"),
        resampling_rule=pd.Timedelta(seconds=18),
        monotonic_channel_range=(29, 46),
        parameters=[f"channel_{i + 1}" for i in range(100)],
        telecommands=[f"telecommand_{i + 1}" for i in range(123)],
        target_channels=[
            f"channel_{i}"
            for i in [
                *list(range(9, 29)),
                *list(range(58, 60)),
                *list(range(70, 92)),
                *list(range(96, 99)),
            ]
        ],
    )


class ESA(
    AnomalyDataset,
):  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """ESA benchmark dataset for anomaly detection.

    The dataset consists of multivariate time series data collected from ESA's
    spacecrafts telemetry data. The data is used to detect anomalies in the spacecrafts'
    telemetry data and evaluate the performance of anomaly detection algorithms.
    """

    def __init__(
        self,
        root: str,
        mission: ESAMission,
        channel_id: str,
        mode: Literal["prediction", "anomaly", "challenge"],
        overlapping: bool = False,
        seq_length: Optional[int] = 250,
        n_predictions: int = 1,
        train: bool = True,
        download: bool = True,
        challenge_parquet: Optional[str] = None,
        uniform_start_end_date: bool = False,
        drop_last: bool = True,
        use_telecommands: bool = True
    ):  # pylint: disable=useless-parent-delegation, too-many-arguments
        """ESABenchmark class that preprocesses and loads ESA dataset for training and
        testing.

        Args:
            root (str): The root directory of the dataset.
            mission (ESAMission): The mission type of the dataset.
            channel_id (str): The channel ID to be used.
            mode (Literal["prediction", "anomaly", "challenge"]): The mode of the dataset.
            overlapping (bool): The flag that indicates whether the dataset is overlapping.
            seq_length (Optional[int]): The length of the sequence for each sample.
            train (bool): The flag that indicates whether the dataset is for training or testing.
            download (bool): The flag that indicates whether the dataset should be downloaded.
            challenge_parquet (Optional[str]): Optional path to a parquet file used when loading challenge data.
            uniform_start_end_date (bool): The flag that indicates whether the dataset should be resampled to have uniform start and end date.
            drop_last (bool): The flag that indicates whether the last sample should be dropped.
        """
        super().__init__(root)
        if seq_length is None or seq_length < 1:
            raise ValueError(f"Invalid window size: {seq_length}")

        if mode not in ["prediction", "anomaly", "challenge"]:
            raise ValueError(f"Invalid mode {mode}")

        self.root = root
        self.mission = mission
        self.channel_id: str = channel_id
        self._mode: Literal["prediction", "anomaly"] = mode
        self.overlapping: bool = overlapping
        self.window_size: int = seq_length if seq_length else 250
        self.train: bool = train
        self.uniform_start_end_date: bool = uniform_start_end_date
        self.drop_last: bool = drop_last
        self.n_predictions: int = n_predictions
        self.use_telecommands: bool = use_telecommands
        self.challenge_parquet: Optional[str] = challenge_parquet

        if not channel_id in self.mission.all_channels:
            raise ValueError(f"Channel ID {channel_id} is not valid")
        print("1")
        challenge_mode = self._mode == "challenge" and not self.train and self.challenge_parquet is not None
        if download and not challenge_mode:
            self.download()
        print("2")
        if not challenge_mode and not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        if self._mode == "anomaly" and self.overlapping:
            logging.warning(
                f"Channel {channel_id} is in anomaly mode and overlapping is set to True."
                " Anomalies will be repeated in the dataset."
            )

        self.data, self.anomalies, self.communication_gaps = self.load_and_preprocess(channel_id)

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Return the data at the given index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds")
        first_idx = (
            index
            if self.overlapping
            else index * (self.window_size + self.n_predictions - 1)
        )
        last_idx = first_idx + self.window_size
        if last_idx > len(self.data) - self.n_predictions:
            last_idx = len(self.data) - self.n_predictions

        x, y_true = (
            torch.tensor(self.data[first_idx:last_idx]),
            torch.from_numpy(
                np.stack(
                    [
                        self.data[first_idx + i + 1 : last_idx + i + 1, 0]
                        for i in range(self.n_predictions)
                    ]
                )
            ).T,
        )
        return x, y_true

    def __len__(self) -> int:
        if self.overlapping:
            length = self.data.shape[0] - self.window_size - self.n_predictions + 1
            return length
        length = self.data.shape[0] / (self.window_size + self.n_predictions)
        if self.drop_last:
            return math.floor(length)
        return math.ceil(length)

    def download(self):
        """Download the dataset from the given URL and extract it to the given
        directory."""
        if self._check_exists():
            return
        download_and_extract_zip(
            self.mission.url_source,
            os.path.join(self.root, self.mission.dirname),
            cleanup=True,
        )

    def _check_exists(self) -> bool:
        """Check if the dataset exists on the local filesystem."""
        return os.path.exists(os.path.join(self.root, self.mission.dirname))

    def _apply_resampling_rule_(
        self, channel_df: pd.DataFrame, start_date: pd.DataFrame, end_date: pd.DataFrame
    ) -> pd.DataFrame:
        """Resample the dataframe using zero order hold.

        Args:
            channel_df (pd.DataFrame): The dataframe to resample.
            start_date (pd.Timestamp): The start date of the dataframe.
            end_date (pd.Timestamp): The end date of the dataframe.

        Returns:
            pd.DataFrame: The resampled dataframe.
        """
        # Resample using zero order hold
        if self._mode == "challenge":
            end_date = self.mission.end_date
            start_date = self.mission.start_date
        elif self.train:
            if end_date > self.mission.train_test_split:
                end_date = self.mission.train_test_split
        else:
            if start_date < self.mission.train_test_split:
                start_date = self.mission.train_test_split
        first_index_resampled = pd.Timestamp(start_date).floor(
            freq=self.mission.resampling_rule
        )
        last_index_resampled = pd.Timestamp(end_date).ceil(
            freq=self.mission.resampling_rule
        )
        resampled_range = pd.date_range(
            first_index_resampled,
            last_index_resampled,
            freq=self.mission.resampling_rule,
        )
        final_param_df = channel_df.reindex(resampled_range, method="ffill")
        # Initialize the first sample
        final_param_df.iloc[0] = channel_df.iloc[0]
        return final_param_df

    def load_and_preprocess(
        self,
        channel_id: str,
    ) -> pd.DataFrame:
        """Preprocess the channel dataset by loading the raw channel dataset.

        Args:
            channel_id (str): The channel ID to preprocess.

        Returns:
            pd.DataFrame: The preprocessed channel dataset.
        """
        source_folder = os.path.join(self.root, self.mission.inner_dirpath)
        if not self.train and self._mode == "challenge":

            return self.load_challenge_channel(channel_id, self.challenge_parquet)
        # Load and format parameter (channel)
        if channel_id in self.mission.parameters:
            channel_df = pd.read_pickle(
                os.path.join(source_folder, "channels", f"{channel_id}.zip")
            )

        # Load and format telecommand
        if channel_id in self.mission.telecommands:
            channel_df = pd.read_pickle(
                os.path.join(source_folder, "telecommands", f"{channel_id}.zip"),
            )

        channel_df = self._apply_resampling_rule_(
            channel_df,
            channel_df.index[0],
            channel_df.index[-1],
        )

        channel_df = channel_df.ffill().bfill().astype(np.float32)

        if self.use_telecommands:
            telecommands_csv = pd.read_csv(os.path.join(source_folder, "telecommands.csv"))
            prioritized_tcs = telecommands_csv.loc[telecommands_csv["Priority"] >= 3, "Telecommand"].to_numpy().flatten()
            
            telecommand_dfs = []
            
            for tc in prioritized_tcs:
                tc_file = os.path.join(source_folder, "telecommands", f"{tc}.zip")
                if os.path.exists(tc_file):
                    df_tc = pd.read_pickle(tc_file)
                    df_tc_bool = pd.Series(0, index=channel_df.index, name=tc)

                    channel_index = channel_df.index
                    for ts in df_tc.index:
                        pos = channel_index.searchsorted(ts, side="left")
                        if pos < len(channel_index):
                            df_tc_bool.iloc[pos] = 1  
                    telecommand_dfs.append(df_tc_bool.to_frame())
                else:
                    logging.warning(f"Telecommand file {tc_file} not found.")

            if telecommand_dfs:
                telecommands_df = telecommand_dfs[0]
                for df in telecommand_dfs[1:]:
                    telecommands_df = telecommands_df.join(df, how="outer")
                # Assicura che tutti i valori mancanti siano riempiti con 0
                telecommands_df = telecommands_df.fillna(0)
                channel_df = channel_df.join(telecommands_df, how="left")


        map_datetime_index = pd.DataFrame(
            list(range(0, len(channel_df))), index=channel_df.index, columns=["value"]
        )
        min_dt, max_dt = channel_df.index.min(), channel_df.index.max()
        labels_df = pd.read_csv(os.path.join(source_folder, "labels.csv"))

        anomaly_types_df = pd.read_csv(os.path.join(source_folder, "anomaly_types.csv"))
        labels_df = pd.merge(labels_df, anomaly_types_df, how="inner", on="ID")
        
        for dcol in ["StartTime", "EndTime"]:
            labels_df[dcol] = pd.to_datetime(labels_df[dcol]).dt.tz_localize(None)
        labels_df = labels_df.loc[labels_df["Channel"] == channel_id]

        anomalies = []
        communication_gaps = []

        for _, label_row in labels_df.iterrows():
            start_time = label_row["StartTime"].floor(freq=self.mission.resampling_rule)
            end_time = label_row["EndTime"].ceil(freq=self.mission.resampling_rule)
            if end_time < min_dt:
                continue
            if start_time > max_dt:
                continue

            map_datetime_index_range = map_datetime_index[
                np.logical_and(
                    start_time <= map_datetime_index.index,
                    map_datetime_index.index <= end_time,
                )
            ]
            start_idx = map_datetime_index_range.iloc[0]["value"]
            end_idx = map_datetime_index_range.iloc[-1]["value"]
            if label_row["Category"] == "Anomaly" or label_row["Category"] == "Rare Event":
                anomalies.append((start_idx, end_idx))
            elif label_row["Category"] == "Communication Gap":
                communication_gaps.append((start_idx, end_idx))

        if self.uniform_start_end_date:
            channel_df = self._apply_resampling_rule_(
                channel_df,
                self.mission.start_date,
                self.mission.end_date,
            ) 

        channel = channel_df.values.astype(np.float32)
        anomalies = sorted(anomalies, key=lambda x: x[0])
        communication_gaps = sorted(communication_gaps, key=lambda x: x[0])

        return channel, anomalies, communication_gaps 
    
    def load_challenge_channel(self, channel_id: str, parquet_file: Optional[str] = None):

        import pyarrow.parquet as pq
        if parquet_file is None:
            source_folder = os.path.join(self.root, "ESA-Mission1-challenge")
            parquet_file = os.path.join(source_folder, "test.parquet")
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # Seleziona le colonne che iniziano con "telecommand_" e ordinali per numero crescente
        telecommand_cols = [col for col in df.columns if col.startswith("telecommand_")]
        telecommand_cols = sorted(telecommand_cols, key=lambda col: int(col.split("_")[1]))
        
        selected_cols = [channel_id] + telecommand_cols
        channel = df[selected_cols]
        
        return channel.values.astype(np.float32), [], []
        
    @property
    def in_features_size(self) -> str:
        """Return the size of the input features."""
        return self.data.shape[-1]
