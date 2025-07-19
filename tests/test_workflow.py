import glob
import os
import sys
import types
from datetime import datetime, timedelta
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
cython_stub = types.ModuleType("spaceai.segmentators.cython_functions")
def _stub(*args, **kwargs):
    return 0
cython_stub.apply_transformations_to_channel_cython = _stub
cython_stub.calculate_slope = _stub
cython_stub.compute_spectral_centroid = _stub
cython_stub.moving_average_error = _stub
cython_stub.spearman_correlation = _stub
cython_stub.stft_spectral_std = _stub
sys.modules.setdefault("spaceai.segmentators.cython_functions", cython_stub)

import joblib
import numpy as np
import pandas as pd

# Provide compatibility with scikit-optimize on NumPy>=2
np.int = int  # type: ignore[attr-defined]

# Provide a minimal torch stub for dependencies
torch = types.ModuleType("torch")
utils = types.ModuleType("torch.utils")
data_mod = types.ModuleType("torch.utils.data")

class Dataset:
    pass

def from_numpy(arr):
    return arr

torch.Tensor = np.ndarray  # type: ignore[attr-defined]
torch.from_numpy = from_numpy

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self.dataset[i]

    def __len__(self):
        return len(self.dataset)

class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

data_mod.Dataset = Dataset
data_mod.DataLoader = DataLoader
data_mod.Subset = Subset
utils.data = data_mod

torch.utils = utils
sys.modules.setdefault("torch", torch)
sys.modules.setdefault("torch.utils", utils)
sys.modules.setdefault("torch.utils.data", data_mod)

import spaceai.data.esa as esa_module
esa_module.torch = torch

from sklearn.linear_model import LogisticRegression

from spaceai.benchmark.esa_competition_predictor import ESACompetitionPredictor
from spaceai.benchmark.esa_competition_training import (
    ESACompetitionTraining,
    SegmentedModel,
)
from spaceai.data.esa import ESAMission


class DummyShapeletMiner:
    def initialize_kernels(self, *args, **kwargs):
        pass


class DummySegmentator:
    def __init__(self):
        self.shapelet_miner = DummyShapeletMiner()

    def segment_statistical(self, esa_channel, masks, ensemble_id, train_phase=False):
        n = len(esa_channel.data)
        df = pd.DataFrame({
            "start": np.arange(n),
            "end": np.arange(n),
            "val": esa_channel.data[:, 0],
        })
        return df, esa_channel.anomalies

    def add_shapelet_features(self, df, esa_channel, mask, ensemble_id):
        df = df.copy()
        df["shape"] = 0.0
        return df

    def get_event_intervals(self, segments, label):
        labels = [int(s[0]) for s in segments]
        intervals = []
        start = None
        for idx, val in enumerate(labels):
            if val == label:
                if start is None:
                    start = idx
            else:
                if start is not None:
                    intervals.append([start, idx - 1])
                    start = None
        if start is not None:
            intervals.append([start, len(labels) - 1])
        return intervals


def create_dataset(tmpdir, channel_id="channel_12"):
    root = os.path.join(tmpdir, "datasets")
    mission_dir = os.path.join(root, "ESA-Mission1", "ESA-Mission1")
    os.makedirs(os.path.join(mission_dir, "channels"), exist_ok=True)

    channels = [channel_id]
    if channel_id != "channel_12":
        channels.append("channel_12")
    pd.DataFrame({"Channel": channels, "Group": ["G1"] * len(channels)}).to_csv(
        os.path.join(mission_dir, "channels.csv"), index=False
    )
    pd.DataFrame({"Telecommand": ["telecommand_1"], "Priority": [1]}).to_csv(
        os.path.join(mission_dir, "telecommands.csv"), index=False
    )
    start = datetime(2020, 1, 1)
    idx = pd.date_range(start, periods=30, freq="S")
    for ch in channels:
        channel_df = pd.DataFrame({ch: np.random.rand(30)}, index=idx)
        channel_df.to_pickle(os.path.join(mission_dir, "channels", f"{ch}.zip"))
    pd.DataFrame(
        {
            "ID": [1],
            "Channel": [channel_id],
            "StartTime": [start],
            "EndTime": [start + timedelta(seconds=5)],
        }
    ).to_csv(os.path.join(mission_dir, "labels.csv"), index=False)
    pd.DataFrame({"ID": [1], "Category": ["Anomaly"]}).to_csv(
        os.path.join(mission_dir, "anomaly_types.csv"), index=False
    )
    chal_dir = os.path.join(root, "ESA-Mission1-challenge")
    os.makedirs(chal_dir, exist_ok=True)
    chal_cols = {ch: np.random.rand(20) for ch in channels}
    chal_cols["telecommand_1"] = np.zeros(20)
    chal_df = pd.DataFrame(chal_cols)
    chal_df.to_parquet(os.path.join(chal_dir, "test.parquet"))

    mission = ESAMission(
        index=1,
        url_source="",
        dirname="ESA-Mission1",
        train_test_split=pd.to_datetime(start + timedelta(seconds=15)),
        start_date=pd.to_datetime(start),
        end_date=pd.to_datetime(start + timedelta(seconds=40)),
        resampling_rule=pd.Timedelta(seconds=1),
        monotonic_channel_range=(0, 0),
        parameters=channels,
        telecommands=["telecommand_1"],
        target_channels=[channel_id],
    )
    return root, mission


def test_full_workflow(tmp_path):
    data_root, mission = create_dataset(tmp_path)
    segmentator = DummySegmentator()
    benchmark = ESACompetitionTraining(
        run_id="run1",
        exp_dir=str(tmp_path / "exp"),
        segmentator=segmentator,
        data_root=data_root,
    )

    def dummy_run(self, *_args, **_kwargs):
        model_dir = os.path.join(
            self.exp_dir, self.run_id, "models", "channel_channel_12"
        )
        os.makedirs(model_dir, exist_ok=True)
        ch_model = SegmentedModel(
            LogisticRegression().fit([[0, 0, 0], [1, 1, 1]], [0, 1]),
            segmentator,
            ensemble_id="e0",
        )
        joblib.dump(ch_model, os.path.join(model_dir, "internal_0.pkl"))
        meta_model = SegmentedModel(
            LogisticRegression().fit([[0], [1]], [0, 1]),
            segmentator,
            ensemble_id="e1",
        )
        joblib.dump(meta_model, os.path.join(model_dir, "external_0.pkl"))
        links = {"external_0": {"mask_id": "m0", "internal_ids": ["internal_0"]}}
        with open(os.path.join(model_dir, "links.json"), "w") as f:
            json.dump(links, f)
        ev_dir = os.path.join(self.exp_dir, self.run_id, "models")
        os.makedirs(ev_dir, exist_ok=True)
        joblib.dump(
            LogisticRegression().fit([[0], [1]], [0, 1]),
            os.path.join(ev_dir, "event_wise_0.pkl"),
        )

    benchmark.run = types.MethodType(dummy_run, benchmark)
    benchmark.run(mission, None, None, None)

    artifacts_dir = os.path.join(str(tmp_path / "exp"), "run1")
    predictor = ESACompetitionPredictor(artifacts_dir, segmentator)
    predictor.load_models()
    assert "external_0" in predictor.meta_models
    assert "internal_0" in predictor.internal_models


def test_training_run(tmp_path):
    data_root, mission = create_dataset(tmp_path, channel_id="channel_41")
    segmentator = DummySegmentator()
    benchmark = ESACompetitionTraining(
        run_id="run_train",
        exp_dir=str(tmp_path / "exp"),
        segmentator=segmentator,
        data_root=data_root,
    )

    import spaceai.benchmark.esa_competition_training as training_mod
    orig_cv = training_mod.cross_validate

    def single_cv(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return orig_cv(*args, **kwargs)

    training_mod.cross_validate = single_cv

    from skopt import BayesSearchCV
    from sklearn.model_selection import TimeSeriesSplit

    def factory():
        return BayesSearchCV(
            estimator=LogisticRegression(solver="liblinear"),
            search_spaces={"C": [1]},
            cv=TimeSeriesSplit(n_splits=2),
            n_iter=1,
        )

    benchmark.run(
        mission=mission,
        search_cv_factory=factory,
        search_cv_factory2=factory,
        search_cv_factory3=factory,
        perc_eval1=0.2,
        perc_eval2=0.1,
        perc_shapelet=0.1,
        external_estimators=1,
        internal_estimators=1,
        final_estimators=2,
    )

    artifacts = os.path.join(tmp_path, "exp", "run_train", "models")
    channel_dir = os.path.join(artifacts, "channel_channel_41")
    assert glob.glob(os.path.join(channel_dir, "internal_*.pkl"))
    assert glob.glob(os.path.join(channel_dir, "external_*.pkl"))
    assert os.path.exists(os.path.join(channel_dir, "links.json"))
    assert glob.glob(os.path.join(artifacts, "event_wise_*.pkl"))

