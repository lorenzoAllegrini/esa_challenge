from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_esa

from spaceai.benchmark.esa_competition import ESACompetitionBenchmark
from spaceai.data.esa import ESAMissions
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2
from spaceai.segmentators.shapelet_miner import ShapeletMiner


def test_competition_workflow(tmp_path, sample_dataset, monkeypatch):
    """Ensure the benchmark.run workflow executes end-to-end."""

    mission = ESAMissions.MISSION_1.value
    mission.target_channels = ["channel_41"]

    shapelet_miner = ShapeletMiner(
        k_min_length=2,
        k_max_length=2,
        num_kernels=1,
        segment_duration=2,
        step_duration=1,
        run_id="wf",
        exp_dir=str(tmp_path / "exp"),
        skip=True,
    )
    segmentator = EsaDatasetSegmentator2(
        transformations=["mean", "max"],
        run_id="wf",
        exp_dir=str(tmp_path / "exp"),
        shapelet_miner=shapelet_miner,
        segment_duration=2,
        step_duration=1,
        telecommands=True,
        poolings=[],
        use_shapelets=False,
        step_difference_feature=False,
    )

    benchmark = ESACompetitionBenchmark(
        run_id="wf",
        exp_dir=str(tmp_path / "exp"),
        data_root=str(sample_dataset),
        segmentator=segmentator,
        seed=0,
    )

    # --- Patch heavy methods with lightweight stubs ---
    def fake_load_channel(self, mission, channel_id, overlapping_train=True):
        return make_esa(sample_dataset, channel_id), make_esa(
            sample_dataset, channel_id
        )

    def fake_channel_ensemble(*args, **kwargs):
        train_df = pd.DataFrame(
            {"start": [0], "end": [1], "group_0": [0.0], "anomaly": [0]}
        )
        test_df = pd.DataFrame({"start": [0], "end": [1], "group_0": [0.0]})
        return train_df, test_df, 0.0

    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.column_stack((np.ones(len(X)), np.zeros(len(X))))

    def fake_event_selection(*args, **kwargs):
        return DummyModel(), 0.0

    def fake_group_activation(self, df, *a, **k):
        return df

    def fake_predict_labels(self, *a, **k):
        return np.zeros(2), np.zeros(2, dtype=int)

    monkeypatch.setattr(ESACompetitionBenchmark, "load_channel", fake_load_channel)
    monkeypatch.setattr(
        ESACompetitionBenchmark, "channel_specific_ensemble", fake_channel_ensemble
    )
    monkeypatch.setattr(
        ESACompetitionBenchmark, "event_wise_model_selection", fake_event_selection
    )
    monkeypatch.setattr(
        ESACompetitionBenchmark, "add_group_activation", fake_group_activation
    )
    monkeypatch.setattr(
        ESACompetitionBenchmark, "predict_challenge_labels", fake_predict_labels
    )
    monkeypatch.setattr(
        ESACompetitionBenchmark, "predict_challenge_labels2", fake_predict_labels
    )
    monkeypatch.setattr(
        ESACompetitionBenchmark, "compute_total_scores", lambda self, m: None
    )

    class DummySearchCV:
        def __init__(self):
            self.best_estimator_ = DummyModel()
            self.best_score_ = 0.0
            self.best_params_ = {}
            self.search_spaces = {}
            self.cv = []

        def fit(self, X, y):
            return self

    labels = benchmark.run(mission, DummySearchCV, DummySearchCV, DummySearchCV)

    assert set(labels.columns) == {"id", "is_anomaly", "pred_binary"}
