import random

import numpy as np
from conftest import make_esa

from spaceai.segmentators.shapelet_miner import ShapeletMiner


def test_shapelet_miner_deterministic(sample_dataset, monkeypatch):
    esa = make_esa(sample_dataset, "channel_12")

    orig_rng = np.random.default_rng

    def fixed_rng(seed=None):
        return orig_rng(0)

    monkeypatch.setattr(np.random, "default_rng", fixed_rng)
    random.seed(0)

    def fake_score(self, *a, **k):
        return [np.array([1.0, 0.0])], np.array([1.0])

    monkeypatch.setattr(ShapeletMiner, "score_anomaly_kernels", fake_score)
    monkeypatch.setattr(
        ShapeletMiner,
        "select_best_kernels",
        lambda self, kernels, scores, num_kernels=None: kernels,
    )

    miner1 = ShapeletMiner(
        k_min_length=2,
        k_max_length=2,
        num_kernels=1,
        segment_duration=2,
        step_duration=1,
        run_id="t",
        exp_dir=str(sample_dataset / "exp"),
        skip=False,
    )
    miner1.initialize_kernels(esa, (0, len(esa.data)), "e1")
    kernels1 = [k.tolist() for k in miner1.kernels]

    monkeypatch.setattr(np.random, "default_rng", fixed_rng)
    random.seed(0)
    miner2 = ShapeletMiner(
        k_min_length=2,
        k_max_length=2,
        num_kernels=1,
        segment_duration=2,
        step_duration=1,
        run_id="t",
        exp_dir=str(sample_dataset / "exp2"),
        skip=False,
    )
    miner2.initialize_kernels(esa, (0, len(esa.data)), "e2")
    kernels2 = [k.tolist() for k in miner2.kernels]

    assert kernels1 == kernels2
