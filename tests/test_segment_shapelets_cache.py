import os
import sys
import types
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub cython functions used by the segmentator
cython_stub = types.ModuleType("spaceai.segmentators.cython_functions")

def _stub(*args, **kwargs):
    return 0

cython_stub.apply_transformations_to_channel_cython = _stub
cython_stub.calculate_slope = _stub
cython_stub.compute_spectral_centroid = _stub
cython_stub.moving_average_error = _stub
cython_stub.spearman_correlation = _stub
cython_stub.stft_spectral_std = _stub

def _kern_stub(X, kernels):
    num_kernels = len(kernels[1])
    return np.zeros((X.shape[0], num_kernels * 2), dtype=np.float32)

cython_stub._apply_kernels2 = _kern_stub
sys.modules.setdefault("spaceai.segmentators.cython_functions", cython_stub)

# Minimal torch stub to satisfy optional imports
torch = types.ModuleType("torch")
utils = types.ModuleType("torch.utils")
data_mod = types.ModuleType("torch.utils.data")
torch.Tensor = np.ndarray  # type: ignore[attr-defined]

def from_numpy(arr):
    return arr

torch.from_numpy = from_numpy

class Dataset:
    pass

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

from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2
from spaceai.segmentators.shapelet_miner import ShapeletMiner


class DummyESA:
    def __init__(self):
        self.channel_id = "ch_1"
        self.data = np.random.rand(60, 1).astype(np.float32)
        self.anomalies = [(20, 30)]


def test_segment_shapelets_saves_parquet(tmp_path):
    miner = ShapeletMiner(
        k_min_length=3,
        k_max_length=5,
        num_kernels=1,
        segment_duration=10,
        step_duration=2,
        run_id="test",
        exp_dir=str(tmp_path),
        skip=True,
    )

    segmentator = EsaDatasetSegmentator2(
        transformations=["mean", "max", "min"],
        run_id="test",
        exp_dir=str(tmp_path),
        shapelet_miner=miner,
        use_shapelets=True,
        poolings=[],
        step_difference_feature=False,
    )

    df = pd.DataFrame(
        {
            "event": [0, 0, 1, 0],
            "start": [0, 10, 20, 30],
            "end": [10, 20, 30, 40],
            "mean": [0.0, 0.0, 0.0, 0.0],
            "max": [0.0, 0.0, 0.0, 0.0],
            "min": [0.0, 0.0, 0.0, 0.0],
        }
    )

    esa = DummyESA()
    ensemble_id = "ens1"
    out_file = (
        tmp_path
        / "test"
        / "channel_segments"
        / esa.channel_id
        / f"{ensemble_id}_shapelets.parquet"
    )

    res_df, anoms = segmentator.segment_shapelets(
        df=df,
        esa_channel=esa,
        shapelet_mask=(0, len(esa.data)),
        ensemble_id=ensemble_id,
        initialize=True,
    )
    assert out_file.exists()

    # Reset kernels to ensure they are reloaded from disk
    segmentator.shapelet_miner.kernels = None
    res_df2, anoms2 = segmentator.segment_shapelets(
        df=df,
        esa_channel=esa,
        shapelet_mask=(0, len(esa.data)),
        ensemble_id=ensemble_id,
        initialize=True,
    )

    pd.testing.assert_frame_equal(res_df, res_df2)
    assert anoms == anoms2
