import sys
import types
import numpy as np

# Provide a minimal torch stub to satisfy optional imports
torch = types.ModuleType("torch")
utils = types.ModuleType("torch.utils")
data_mod = types.ModuleType("torch.utils.data")
torch.Tensor = np.ndarray  # type: ignore[attr-defined]

class Dataset:
    pass

def from_numpy(arr):
    return arr

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

cython_stub = types.ModuleType("spaceai.segmentators.cython_functions")

def _stub(*_args, **_kwargs):
    return 0

cython_stub.apply_transformations_to_channel_cython = _stub
cython_stub.calculate_slope = _stub
cython_stub.compute_spectral_centroid = _stub
cython_stub.moving_average_error = _stub
cython_stub.spearman_correlation = _stub
cython_stub.stft_spectral_std = _stub

def _kern_stub(*_args, **_kwargs):
    X = _args[0]
    kernels = _args[1]
    num_kernels = len(kernels[1])
    return np.zeros((X.shape[0], num_kernels * 2), dtype=np.float32)

cython_stub._apply_kernels2 = _kern_stub
sys.modules.setdefault("spaceai.segmentators.cython_functions", cython_stub)

from spaceai.segmentators.shapelet_miner import ShapeletMiner


class DummyESA:
    def __init__(self, n_samples=200):
        self.channel_id = "ch_1"
        self.data = np.random.rand(n_samples, 1).astype(np.float32)
        self.anomalies = [(50, 60)]


def test_initialize_kernels_skip(tmp_path):
    esa = DummyESA()
    miner = ShapeletMiner(
        k_min_length=3,
        k_max_length=5,
        num_kernels=2,
        segment_duration=10,
        step_duration=2,
        run_id="test",
        exp_dir=str(tmp_path),
        skip=True,
    )
    miner.initialize_kernels(esa, (0, len(esa.data)), ensemble_id="e0")
    assert miner.kernels is not None
    assert len(miner.kernels) == miner.num_kernels

    X = esa.data[:20].reshape(1, 1, -1)
    df = miner._transform(X)
    assert df.shape == (1, miner.num_kernels * 2)


def test_initialize_kernels_save_and_load(tmp_path):
    esa = DummyESA()
    miner = ShapeletMiner(
        k_min_length=3,
        k_max_length=5,
        num_kernels=2,
        segment_duration=10,
        step_duration=2,
        run_id="test",
        exp_dir=str(tmp_path),
        skip=False,
    )
    miner.initialize_kernels(esa, (0, len(esa.data)), ensemble_id="e0")
    first = [k.copy() for k in miner.kernels]

    miner2 = ShapeletMiner(
        k_min_length=3,
        k_max_length=5,
        num_kernels=2,
        segment_duration=10,
        step_duration=2,
        run_id="test",
        exp_dir=str(tmp_path),
        skip=True,
    )
    miner2.initialize_kernels(esa, (0, len(esa.data)), ensemble_id="e0")

    assert miner2.kernels is not None
    for k1, k2 in zip(first, miner2.kernels):
        assert np.allclose(k1, k2)
