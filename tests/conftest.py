import sys
import types

mod = types.ModuleType("spaceai.data.ops_sat")
setattr(mod, "OPSSAT", object())
sys.modules.setdefault("spaceai.data.ops_sat", mod)
bench_mod = types.ModuleType("spaceai.benchmark.ops_sat")
setattr(bench_mod, "OPSSATBenchmark", object)
sys.modules.setdefault("spaceai.benchmark.ops_sat", bench_mod)

# Provide a minimal torch substitute for tests
torch_mod = types.ModuleType("torch")
utils_mod = types.ModuleType("torch.utils")
data_mod = types.ModuleType("torch.utils.data")
setattr(data_mod, "Dataset", object)
setattr(data_mod, "DataLoader", object)
setattr(data_mod, "Subset", object)
setattr(utils_mod, "data", data_mod)
setattr(torch_mod, "utils", utils_mod)
setattr(torch_mod, "tensor", lambda x: x)
setattr(torch_mod, "Tensor", object)
sys.modules.setdefault("torch", torch_mod)
sys.modules.setdefault("torch.utils", utils_mod)
sys.modules.setdefault("torch.utils.data", data_mod)

# Stub heavy optional dependencies used only during imports
rocket_mod = types.ModuleType("sktime.transformations.panel.rocket")
setattr(rocket_mod, "Rocket", object)
sys.modules.setdefault("sktime", types.ModuleType("sktime"))
transforms_mod = types.ModuleType("sktime.transformations")
sys.modules.setdefault("sktime.transformations", transforms_mod)
sys.modules.setdefault(
    "sktime.transformations.base", types.ModuleType("sktime.transformations.base")
)
setattr(sys.modules["sktime.transformations.base"], "BaseTransformer", object)
sys.modules.setdefault(
    "sktime.transformations.panel", types.ModuleType("sktime.transformations.panel")
)
sys.modules.setdefault("sktime.transformations.panel.rocket", rocket_mod)

numba_mod = types.ModuleType("numba")
setattr(numba_mod, "prange", range)
sys.modules.setdefault("numba", numba_mod)

tslearn_mod = types.ModuleType("tslearn")
cluster_mod = types.ModuleType("tslearn.clustering")
setattr(cluster_mod, "TimeSeriesKMeans", object)
setattr(tslearn_mod, "clustering", cluster_mod)
sys.modules.setdefault("tslearn", tslearn_mod)
sys.modules.setdefault("tslearn.clustering", cluster_mod)

# skopt is used for Bayesian optimization; a minimal stub suffices
skopt_space_mod = types.ModuleType("skopt.space")
setattr(skopt_space_mod, "Dimension", object)
sys.modules.setdefault("skopt", types.ModuleType("skopt"))
sys.modules.setdefault("skopt.space", skopt_space_mod)

# simple placeholder for pymannkendall used in feature functions
pymk_mod = types.ModuleType("pymannkendall")
sys.modules.setdefault("pymannkendall", pymk_mod)

# Minimal pyarrow parquet stub to avoid optional dependency
pyarrow_parquet_mod = types.ModuleType("pyarrow.parquet")


def _fake_read_table(path):
    import pandas as pd

    dates = pd.date_range("2000-01-01", periods=20, freq="30S")
    return pd.DataFrame(
        {"channel_12": range(20), "channel_41": range(20), "telecommand_1": 0},
        index=dates,
    )


setattr(
    pyarrow_parquet_mod,
    "read_table",
    lambda path: types.SimpleNamespace(to_pandas=lambda: _fake_read_table(path)),
)
pyarrow_mod = types.ModuleType("pyarrow")
setattr(pyarrow_mod, "__version__", "0.0")
sys.modules.setdefault("pyarrow", pyarrow_mod)
sys.modules.setdefault("pyarrow.parquet", pyarrow_parquet_mod)
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Stub out the compiled cython extension with simple numpy fallbacks
cython_mod = types.ModuleType("spaceai.segmentators.cython_functions")


def _fake_apply(self, data, anomalies, masks, train=False):
    arr = np.asarray(data)
    segments = []
    for i in range(len(arr) - 1):
        segments.append([0, i, i + 1, arr[i, 0], arr[i, 1]])
    return segments


setattr(cython_mod, "apply_transformations_to_channel_cython", _fake_apply)
setattr(cython_mod, "calculate_slope", lambda arr: np.gradient(arr))
setattr(cython_mod, "compute_spectral_centroid", lambda arr, *a, **k: 0.0)
setattr(cython_mod, "moving_average_error", lambda a, b: 0.0)
setattr(cython_mod, "spearman_correlation", lambda a, b: 0.0)
setattr(cython_mod, "stft_spectral_std", lambda arr, *a, **k: 0.0)
setattr(
    cython_mod,
    "_apply_kernels2",
    lambda data, kernels: np.zeros((data.shape[0], len(kernels) * 2)),
)
sys.modules.setdefault("spaceai.segmentators.cython_functions", cython_mod)

from spaceai.data.esa import (
    ESA,
    ESAMissions,
)


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a tiny ESA-like dataset with two channels and one telecommand."""

    mission = ESAMissions.MISSION_1.value
    base = tmp_path / mission.dirname / mission.dirname
    channels = base / "channels"
    telecommands_dir = base / "telecommands"
    channels.mkdir(parents=True)
    telecommands_dir.mkdir(parents=True)

    # 20 timesteps spaced according to mission sampling rule
    dates = pd.date_range(mission.start_date, periods=20, freq=mission.resampling_rule)

    ch12 = pd.DataFrame({"channel_12": np.arange(20)}, index=dates)
    ch41 = pd.DataFrame({"channel_41": np.arange(20) * 2}, index=dates)
    ch12.to_pickle(channels / "channel_12.zip")
    ch41.to_pickle(channels / "channel_41.zip")

    # telecommand executed twice
    tc_df = pd.DataFrame({"value": [1, 1]}, index=[dates[5], dates[15]])
    tc_df.to_pickle(telecommands_dir / "telecommand_1.zip")

    labels = pd.DataFrame(
        {
            "ID": [1],
            "Channel": ["channel_12"],
            "StartTime": [dates[8]],
            "EndTime": [dates[10]],
        }
    )
    labels.to_csv(base / "labels.csv", index=False)
    anomaly_types = pd.DataFrame({"ID": [1], "Category": ["Anomaly"]})
    anomaly_types.to_csv(base / "anomaly_types.csv", index=False)

    telecommands_csv = pd.DataFrame({"Telecommand": ["telecommand_1"], "Priority": [3]})
    telecommands_csv.to_csv(base / "telecommands.csv", index=False)

    channels_csv = pd.DataFrame(
        {"Channel": ["channel_12", "channel_41"], "Group": [0, 0]}
    )
    channels_csv.to_csv(base / "channels.csv", index=False)

    return tmp_path


def make_esa(root: Path, channel_id: str = "channel_12") -> ESA:
    """Return an ESA object pointing to the temporary dataset."""

    mission = ESAMissions.MISSION_1.value
    return ESA(
        root=str(root),
        mission=mission,
        channel_id=channel_id,
        mode="anomaly",
        overlapping=False,
        seq_length=2,
        n_predictions=1,
        train=True,
        download=False,
        use_telecommands=True,
    )
