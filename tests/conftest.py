import sys
import types

mod = types.ModuleType("spaceai.data.ops_sat")
setattr(mod, "OPSSAT", object())
sys.modules.setdefault("spaceai.data.ops_sat", mod)
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spaceai.data.esa import (
    ESA,
    ESAMissions,
)


@pytest.fixture
def sample_dataset(tmp_path):
    mission = ESAMissions.MISSION_1.value
    base = tmp_path / mission.dirname / mission.dirname
    channels = base / "channels"
    channels.mkdir(parents=True)
    dates = pd.date_range(mission.start_date, periods=5, freq=mission.resampling_rule)
    df = pd.DataFrame({"channel_1": np.arange(5)}, index=dates)
    df.to_pickle(channels / "channel_1.zip")
    labels = pd.DataFrame(
        {
            "ID": [1],
            "Channel": ["channel_1"],
            "StartTime": [dates[1]],
            "EndTime": [dates[2]],
        }
    )
    labels.to_csv(base / "labels.csv", index=False)
    anomaly_types = pd.DataFrame({"ID": [1], "Category": ["Anomaly"]})
    anomaly_types.to_csv(base / "anomaly_types.csv", index=False)
    return tmp_path


def make_esa(root):
    mission = ESAMissions.MISSION_1.value
    return ESA(
        root=str(root),
        mission=mission,
        channel_id="channel_1",
        mode="anomaly",
        overlapping=False,
        seq_length=2,
        n_predictions=1,
        train=True,
        download=False,
        use_telecommands=False,
    )
