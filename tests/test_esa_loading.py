from conftest import make_esa

from spaceai.data.esa import ESAMissions


def test_load_channel(sample_dataset):
    esa = make_esa(sample_dataset, "channel_12")
    assert esa.data.shape == (20, 2)
    assert esa.anomalies == [(8, 10)]
