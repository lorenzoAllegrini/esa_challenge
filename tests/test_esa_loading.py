from conftest import make_esa

from spaceai.data.esa import ESAMissions


def test_load_channel(sample_dataset):
    esa = make_esa(sample_dataset)
    assert esa.data.shape[0] == 5
    assert esa.anomalies == [(1, 2)]
