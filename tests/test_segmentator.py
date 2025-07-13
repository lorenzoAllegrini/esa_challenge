from conftest import make_esa

from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2


def test_segmentator_columns(sample_dataset):
    esa = make_esa(sample_dataset)
    segmentator = EsaDatasetSegmentator2(
        transformations=["mean", "max"],
        run_id="t",
        exp_dir=str(sample_dataset / "exp"),
        segment_duration=2,
        step_duration=1,
        telecommands=False,
        poolings=[],
        use_shapelets=False,
        step_difference_feature=False,
    )
    df, _ = segmentator.segment(esa, [], "e1", train_phase=True)
    assert set(["start", "end", "mean", "max"]).issubset(df.columns)
