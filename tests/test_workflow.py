from spaceai.benchmark.esa_competition import ESACompetitionBenchmark
from spaceai.data.esa import ESAMissions
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2
from spaceai.segmentators.shapelet_miner import ShapeletMiner


def test_competition_workflow(tmp_path, sample_dataset, monkeypatch):
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

    train_ch, test_ch = benchmark.load_channel(mission, "channel_41")
    assert train_ch.channel_id == "channel_41"
    assert test_ch.channel_id == "channel_41"
