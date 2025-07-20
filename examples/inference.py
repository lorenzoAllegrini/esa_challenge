import argparse

from spaceai.benchmark.esa_competition_predictor import ESACompetitionPredictor
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2
from spaceai.segmentators.shapelet_miner import ShapeletMiner


def main():
    parser = argparse.ArgumentParser(description="ESA competition inference")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    shapelet_miner = ShapeletMiner(
        k_min_length=30,
        k_max_length=40,
        num_kernels=10,
        segment_duration=50,
        step_duration=10,
        run_id=args.artifacts_dir.split("/")[-1],
        exp_dir=args.artifacts_dir,
        skip=True,
    )

    segmentator = EsaDatasetSegmentator2(
        transformations=[
            "min",
            "max",
            "mean",
            "std",
            "var",
            "stft",
            "sc",
            "slope",
            "diff_var",
        ],
        segment_duration=50,
        step_duration=10,
        shapelet_miner=shapelet_miner,
        telecommands=False,
        pooling_segment_len=200,
        pooling_segment_stride=20,
        poolings=["max", "min"],
        run_id=args.artifacts_dir.split("/")[-1],
        exp_dir=args.artifacts_dir,
        use_shapelets=True,
    )

    predictor = ESACompetitionPredictor(args.artifacts_dir, segmentator)
    predictor.load_models()
    predictor.predict(args.test_parquet, args.output)


if __name__ == "__main__":
    main()
