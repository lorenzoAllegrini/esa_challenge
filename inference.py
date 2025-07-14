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
        k_min_length=1599,
        k_max_length=1600,
        num_kernels=5,
        segment_duration=1000,
        step_duration=200,
        run_id=args.artifacts_dir.split("/")[-1],
        exp_dir=args.artifacts_dir,
        skip=True,
    )

    segmentator = EsaDatasetSegmentator2(
        transformations=["min", "max", "mean", "std"],
        segment_duration=1000,
        step_duration=200,
        shapelet_miner=shapelet_miner,
        telecommands=False,
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
