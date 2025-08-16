import argparse
import glob
import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from spaceai.benchmark.esa_competition_predictor import ESACompetitionPredictor
from spaceai.data import ESAMissions


def load_segmentator(artifacts_dir: str):
    pattern = os.path.join(artifacts_dir, "models", "channel_*", "internal_*.pkl")
    internal_models = glob.glob(pattern)
    if not internal_models:
        raise FileNotFoundError(f"No internal models found in {artifacts_dir}")
    model = joblib.load(internal_models[0])
    segmentator = getattr(model, "segmentator", None)
    if segmentator is None:
        raise AttributeError("Loaded model does not contain a segmentator")
    return segmentator


def aggregate_probas(probas: List[np.ndarray], mode: str = "max") -> np.ndarray:
    arr = np.vstack(probas)
    if mode == "max":
        return np.max(arr, axis=0)
    elif mode == "mean":
        return np.mean(arr, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation mode: {mode}")


def binary_from_proba(proba: np.ndarray, peak_height: float, buffer_size: int) -> np.ndarray:
    peaks, _ = find_peaks(proba, height=peak_height)
    y_binary = np.zeros_like(proba, dtype=int)
    for idx in peaks:
        a = max(0, idx - buffer_size)
        b = min(len(proba) - 1, idx + buffer_size)
        y_binary[a : b + 1] = 1
    return y_binary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble inference over multiple experiments")
    parser.add_argument("--experiments", nargs="+", required=True, help="List of experiment directories or IDs under experiments/")
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--mission", type=int, default=1)
    parser.add_argument("--agg", choices=["max", "mean"], default="max")
    parser.add_argument("--peak-height", type=float, default=0.5)
    parser.add_argument("--buffer-size", type=int, default=100)
    args = parser.parse_args()

    mission = getattr(ESAMissions, f"MISSION_{args.mission}").value

    probas = []
    ids = None
    for exp in args.experiments:
        artifacts_dir = exp
        if not os.path.isabs(artifacts_dir) and not artifacts_dir.startswith("experiments/"):
            artifacts_dir = os.path.join("experiments", artifacts_dir)
        segmentator = load_segmentator(artifacts_dir)
        predictor = ESACompetitionPredictor(artifacts_dir, segmentator, data_root=args.data_root)
        df = predictor.run(mission, test_parquet=args.test_parquet)
        probas.append(df["is_anomaly"].to_numpy())
        if ids is None:
            ids = df["id"].to_numpy()

    final_proba = aggregate_probas(probas, mode=args.agg)
    y_binary = binary_from_proba(final_proba, peak_height=args.peak_height, buffer_size=args.buffer_size)

    submission = pd.DataFrame({"id": ids, "is_anomaly": final_proba, "pred_binary": y_binary})
    submission.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
