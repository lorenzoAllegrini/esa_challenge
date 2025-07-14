import argparse
import os
from functools import partial

import more_itertools as mit
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import (
    Integer,
    Real,
)

from spaceai.benchmark import ESACompetitionBenchmark
from spaceai.data import ESAMissions
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2
from spaceai.segmentators.shapelet_miner import ShapeletMiner
from spaceai.utils.callbacks import SystemMonitorCallback


def esa_scorer(y_val, y_pred, benchmark):
    pred_anomalies = benchmark.process_pred_anomalies(y_pred, 0)
    indices = np.where(y_val == 1)[0]
    groups = [list(group) for group in mit.consecutive_groups(indices)]
    true_anomalies = [[group[0], group[-1]] for group in groups]
    res = benchmark.compute_classification_metrics(true_anomalies, pred_anomalies)
    esa_res = benchmark.compute_esa_classification_metrics(
        res, true_anomalies, pred_anomalies, len(y_val)
    )
    return esa_res["f0.5"]


def main():
    parser = argparse.ArgumentParser(description="ESA competition training")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--exp-dir", default="experiments")
    parser.add_argument("--data-root", default="datasets")
    args = parser.parse_args()

    shapelet_miner = ShapeletMiner(
        k_min_length=1599,
        k_max_length=1600,
        num_kernels=5,
        segment_duration=1000,
        step_duration=200,
        run_id=args.run_id,
        exp_dir=args.exp_dir,
    )

    segmentator = EsaDatasetSegmentator2(
        transformations=["min", "max", "mean", "std"],
        segment_duration=1000,
        step_duration=200,
        shapelet_miner=shapelet_miner,
        telecommands=False,
        poolings=["max", "min"],
        run_id=args.run_id,
        exp_dir=args.exp_dir,
        use_shapelets=True,
    )

    benchmark = ESACompetitionBenchmark(
        run_id=args.run_id,
        exp_dir=args.exp_dir,
        data_root=args.data_root,
        segmentator=segmentator,
    )

    lr_param_space = {
        "C": Real(1e-4, 1e2, prior="log-uniform"),
    }
    lr_classifier = LogisticRegression(max_iter=1000, solver="liblinear")
    search_cv_factory = lambda: BayesSearchCV(
        estimator=lr_classifier,
        search_spaces=lr_param_space,
        scoring=make_scorer(
            partial(esa_scorer, benchmark=benchmark), greater_is_better=True
        ),
        cv=TimeSeriesSplit(n_splits=3),
        n_iter=10,
        n_jobs=-1,
    )

    mission = ESAMissions.MISSION_1.value
    benchmark.run(
        mission=mission,
        search_cv_factory=search_cv_factory,
        search_cv_factory2=search_cv_factory,
        search_cv_factory3=search_cv_factory,
        skip_channel_training=False,
    )


if __name__ == "__main__":
    main()
