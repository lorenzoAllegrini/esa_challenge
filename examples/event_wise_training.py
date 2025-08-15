import argparse
from functools import partial

import more_itertools as mit
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Categorical, Real
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit

from spaceai.benchmark import ESACompetitionBenchmark


def esa_scorer(y_val, y_pred, benchmark):
    """Compute the ESA f0.5 score used during model selection."""
    pred_anomalies = benchmark.process_pred_anomalies(y_pred, 0)
    indices = np.where(y_val == 1)[0]
    groups = [list(group) for group in mit.consecutive_groups(indices)]
    true_anomalies = [[group[0], group[-1]] for group in groups]
    res = benchmark.compute_classification_metrics(true_anomalies, pred_anomalies)
    esa_res = benchmark.compute_esa_classification_metrics(
        res, true_anomalies, pred_anomalies, len(y_val)
    )
    return esa_res["f0.5"]


def make_logistic_search_cv(estimator, space, scorer):
    return BayesSearchCV(
        estimator=estimator,
        search_spaces=space,
        scoring=scorer,
        cv=TimeSeriesSplit(n_splits=3),
        n_iter=50,
        n_jobs=-1,
        verbose=0,
        error_score=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an event-wise anomaly classifier from a features CSV."
    )
    parser.add_argument("--run-id", required=True, help="Identifier of the training run.")
    parser.add_argument(
        "--train-csv",
        required=True,
        help="Path to the CSV file containing event-wise training features.",
    )
    parser.add_argument(
        "--exp-dir",
        default="experiments",
        help="Directory where trained models will be stored.",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)

    benchmark = ESACompetitionBenchmark(
        run_id=args.run_id, exp_dir=args.exp_dir, segmentator=None
    )

    scorer = make_scorer(partial(esa_scorer, benchmark=benchmark), greater_is_better=True)

    lr_classifier = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )
    lr_param_space = {
        "C": Real(1e-4, 1e2, prior="log-uniform"),
        "penalty": Categorical(["l1", "l2"]),
        "class_weight": Categorical([None, "balanced"]),
    }
    search_cv = make_logistic_search_cv(lr_classifier, lr_param_space, scorer)

    benchmark.event_wise_model_selection(
        train_set=train_df,
        search_cv=search_cv,
        run_id=args.run_id,
        callbacks=None,
        call_every_ms=100,
        flat=True,
    )


if __name__ == "__main__":
    main()
