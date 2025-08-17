import argparse
import os
from functools import partial

import more_itertools as mit
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from spaceai.models.lstm_classifier import LSTMClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import (
    Integer,
    Categorical,
    Real,
)

from spaceai.benchmark import ESACompetitionTraining

from spaceai.data import ESAMissions
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2
from spaceai.segmentators.shapelet_miner import ShapeletMiner
from spaceai.utils.callbacks import SystemMonitorCallback
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from spaceai.utils.tools import kernel_column_selector



def make_logistic_search_cv(pipeline, space, scorer):
    return BayesSearchCV(
        estimator=pipeline,
        search_spaces=space,
        scoring=scorer,
        cv=TimeSeriesSplit(n_splits=3),
        n_iter=50,
        n_jobs=-1,
        verbose=0,
        error_score=0.0
    )

def make_xgb_search_cv(pipeline, space, scorer):
    return BayesSearchCV(
        estimator=pipeline,
        search_spaces=space,
        scoring=scorer,
        cv=TimeSeriesSplit(n_splits=3),
        verbose=0,
        n_jobs=-1,
        n_iter=40,
        error_score=0.0
    )


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
    parser.add_argument("--run-id", default="esa_training")
    parser.add_argument("--exp-dir", default="experiments")
    parser.add_argument("--data-root", default="datasets")
    args = parser.parse_args()

    num_kernels = 10
    segment_duration = 50
    step_duration = 10
    # --- Section 1: Shapelet Miner Setup ---
    shapelet_miner = ShapeletMiner(
        k_min_length=30,
        k_max_length=40,
        num_kernels=num_kernels,
        segment_duration=segment_duration,
        step_duration=step_duration,
        run_id="esa_training", 
        exp_dir="experiments",
        skip=False
    )

    segmentator = EsaDatasetSegmentator2(
        transformations=["min", "max", "mean", "std", "var", "stft", "sc", "slope", "diff_var"],
        segment_duration=segment_duration,
        step_duration=step_duration,
        shapelet_miner=shapelet_miner,
        telecommands=False,
        pooling_segment_len=200,
        pooling_segment_stride=20,
        poolings=["max", "min"],
        run_id="esa_training", 
        exp_dir="experiments", 
        use_shapelets=True
    )


    benchmark = ESACompetitionTraining(
        run_id=args.run_id,
        exp_dir=args.exp_dir,
        data_root=args.data_root,
        segmentator=segmentator,
    )
    
    # Parameters are passed directly to ``XGBClassifier`` so they should not
    # include any pipeline prefix such as ``classifier__``. Otherwise, XGBoost
    # will ignore them and emit warnings during training.
    selected_features = [
        "max_std",
        "min_slope",
        "max_slope",
        "max_var",
        "max_diff_var",
        "max_stft",
        "max_sc",
    ]
    feature_selector = ColumnTransformer(
        transformers=[
            ("manual", "passthrough", selected_features),
            ("kernels", "passthrough", kernel_column_selector),
        ],
        remainder="drop",
    )
    pipeline = Pipeline([
        ("selector", feature_selector),
        ("classifier", XGBClassifier(base_score=0.5))
    ])

    # --- Section 5: Hyperparameter Search for XGBoost ---
    
    xgb_param_space = {
        "classifier__scale_pos_weight": Real(0.2, 1.0),
        "classifier__n_estimators": Integer(1000, 1500),
        "classifier__max_depth": Integer(6, 9),
        "classifier__learning_rate": Real(0.001, 0.01),
        "classifier__min_child_weight": Integer(1, 3),
    }
    xgb_scorer = make_scorer(partial(esa_scorer, benchmark=benchmark), greater_is_better=True)
    
    search_cv_factory = partial(
        make_xgb_search_cv,
        pipeline=pipeline,
        space=xgb_param_space,
        scorer=xgb_scorer)
        
    lr_param_space = {
        "C": Real(1e-4, 1e+2, prior="log-uniform"),
        "penalty": Categorical(["l1", "l2"]),
        "class_weight": Categorical([None, "balanced"]),
    }
    
    lr_classifier = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )
    search_cv_factory2 = partial(
        make_logistic_search_cv,
        pipeline=lr_classifier,
        space=lr_param_space,
        scorer=xgb_scorer)
    
    lstm_param_space = {
        "learning_rate": Real(1e-5, 1e-4, prior="log-uniform"),
        "batch_size": Integer(100, 500),
    }
    base_lstm = LSTMClassifier(
        input_size=13,     # da sostituire con il numero di feature
        output_size=1,             # per classificazione binaria
        hidden_sizes=[80,80],         # un valore base, verrà esplorato
        seq_len=3,                 # importante: lunghezza delle finestre
        device="cpu", 
        reduce_out="first",          # o "cpu"
        epochs=100,
        dropout=0.0
    )

    search_cv_factory3 = lambda: BayesSearchCV(
        estimator=base_lstm,
        search_spaces=lstm_param_space,
        scoring=make_scorer(partial(esa_scorer, benchmark=benchmark), greater_is_better=True),
        cv=TimeSeriesSplit(n_splits=2),  # standard CV con shuffle
        n_iter=1,        # meno iterazioni rispetto a modelli più veloci
        n_jobs=1,         # ⚠️ PyTorch non ama il multiprocessing
        verbose=3,
        error_score='raise'
    )
    mission = ESAMissions.MISSION_1.value
    benchmark.run(
        mission=mission,
        search_cv_factory=search_cv_factory,
        search_cv_factory2=search_cv_factory2,
        search_cv_factory3=search_cv_factory2,
        skip_channel_training=False,
        final_estimators=2
        
    )


if __name__ == "__main__":
    main()
