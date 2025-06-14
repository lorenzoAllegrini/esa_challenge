import os
import numpy as np
import pandas as pd
from spaceai.data import ESA, ESAMissions
from spaceai.benchmark import ESACompetitionBenchmark
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from spaceai.utils.callbacks import SystemMonitorCallback
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2
from skopt import BayesSearchCV
from functools import partial
from sklearn.model_selection import TimeSeriesSplit
import more_itertools as mit
from sklearn.metrics import make_scorer
from skopt.space import Real, Categorical, Integer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn_crfsuite import CRF
from sklearn.base import BaseEstimator, TransformerMixin
from spaceai.segmentators.shapelet_miner import ShapeletMiner
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def esa_scorer(y_val, y_pred, benchmark):
    pred_anomalies = benchmark.process_pred_anomalies(y_pred, 0)

    indices = np.where(y_val == 1)[0]
    groups = [list(group) for group in mit.consecutive_groups(indices)]
    true_anomalies = [[group[0], group[-1]] for group in groups]
    res = benchmark.compute_classification_metrics(true_anomalies, pred_anomalies)
    esa_res = benchmark.compute_esa_classification_metrics(res,true_anomalies, pred_anomalies, len(y_val))
    return esa_res["f0.5"]


def main():
    num_kernels = 10
    segment_duration = 50
    step_duration = 10
    # --- Section 1: Shapelet Miner Setup ---
    esa_shapelet_miner = ShapeletMiner(
        k_min_length=30,
        k_max_length=40,
        num_kernels=num_kernels,
        segment_duration=segment_duration,
        step_duration=step_duration,
        run_id="esa_competition", 
        exp_dir="experiments",
    )

    # --- Section 2: Segmentator Setup ---
    esa_segmentator = EsaDatasetSegmentator2(
        transformations=["min", "max", "mean", "std", "var", "stft", "sc", "slope", "diff_var"],
        segment_duration=segment_duration,
        step_duration=step_duration,
        shapelet_miner=esa_shapelet_miner,
        telecommands=False,
        pooling_segment_len=200,
        pooling_segment_stride=20,
        poolings=["max", "min"],
        run_id="esa_competition", 
        exp_dir="experiments", 
    )

    # --- Section 3: Benchmark Initialization ---
    benchmark = ESACompetitionBenchmark(
        run_id="esa_competition", 
        exp_dir="experiments", 
        data_root="datasets",
        segmentator=esa_segmentator
    )

    # --- Section 4: Feature Selection & Pipeline ---
    
    selected_features = ["max_std", "min_slope", "max_slope", "max_var", "max_diff_var", "max_stft", "max_sc"]
    feature_selector = ColumnTransformer(
        transformers=[
            ("manual", "passthrough", selected_features),
            ("kernels", "passthrough", 
                lambda X: [c for c in X.columns 
                           if (c.startswith("max_kernel") and c.endswith("max_convolution"))
                           or (c.startswith("min_kernel") and c.endswith("min_convolution"))]
            )
        ],
        remainder="drop"
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
    
    search_cv_factory = lambda: BayesSearchCV(
        estimator=pipeline,
        search_spaces=xgb_param_space,
        scoring=xgb_scorer,
        cv=TimeSeriesSplit(n_splits=3),
        verbose=0,
        n_jobs=-1,
        n_iter=40,
        error_score=0.0
    )

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
    search_cv_factory2 = lambda: BayesSearchCV(
        estimator=lr_classifier,
        search_spaces=lr_param_space,
        scoring=make_scorer(partial(esa_scorer, benchmark=benchmark), greater_is_better=True),
        cv=TimeSeriesSplit(n_splits=3),
        n_iter=50,
        n_jobs=-1,
        verbose=3,
        error_score='raise'
    )
    
    xgb_param_space3 = {
        "scale_pos_weight": Real(0.1, 1.0),
        "n_estimators": Integer(1000, 1500),
        "max_depth": Integer(7, 12),
        "learning_rate": Real(0.001, 0.01),
        "min_child_weight": Integer(1, 3),
        "colsample_bytree": Real(0.7,1.0)
    }

    rf_param_space = {
        "n_estimators": Integer(100, 300),
        "max_depth": Integer(5, 15),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 5),
        "max_features": Real(0.5, 1.0)
    }

    xgb_scorer3 = make_scorer(partial(esa_scorer, benchmark=benchmark), greater_is_better=True)
    
    

    mission1 = ESAMissions.MISSION_1.value

    benchmark.run(
        mission=mission1,
        search_cv_factory=search_cv_factory,
        search_cv_factory2=search_cv_factory2,
        search_cv_factory3=search_cv_factory2,
        perc_eval1=0.2,
        perc_eval2=0.2,
        perc_shapelet=0.1,
        internal_estimators=5,
        external_estimators=2,
        final_estimators=3,
        skip_channel_training=True,
        gamma=1.5,
        delta=0.05,
        beta = 0.4
        )

if __name__ == "__main__":
    main()