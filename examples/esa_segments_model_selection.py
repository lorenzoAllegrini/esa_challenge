import os
import numpy as np
import pandas as pd
from spaceai.data import ESA, ESAMissions
from spaceai.benchmark import ESABenchmark
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

class SequenceGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, chunk_size: int):
        """
        chunk_size: numero di righe da raggruppare in ogni sequenza
        """
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        # non serve imparare nulla
        return self

    def transform(self, X):
        """
        X: pandas.DataFrame (con solo le colonne feature già selezionate)
        Ritorna: list of sequences, dove ogni sequence è una list di dict
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SequenceGrouper richiede un DataFrame in input")
        
        records = X.to_dict(orient="records")
        seqs = [
            records[i : i + self.chunk_size]
            for i in range(0, len(records), self.chunk_size)
        ]
        return seqs

def esa_scorer(y_val, y_pred, benchmark):
    pred_anomalies = benchmark.process_pred_anomalies(y_pred, 0)

    indices = np.where(y_val == 1)[0]
    groups = [list(group) for group in mit.consecutive_groups(indices)]
    true_anomalies = [[group[0], group[-1]] for group in groups]
    #print(f"pred_anomalies: {pred_anomalies}")
    #print(f"true_anomalies: {true_anomalies}")
    res = benchmark.compute_classification_metrics(true_anomalies, pred_anomalies)
    esa_res = benchmark.compute_esa_classification_metrics(res,true_anomalies, pred_anomalies, len(y_val))
    return esa_res["f0.5"]

def esa_crf_scorer(y_val, y_pred, benchmark):
    # y_val e y_pred arrivano come list[list[str]]
    # 1) appiattisco e converto a interi
    y_val_flat  = np.concatenate([np.array(seq, dtype=int) for seq in y_val])
    y_pred_flat = np.concatenate([np.array(seq, dtype=int) for seq in y_pred])
    
    # 2) estraggo gli intervalli sulle predizioni binarie
    pred_anomalies = benchmark.process_pred_anomalies(y_pred_flat, pred_buffer=0)

    # 3) estraggo gli intervalli veri
    ones = np.where(y_val_flat == 1)[0]
    groups = [list(g) for g in mit.consecutive_groups(ones)]
    true_anomalies = [[g[0], g[-1]] for g in groups]

    print(pred_anomalies)
    print(true_anomalies)
    res     = benchmark.compute_classification_metrics(true_anomalies, pred_anomalies)
    esa_res = benchmark.compute_esa_classification_metrics(
                   res, true_anomalies, pred_anomalies, total_length=len(y_val_flat)
              )

    return esa_res["f0.5"]

def precision_ratio_score(X, y):
    eps = 1e-6
    scores = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        mean_ano = feature[y == 1].mean()
        mean_nom = feature[y == 0].mean()
        ratio = mean_ano / (mean_nom + eps)
        scores.append(ratio)
    return np.array(scores), None

def main():
    num_kernels = 10
    segment_duration=50
    step_duration=10

    esa_shapelet_miner = ShapeletMiner(
        k_min_length=30,
        k_max_length=40,
        num_kernels=num_kernels,
        segment_duration=segment_duration,
        step_duration=step_duration
    )

    esa_segmentator = EsaDatasetSegmentator2(
        transformations=["mean","std", "var", "stft", "sc", "slope", "diff_var"],
        segment_duration=50,
        step_duration=10,
        shapelet_miner=esa_shapelet_miner,
        telecommands=False,
        pooling_segment_len= 200,
        pooling_segment_stride=20,
        poolings=["max", "min"],
    )

    benchmark = ESABenchmark(
        run_id="esa_segments", 
        exp_dir="experiments", 
        data_root="datasets",
        segmentator = esa_segmentator
    )
    param_space = {
        "classifier__scale_pos_weight": Real(0.4, 1.0),
        "classifier__n_estimators": Integer(1000, 1500),
        "classifier__max_depth": Integer(4, 9),
        "classifier__learning_rate": Real(0.001, 0.01),
        "classifier__min_child_weight": Integer(1, 3),
    }

    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        for channel_id in mission.target_channels:
            scorer = make_scorer(partial(esa_scorer, benchmark=benchmark), greater_is_better=True)

            selected_features = [
                "max_std", "min_slope", "max_var", "max_diff_var", "max_stft", "max_sc"
            ]

            feature_selector = ColumnTransformer(
                transformers=[
                    ("manual",   "passthrough", selected_features),
                    ("kernels",  "passthrough",
                        lambda X: [c for c in X.columns if (c.startswith("max_kernel") and c.endswith("max_convolution"))
                                                            or (c.startswith("min_kernel") and c.endswith("min_convolution"))]
                    )
                ],
                remainder="drop"   # butta via tutto il resto
            )
            
            pipeline = Pipeline([
                ("selector", feature_selector),
                ("classifier", XGBClassifier(base_score=0.5))
            ])

            search_cv_factory = lambda: BayesSearchCV(
                estimator=pipeline,
                search_spaces=param_space,
                scoring=scorer,
                cv=TimeSeriesSplit(n_splits=3),
                verbose=1,
                n_jobs=-1,
                n_iter=200,
                error_score=np.nan
            )

            res = benchmark.channel_specific_ensemble(
                mission=mission,
                channel_id=channel_id,
                search_cv_factory=search_cv_factory,
                perc_eval2 = 0.25,
                perc_eval1 = 0.1,
                perc_shapelet = 0.15,
                external_estimators = 5,
                internal_estimators = 5
            )

        param_space = {
            # qui “clf__” punta al passo "clf" della pipeline
            "clf__C":        Real(1e-4, 1e+2,   prior="log-uniform"),
            "clf__penalty":  Categorical(["l1", "l2"]),
            "clf__class_weight": Categorical([None, "balanced"]),
        }

        classifier = LogisticRegression(
            penalty="l2",
            solver="liblinear",     # adatto a l1/l2
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
                                
        search = BayesSearchCV(
            estimator=classifier,
            search_spaces=param_space,
            scoring=make_scorer(partial(esa_scorer, benchmark=benchmark), greater_is_better=True),
            cv=TimeSeriesSplit(n_splits=2),
            n_iter=50,
            n_jobs=-1,
            verbose=1,
            error_score="raise"
        )
        res = benchmark.event_wise_model_selection(
            search_cv=search,
            callbacks=[SystemMonitorCallback()],
            call_every_ms=100
        )
        print(f"final score: {res[1]}")
        print(f"best_model: {res[0]}")


if __name__ == "__main__":
    main()
