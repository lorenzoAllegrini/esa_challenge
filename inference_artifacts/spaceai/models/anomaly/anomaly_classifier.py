from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Optional,
    Union,
)

import numpy as np
import torch

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel

class AnomalyClassifier:
    def __init__(self, model: BaseEstimator):
        """
        Classificatore diretto per rilevamento anomalie sui segmenti.
        
        Args:
            model (BaseEstimator): un modello sklearn-like con .fit() e .predict()
        """
        self.model = model

    def fit(self, X, y=None):
        """
        Allena il classificatore sui dati segmentati.
        
        Args:
            X (np.ndarray): dati (n_segmenti, n_feature)
            y (np.ndarray): etichette binarie (0 = normale, 1 = anomalo)
        """
        self.model.fit(X)

    def predict(self, X):
        """
        Predice anomalie sui segmenti.
        
        Args:
            X (np.ndarray): dati (n_segmenti, n_feature)
        
        Returns:
            np.ndarray: predizioni binarie (0 o 1)
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Se disponibile, restituisce le probabilità predette.
        
        Returns:
            np.ndarray: probabilità di classe positiva
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        else:
            raise NotImplementedError("Il modello non supporta predict_proba")

    def score(self, X, y_true):
        """
        Valuta le prestazioni del classificatore.
        
        Returns:
            dict: metriche di classificazione
        """
        y_pred = self.predict(X)
        return {
            "precision": precision_score(y_true, y_pred, zero_division=1),
            "recall": recall_score(y_true, y_pred, zero_division=1),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }