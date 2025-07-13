from typing import (
    List,
    Literal,
    Optional,
)

import numpy as np
import torch
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
)
from sklearn.preprocessing import StandardScaler
from torch import (
    nn,
    optim,
)
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)
from tqdm import tqdm

from .predictors import LSTM


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.0001, beta: float = 8.0, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits:   (N,*) numeri reali in output prima della sigmoid
        targets:  stesse shape, 0/1 labels float32
        """
        probs = torch.sigmoid(logits)
        # flatten
        p = probs.view(-1)
        t = targets.view(-1)

        # veri positivi, false neg, false pos
        TP = (p * t).sum()
        FN = (t * (1 - p)).sum()
        FP = ((1 - t) * p).sum()

        ti = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        return 1 - ti


class LSTMClassifier(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.3,
        reduce_out: Optional[Literal["first", "mean"]] = None,
        device: Literal["cpu", "cuda"] = "cpu",
        batch_size: int = 64,
        learning_rate: float = 0.001,
        washout: int = 0,
        patience: Optional[int] = None,
        epochs: int = 20,
        seq_len: int = 5,
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.reduce_out = reduce_out
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.washout = washout
        self.patience = patience
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.stateful = False

        self.model = LSTM(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout=dropout,
            reduce_out=reduce_out,
            device=device,
            washout=washout,
        )
        self.model.build()

    def fit(self, train_set, train_labels):
        # train_set = self.scaler.fit_transform(train_set)
        X, y = [], []
        for i in range(len(train_set) - self.seq_len + 1):
            X.append(train_set[i : i + self.seq_len])
            y.append(train_labels[i : i + self.seq_len])
        X = np.array(X)
        y = np.array(y)
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = optim.Adam(self.model.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        return self.model.fit(
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=self.epochs,
            patience_before_stopping=self.patience,
        )

    def predict_proba(self, test_set: np.ndarray) -> np.ndarray:
        # test_set = self.scaler.transform(test_set)
        if self.model.model is not None:
            self.model.model.eval()
        self.stateful = False  # disattiva stati interni

        # Costruzione finestre di input
        X = [
            test_set[i : i + self.seq_len]
            for i in range(len(test_set) - self.seq_len + 1)
        ]
        X = torch.tensor(np.array(X), dtype=torch.float32)

        all_logits = []
        with torch.no_grad():
            for x in X:
                x = x.unsqueeze(0).to(self.device)  # [1, seq_len, D]
                out = self.model(x)  # logit o array di logit
                logit = out.detach().cpu().squeeze().numpy()
                all_logits.append(logit)

        y_logit = np.array(all_logits).reshape(-1)  # shape: (n_samples,)

        # Applica sigmoid per ottenere P(y=1)
        p1 = 1.0 / (1.0 + np.exp(-y_logit))  # shape: (n_samples,)

        # Costruisci la matrice (n_samples, 2): [P(y=0), P(y=1)]
        p0 = 1.0 - p1
        proba = np.vstack([p0, p1]).T  # shape: (n_samples, 2)

        return proba

    def predict(self, test_set: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(test_set)
        return (probas >= 0.5)[:, 1].astype(int)
