from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate
# ``scikit-optimize`` provides the ``Dimension`` class used for defining search
# spaces.  The package is optional and not available in some lightweight
# execution environments, hence we attempt to import it lazily and provide a
# minimal stub when missing so that the rest of the module can still be
# imported.
try:  # pragma: no cover - behaviour depends on environment
    from skopt.space import Dimension  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - executed without scikit-optimize
    from typing import Any as Dimension  # type: ignore[assignment]
from torch.utils.data import (
    DataLoader,
    Subset,
)

from spaceai.data import (
    ESA,
    ESAMission,
)
from spaceai.utils import data
from spaceai.utils.callbacks import (
    Callback,
    CallbackHandler,
)
from spaceai.utils.tools import (
    generate_kfold_masks,
    make_smart_masks,
    sample_smart_masks,
)

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel
    from spaceai.models.anomaly import AnomalyDetector
    from spaceai.models.anomaly_classifier import AnomalyClassifier

import hashlib
import json
import random
import warnings

import more_itertools as mit
from scipy.signal import find_peaks
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import (
    QuantileTransformer,
    StandardScaler,
)
from tqdm import tqdm

###aggiunti questi
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2
from spaceai.segmentators.shapelet_miner import ShapeletMiner

from .benchmark import Benchmark


def make_esa_scorer(benchmark):
    """Return a scikit-learn scorer computing ESA precision."""

    def scorer(y_true, y_pred):
        """Inner scoring function used by ``make_scorer``."""
        pred_anomalies = benchmark.process_pred_anomalies(y_pred, 0)
        indices = np.where(y_true == 1)[0]
        groups = [list(group) for group in mit.consecutive_groups(indices)]
        true_anomalies = [[group[0], group[-1]] for group in groups]
        res = benchmark.compute_classification_metrics(true_anomalies, pred_anomalies)
        esa_res = benchmark.compute_esa_classification_metrics(
            res, true_anomalies, pred_anomalies, len(y_true)
        )
        return esa_res["precision_corrected"]

    return make_scorer(scorer, greater_is_better=True)


class ESACompetitionBenchmark(Benchmark):

    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        segmentator: Optional[EsaDatasetSegmentator2],
        data_root: str = "datasets",
        id_offset=14728321,
        seed: int = 42,
    ):
        """Initializes a new ESA Competition benchmark run.

        Args:
            run_id (str): A unique identifier for this run.
            exp_dir (str): The directory where the results of this run are stored.
            seq_length (int): The length of the sequences used for training and testing.
            data_root (str): The root directory of the ESA dataset.
        """
        super().__init__(run_id, exp_dir)
        self.data_root: str = data_root
        self.all_results: List[Dict[str, Any]] = []
        self.segmentator: EsaDatasetSegmentator2 = segmentator
        self.all_logs: List[Dict[str, Any]] = []
        self.id_offset = id_offset
        self.seed = seed
        random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

    def add_group_activation(
        self,
        df: pd.DataFrame,
        group_map: dict[int, list[str]],
        cv_score: dict[str, float],
        gamma: float = 1.8,
        eps: float = 1e-6,
        delta: float = 0.05,
        beta: float = 0.5,
        alpha: float = 1.5,  # bonus per i canali extra
    ) -> pd.DataFrame:
        """
        Costruisce colonne "group_{gid}" con la regola:
            score(t) = max_c (w_c * F_{t,c}) +
                    beta * Σ_{c ≠ c*} (w_c * F_{t,c})

        dove
            F_{t,c} = min(p_{t,c} + δ_if_(p>0.5), 1.0) ^ γ
            w_c     = cv_score_c + eps

        Parametri:
            gamma : esponente per accentuare differenze di probabilità
            delta : extra boost aggiunto SOLO se p > 0.5 (capped a 1.0)
            beta  : 0 (solo canale migliore) … 1 (somma completa)
        """
        out = pd.DataFrame(index=df.index)
        min_val = 1e-6
        beta = max(0.0, min(beta, 1.0))  # mantieni β in [0,1]
        print(cv_score)

        for gid, ch_list in group_map.items():
            chs = [c for c in ch_list if c in df.columns and c.startswith("channel_")]
            if not chs:
                continue

            # pesi dai cv-score
            w = (
                np.array([cv_score.get(c, 1.0) for c in chs], dtype=float) + eps
            ) ** alpha

            # matrice probabilità
            P = df[chs].to_numpy()
            print(f"w: {w}")
          
            # delta solo se p>0.5, clamp finale a 1.0
            P_adj = np.minimum(P + (P > 0.5) * delta, 1.0)
            #print(f"P_adj: {P_adj}")
            # F = (P_adj)^gamma   (θ rimosso)
            F = P_adj**gamma

            contrib = F * w  # contributi pesati
         
            max_val = contrib.max(axis=1)
            bonus = (contrib.sum(axis=1) - max_val) * beta

            A = np.maximum(max_val + bonus, min_val)
            out[f"group_{gid}"] = A

        return pd.concat([df, out], axis=1)

    def add_pooling_features(
        self,
        df: pd.DataFrame,
        base_prefix: str = "channel_",
        k: int = 3,
        stride: int = 1,
    ) -> pd.DataFrame:
        """Add max-pooling features over channel probabilities."""

        # 1) individua le colonne channel_…
        cols = [c for c in df.columns if c.startswith(base_prefix)]
        values = df[cols].to_numpy()
        T, C = values.shape

        # 2) costruisci il dizionario dei pool feats
        pool_feats = {}
        for j, col in enumerate(cols):
            arr = values[:, j]
            pooled = np.empty_like(arr)
            for t in range(T):
                start = max(0, t - k + 1)
                pooled[t] = arr[start : t + 1].max()
            # manteniamo lo stesso nome colonna
            pool_feats[col] = pooled

        # 3) DataFrame dei pool feats, stesse colonne originali
        pool_df = pd.DataFrame(pool_feats, index=df.index, columns=cols)

        # 4) togli le colonne originali e metti quelle poolate
        df_no_channels = df.drop(cols, axis=1)
        return pd.concat([df_no_channels, pool_df], axis=1)

    def run(
        self,
        mission: ESAMission,
        search_cv_factory: Any,
        search_cv_factory2: Any,
        search_cv_factory3: Any,
        callbacks: Optional[List["Callback"]] = None,
        call_every_ms: int = 100,
        perc_eval2: float = 0.2,
        perc_eval1: float = 0.2,
        perc_shapelet: float = 0.1,
        external_estimators: int = 2,
        internal_estimators: int = 5,
        final_estimators: int = 3,
        peak_height: float = 0.5,
        buffer_size: int = 100,
        flat: bool = True,
        skip_channel_training: bool = False,
        gamma: float = 1.5,
        delta: float = 0.05,
        beta: float = 0.3,
    ):
        """Execute the full ESA competition training pipeline."""
        source_folder = os.path.join(self.data_root, mission.inner_dirpath)
        meta = pd.read_csv(os.path.join(source_folder, "channels.csv")).assign(
            Channel=lambda d: d.Channel.str.strip()
        )
        groups = (
            meta[meta["Channel"].isin(mission.target_channels)]
            .groupby("Group")["Channel"]
            .apply(list)
            .to_dict()
        )

        train_channel, challenge_channel = self.load_channel(
            mission, "channel_12", overlapping_train=True
        )
        total_len = len(train_channel.data)
        masks1 = make_smart_masks(
            total_len=total_len,
            invalid_masks=[],
            mask_len=int(perc_eval1 * total_len),
            n_masks=final_estimators,
            rng=self.rng,
        )
        binary_predictions = []
        challenge_probas = []
        for fold_idx, mask1 in enumerate(masks1):
            if fold_idx == 0:
                continue
            feat_dir = os.path.join(self.exp_dir, self.run_id)
            os.makedirs(feat_dir, exist_ok=True)

            # il run_id “locale” per questo mask1
            mask_run_id = self.make_run_id([tuple(mask1)])

            if skip_channel_training:
                # 1a) carica cv_scores per questo fold
                cv_csv = os.path.join(feat_dir, f"cv_scores_fold{fold_idx:02d}.csv")
                df_cv = pd.read_csv(cv_csv)  # colonne: "channel","cv_score"
                channel_cv = dict(zip(df_cv["channel"], df_cv["cv_score"]))

                global_train_csv = os.path.join(
                    feat_dir, f"train_ensemble_probas_{mask_run_id}.csv"
                )
                global_test_csv = os.path.join(
                    feat_dir, f"test_ensemble_probas_{mask_run_id}.csv"
                )

                # ora leggi i DataFrame veri
                final_train = pd.read_csv(global_train_csv)
                challenge_test = pd.read_csv(global_test_csv)

            else:
                final_train = None
                challenge_test = None
                channel_cv = {}
                for channel_id in mission.target_channels:
                    if (
                        int(channel_id.split("_")[1]) < 41
                        or int(channel_id.split("_")[1]) > 48
                    ):
                        continue
                    train_channel, challenge_channel = self.load_channel(
                        mission, channel_id, overlapping_train=True
                    )
                    final_train, challenge_test, channel_score = (
                        self.channel_specific_ensemble(
                            train_channel=train_channel,
                            challenge_channel=challenge_channel,
                            mask1=mask1,
                            channel_id=channel_id,
                            search_cv_factory=search_cv_factory,
                            search_cv_factory2=search_cv_factory2,
                            callbacks=callbacks,
                            call_every_ms=call_every_ms,
                            perc_eval2=perc_eval2,
                            perc_shapelet=perc_shapelet,
                            external_estimators=external_estimators,
                            internal_estimators=internal_estimators,
                        )
                    )
                    channel_cv[channel_id] = channel_score
                df_cv = pd.DataFrame(
                    list(channel_cv.items()), columns=["channel", "cv_score"]
                )
                cv_csv = os.path.join(feat_dir, f"cv_scores_fold{fold_idx:02d}.csv")
                df_cv.to_csv(cv_csv, index=False)

            # final_train  = self.add_pooling_features(final_train, k=5, stride=1)
            # challenge_test = self.add_pooling_features(challenge_test,  k=5, stride=1)
            final_train = self.add_group_activation(
                final_train, groups, channel_cv, gamma=gamma, delta=delta, beta=beta
            )
            challenge_test = self.add_group_activation(
                challenge_test, groups, channel_cv
            )

            feat_dir = os.path.join(self.exp_dir, self.run_id, "fold_features")
            os.makedirs(feat_dir, exist_ok=True)

            # nome file distinguendo il fold corrente
            train_csv = os.path.join(feat_dir, f"fold{fold_idx:02d}_train_feats.csv")
            test_csv = os.path.join(feat_dir, f"fold{fold_idx:02d}_challenge_feats.csv")

            # ------------------------------------------------------------
            # salva
            # ------------------------------------------------------------
            final_train.to_csv(train_csv, index=False)
            challenge_test.to_csv(test_csv, index=False)
            best_model, best_score = self.event_wise_model_selection(
                train_set=final_train,
                search_cv=search_cv_factory3(),
                callbacks=callbacks,
                call_every_ms=100,
                flat=flat,
                run_id=self.make_run_id([tuple(mask1)]),
            )
            feat = [c for c in challenge_test.columns if c.startswith("group_")]
            X_test = challenge_test[feat]
            predictions = best_model.predict(X_test)
            probas = best_model.predict_proba(X_test)[:, 1]
            binary_predictions.append(predictions)
            challenge_probas.append(probas)
            probas_df = pd.DataFrame(
                {"pred_proba": best_model.predict_proba(X_test)[:, 1]}
            )
            proba_csv = os.path.join(
                self.exp_dir, self.run_id, f"predicted_probas_fold{fold_idx:02d}.csv"
            )
            probas_df.to_csv(proba_csv, index=False)

        final_probas = np.max(challenge_probas, axis=0) - np.var(
            challenge_probas, axis=0
        )
        final_predictions = np.max(binary_predictions, axis=0)

        y_full, y_binary = self.predict_challenge_labels(
            challenge_test=challenge_test,
            challenge_probas=final_probas,
            peak_height=peak_height,
            buffer_size=buffer_size,
        )
        labels_df = pd.DataFrame(
            {
                "id": np.arange(len(y_full)) + self.id_offset,
                "is_anomaly": y_full,
                "pred_binary": y_binary,
            }
        )
        y_full, y_binary = self.predict_challenge_labels2(
            challenge_test=challenge_test,
            challenge_preds=final_predictions,
        )

        self.compute_total_scores(mission)
        return labels_df

    def channel_specific_ensemble(
        self,
        train_channel: ESA,
        challenge_channel: ESA,
        mask1: Tuple[int, int],
        channel_id: str,
        search_cv_factory: Any,
        search_cv_factory2: Any,
        callbacks: Optional[List["Callback"]] = None,
        call_every_ms: int = 100,
        perc_eval2: float = 0.25,
        perc_shapelet: float = 0.15,
        external_estimators: int = 5,
        internal_estimators: int = 5,
    ):
        """Full hierarchical ensemble for **one** channel.

        1. Split channel data into *base‑train* (perc_train) and *test*.
        2. For each **external** estimator:
        • Sample a random contiguous slice (`data_range`) of length `perc_eval1` from
            base‑train.
        • Train `internal_estimators` models via `channel_specific_model_selection`
            on that slice; each gets its own random `shapelet_range`.
        • Fit a meta‑classifier (LogReg) on the *eval* proba outputs of the internals.
        3. Aggregate the meta‑probabilities of all external estimators (simple mean)
        to obtain `final_proba`, saved as `final_channel_<id>.npy`.
        """

        len_data = len(train_channel.data)

        # -------------------- ENSEMBLE TRAINING -------------------------------
        external_eval1_probas = []
        external_challenge_probas = []
        eval2_masks = sample_smart_masks(
            len_data,
            [mask1],
            int(perc_eval2 * len_data),
            external_estimators,
            rng=np.random.default_rng(int(channel_id.split("_")[1])),
        )

        scores = []

        for ext_idx, mask2 in enumerate(tqdm(eval2_masks, desc="External estimators")):
            shapelet_masks = sample_smart_masks(
                len_data,
                [tuple(mask1), tuple(mask2)],
                int(perc_shapelet * len_data),
                internal_estimators,
                rng=np.random.default_rng(int(channel_id.split("_")[1])),
            )
            internal_eval2_probas = []
            internal_eval1_probas = []
            internal_challenge_probas = []
            for int_idx, shapelet_mask in enumerate(
                tqdm(shapelet_masks, desc=f"  Internals for ext={ext_idx}", leave=False)
            ):
                if self.segmentator is not None:
                    self.segmentator.shapelet_miner.initialize_kernels(
                        train_channel, shapelet_mask, self.make_run_id([shapelet_mask])
                    )
                    internal_train_channel, internal_train_anomalies = (
                        self.segmentator.segment(
                            train_channel,
                            masks=[tuple(mask1), tuple(mask2), tuple(shapelet_mask)],
                            train_phase=True,
                            ensemble_id=f"train_{self.make_run_id([tuple(mask1), tuple(mask2), tuple(shapelet_mask)])}",
                        )
                    )

                    eval2_channel, eval2_anomalies = self.segmentator.segment(
                        train_channel,
                        masks=[mask2],
                        ensemble_id=f"eval2_{self.make_run_id([tuple(mask2), tuple(shapelet_mask)])}",
                    )
                    eval1_channel, eval1_anomalies = self.segmentator.segment(
                        train_channel,
                        masks=[mask1],
                        ensemble_id=f"eval1_{self.make_run_id([tuple(mask1), tuple(shapelet_mask)])}",
                    )
                    challenge_channel_segments, _ = self.segmentator.segment(
                        challenge_channel,
                        masks=[],
                        ensemble_id=f"challenge_{self.make_run_id([tuple(shapelet_mask)])}",
                        train_phase=True,
                    )

                internal_estimator, _ = self.channel_specific_model_selection(
                    train_channel=internal_train_channel,
                    train_anomalies=internal_train_anomalies,
                    search_cv=search_cv_factory(),
                    callbacks=callbacks,
                    run_id=f"internal_{self.make_run_id([tuple(mask1), tuple(mask2), tuple(shapelet_mask)])}",
                    channel_id=channel_id,
                    call_every_ms=call_every_ms,
                )
                # -------- Save predicted internal probas ----------
                proba2 = internal_estimator.predict_proba(eval2_channel)
                if proba2.shape[1] == 1:
                    cls2 = internal_estimator.classes_[0]
                    p2 = np.full(len(proba2), 1.0 if cls2 == 1 else 0.0)
                else:
                    p2 = proba2[:, 1]
                internal_eval2_probas.append(p2)

                # 2) Eval1
                proba1 = internal_estimator.predict_proba(eval1_channel)
                if proba1.shape[1] == 1:
                    cls1 = internal_estimator.classes_[0]
                    p1 = np.full(len(proba1), 1.0 if cls1 == 1 else 0.0)
                else:
                    p1 = proba1[:, 1]
                internal_eval1_probas.append(p1)

                # 3) Challenge
                probac = internal_estimator.predict_proba(challenge_channel_segments)
                if probac.shape[1] == 1:
                    clsc = internal_estimator.classes_[0]
                    pc = np.full(len(probac), 1.0 if clsc == 1 else 0.0)
                else:
                    pc = probac[:, 1]
                internal_challenge_probas.append(pc)

            # -------- Meta‑classifier over internal probs (eval split) ----------
            meta_train_df = pd.DataFrame(
                {
                    f"channel_{i}": internal_eval2_probas[i]
                    for i in range(len(internal_eval2_probas))
                }
            )
            meta_eval1_df = pd.DataFrame(
                {
                    f"channel_{i}": internal_eval1_probas[i]
                    for i in range(len(internal_eval1_probas))
                }
            )
            meta_challenge_df = pd.DataFrame(
                {
                    f"channel_{i}": internal_challenge_probas[i]
                    for i in range(len(internal_challenge_probas))
                }
            )

            meta_estimator, meta_score = self.channel_specific_model_selection(
                train_channel=meta_train_df,
                train_anomalies=eval2_anomalies,
                search_cv=search_cv_factory2(),
                callbacks=callbacks,
                run_id=f"external_{self.make_run_id([tuple(mask2), tuple(shapelet_mask)])}",
                channel_id=channel_id,
                call_every_ms=call_every_ms,
            )
            scores.append(meta_score)
            probas_eval1 = meta_estimator.predict_proba(meta_eval1_df)
            if probas_eval1.shape[1] == 1:
                single_class = meta_estimator.classes_[0]
                ext1 = np.full(len(probas_eval1), 1.0 if single_class == 1 else 0.0)
            else:
                ext1 = probas_eval1[:, 1]
            external_eval1_probas.append(ext1)

            # 2) Challenge
            probas_ch = meta_estimator.predict_proba(meta_challenge_df)
            if probas_ch.shape[1] == 1:
                single_class = meta_estimator.classes_[0]
                ext_ch = np.full(len(probas_ch), 1.0 if single_class == 1 else 0.0)
            else:
                ext_ch = probas_ch[:, 1]
            external_challenge_probas.append(ext_ch)

        # -------------------- Ensemble of Meta-classifiers prediction for channel -------------------

        eval1_probas = np.mean(external_eval1_probas, axis=0)
        """qt = QuantileTransformer(
        output_distribution="uniform",
        n_quantiles=min(1000, len(eval1_probas)),
        random_state=0
        )
        # fit + transform sul vettore  (reshape(-1,1) per sklearn)
        eval1_probas = qt.fit_transform(eval1_probas.reshape(-1, 1)).ravel() """
        starts_test = eval1_channel["start"].tolist()
        ends_test = eval1_channel["end"].tolist()

        eval1_labels = np.zeros(len(eval1_channel), dtype=int)

        for s, e in eval1_anomalies:
            s = max(0, s)
            e = min(len(eval1_channel) - 1, e)
            eval1_labels[s : e + 1] = 1

        final_train = self.save_channel_probas(
            proba_list=eval1_probas,
            fname=f"train_ensemble_probas_{self.make_run_id([tuple(mask1)])}.csv",
            channel_id=channel_id,
            start_list=starts_test,
            end_list=ends_test,
            true_list=eval1_labels,
        )

        # -------------------- Ensemble of Meta-classifiers prediction for channel ON CHALLENGE -------------------

        challenge_probas = np.mean(external_challenge_probas, axis=0)
        # challenge_probas = qt.transform(challenge_probas.reshape(-1, 1)).ravel()
        starts_test = challenge_channel_segments["start"].tolist()
        ends_test = challenge_channel_segments["end"].tolist()

        challenge_test = self.save_channel_probas(
            proba_list=challenge_probas,
            fname=f"test_ensemble_probas_{self.make_run_id([tuple(mask1)])}.csv",
            channel_id=channel_id,
            start_list=starts_test,
            end_list=ends_test,
        )

        return final_train, challenge_test, np.mean(np.array(scores))

    def channel_specific_model_selection(
        self,
        train_channel: Any,
        train_anomalies: List[Tuple[int, int]],
        search_cv: Any,
        run_id: str,  # <<< nuovo argomento
        channel_id: str,  # <<< serve per il filename
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
    ):
        """Select the best model for a single channel using BayesSearchCV."""
        warnings.filterwarnings("ignore", category=FitFailedWarning)
        # prepara directory & JSON
        sel_dir = os.path.join(
            self.exp_dir, self.run_id, "channel_segments", channel_id
        )
        os.makedirs(sel_dir, exist_ok=True)
        json_path = os.path.join(sel_dir, f".json")

        model_dir = os.path.join(
            self.exp_dir,
            self.run_id,
            "models",
            f"channel_{channel_id}",
        )
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{run_id}.pkl")

        # carica esistente o inizializza vuoto
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                history = json.load(f)
        else:
            history = {}

        # numero di iterazioni programmate per questo search_cv
        desired_n = getattr(search_cv, "n_iter", None) or getattr(
            search_cv, "n_iter_search", None
        )

        full_train = train_channel.copy(deep=True).reset_index(drop=True)
        labels_train = np.zeros(len(full_train), dtype=int)
        for s, e in train_anomalies:
            s, e = max(0, s), min(len(full_train) - 1, e)
            labels_train[s : e + 1] = 1

        # --- NUOVA PARTE DI CODICE -----------------------------------
        unique = np.unique(labels_train)
        if unique.size < 2:
            # se c'è solo una classe, salta la CV e fai un fit diretto
            estimator = DummyClassifier(strategy="constant", constant=0)
            estimator.fit(full_train, labels_train)
            joblib.dump(estimator, model_path)
            return estimator, 0.0

        # se abbiamo già un risultato valido, ricicla
        prev = history.get(run_id)

        current_space_keys = sorted(search_cv.search_spaces.keys())
        if (
            prev is not None
            and desired_n is not None
            and prev.get("n_iter", 0) >= desired_n
        ):
            best_params = prev["best_params"]

            # ricrea e refit
            estimator = clone(search_cv.estimator).set_params(**best_params)
            estimator.fit(full_train, labels_train)
            cv_res = cross_validate(
                estimator,
                full_train,
                labels_train,
                cv=search_cv.cv,
                scoring=make_esa_scorer(self),
                n_jobs=-1,
                error_score=0.0,
            )
            metric_key = [k for k in cv_res if k.startswith("test_")][0]
            new_score = float(np.mean(cv_res[metric_key]))

            # se score diverso (o non salvato) → aggiorna file
            if abs(new_score - prev.get("cv_score", np.nan)) > 1e-6:
                history[run_id]["cv_score"] = new_score
                with open(json_path, "w") as f:
                    json.dump(history, f, indent=2)
            joblib.dump(estimator, model_path)
            return estimator, new_score

        # altrimenti: esegui la BayesSearchCV
        callback_handler = CallbackHandler(callbacks or [], call_every_ms)
        callback_handler.start()
        try:
            search_cv.fit(full_train, labels_train)
        except ValueError:
            # ------------------- fallback -------------------------------
            estimator = clone(search_cv.estimator)
            estimator.fit(full_train, labels_train)

            # ---- verifica se ogni fold ha entrambe le classi -----------
            cv_ok = True
            for tr_idx, te_idx in search_cv.cv.split(full_train, labels_train):
                if (
                    np.unique(labels_train[tr_idx]).size < 2
                    or np.unique(labels_train[te_idx]).size < 2
                ):
                    cv_ok = False
                    break

            if not cv_ok:
                joblib.dump(estimator, model_path)
                return estimator, 0.0  # oppure 0.0

            # ---- calcola comunque lo score -----------------------------
            cv_res = cross_validate(
                estimator,
                full_train,
                labels_train,
                cv=search_cv.cv,
                scoring=make_esa_scorer(self),
                n_jobs=1,
                error_score=0.0,
            )
            metric_key = [k for k in cv_res if k.startswith("test_")][0]
            fallback_score = float(np.mean(cv_res[metric_key]))
            joblib.dump(estimator, model_path)
            return estimator, fallback_score

        # ----- caso "normale" dopo fit riuscito -----
        best_estimator = search_cv.best_estimator_
        best_params = search_cv.best_params_

        # ---> qui si calcola cv_res sul best_estimator
        cv_res = cross_validate(
            best_estimator,
            full_train,
            labels_train,
            cv=search_cv.cv,
            scoring=make_esa_scorer(self),
            n_jobs=-1,
            error_score=0.0,
        )

        metric_key = [k for k in cv_res if k.startswith("test_")][0]
        best_score = float(np.mean(cv_res[metric_key]))
        n_done = (
            desired_n if desired_n is not None else len(search_cv.cv_results_["params"])
        )

        # salva nel JSON
        history[run_id] = {
            "cv_score": best_score,
            "n_iter": n_done,
            "best_params": best_params,
        }
        with open(json_path, "w") as f:
            json.dump(history, f, indent=2)
        joblib.dump(best_estimator, model_path)

        return best_estimator, best_score

    def event_wise_model_selection(
        self,
        train_set: pd.DataFrame,
        search_cv: Any,
        run_id: str,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
        flat: bool = True,
    ) -> Tuple[Any, float]:
        """
        Allena un meta‐modello segment‐wise sui dati di train_set.
        Salva (o ricarica) i migliori iperparametri in un file JSON di storico.

        Args:
            train_set:   DataFrame con colonne ["start","end","anomaly"] + "group_*"
            search_cv:   Un BayesSearchCV (o altro SearchCV) già configurato.
            run_id:      Identificativo univoco per questa cella di ricerca (es. "fold01_XGB").
            callbacks:   Lista di callback (opzionale).
            call_every_ms: Frequenza (ms) per i callback (opzionale).
            flat:        Se True, X e y sono già piatti; altrimenti si creano sequenze.

        Returns:
            best_model, best_score
        """
        # --- 1) prelevo X e y da train_set ---
        feat = [c for c in train_set.columns if c.startswith("group_")]
        if flat:
            X = train_set[feat]
            y = train_set["anomaly"].astype(int)
        else:
            seq_len = search_cv.search_spaces.get("lstm__seq_len", 5)
            X, y = self.create_sliding_sequences(train_set, feat, seq_len)

        # In rare cases all labels may belong to a single class, causing
        # estimators such as ``LogisticRegression`` to fail during fitting.
        # When this happens, fall back to a dummy classifier that always
        # predicts the majority class.
        unique = np.unique(y)
        if unique.size < 2:
            majority = int(unique[0]) if unique.size == 1 else 0
            estimator = DummyClassifier(strategy="constant", constant=majority)
            estimator.fit(X, y)
            model_dir = os.path.join(self.exp_dir, self.run_id, "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"event_wise_{run_id}.pkl")
            joblib.dump(estimator, model_path)
            return estimator, 0.0

        # skip hyperparameter search when any CV split lacks both classes
        cv_ok = True
        for tr_idx, te_idx in search_cv.cv.split(X, y):
            if (
                np.unique(y[tr_idx]).size < 2
                or np.unique(y[te_idx]).size < 2
            ):
                cv_ok = False
                break

        if not cv_ok:
            majority = int(np.bincount(y).argmax())
            estimator = DummyClassifier(strategy="constant", constant=majority)
            estimator.fit(X, y)
            model_dir = os.path.join(self.exp_dir, self.run_id, "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"event_wise_{run_id}.pkl")
            joblib.dump(estimator, model_path)
            return estimator, 0.0

        # --- 2) preparo path e carico (o inizializzo) il file di storicizzazione ---
        sel_dir = os.path.join(self.exp_dir, self.run_id)
        os.makedirs(sel_dir, exist_ok=True)
        json_path = os.path.join(sel_dir, "event_wise_params.json")

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                history = json.load(f)
        else:
            history = {}

        # numero di iterazioni attese della BayesSearchCV
        desired_n = getattr(search_cv, "n_iter", None) or getattr(
            search_cv, "n_iter_search", None
        )
        prev = history.get(run_id)

        # estraggo chiavi dello search space ordinandole
        current_space_keys = sorted(search_cv.search_spaces.keys())

        # --- 3) se esistono già risultati per questo run_id e n_iter è soddisfatto, li riuso ---
        if (
            prev is not None
            and desired_n is not None
            and prev.get("n_iter", 0) >= desired_n
        ):
            best_params = prev["best_params"]
            estimator = clone(search_cv.estimator).set_params(**best_params)
            estimator.fit(X, y)
            model_dir = os.path.join(self.exp_dir, self.run_id, "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"event_wise_{run_id}.pkl")
            joblib.dump(estimator, model_path)
            return estimator, prev["cv_score"]

        # --- 4) altrimenti: eseguo davvero la BayesSearchCV e salvo i nuovi params ---
        callback_handler = CallbackHandler(callbacks or [], call_every_ms)
        callback_handler.start()
        search_cv.set_params(error_score=0.0)
        try:
            search_cv.fit(X, y)
        except ValueError:
            # fall back to a constant classifier when CV cannot be performed
            callback_handler.stop()
            estimator = DummyClassifier(strategy="constant", constant=0)
            estimator.fit(X, y)
            model_dir = os.path.join(self.exp_dir, self.run_id, "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"event_wise_{run_id}.pkl")
            joblib.dump(estimator, model_path)
            return estimator, 0.0
        finally:
            callback_handler.stop()

        best_estimator = search_cv.best_estimator_
        best_params = search_cv.best_params_
        best_score = float(search_cv.best_score_)
        n_done = (
            desired_n if desired_n is not None else len(search_cv.cv_results_["params"])
        )

        # ----------------- SALVATAGGIO nel JSON -----------------
        history[run_id] = {
            "cv_score": best_score,
            "n_iter": n_done,
            "best_params": best_params,
        }
        if "batch_size" in search_cv.search_spaces.keys():
            with open(json_path, "w") as f:
                json.dump(history, f, indent=2)
        # ---------------------------------------------------------
        model_dir = os.path.join(self.exp_dir, self.run_id, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"event_wise_{run_id}.pkl")
        joblib.dump(best_estimator, model_path)
        print(f"best_estimator: {best_estimator}, best score: {best_score}")
        return best_estimator, best_score

    def create_sliding_sequences(self, df, feature_cols, seq_len, for_crf=True):
        """Convert a feature DataFrame into overlapping sliding windows."""

        X_seq, y_seq = [], []
        for i in range(len(df) - seq_len + 1):
            seq_df = df.iloc[i : i + seq_len]
            seq_x = seq_df[feature_cols].values

            if for_crf:
                # Costruzione X: lista di dizionari (uno per timestep)
                seq_x_dicts = [dict(zip(feature_cols, row)) for row in seq_x]
                X_seq.append(seq_x_dicts)

                # Costruzione Y: lista di etichette reali per ogni timestep, convertite in stringa
                labels = seq_df["anomaly"].astype(str).tolist()
                y_seq.append(labels)
            else:
                # X = array (per modelli come LSTM), y = solo ultima etichetta
                X_seq.append(seq_x)
                y_seq.append(df["anomaly"].iloc[i + seq_len - 1])
        return X_seq, y_seq

    def predict_challenge_labels(
        self,
        challenge_test: pd.DataFrame,
        challenge_probas: Any,
        peak_height: float = 0.4,
        buffer_size: int = 400,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applica il modello segment‐wise a challenge_test, ricostruisce
        serie di probabilità e genera la serie binaria attorno ai picchi.

        Salva:
          - predicted_proba_labels.csv
          - predicted_binary_labels.csv

        Returns:
            y_full (float proba), y_binary (0/1)
        """

        max_idx = int(challenge_test["end"].max())
        T = max_idx + 40
        sum_p = np.zeros(T, float)
        cnt = np.zeros(T, int)

        for (_, row), p in zip(challenge_test.iterrows(), challenge_probas):
            s, e = int(row["start"]), int(row["end"])
            sum_p[s : e + 1] += p
            cnt[s : e + 1] += 1

        mask = cnt > 0
        y_full = np.zeros(T, float)
        y_full[mask] = sum_p[mask] / cnt[mask]

        peaks, _ = find_peaks(y_full, height=peak_height)
        y_binary = np.zeros(T, int)
        for idx in peaks:
            a = max(0, idx - buffer_size)
            b = min(T - 1, idx + buffer_size)
            y_binary[a : b + 1] = 1

        ids = np.arange(T) + self.id_offset
        pd.DataFrame({"id": ids, "is_anomaly": y_full}).to_csv(
            os.path.join(self.run_dir, "predicted_proba_labels.csv"), index=False
        )
        pd.DataFrame({"id": ids, "pred_binary": y_binary}).to_csv(
            os.path.join(self.run_dir, "predicted_binary_labels.csv"), index=False
        )

        return y_full, y_binary

    def predict_challenge_labels2(
        self,
        challenge_test: pd.DataFrame,
        challenge_preds: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Riduce la zona anomala al minimo: fra l'end del primo segmento
        NON anomalo prima del blocco positivo e lo start del primo segmento
        NON anomalo dopo il blocco.
        """
        # lunghezza serie completa ------------------------------------------------
        max_idx = int(challenge_test["end"].max())
        T = max_idx + 40
        y_full = np.zeros(T, dtype=float)

        # array comodi ------------------------------------------------------------
        starts = challenge_test["start"].astype(int).to_numpy()
        ends = challenge_test["end"].astype(int).to_numpy()
        preds = np.asarray(challenge_preds, dtype=int)

        # indici dei segmenti positivi e gruppi contigui --------------------------
        pos_idx = np.where(preds > 0)[0]
        groups = [list(g) for g in mit.consecutive_groups(pos_idx)]

        for g in groups:
            first_pos, last_pos = g[0], g[-1]

            # ------ limite inferiore: end del segmento negativo precedente -------
            if first_pos > 0 and preds[first_pos - 1] == 0:
                new_start = ends[first_pos - 1]
            else:  # non c'è negativo prima
                new_start = starts[first_pos]

            # ------ limite superiore: start del segmento negativo successivo -----
            if last_pos < len(preds) - 1 and preds[last_pos + 1] == 0:
                new_end = starts[last_pos + 1]
            else:  # non c'è negativo dopo
                new_end = ends[last_pos]

            # sempre almeno 1 time-step
            if new_end <= new_start:
                new_end = new_start + 200
                new_start = new_end - 200

            # clamp ai bordi array
            new_start = max(0, new_start)
            new_end = min(T, new_end)

            y_full[new_start:new_end] = 1.0

        # salvataggio -------------------------------------------------------------
        ids = np.arange(T) + getattr(self, "id_offset", 0)
        out = pd.DataFrame({"id": ids, "is_anomaly": y_full.astype(int)})
        out.to_csv(os.path.join(self.run_dir, "predicted_labels.csv"), index=False)

        return y_full, y_full.astype(int)

    def load_channel(
        self, mission: ESAMission, channel_id: str, overlapping_train: bool = True
    ) -> Tuple[ESA, ESA]:
        """Load the training and testing datasets for a given channel.

        Args:
            channel_id (str): the ID of the channel to be used
            overlapping_train (bool): whether to use overlapping sequences for the training dataset

        Returns:
            Tuple[ESA, ESA]: training and testing datasets

        """
        train_channel = ESA(
            root=self.data_root,
            mission=mission,
            channel_id=channel_id,
            mode="anomaly",
            overlapping=overlapping_train,
            seq_length=250,
            n_predictions=1,
        )
        test_channel = ESA(
            root=self.data_root,
            mission=mission,
            channel_id=channel_id,
            mode="anomaly",
            overlapping=False,
            seq_length=250,
            train=False,
            drop_last=False,
            n_predictions=1,
        )

        return train_channel, test_channel

    def compute_classification_metrics(self, true_anomalies, pred_anomalies):
        """Basic precision/recall metrics on anomaly intervals."""
        results = {
            "n_anomalies": len(true_anomalies),
            "n_detected": len(pred_anomalies),
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        matched_true_seqs = []
        true_indices_grouped = [list(range(e[0], e[1] + 1)) for e in true_anomalies]
        true_indices_flat = set([i for group in true_indices_grouped for i in group])
        for e_seq in pred_anomalies:
            i_anom_predicted = set(range(e_seq[0], e_seq[1] + 1))

            matched_indices = list(i_anom_predicted & true_indices_flat)
            valid = True if len(matched_indices) > 0 else False

            if valid:
                true_seq_index = [
                    i
                    for i in range(len(true_indices_grouped))
                    if len(
                        np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])
                    )
                    > 0
                ]

                if not true_seq_index[0] in matched_true_seqs:
                    matched_true_seqs.append(true_seq_index[0])
                    results["true_positives"] += 1

            else:
                results["false_positives"] += 1

        results["false_negatives"] = len(
            np.delete(true_anomalies, matched_true_seqs, axis=0)
        )

        tpfp = results["true_positives"] + results["false_positives"]
        results["precision"] = results["true_positives"] / tpfp if tpfp > 0 else 1
        tpfn = results["true_positives"] + results["false_negatives"]
        results["recall"] = results["true_positives"] / tpfn if tpfn > 0 else 1
        results["f1"] = (
            (
                (results["precision"] * results["recall"])
                / (results["precision"] + results["recall"])
            )
            if results["precision"] + results["recall"] > 0
            else 0
        )
        return results

    def compute_esa_classification_metrics(
        self,
        results: Dict[str, Any],
        true_anomalies: List[Tuple[int, int]],
        pred_anomalies: List[Tuple[int, int]],
        total_length: int,
    ) -> Dict[str, Any]:
        """Compute ESA classification metrics.

        Args:
            results (Dict[str, Any]): the classification results
            true_anomalies (List[Tuple[int, int]]): the true anomalies
            pred_anomalies (List[Tuple[int, int]]): the predicted anomalies
            total_length (int): the total length of the sequence

        Returns:
            Dict[str, Any]: the ESA metrics results
        """
        esa_results = {}
        indices_true_grouped = [list(range(e[0], e[1] + 1)) for e in true_anomalies]
        indices_true_flat = set([i for group in indices_true_grouped for i in group])
        indices_pred_grouped = [list(range(e[0], e[1] + 1)) for e in pred_anomalies]
        indices_pred_flat = set([i for group in indices_pred_grouped for i in group])
        indices_all_flat = indices_true_flat.union(indices_pred_flat)
        n_e = total_length - len(indices_true_flat)
        tn_e = total_length - len(indices_all_flat)
        esa_results["tnr"] = tn_e / n_e if n_e > 0 else 1
        esa_results["precision_corrected"] = results["precision"] * esa_results["tnr"]
        esa_results["f0.5"] = (
            (
                (1 + 0.5**2)
                * (esa_results["precision_corrected"] * results["recall"])
                / (0.5**2 * esa_results["precision_corrected"] + results["recall"])
            )
            if esa_results["precision_corrected"] + results["recall"] > 0
            else 0
        )
        return esa_results

    def process_pred_anomalies(
        self, y_pred: np.ndarray, pred_buffer: int
    ) -> List[List[int]]:
        """Merge binary predictions into buffered anomaly intervals."""
        pred_anomalies = np.where(y_pred == 1)[0]

        if len(pred_anomalies) > 0:

            groups = [list(group) for group in mit.consecutive_groups(pred_anomalies)]
            buffered_intervals = [
                [max(0, int(group[0] - pred_buffer)), int(group[-1] + pred_buffer)]
                for group in groups
            ]

            merged_intervals = []
            for interval in sorted(buffered_intervals, key=lambda x: x[0]):
                if not merged_intervals or interval[0] > merged_intervals[-1][1]:
                    merged_intervals.append(interval)
                else:
                    merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])

            return merged_intervals
        else:
            return []

    def save_channel_probas(
        self,
        proba_list: List[float],
        fname: str,
        channel_id: str,
        start_list: List[int],
        end_list: List[int],
        true_list: Optional[List[int]] = None,
    ):
        """
        Salva (in append/overwrite) le probabilità out‐of‐fold del canale `channel_id`
        in un CSV wide in cui ogni colonna è un channel e le righe corrispondono
        all'indice di segmento OOF.
        """
        csv_path = os.path.join(self.run_dir, fname)

        if os.path.exists(csv_path):
            df_all = pd.read_csv(csv_path)
        else:
            df_all = pd.DataFrame()

        needed = len(proba_list)
        if df_all.shape[0] < needed:
            # allungo l’indice a RangeIndex(0..needed-1)
            df_all = df_all.reindex(pd.RangeIndex(needed))

        if "start" not in df_all.columns:
            df_all["start"] = pd.Series(start_list, index=pd.RangeIndex(needed))
        if "end" not in df_all.columns:
            df_all["end"] = pd.Series(end_list, index=pd.RangeIndex(needed))

        # 3) Scrivo la colonna del channel corrente
        df_all[channel_id] = pd.Series(proba_list, index=pd.RangeIndex(needed))

        if true_list is not None:
            true_series = pd.Series(true_list, index=pd.RangeIndex(needed)).astype(int)
            if "anomaly" in df_all.columns:
                # OR logico (bitwise)
                df_all["anomaly"] = (
                    df_all["anomaly"].fillna(0).astype(int) | true_series
                )
            else:
                df_all["anomaly"] = true_series

        # 4) Salvo su disco
        df_all.to_csv(csv_path, index=False)

        return df_all

    def make_run_id(self, masks: List[Tuple[int, int]]) -> str:
        """Create a short hash identifier for a list of masks."""
        # Canonicalizza e genera hash a 8 caratteri
        sorted_masks = sorted(masks, key=lambda x: (x[0], x[1]))
        key = "|".join(f"{s}_{e}" for s, e in sorted_masks)
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]

    def compute_total_scores(self, mission):
        """Aggregate and persist CV scores for all processed channels."""
        rows = []
        for channel_id in mission.target_channels:
            sel_dir = os.path.join(
                self.exp_dir, self.run_id, "channel_segments", channel_id
            )
            json_path = os.path.join(sel_dir, ".json")
            if not os.path.exists(json_path):
                continue  # il canale non è mai stato processato

            with open(json_path, "r") as f:
                history = json.load(f)

            # raccogli tutti gli score disponibili
            internal_scores = [
                v["cv_score"]  # ← assume che ora tu salvi cv_score
                for k, v in history.items()
                if k.startswith("internal_") and "cv_score" in v
            ]
            external_scores = [
                v["cv_score"]
                for k, v in history.items()
                if k.startswith("external_") and "cv_score" in v
            ]

            # media o NaN se non presenti
            internal_mean = (
                float(np.mean(internal_scores)) if internal_scores else float("nan")
            )
            external_mean = (
                float(np.mean(external_scores)) if external_scores else float("nan")
            )

            rows.append(
                {
                    "channel_id": channel_id,
                    "mean_internal_cv": internal_mean,
                    "mean_external_cv": external_mean,
                }
            )

        if rows:  # scrivi solo se c’è qualcosa
            score_df = pd.DataFrame(rows)
            score_csv = os.path.join(self.run_dir, "cv_score_summary.csv")
            score_df.to_csv(score_csv, index=False)
