from __future__ import annotations

import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Literal,
    Callable
)

import numpy as np
import pandas as pd
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
from spaceai.utils.tools import make_smart_masks

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel
    from spaceai.models.anomaly import AnomalyDetector
    from spaceai.models.anomaly_classifier import AnomalyClassifier

from tqdm import tqdm
from .benchmark import Benchmark

###aggiunti questi
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2
import more_itertools as mit
from sklearn.preprocessing import StandardScaler
import json
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from spaceai.segmentators.shapelet_miner import ShapeletMiner


class ESABenchmark(Benchmark):

    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        segmentator: Optional[EsaDatasetSegmentator2],
        seq_length: int = 250,
        n_predictions: int = 1,
        data_root: str = "datasets", 
    ):
        """Initializes a new ESA benchmark run.

        Args:
            run_id (str): A unique identifier for this run.
            exp_dir (str): The directory where the results of this run are stored.
            seq_length (int): The length of the sequences used for training and testing.
            data_root (str): The root directory of the ESA dataset.
        """
        super().__init__(run_id, exp_dir)
        self.data_root: str = data_root
        self.seq_length: int = seq_length
        self.n_predictions: int = n_predictions
        self.all_results: List[Dict[str, Any]] = []
        self.segmentator: EsaDatasetSegmentator2 = segmentator
        self.all_logs: List[Dict[str, Any]] = []

    def run(
        self,
        mission: ESAMission,
        channel_id: str,
        predictor: SequenceModel,
        detector: AnomalyDetector,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
        perc_eval: Optional[float] = 0.2,
        restore_predictor: bool = False,
        overlapping_train: bool = True,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
    ):
        """Runs the benchmark for a given channel.

        Args:
            mission (ESAMission): the mission to be used
            channel_id (str): the ID of the channel to be used
            predictor (SequenceModel): the sequence model to be trained
            detector (AnomalyDetector): the anomaly detector to be used
            fit_predictor_args (Optional[Dict[str, Any]]): additional arguments for the predictor's fit method
            perc_eval (Optional[float]): the percentage of the training data to be used for evaluation
            restore_predictor (bool): whether to restore the predictor from a previous run
            overlapping_train (bool): whether to use overlapping sequences for the training dataset
        """
        callback_handler = CallbackHandler(
            callbacks=callbacks if callbacks is not None else [],
            call_every_ms=call_every_ms,
        )
        train_channel, test_channel = self.load_channel(
            mission,
            channel_id,
            overlapping_train=overlapping_train,
        )
        os.makedirs(self.run_dir, exist_ok=True)

        results: Dict[str, Any] = {"channel_id": channel_id}
        train_history = None
        if (
            os.path.exists(os.path.join(self.run_dir, f"predictor-{channel_id}.pt"))
            and restore_predictor
        ):
            logging.info(f"Restoring predictor for channel {channel_id}...")
            predictor.load(os.path.join(self.run_dir, f"predictor-{channel_id}.pt"))

        elif fit_predictor_args is not None:
            logging.info(f"Fitting the predictor for channel {channel_id}...")
            # Training the predictor
            batch_size = fit_predictor_args.pop("batch_size", 64)
            eval_channel = None
            if perc_eval is not None:
                # Split the training data into training and evaluation sets
                indices = np.arange(len(train_channel))
                np.random.shuffle(indices)
                eval_size = int(len(train_channel) * perc_eval)
                eval_channel = Subset(train_channel, indices[:eval_size])
                train_channel = Subset(train_channel, indices[eval_size:])
            train_loader = DataLoader(
                train_channel,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=data.seq_collate_fn(n_inputs=2, mode="batch"),
            )
            eval_loader = (
                DataLoader(
                    eval_channel,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=data.seq_collate_fn(n_inputs=2, mode="batch"),
                )
                if eval_channel is not None
                else None
            )
            callback_handler.start()
            predictor.stateful = False
            train_history = predictor.fit(
                train_loader=train_loader,
                valid_loader=eval_loader,
                **fit_predictor_args,
            )
            callback_handler.stop()
            results.update(
                {
                    f"train_{k}": v
                    for k, v in callback_handler.collect(reset=True).items()
                }
            )
            logging.info(
                f"Training time on channel {channel_id}: {results['train_time']}"
            )
            train_history = pd.DataFrame.from_records(train_history).to_csv(
                os.path.join(self.run_dir, f"train_history-{channel_id}.csv"),
                index=False,
            )
            predictor_path = os.path.join(self.run_dir, f"predictor-{channel_id}.pt")
            predictor.save(predictor_path)
            results["disk_usage"] = os.path.getsize(predictor_path)

        if predictor.model is not None:
            predictor.model.eval()
        logging.info(f"Predicting the test data for channel {channel_id}...")
        test_loader = DataLoader(
            test_channel,
            batch_size=1,
            shuffle=False,
            collate_fn=data.seq_collate_fn(n_inputs=2, mode="time"),
        )
        callback_handler.start()
        predictor.stateful = True
        y_pred, y_trg = zip(
            *[
                (
                    predictor(x.to(predictor.device)).detach().cpu().squeeze().numpy(),
                    y.detach().cpu().squeeze().numpy(),
                )
                for x, y in tqdm(test_loader, desc="Predicting")
            ]
        )
        y_pred, y_trg = [
            np.concatenate(seq)[test_channel.window_size - 1 :]
            for seq in [y_pred, y_trg]
        ]
        callback_handler.stop()
        results.update(
            {f"predict_{k}": v for k, v in callback_handler.collect(reset=True).items()}
        )
        results["test_loss"] = np.mean(((y_pred - y_trg) ** 2))  # type: ignore[operator]
        logging.info(f"Test loss for channel {channel_id}: {results['test_loss']}")
        logging.info(
            f"Prediction time for channel {channel_id}: {results['predict_time']}"
        )

        # Testing the detector
        logging.info(f"Detecting anomalies for channel {channel_id}")
        callback_handler.start()
        if len(y_trg) < 2500:
            detector.ignore_first_n_factor = 1
        if len(y_trg) < 1800:
            detector.ignore_first_n_factor = 0
        pred_anomalies = detector.detect_anomalies(y_pred, y_trg)
        pred_anomalies += detector.flush_detector()
        callback_handler.stop()
        results.update(
            {f"detect_{k}": v for k, v in callback_handler.collect(reset=True).items()}
        )
        logging.info(
            f"Detection time for channel {channel_id}: {results['detect_time']}"
        )

        true_anomalies = test_channel.anomalies

        classification_results = self.compute_classification_metrics(
            true_anomalies, pred_anomalies
        )
        esa_classification_results = self.compute_esa_classification_metrics(
            classification_results,
            true_anomalies,
            pred_anomalies,
            total_length=len(y_trg),
        )
        classification_results.update(esa_classification_results)
        results.update(classification_results)
        if train_history is not None:
            results["train_loss"] = train_history[-1]["loss_train"]
            if eval_loader is not None:
                results["eval_loss"] = train_history[-1]["loss_eval"]

        logging.info(f"Results for channel {channel_id}")

        self.all_results.append(results)

        pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )

            
    def channel_specific_ensemble(
        self,
        mission: "ESAMission",
        channel_id: str,
        search_cv_factory: Any,
        overlapping_train: bool = True,
        callbacks: Optional[List["Callback"]] = None,
        call_every_ms: int = 100,
        perc_eval2: float = 0.25,
        perc_eval1: float = 0.1,
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


        # -------------------- BASE DATA LOAD & GLOBAL SPLIT -------------------
        train_channel, challenge_channel = self.load_channel(mission, channel_id, overlapping_train=overlapping_train)
        len_data = len(train_channel.data)
        mask1 = [int((1-perc_eval1)*len(train_channel.data)), len(train_channel.data)]

        # -------------------- ENSEMBLE TRAINING -------------------------------
        external_objects: List[Tuple[List[Any], LogisticRegression]] = []
        external_eval1_probas = []
        external_challenge_probas = []
        eval2_masks = make_smart_masks(len_data, [mask1], int(perc_eval2*len_data), external_estimators)

        for ext_idx, mask2 in enumerate(eval2_masks):
            shapelet_masks = make_smart_masks(len_data, [tuple(mask1), tuple(mask2)], int(perc_shapelet*len_data), internal_estimators)
            internal_models: List[Any] = []
            internal_eval2_probas = []
            internal_eval1_probas = []
            internal_challenge_probas = []
            for int_idx, shapelet_mask in enumerate(shapelet_masks):
                if self.segmentator is not None:
                    self.segmentator.shapelet_miner.initialize_kernels(train_channel, shapelet_mask)
                    train_channel, train_anomalies = self.segmentator.segment(train_channel, masks=[tuple(mask1), tuple(mask2), tuple(shapelet_mask)], train_phase=True, ensemble_id=f"train_{ext_idx}_{int_idx}")
                    eval2_channel, eval2_anomalies = self.segmentator.segment(train_channel, masks=[mask2], ensemble_id=f"eval2_{ext_idx}_{int_idx}")
                    eval1_channel, eval1_anomalies = self.segmentator.segment(train_channel, masks=[mask1], ensemble_id=f"eval1_{ext_idx}_{int_idx}")
                    challenge_channel, _ = self.segmentator.segment(challenge_channel, masks=[], ensemble_id=f"challenge_{ext_idx}_{int_idx}")
                
                internal_estimator = self.channel_specific_model_selection(
                    train_channel=train_channel,
                    train_anomalies=train_anomalies,
                    channel_id=channel_id,
                    search_cv=search_cv_factory(),
                    invalid_masks=[tuple(mask1), tuple(mask2), tuple(shapelet_mask)],
                    callbacks=callbacks,
                    call_every_ms=call_every_ms,
                )
                internal_models.append(internal_estimator)
                internal_eval2_probas.append(internal_estimator.predict_proba(eval2_channel))
                internal_eval1_probas.append(internal_estimator.predict_proba(eval1_channel))
                internal_challenge_probas.append(internal_estimator.predict_proba(challenge_channel))

            # -------- Meta‑classifier over internal probs (eval split) ----------
            labels_eval2 = np.zeros(len(eval2_channel), dtype=int)
        
            for s, e in eval2_anomalies:
                s = max(0, s)
                e = min(len(eval2_channel)-1, e)
                labels_eval2[s : e+1] = 1
             
            meta_train = np.column_stack(internal_eval2_probas)
            meta_clf = LogisticRegression(max_iter=1000)
            meta_clf.fit(meta_train, labels_eval2)

            meta_eval1 = np.column_stack(internal_eval1_probas)
            external_eval1_probas.append(meta_clf.predict_proba(meta_eval1))

            meta_challenge = np.column_stack(internal_challenge_probas)
            external_challenge_probas.append(meta_clf.predict_proba(meta_challenge))

        # -------------------- Ensemble of Meta-classifiers prediction for channel -------------------
        external_eval_probas = []
        external_challenge_probas = []
        for internal_models, meta_clf in external_objects:
            probs1 = [m.predict_proba(eval1_channel)[:,1] for m in internal_models]
            X_meta1 = np.column_stack(probs1)
            external_eval_probas.append(meta_clf.predict_proba(X_meta1)[:,1])

            probsc = [m.predict_proba(challenge_channel)[:,1] for m in internal_models]
            X_metac = np.column_stack(probsc)
            external_challenge_probas.append(meta_clf.predict_proba(X_metac)[:,1])

        eval1_probas = np.mean(external_eval_probas, axis=0)
        starts_test = eval1_channel["start"].tolist()
        ends_test = eval1_channel["end"].tolist()

        eval1_labels = np.zeros(len(eval1_channel), dtype=int)
      
        for s, e in eval1_anomalies:
            s = max(0, s)
            e = min(len(eval1_channel)-1, e)
            eval1_labels[s : e+1] = 1
        
        self.save_channel_probas(
            proba_list=eval1_probas,
            fname=f"train_ensemble_probas.csv",
            channel_id=channel_id,
            start_list=starts_test,
            end_list=ends_test,
            true_list=eval1_labels
        )
        
        # -------------------- Ensemble of Meta-classifiers prediction for channel ON CHALLENGE -------------------
        
        challenge_probas = np.mean(external_challenge_probas, axis=0)
        starts_test = challenge_channel["start"].tolist()
        ends_test = challenge_channel["end"].tolist()
        
        self.save_channel_probas(
            proba_list=challenge_probas,
            fname=f"test_ensemble_probas.csv",
            channel_id=channel_id,
            start_list=starts_test,
            end_list=ends_test,
        )


    def channel_specific_model_selection( 
        self,
        train_channel: Any,
        train_anomalies:List[Tuple[int,int]],
        eval_channel: Any,
        channel_id: str,
        search_cv: Any,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
    ):
        callback_handler = CallbackHandler(
            callbacks=callbacks or [],
            call_every_ms=call_every_ms,
        )

        full_train = train_channel.copy(deep=True).reset_index(drop=True)

        labels_train = np.zeros(len(full_train), dtype=int)
      
        for s, e in train_anomalies:
            s = max(0, s)
            e = min(len(full_train)-1, e)
            labels_train[s : e+1] = 1

        callback_handler.start()
        search_cv.fit(X=full_train, y=labels_train)
        callback_handler.stop()

        best_estimator = search_cv.best_estimator_
        
        return best_estimator
    
    def event_wise_model_selection(
        self,
        search_cv: Any,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
    ):
        callback_handler = CallbackHandler(
            callbacks=callbacks or [],
            call_every_ms=call_every_ms,
        )

        train_path = os.path.join(self.run_dir, f"ensemble_probas.csv")
        test_path  = os.path.join(self.run_dir, f"all_channels_challenge_probas_{perc_train}.csv")
        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)

        feature_cols = [c for c in df_train.columns if c.startswith("channel_")]
        X_train = df_train[feature_cols]
        y_train = df_train["anomaly"].astype(int)    # 0/1

        callback_handler.start()
        search_cv.fit(X_train, y_train)
        callback_handler.stop()
        best_model  = search_cv.best_estimator_
        best_score  = search_cv.best_score_
        
        X_test       = df_test[feature_cols]
        seg_probas   = best_model.predict_proba(X_test)[:, 1]  

        max_end = int(df_test["end"].iloc[-1])
        T       = max_end + 40  # lunghezza reale
        sum_p   = np.zeros(T, dtype=float)
        cnt     = np.zeros(T, dtype=int)

        for i, p in enumerate(seg_probas):
            s = int(df_test.iloc[i]["start"])
            e = int(df_test.iloc[i]["end"])
            sum_p[s:e+1] += p
            cnt[s:e+1] += 1

        mask = cnt > 0
        y_full  = np.zeros(T, dtype=float)
        y_full[mask] = sum_p[mask] / cnt[mask]

        df_out = pd.DataFrame({
            "id": np.arange(len(y_full)) + 14728321,
            "is_anomaly":  y_full
        })
        df_out.to_csv(os.path.join(self.run_dir, "predicted_proba_labels.csv"), index=False)

        # --- 7) Ricavo i “picchi” e do un buffer attorno ---
        peaks, _ = find_peaks(y_full, height=0.5)
        binary  = np.zeros_like(y_full, dtype=int)
        buffer_ = 400
        for i in peaks:
            a = max(0, i - buffer_)
            b = min(T-1, i + buffer_)
            binary[a:b+1] = 1

        # --- 8) Salvo i label binari evento-wise ---
        df_bin = pd.DataFrame({
            "id":          df_out["id"],
            "pred_binary": binary
        })
        df_bin.to_csv(os.path.join(self.run_dir, "predicted_binary_labels.csv"), index=False)

        return best_model, best_score
        

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
            mode="challenge",
            overlapping=overlapping_train,
            seq_length=self.seq_length,
            n_predictions=self.n_predictions,
        )
        test_channel = ESA(
            root=self.data_root,
            mission=mission,
            channel_id=channel_id,
            mode="challenge",
            overlapping=False,
            seq_length=self.seq_length,
            train=False,
            drop_last=False,
            n_predictions=1,
        )

        return train_channel, test_channel

    def compute_classification_metrics(self, true_anomalies, pred_anomalies):
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
                2
                * (results["precision"] * results["recall"])
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
    
    def process_pred_anomalies(self, y_pred: np.ndarray, pred_buffer: int) -> List[List[int]]:
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
        start_list:   List[int],
        end_list:     List[int],
        true_list: Optional[List[int]] = None
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
            df_all["end"]   = pd.Series(end_list,   index=pd.RangeIndex(needed))

        # 3) Scrivo la colonna del channel corrente
        df_all[channel_id] = pd.Series(proba_list, index=pd.RangeIndex(needed))

        if true_list is not None:
            true_series = pd.Series(true_list, index=pd.RangeIndex(needed)).astype(int)
            if "anomaly" in df_all.columns:
                # OR logico (bitwise)
                df_all["anomaly"] = (df_all["anomaly"].fillna(0).astype(int) | true_series)
            else:
                df_all["anomaly"] = true_series

        # 4) Salvo su disco
        df_all.to_csv(csv_path, index=False)

    def chanel_specific_retraining(self, 
        search_cv: Any,
        train_channel: pd.DataFrame,
        labels: np.ndarray,
        best_model: Any,
        channel_id: str
    ):
        cv = search_cv.cv
        proba_list = []
        false_positives = 0
        true_list  = [] 
        starts_list = []
        ends_list   = []
        precisions = []
        recalls = []
        all_true_intervals = []
        all_pred_intervals = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(train_channel, labels)):
            X_train, y_train = train_channel.iloc[train_idx], labels[train_idx]
            X_test, y_test = train_channel.iloc[test_idx], labels[test_idx]

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            proba = best_model.predict_proba(X_test)[:, 1]
            proba_list.extend(proba.tolist())
            true_list.extend(y_test.tolist())
            starts_list.extend(X_test["start"].tolist())
            ends_list.extend(X_test["end"].tolist())

            y_test = [[int(X_test.iloc[anom[0]]["start"]), int(X_test.iloc[anom[1]]["end"])] for anom in self.process_pred_anomalies(y_test, 0)]
            y_pred = [[int(X_test.iloc[pred[0]]["start"]), int(X_test.iloc[pred[1]]["end"])] for pred in self.process_pred_anomalies(y_pred, 0)]

            fold_res = self.compute_classification_metrics(y_test, y_pred)
            fold_esa = self.compute_esa_classification_metrics(fold_res, y_test, y_pred, len(y_test))

            precisions.append(fold_esa.get("precision_corrected"))
            recalls.append(fold_res.get("recall"))
            all_true_intervals.extend(y_test)
            all_pred_intervals.extend(y_pred)
            false_positives += fold_res["false_positives"]
            
        avg_precision = float(np.nanmean(precisions))
        avg_recall = float(np.nanmean(recalls))
        
        self.save_channel_probas(proba_list, fname="all_channels_train_probas.csv", channel_id=channel_id, true_list=true_list,
                                  start_list=starts_list, end_list=ends_list)

        channel_log = {
            "channel_id":       channel_id,
            "avg_precision":    avg_precision,
            "avg_recall":       avg_recall,
            "false_positives":  false_positives,
            "true_intervals":   all_true_intervals,
            "pred_intervals":   all_pred_intervals,
            "best_params":      search_cv.best_params_
        }
        return channel_log