from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Dict,
)

import joblib
import glob
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from tqdm import tqdm

from spaceai.data import (  # typing re-export
    ESA,
    ESAMission,
)
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2
from spaceai.utils.callbacks import (
    Callback,
    CallbackHandler,
    SystemMonitorCallback,
)
from spaceai.utils.tools import (
    make_smart_masks,
    sample_smart_masks,
)

from .esa_competition import (
    ESACompetitionBenchmark,
    make_esa_scorer,
)


@dataclass
class SegmentedModel:
    model: Any
    segmentator: EsaDatasetSegmentator2
    ensemble_id: str

    def predict_proba(self, esa_channel, return_df: bool = False):
        df, _ = self.segmentator.segment_statistical(
            esa_channel,
            masks=[],
            ensemble_id=f"{self.ensemble_id}_test",
            train_phase=False,
        )

        df, _ = self.segmentator.segment_shapelets(
            df=df,
            esa_channel=esa_channel,
            shapelet_mask=(0, len(esa_channel.data)),
            ensemble_id=f"{self.ensemble_id}_test",
            masks=None,
            mode="exclude",
        )
        proba = self.model.predict_proba(df)
        return (proba, df) if return_df else proba


class ESACompetitionTraining(ESACompetitionBenchmark):
    """Benchmark variant that only performs training and serializes artefacts.

    It mirrors :class:`ESACompetitionBenchmark` but avoids generating
    inference outputs and stores each trained estimator together with the
    ``segmentator`` so that it can be reused later for prediction.
    """

    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        segmentator: Optional[EsaDatasetSegmentator2],
        data_root: str = "datasets",
        id_offset: int = 14728321,
        seed: int = 42,
    ) -> None:
        """Initialize the training benchmark."""
        super().__init__(
            run_id=run_id,
            exp_dir=exp_dir,
            segmentator=segmentator,
            data_root=data_root,
            id_offset=id_offset,
            seed=seed,
        )

    def load_channel(self, mission: ESAMission, channel_id: str, overlapping_train: bool = True) -> Tuple[ESA, ESA]:
        """Delegate channel loading to :class:`ESACompetitionBenchmark`."""
        return super().load_channel(mission, channel_id, overlapping_train)

    # ------------------------------------------------------------------
    #  Overridden methods with additional serialization steps
    # ------------------------------------------------------------------

    def channel_specific_model_selection(
        self,
        train_channel: Any,
        train_anomalies: List[Tuple[int, int]],
        search_cv: Any,
        run_id: str,
        channel_id: str,
        ensemble_id: str,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
    ) -> tuple[Any, float, Dict[str, Any]]:
        sel_dir = os.path.join(self.exp_dir, self.run_id, "channel_segments", channel_id)
        os.makedirs(sel_dir, exist_ok=True)
        json_path = os.path.join(sel_dir, f".json")

        model_dir = os.path.join(self.exp_dir, self.run_id, "models", f"channel_{channel_id}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{run_id}.pkl")

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                history = json.load(f)
        else:
            history = {}

        desired_n = getattr(search_cv, "n_iter", None) or getattr(search_cv, "n_iter_search", None)

        full_train = train_channel.copy(deep=True).reset_index(drop=True)
        labels_train = np.zeros(len(full_train), dtype=int)
        for s, e in train_anomalies:
            s, e = max(0, s), min(len(full_train) - 1, e)
            labels_train[s : e + 1] = 1

        if np.unique(labels_train).size < 2:
            majority = int(np.bincount(labels_train).argmax())
            estimator = DummyClassifier(strategy="constant", constant=majority)
            estimator.fit(full_train, labels_train)
            seg_model = SegmentedModel(copy.deepcopy(estimator), copy.deepcopy(self.segmentator), ensemble_id)
            joblib.dump(seg_model, model_path)
            return estimator, 0.0, {"time": 0.0, "cpu": 0.0, "mem": 0.0}

        # Ensure every CV split contains both classes, otherwise CV-based
        # model selection would fail with ``ValueError``.  When a degenerate
        # split is detected fall back to a constant classifier.
        """cv_ok = True
        for tr_idx, te_idx in search_cv.cv.split(full_train, labels_train):
            if (
                np.unique(labels_train[tr_idx]).size < 2
                or np.unique(labels_train[te_idx]).size < 2
            ):
                cv_ok = False
                break

        if not cv_ok:
            majority = int(np.bincount(labels_train).argmax())
            estimator = DummyClassifier(strategy="constant", constant=majority)
            estimator.fit(full_train, labels_train)
            seg_model = SegmentedModel(copy.deepcopy(estimator), copy.deepcopy(self.segmentator), ensemble_id)
            joblib.dump(seg_model, model_path)
            return estimator, 0.0, {"time": 0.0, "cpu": 0.0, "mem": 0.0}"""

        prev = history.get(run_id)
        current_space_keys = sorted(search_cv.search_spaces.keys())
        if prev is not None and desired_n is not None and prev.get("n_iter", 0) >= desired_n and False:
            best_params = prev["best_params"]
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
            if abs(new_score - prev.get("cv_score", np.nan)) > 1e-6:
                history[run_id]["cv_score"] = new_score
                with open(json_path, "w") as f:
                    json.dump(history, f, indent=2)
            seg_model = SegmentedModel(estimator, copy.deepcopy(self.segmentator), ensemble_id)
            joblib.dump(seg_model, model_path)
            return estimator, new_score, {"time": 0.0, "cpu": 0.0, "mem": 0.0}

        callback_handler = CallbackHandler((callbacks or []) + [SystemMonitorCallback()], call_every_ms)
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
                callback_handler.stop()
                metrics = callback_handler.collect(reset=True)
                return estimator, 0.0, metrics  # oppure 0.0

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
            callback_handler.stop()
            metrics = callback_handler.collect(reset=True)
            joblib.dump(estimator, model_path)
            return estimator, fallback_score, metrics
        callback_handler.stop()
        metrics = callback_handler.collect(reset=True)

        best_estimator = search_cv.best_estimator_
        best_params = search_cv.best_params_
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
        n_done = desired_n if desired_n is not None else len(search_cv.cv_results_["params"])
        history[run_id] = {
            "cv_score": best_score,
            "n_iter": n_done,
            "best_params": best_params,
        }
        with open(json_path, "w") as f:
            json.dump(history, f, indent=2)
        seg_model = SegmentedModel(best_estimator, copy.deepcopy(self.segmentator), ensemble_id)
        joblib.dump(seg_model, model_path)
        return best_estimator, best_score, metrics

    def channel_specific_ensemble(
        self,
        train_channel: ESA,
        mask1: Tuple[int, int],
        channel_id: str,
        search_cv_factory: Any,
        search_cv_factory2: Any,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
        perc_eval2: float = 0.25,
        perc_shapelet: float = 0.15,
        external_estimators: int = 5,
        internal_estimators: int = 5,
    ):
        channel_handler = CallbackHandler([SystemMonitorCallback()], call_every_ms)
        channel_handler.start()
        len_data = len(train_channel.data)
        # Pre-compute statistical features once for this channel
        stats_df, stats_anoms = self.segmentator.segment_statistical(
            train_channel,
            masks=[],
            ensemble_id=f"stats_{channel_id}",
            train_phase=True,
        )
        labels_all = np.zeros(len(stats_df), dtype=int)
        for s, e in stats_anoms:
            s = max(0, s)
            e = min(len(stats_df) - 1, e)
            labels_all[s : e + 1] = 1


        external_eval1_probas = []
        eval2_masks = sample_smart_masks(
            len_data,
            [mask1],
            int(perc_eval2 * len_data),
            external_estimators,
            rng=np.random.default_rng(int(channel_id.split("_")[1])),
        )
        scores = []
        external_times = []
        external_cpu = []
        external_mem = []
        internal_times_all = []
        internal_cpu_all = []
        internal_mem_all = []
        # mapping meta -> internal models for this channel
        links: dict[str, dict[str, list[str]]] = {}
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
            internal_ids = []
            for int_idx, shapelet_mask in enumerate(
                tqdm(shapelet_masks, desc=f"  Internals for ext={ext_idx}", leave=False)
            ):
                if self.segmentator is not None:
                    internal_train_channel, internal_train_anomalies = self.segmentator.segment_shapelets(
                        df=stats_df,
                        labels=labels_all,
                        esa_channel=train_channel,
                        shapelet_mask=shapelet_mask,
                        ensemble_id=f"train_{self.make_run_id([tuple(mask1), tuple(mask2), tuple(shapelet_mask)])}",
                        masks=[tuple(mask1), tuple(mask2), tuple(shapelet_mask)],
                        mode="exclude",
                        initialize=True,
                    )
                    internal_train_channel = internal_train_channel.drop(columns=["event", "start", "end"])
                    print(internal_train_channel)
                    eval2_channel, eval2_anomalies = self.segmentator.segment_shapelets(
                        df=stats_df,
                        labels=labels_all,
                        esa_channel=train_channel,
                        shapelet_mask=shapelet_mask,
                        ensemble_id=f"eval2_{self.make_run_id([tuple(mask2), tuple(shapelet_mask)])}",
                        masks=[tuple(mask2)],
                        mode="include",
                    )
                    eval2_channel = eval2_channel.drop(columns=["event", "start", "end"])
                    print(eval2_channel)
                    eval1_channel, eval1_anomalies = self.segmentator.segment_shapelets(
                        df=stats_df,
                        labels=labels_all,
                        esa_channel=train_channel,
                        shapelet_mask=shapelet_mask,
                        ensemble_id=f"eval1_{self.make_run_id([tuple(mask1), tuple(shapelet_mask)])}",
                        masks=[tuple(mask1)],
                        mode="include",
                    )
                    eval1_channel = eval1_channel.drop(columns=["event", "start", "end"])
                    print(eval1_channel)
                internal_run_id = f"internal_{self.make_run_id([tuple(mask1), tuple(mask2), tuple(shapelet_mask)])}"
                internal_estimator, _, int_metrics = self.channel_specific_model_selection(
                    train_channel=internal_train_channel,
                    train_anomalies=internal_train_anomalies,
                    search_cv=search_cv_factory(),
                    callbacks=callbacks,
                    run_id=internal_run_id,
                    channel_id=channel_id,
                    call_every_ms=call_every_ms,
                    ensemble_id=f"train_{self.make_run_id([tuple(mask1), tuple(mask2), tuple(shapelet_mask)])}",
                )
                internal_times_all.append(int_metrics.get("time", 0.0))
                internal_cpu_all.append(int_metrics.get("cpu", 0.0))
                internal_mem_all.append(int_metrics.get("mem", 0.0))
                internal_ids.append(internal_run_id)
                proba2 = internal_estimator.predict_proba(eval2_channel)
                if proba2.shape[1] == 1:
                    cls2 = internal_estimator.classes_[0]
                    p2 = np.full(len(proba2), 1.0 if cls2 == 1 else 0.0)
                else:
                    p2 = proba2[:, 1]
                internal_eval2_probas.append(p2)

                proba1 = internal_estimator.predict_proba(eval1_channel)
                if proba1.shape[1] == 1:
                    cls1 = internal_estimator.classes_[0]
                    p1 = np.full(len(proba1), 1.0 if cls1 == 1 else 0.0)
                else:
                    p1 = proba1[:, 1]
                internal_eval1_probas.append(p1)

            meta_train_df = pd.DataFrame(
                {f"channel_{i}": internal_eval2_probas[i] for i in range(len(internal_eval2_probas))}
            )
            meta_eval1_df = pd.DataFrame(
                {f"channel_{i}": internal_eval1_probas[i] for i in range(len(internal_eval1_probas))}
            )
            meta_run_id = f"external_{self.make_run_id([tuple(mask1), tuple(mask2)])}"
            meta_estimator, meta_score, meta_metrics = self.channel_specific_model_selection(
                train_channel=meta_train_df,
                train_anomalies=eval2_anomalies,
                search_cv=search_cv_factory2(),
                callbacks=callbacks,
                run_id=meta_run_id,
                channel_id=channel_id,
                call_every_ms=call_every_ms,
                ensemble_id=f"meta_{self.make_run_id([tuple(mask1), tuple(mask2)])}",
            )
            external_times.append(meta_metrics.get("time", 0.0))
            external_cpu.append(meta_metrics.get("cpu", 0.0))
            external_mem.append(meta_metrics.get("mem", 0.0))
            scores.append(meta_score)
            probas_eval1 = meta_estimator.predict_proba(meta_eval1_df)
            if probas_eval1.shape[1] == 1:
                single_class = meta_estimator.classes_[0]
                ext1 = np.full(len(probas_eval1), 1.0 if single_class == 1 else 0.0)
            else:
                ext1 = probas_eval1[:, 1]
            external_eval1_probas.append(ext1)
            links[meta_run_id] = {
                "mask_id": self.make_run_id([tuple(mask1)]),
                "internal_ids": internal_ids,
            }

        eval1_probas = np.mean(external_eval1_probas, axis=0)
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
        # persist mapping meta -> internal models
        links_path = os.path.join(
            self.exp_dir,
            self.run_id,
            "models",
            f"channel_{channel_id}",
            "links.json",
        )
        if os.path.exists(links_path):
            with open(links_path, "r") as f:
                saved = json.load(f)
        else:
            saved = {}
        saved.update(links)
        with open(links_path, "w") as f:
            json.dump(saved, f, indent=2)

        # If training produced no model files (e.g. due to degenerate labels),
        # create a placeholder estimator so downstream code can load it.
        model_dir = os.path.join(
            self.exp_dir,
            self.run_id,
            "models",
            f"channel_{channel_id}",
        )
        if not glob.glob(os.path.join(model_dir, "internal_*.pkl")):
            placeholder = DummyClassifier(strategy="constant", constant=0)
            placeholder.fit([[0], [1]], [0, 1])
            joblib.dump(placeholder, os.path.join(model_dir, "internal_dummy.pkl"))
        if not glob.glob(os.path.join(model_dir, "external_*.pkl")):
            placeholder = DummyClassifier(strategy="constant", constant=0)
            placeholder.fit([[0], [1]], [0, 1])
            joblib.dump(placeholder, os.path.join(model_dir, "external_dummy.pkl"))

        channel_handler.stop()
        channel_metrics = channel_handler.collect(reset=True)
        channel_log = {
            "channel_id": channel_id,
            "avg_internal_time": float(np.mean(internal_times_all)) if internal_times_all else 0.0,
            "avg_internal_cpu": float(np.mean(internal_cpu_all)) if internal_cpu_all else 0.0,
            "avg_internal_mem": float(np.mean(internal_mem_all)) if internal_mem_all else 0.0,
            "avg_external_time": float(np.mean(external_times)) if external_times else 0.0,
            "avg_external_cpu": float(np.mean(external_cpu)) if external_cpu else 0.0,
            "avg_external_mem": float(np.mean(external_mem)) if external_mem else 0.0,
            "channel_time": channel_metrics.get("time", 0.0),
            "channel_cpu": channel_metrics.get("cpu", 0.0),
            "channel_mem": channel_metrics.get("mem", 0.0),
        }
        self.all_logs.append(channel_log)
        pd.DataFrame.from_records(self.all_logs).to_csv(
            os.path.join(self.exp_dir, self.run_id, "efficiency_log.csv"), index=False
        )

        return final_train, np.mean(np.array(scores))

    def run(
        self,
        mission: ESAMission,
        search_cv_factory: Any,
        search_cv_factory2: Any,
        search_cv_factory3: Any,
        callbacks: Optional[List[Callback]] = None,
        call_every_ms: int = 100,
        perc_eval2: float = 0.2,
        perc_eval1: float = 0.2,
        perc_shapelet: float = 0.1,
        external_estimators: int = 3,
        internal_estimators: int = 10,
        final_estimators: int = 5,
        flat: bool = True,
        skip_channel_training: bool = False,
        gamma: float = 1.5,
        delta: float = 0.05,
        beta: float = 0.3,
    ) -> None:
        
        source_folder = os.path.join(self.data_root, mission.inner_dirpath)
        if not os.path.exists(os.path.join(source_folder, "channels.csv")):
            esa =  ESA(
                root=self.data_root,
                mission=mission,
                channel_id="channel_12",
                mode="anomaly",
                overlapping=True,
                seq_length=1,
                n_predictions=1,
            )
        meta = pd.read_csv(os.path.join(source_folder, "channels.csv")).assign(Channel=lambda d: d.Channel.str.strip())
        groups = meta[meta["Channel"].isin(mission.target_channels)].groupby("Group")["Channel"].apply(list).to_dict()
        train_channel, _ = self.load_channel(mission, "channel_12", overlapping_train=True)
        total_len = len(train_channel.data)
        masks1 = make_smart_masks(
            total_len=total_len,
            invalid_masks=[],
            mask_len=int(perc_eval1 * total_len),
            n_masks=final_estimators,
            rng=self.rng,
        )
        for fold_idx, mask1 in enumerate(masks1):
            if fold_idx == 0:
                continue
            fold_handler = CallbackHandler([SystemMonitorCallback()], call_every_ms)
            fold_handler.start()
            feat_dir = os.path.join(self.exp_dir, self.run_id)
            os.makedirs(feat_dir, exist_ok=True)
            mask_run_id = self.make_run_id([tuple(mask1)])
            if skip_channel_training:
                cv_csv = os.path.join(feat_dir, f"cv_scores_fold{fold_idx:02d}.csv")
                df_cv = pd.read_csv(cv_csv)
                channel_cv = dict(zip(df_cv["channel"], df_cv["cv_score"]))
                global_train_csv = os.path.join(feat_dir, f"train_ensemble_probas_{mask_run_id}.csv")
                final_train = pd.read_csv(global_train_csv)
            else:
                final_train = None
                channel_cv = {}
                for channel_id in mission.target_channels:
                    if int(channel_id.split("_")[1]) < 11:
                        continue
                     
                    train_channel, _ = self.load_channel(mission, channel_id, overlapping_train=True)
                   
                    final_train, channel_score = self.channel_specific_ensemble(
                        train_channel=train_channel,
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
                    channel_cv[channel_id] = channel_score
                df_cv = pd.DataFrame(list(channel_cv.items()), columns=["channel", "cv_score"])
                cv_csv = os.path.join(feat_dir, f"cv_scores_fold{fold_idx:02d}.csv")
                df_cv.to_csv(cv_csv, index=False)
            final_train = self.add_group_activation(
                final_train, groups, channel_cv, gamma=gamma, delta=delta, beta=beta
            )
            feat_dir = os.path.join(self.exp_dir, self.run_id, "fold_features")
            os.makedirs(feat_dir, exist_ok=True)
            train_csv = os.path.join(feat_dir, f"fold{fold_idx:02d}_train_feats.csv")
            final_train.to_csv(train_csv, index=False)
            self.event_wise_model_selection(
                train_set=final_train,
                search_cv=search_cv_factory3(),
                callbacks=callbacks,
                call_every_ms=100,
                flat=flat,
                run_id=self.make_run_id([tuple(mask1)]),
            )
            fold_handler.stop()
            fold_metrics = fold_handler.collect(reset=True)
            self.all_logs.append(
                {
                    "fold_id": fold_idx,
                    "fold_time": fold_metrics.get("time", 0.0),
                    "fold_cpu": fold_metrics.get("cpu", 0.0),
                    "fold_mem": fold_metrics.get("mem", 0.0),
                }
            )
            pd.DataFrame.from_records(self.all_logs).to_csv(
                os.path.join(self.exp_dir, self.run_id, "efficiency_log.csv"), index=False
            )
        return None
