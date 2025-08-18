from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from spaceai.data import ESA, ESAMission, ESAMissions
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2

from .esa_competition import ESACompetitionBenchmark


class ESACompetitionPredictor(ESACompetitionBenchmark):
    """Benchmark variant used only for inference.

    It mirrors :class:`ESACompetitionBenchmark` but loads the models
    previously saved by :class:`ESACompetitionTraining` and applies them
    to new challenge data.
    """

    def __init__(
        self,
        artifacts_dir: str,
        segmentator: Optional[EsaDatasetSegmentator2],
        data_root: str = "datasets",
        id_offset: int = 14728321,
        seed: int = 42,
        mission=ESAMissions.MISSION_1.value,
    ) -> None:
        run_id = os.path.basename(os.path.normpath(artifacts_dir))
        super().__init__(
            run_id=run_id,
            exp_dir=os.path.dirname(artifacts_dir),
            segmentator=segmentator,
            data_root=data_root,
            id_offset=id_offset,
            seed=seed,
        )
        self.mission = mission
        self.artifacts_dir = artifacts_dir
        self.internal_models: Dict[str, Any] = {}
        self.internal_models_by_channel: Dict[str, Dict[str, Any]] = {}
        for ch in mission.target_channels:
            self.internal_models_by_channel[ch] = {}
        self.meta_models: Dict[str, Any] = {}
        self.meta_models_by_channel: Dict[str, Dict[str, Any]] = {}
        for ch in mission.target_channels:
            self.meta_models_by_channel[ch] = {}
        self.event_models: List[Any] = []
        self.event_models_by_mask: Dict[str, List[Any]] = {}
        self.channel_links: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def load_models(self) -> None:
        """Load serialized models from ``artifacts_dir``."""

        model_base = os.path.join(self.artifacts_dir, "models")

        files = glob.glob(model_base, recursive=True)

        for ch_dir in glob.glob(os.path.join(model_base, "channel_*")):

            base = os.path.basename(ch_dir)
            ch_id = base[len("channel_") :]

            used_file = os.path.join(ch_dir, "used_models.json")            
            allowed_internal: Optional[Set[str]] = None
            allowed_meta: Optional[Set[str]] = None
            if os.path.exists(used_file):
                with open(used_file, "r") as f:
                    used = json.load(f)
                if "internal_ids" in used or "meta_ids" in used:
                    allowed_internal = set(used.get("internal_ids", []))
                    allowed_meta = set(used.get("meta_ids", []))              
                else:
                    allowed_internal = set()
                    allowed_meta = set()
                    for info in used.values():
                        allowed_internal.update(info.get("internal_ids", []))
                        allowed_meta.update(info.get("meta_ids", []))

            links_file = os.path.join(ch_dir, "links.json")
            if os.path.exists(links_file):
                with open(links_file, "r") as f:
                    links_data = json.load(f)
                    #print(f"channel_id: {ch_id}, links_data: {links_data}")
                if allowed_meta is not None:
                    links_data = {
                        k: v for k, v in links_data.items() if k in allowed_meta
                    }
                self.channel_links[ch_id] = links_data
                #print(f"channel_id: {ch_id}, links_data: {links_data}")

            for p in glob.glob(os.path.join(ch_dir, "internal_*.pkl")):
  
                mid = os.path.splitext(os.path.basename(p))[0]
                if allowed_internal is not None and mid not in allowed_internal:
                    continue
                model = joblib.load(p)
                estimator = getattr(model, "model", model)
                feature_names: List[str] = []
                if hasattr(estimator, "feature_names_in_"):
                    feature_names = list(estimator.feature_names_in_)
                else:
                    booster = getattr(estimator, "get_booster", None)
                    if booster is not None:
                        try:
                            feature_names = booster().feature_names or []
                        except Exception:  # pragma: no cover - safeguard
                            feature_names = []
                if any(f in {"event", "start", "end"} for f in feature_names):
                    # skip legacy models expecting removed features
                    continue
                self.internal_models[mid] = model
                self.internal_models_by_channel[ch_id][mid] = model
            for p in glob.glob(os.path.join(ch_dir, "external_*.pkl")):
                mid = os.path.splitext(os.path.basename(p))[0]
                if allowed_meta is not None and mid not in allowed_meta:
                    continue
                model = joblib.load(p)
                self.meta_models[mid] = model
                self.meta_models_by_channel[ch_id][mid] = model

        self.event_models_by_mask = defaultdict(list)
   
        for p in glob.glob(os.path.join(model_base, "event_wise_*.pkl")):
       
            mask_id = os.path.splitext(os.path.basename(p))[0].split("event_wise_")[1]
            model = joblib.load(p)
            self.event_models.append(model)
            self.event_models_by_mask[mask_id].append(model)

    # ------------------------------------------------------------------
    #  Inference helpers
    # ------------------------------------------------------------------
    def channel_specific_ensemble(
        self,
        challenge_channel: ESA,
        channel_id: str,
        mask_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply saved estimators for ``channel_id`` on ``challenge_channel``.

        If ``mask_id`` is provided only models linked to that mask are considered.
        This mirrors the hierarchical structure used during training where each
        external mask holds a subset of meta models and related internal
        estimators.
        """
        links = self.channel_links.get(channel_id, {})
  
        if not links:
            raise RuntimeError(f"No model links found for channel {channel_id}")

        valid_links = {}
        for meta_id, info in links.items():
          
            if mask_id is not None and info.get("mask_id", "default") != mask_id:
                continue
            if meta_id not in self.meta_models_by_channel[channel_id]:
                continue
            internal_ids_full = info.get("internal_ids", [])
            internal_ids = [
                iid
                for iid in internal_ids_full
                if iid in self.internal_models_by_channel[channel_id]
            ]
            if len(internal_ids) != len(internal_ids_full):
                continue
            if internal_ids:
                valid_links[meta_id] = {
                    "internal_ids": internal_ids,
                    "mask_id": info.get("mask_id", "default"),
                }
        if not valid_links:
            raise RuntimeError(
                f"No serialized models available for channel {channel_id} and mask {mask_id}"
            )

        df_out: Optional[pd.DataFrame] = None
        mask_probas: Dict[str, List[np.ndarray]] = defaultdict(list)

        stats_df, _ = self.segmentator.segment_statistical(
            challenge_channel,
            masks=[],
            ensemble_id=f"stats_{channel_id}_test",
            train_phase=False,
        )

        for meta_id, info in valid_links.items():
            internal_ids = info.get("internal_ids", [])
            curr_mask = info.get("mask_id", "default")
            internal_probas = []
            first_df: Optional[pd.DataFrame] = None
            skip_meta = False
            for iid in internal_ids:
                mdl = self.internal_models_by_channel[channel_id][iid]
                estimator = getattr(mdl, "model", mdl)
                feature_names: List[str] = []
                if hasattr(estimator, "feature_names_in_"):
                    feature_names = list(estimator.feature_names_in_)
                else:
                    booster = getattr(estimator, "get_booster", None)
                    if booster is not None:
                        try:
                            feature_names = booster().feature_names or []
                        except Exception:  # pragma: no cover
                            feature_names = []
                if any(f in {"event", "start", "end"} for f in feature_names):
                    skip_meta = True
                    break

                df_curr, _ = mdl.segmentator.segment_shapelets(
                    df=stats_df,
                    esa_channel=challenge_channel,
                    shapelet_mask=(0, len(challenge_channel.data)),
                    ensemble_id=f"{mdl.ensemble_id}_test",
                    masks=None,
                    mode="exclude",
                    initialize=False,
                )
                if first_df is None:
                    first_df = df_curr

                df_curr = df_curr.drop(
                    columns=["event", "start", "end"], errors="ignore"
                )

                p = mdl.model.predict_proba(df_curr)

                if p.shape[1] == 1:
                    cls = mdl.model.classes_[0]
                    internal_probas.append(
                        np.full(len(df_curr), 1.0 if cls == 1 else 0.0)
                    )
                else:
                    internal_probas.append(p[:, 1])
            if skip_meta or not internal_probas:
                continue
            meta_df = pd.DataFrame(
                {
                    f"channel_{i}": internal_probas[i]
                    for i in range(len(internal_probas))
                }
            )
            meta_mdl = self.meta_models_by_channel[channel_id][meta_id]
            mp = meta_mdl.model.predict_proba(meta_df)
            if mp.shape[1] == 1:
                cls = meta_mdl.model.classes_[0]
                mp = np.full(len(meta_df), 1.0 if cls == 1 else 0.0)
            else:
                mp = mp[:, 1]
            mask_probas[curr_mask].append(mp)
            if df_out is None and first_df is not None:
                df_out = first_df[["start", "end"]].copy()

        if not mask_probas or df_out is None:
            raise RuntimeError("No channel data processed")
        for m_id, probas in mask_probas.items():
            mean_proba = np.mean(probas, axis=0)
            df_out[f"{channel_id}_{m_id}"] = mean_proba
        return df_out

    def run(
        self,
        mission: ESAMission,
        test_parquet: Optional[str] = None,
        peak_height: float = 0.5,
        buffer_size: int = 100,
        gamma: float = 1.5,
        delta: float = 0.05,
        beta: float = 0.3,
    ) -> pd.DataFrame:
        """Execute the inference pipeline on the challenge data.

        Args:
            mission: ESA mission metadata.
            test_parquet: Optional path to a challenge parquet file.
        """
        if not self.meta_models:
            self.load_models()

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

        mask_ids = sorted(self.event_models_by_mask.keys())
   
        mask_dfs: Dict[str, pd.DataFrame] = {}

        # Load cross-validation scores saved during training. Multiple folds
        # may exist, therefore average them for each channel.
        cv_files = glob.glob(os.path.join(self.artifacts_dir, "cv_scores_fold*.csv"))
        cv_raw: Dict[str, List[float]] = defaultdict(list)
        for csv_path in cv_files:
            df_cv = pd.read_csv(csv_path)
            for _, row in df_cv.iterrows():
                ch = str(row["channel"])
                cv_raw[ch].append(float(row["cv_score"]))

        challenge_channels: Dict[str, ESA] = {}
        
        for channel_id in mission.target_channels:
            
        
            challenge_channels[channel_id] = ESA(
                root=self.data_root,
                mission=mission,
                channel_id=channel_id,
                mode="challenge",
                overlapping=False,
                seq_length=250,
                train=False,
                drop_last=False,
                n_predictions=1,
                challenge_parquet=test_parquet,
                download=False,
            )
        if len(mask_ids) == 0:
            raise RuntimeError("len 0")

        for mask_id in tqdm(mask_ids, desc="Masks"):
          
            for channel_id, challenge_channel in tqdm(
                challenge_channels.items(), desc="Channels", leave=False
            ):
                try:
                        df_ch = self.channel_specific_ensemble(
                            challenge_channel, channel_id, mask_id=mask_id
                        )

                
                except RuntimeError as e:
                    raise RuntimeError(f"errore channel specific ensemble: {e}")
                if mask_id not in mask_dfs:
                    mask_dfs[mask_id] = df_ch[["start", "end"]].copy()
                for col in [c for c in df_ch.columns if c not in {"start", "end"}]:
                    mask_dfs[mask_id][col] = df_ch[col]
                #mask_dfs[mask_id].to_csv("mask_dfs")

        if not mask_dfs:
            raise RuntimeError("No channels processed")
            

        fold_probas = []
        challenge_df: Optional[pd.DataFrame] = None
        for mask_id, df_mask in mask_dfs.items():
            channel_cv = pd.read_csv(
                os.path.join(self.artifacts_dir, f"cv_scores_fold{mask_id}.csv")
            )
            channel_cv_set = {}
            for _,row in channel_cv.iterrows():
                channel_cv_set[row["channel"]] = row["cv_score"]
            rename_map = {
                c: c.rsplit("_", 1)[0]
                for c in df_mask.columns
                if c.startswith("channel_")
            }
            df_renamed = df_mask.rename(columns=rename_map)
            df_aug = self.add_group_activation(
                df_renamed, groups, channel_cv_set, gamma=gamma, delta=delta, beta=beta
            )
            #df_aug.to_csv("df_aug.csv")
            X = df_aug[[c for c in df_aug.columns if c.startswith("group_")]]
            for mdl in self.event_models_by_mask.get(mask_id, []):
             
                p = mdl.predict_proba(X)
            
                if p.shape[1] == 1:
                    single = mdl.classes_[0]
                    p = np.full(len(X), 1.0 if single == 1 else 0.0)
                else:
                    p = p[:, 1]
                fold_probas.append(p)
            if challenge_df is None:
                challenge_df = df_mask
       
        final_probas = np.max(fold_probas, axis=0) #- np.var(fold_probas, axis=0)
        y_full, y_binary = self.predict_challenge_labels(
            challenge_test=(
                challenge_df
                if challenge_df is not None
                else next(iter(mask_dfs.values()))
            ),
            challenge_probas=final_probas,
            peak_height=peak_height,
            buffer_size=buffer_size,
        )
        return pd.DataFrame(
            {
                "id": np.arange(len(y_full)) + self.id_offset,
                "is_anomaly": y_binary,
            }
        )

    def predict(
        self,
        test_parquet: str,
        output: str,
        mission: Optional[ESAMission] = None,
    ) -> None:
        """Convenience wrapper to run inference given a parquet file path."""
        if mission is None:
            mission = ESAMissions.MISSION_1.value
        df = self.run(mission, test_parquet=test_parquet)
        #df.to_csv(output, index=False)
        return df
