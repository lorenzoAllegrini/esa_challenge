from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

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
        self.artifacts_dir = artifacts_dir
        self.internal_models: Dict[str, Any] = {}
        self.meta_models: Dict[str, Any] = {}
        self.event_models: List[Any] = []
        self.channel_links: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def load_models(self) -> None:
        """Load serialized models from ``artifacts_dir``."""

        model_base = os.path.join("experiments", self.artifacts_dir, "models")

        for ch_dir in glob.glob(os.path.join(model_base, "channel_*")):
         
            base = os.path.basename(ch_dir)
            ch_id = base[len("channel_") :]
            links_file = os.path.join(ch_dir, "links.json")
            if os.path.exists(links_file):
                with open(links_file, "r") as f:
                    self.channel_links[ch_id] = json.load(f)
              
            for p in glob.glob(os.path.join(ch_dir, "internal_*.pkl")):
                mid = os.path.splitext(os.path.basename(p))[0]
                self.internal_models[mid] = joblib.load(p)
            for p in glob.glob(os.path.join(ch_dir, "external_*.pkl")):
                mid = os.path.splitext(os.path.basename(p))[0]
                self.meta_models[mid] = joblib.load(p)
        for p in glob.glob(os.path.join(model_base, "event_wise_*.pkl")):
            self.event_models.append(joblib.load(p))

    # ------------------------------------------------------------------
    #  Inference helpers
    # ------------------------------------------------------------------
    def channel_specific_ensemble(self, challenge_channel: ESA, channel_id: str) -> pd.DataFrame:
        """Apply all saved estimators for ``channel_id`` on ``challenge_channel``."""
        links = self.channel_links.get(channel_id, {})
    
        if not links:
            raise RuntimeError(f"No model links found for channel {channel_id}")

        df_out: Optional[pd.DataFrame] = None
        meta_probas = []

        stats_df, _ = self.segmentator.segment_statistical(
            challenge_channel,
            masks=[],
            ensemble_id=f"stats_{channel_id}_test",
            train_phase=False,
        )
        for meta_id, info in links.items():
            internal_ids = info.get("internal_ids", [])
            internal_probas = []
            first_df: Optional[pd.DataFrame] = None
            for iid in internal_ids:
                mdl = self.internal_models[iid]
                
                df_curr, _ = mdl.segmentator.segment_shapelets(
                    df=stats_df,
                    esa_channel=challenge_channel,
                    shapelet_mask=(0, len(challenge_channel.data)),
                    ensemble_id=f"{mdl.ensemble_id}_test",
                    masks=None,
                    mode="exclude",
                    initialize=False,
                )
                p = mdl.model.predict_proba(df_curr)
                if first_df is None:
                    first_df = df_curr
                if p.shape[1] == 1:
                    cls = mdl.model.classes_[0]
                    internal_probas.append(
                        np.full(len(df_curr), 1.0 if cls == 1 else 0.0)
                    )
                else:
                    internal_probas.append(p[:, 1])

            meta_df = pd.DataFrame({f"internal_model_{i}": internal_probas[i] for i in range(len(internal_probas))})
            meta_mdl = self.meta_models[meta_id]
            mp = meta_mdl.model.predict_proba(meta_df)
            if mp.shape[1] == 1:
                cls = meta_mdl.model.classes_[0]
                mp = np.full(len(meta_df), 1.0 if cls == 1 else 0.0)
            else:
                mp = mp[:, 1]
            meta_probas.append(mp)
            if df_out is None and first_df is not None:
                df_out = first_df[["start", "end"]].copy()

        mean_proba = np.mean(meta_probas, axis=0)
        if df_out is None:
            raise RuntimeError("No channel data processed")
        df_out[channel_id] = mean_proba
        return df_out

    def run(
        self,
        mission: ESAMission,
        peak_height: float = 0.5,
        buffer_size: int = 100,
        gamma: float = 1.5,
        delta: float = 0.05,
        beta: float = 0.3,
    ) -> pd.DataFrame:
        """Execute the inference pipeline on the challenge data."""
        if not self.meta_models:
            self.load_models()

        source_folder = os.path.join(self.data_root, mission.inner_dirpath)
        meta = pd.read_csv(os.path.join(source_folder, "channels.csv")).assign(Channel=lambda d: d.Channel.str.strip())
        groups = meta[meta["Channel"].isin(mission.target_channels)].groupby("Group")["Channel"].apply(list).to_dict()

        global_df: Optional[pd.DataFrame] = None
        channel_cv: Dict[str, float] = {}
        for channel_id in mission.target_channels:
            _, challenge_channel = self.load_channel(mission, channel_id, overlapping_train=False)
            df_ch = self.channel_specific_ensemble(challenge_channel, channel_id)
            print(f"df_ch: {df_ch}")
            if global_df is None:
                global_df = df_ch
            else:
                print("else")
                global_df = global_df.merge(df_ch, on=["start", "end"], how="outer")
                print(f"global_df: {global_df}")
            channel_cv[channel_id] = 1.0

        if global_df is None:
            raise RuntimeError("No channels processed")

        global_df = self.add_group_activation(global_df, groups, channel_cv, gamma=gamma, delta=delta, beta=beta)
        X = global_df[[c for c in global_df.columns if c.startswith("group_")]]
        fold_probas = []
        for mdl in self.event_models:
            p = mdl.predict_proba(X)
            if p.shape[1] == 1:
                single = mdl.classes_[0]
                p = np.full(len(X), 1.0 if single == 1 else 0.0)
            else:
                p = p[:, 1]
            fold_probas.append(p)

        final_probas = np.max(fold_probas, axis=0) - np.var(fold_probas, axis=0)
        y_full, y_binary = self.predict_challenge_labels(
            challenge_test=global_df,
            challenge_probas=final_probas,
            peak_height=peak_height,
            buffer_size=buffer_size,
        )
        return pd.DataFrame(
            {
                "id": np.arange(len(y_full)) + self.id_offset,
                "is_anomaly": y_full,
                "pred_binary": y_binary,
            }
        )

    def predict(
        self,
        test_parquet: str,
        output: str,
        mission: Optional[ESAMission] = None,
    ) -> None:
        """Convenience wrapper to run inference given a parquet file path."""
        data_root = os.path.dirname(os.path.dirname(test_parquet))
        if mission is None:
            mission = ESAMissions.MISSION_1.value
        self.data_root = data_root
        df = self.run(mission)
        df.to_csv(output, index=False)
