import glob
import os

import joblib
import numpy as np
import pandas as pd

from spaceai.data import (
    ESA,
    ESAMissions,
)
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2

from .esa_competition import ESACompetitionBenchmark


class ESACompetitionPredictor:
    """Load training artefacts and run inference on new data."""

    def __init__(self, artifacts_dir: str, segmentator: EsaDatasetSegmentator2) -> None:
        self.artifacts_dir = artifacts_dir
        self.segmentator = segmentator
        self.channel_models: dict[str, list] = {}
        self.event_models: list = []
        self.run_id = os.path.basename(os.path.normpath(artifacts_dir))
        self.benchmark = ESACompetitionBenchmark(
            run_id=self.run_id,
            exp_dir=os.path.dirname(artifacts_dir),
            segmentator=segmentator,
        )

    def load_models(self) -> None:
        """Load serialized models from ``artifacts_dir``."""
        model_base = os.path.join(self.artifacts_dir, "models")
        for ch_dir in glob.glob(os.path.join(model_base, "channel_*")):
            ch_id = os.path.basename(ch_dir).split("_")[1]
            models = [
                joblib.load(p) for p in sorted(glob.glob(os.path.join(ch_dir, "*.pkl")))
            ]
            self.channel_models[ch_id] = models
        self.event_models = [
            joblib.load(p)
            for p in sorted(glob.glob(os.path.join(model_base, "event_wise_*.pkl")))
        ]

    def predict(self, test_parquet: str, output: str) -> None:
        """Run the inference pipeline and produce ``submission.csv``."""
        data_root = os.path.dirname(os.path.dirname(test_parquet))
        mission = ESAMissions.MISSION_1.value

        source_folder = os.path.join(data_root, mission.inner_dirpath)
        meta = pd.read_csv(os.path.join(source_folder, "channels.csv")).assign(
            Channel=lambda d: d.Channel.str.strip()
        )
        groups = (
            meta[meta["Channel"].isin(mission.target_channels)]
            .groupby("Group")["Channel"]
            .apply(list)
            .to_dict()
        )

        global_df = None
        channel_cv = {}
        for channel_id, models in self.channel_models.items():
            esa_channel = ESA(
                root=data_root,
                mission=mission,
                channel_id=f"{channel_id}",
                mode="challenge",
                train=False,
            )
            df, _ = self.segmentator.segment(
                esa_channel, masks=[], ensemble_id="challenge", train_phase=False
            )
            proba = np.mean(
                [
                    (
                        mdl.predict_proba(df)[:, 1]
                        if mdl.predict_proba(df).shape[1] > 1
                        else np.full(len(df), 1.0 if mdl.classes_[0] == 1 else 0.0)
                    )
                    for mdl in models
                ],
                axis=0,
            )
            if global_df is None:
                global_df = pd.DataFrame({"start": df["start"], "end": df["end"]})
            global_df[channel_id] = proba
            channel_cv[channel_id] = 1.0

        if global_df is None:
            raise RuntimeError("No channels loaded")

        global_df = self.benchmark.add_group_activation(global_df, groups, channel_cv)

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
        y_full, y_binary = self.benchmark.predict_challenge_labels(
            challenge_test=global_df,
            challenge_probas=final_probas,
        )
        out_df = pd.DataFrame(
            {
                "id": np.arange(len(y_full)) + self.benchmark.id_offset,
                "is_anomaly": y_full,
                "pred_binary": y_binary,
            }
        )
        out_df.to_csv(output, index=False)
