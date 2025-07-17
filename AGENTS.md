# Development Guidelines

This project implements a hierarchical ensemble for the ESA Anomaly Detection Benchmark. The main training entry point is `ESACompetitionTraining` in `spaceai/benchmark/esa_competition_training.py` or the script `examples/esa_competition_training.py`.

## Training workflow
1. **Dataset preparation** – download the ESA-ADB dataset and place it inside the `datasets/` directory. Missions and channel information are structured as described in the repository.
2. **Feature extraction** – telemetry channels are segmented with `EsaDatasetSegmentator2`. Statistical features and shapelet responses are extracted. Rolling pooling can be enabled to smooth the series.
3. **Channel ensembles** – for each target channel, multiple `internal` estimators are trained on masked subsequences. Their outputs are combined by a `meta` model. Several repetitions produce `external` estimators.
4. **Cross-channel aggregation** – probability scores of all channels are normalized and aggregated by group. A final event-wise classifier detects start and end of anomaly intervals.
5. **Serialization** – trained models together with the segmentator are stored under `experiments/<run_id>/models` so that they can be reused for inference.

A single training run is triggered via `python esa_competition_training.py --run-id <id> --data-root <datasets>` or by calling `ESACompetitionTraining.run()` programmatically.

## Testing the workflow
After any modification to the repository **always run the tests** to verify that the training and inference pipeline still works. The suite in `tests/test_workflow.py` creates a small synthetic dataset, runs a complete training round and checks that all artefacts are generated correctly.

Execute the following commands before committing:

```bash
poetry install --with test  # install optional test dependencies
pytest
```

Only commit changes once all tests pass. Keeping the workflow green ensures that the reference implementation remains reliable for future research and competition submissions.

## Deployment
To make your solution portable provide a Docker image containing the trained models and inference code. Share the image privately (e.g. with a link to Google Drive, OneDrive or Dropbox). When executed the container must load `test.parquet` from a mounted directory and produce `submission.csv` in the same location. The repository includes a sample `Dockerfile` that installs all dependencies and runs `inference.py`. Customize it as needed before building the image:

```bash
docker build -t esa-inference .
# usage example
docker run --rm -v /path/to/data:/data -v /path/to/artifacts:/artifacts esa-inference
```
