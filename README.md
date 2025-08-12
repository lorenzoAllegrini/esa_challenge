# ESA Anomaly Detection Challenge

This repository contains the reference implementation used for the ESA Anomaly Detection Benchmark (ESA-ADB) competition. The project implements a hierarchical ensemble algorithm for detecting anomalies in multivariate satellite telemetry.

## Abstract
This work presents a hierarchical ensemble pipeline for anomaly detection in multivariate satellite telemetry data provided by the European Space Agency (ESA). The method integrates shapelet-based and statistical feature extraction, per-channel modeling, intra-channel stacking, and a final cross-channel aggregation. The pipeline is trained and validated using time-series cross-validation and two-level masking strategies to prevent information leakage. Results on the European Space Agency Anomaly Detection Benchmark (ESA-ADB) challenge demonstrate strong generalization, highlighting the effectiveness of hierarchical modeling in detecting subtle anomalies in realistic satellite telemetry.

## Algorithm Overview
The training procedure is composed of three stages:

1. **Feature extraction** – Each telemetry channel is segmented using `EsaDatasetSegmentator2`. The segmentator extracts statistical features (mean, std, min, max, etc.) and shapelet responses. Rolling pooling can optionally be applied to smooth the series.
2. **Channel ensembles** – For every target channel a pool of `internal` models is fitted on random subsequences (masking prevents leakage). Their predicted probabilities are then combined by a `meta` classifier (e.g. logistic regression). The process is repeated multiple times to obtain `external` estimators.
3. **Cross-channel aggregation** – The probability outputs from all channels are normalized and aggregated into group scores. A final event-wise classifier learns to detect the start and end of anomaly intervals based on these aggregated features.

Two nested levels of masking along with time series cross validation are used during training to mimic the unseen challenge set and to avoid any form of information leakage.

## Usage
Install the dependencies with [Poetry](https://python-poetry.org/):
```bash
pip install poetry
poetry install
```

### Training
Download the ESA-ADB dataset and run:
```bash
python esa_competition_training.py --run-id my_run --data-root /path/to/datasets
```
The script will create an `experiments/my_run` directory containing the trained models and intermediate files.

### Inference
Once training has completed you can generate predictions on the challenge set with:
```bash
python inference.py --artifacts-dir experiments/my_run --test-parquet /path/to/test.parquet --output submission.csv
```
The resulting `submission.csv` follows the format required by the competition.

### External masks
During training multiple external masking strategies may be applied. Each meta-model and event-wise model is tagged with a
`mask_id` identifying the mask used during its creation. At inference time the predictor groups meta-models by `mask_id`,
averages their probabilities and creates one channel column per mask (e.g. `channel_41_ab12cd34`). Event-wise models stored as
`event_wise_<mask_id>.pkl` are matched with their corresponding columns to produce the final anomaly probabilities.

## Repository Structure
- `esa_competition_training.py` – entry point for training the hierarchical ensemble.
- `inference.py` – script used to run inference and produce the final submission.
- `spaceai/` – core library containing dataset loaders, segmentators and model utilities.
- `examples/` – additional experiment scripts.

This implementation is independent of the original SpaceAI project and focuses solely on the ESA Anomaly Detection Benchmark.
