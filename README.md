# ESA Challenge Benchmark
Repository providing utilities and reference implementations for the ESA telemetry anomaly detection challenge.


### Getting started

**Installing this codebase requires Python 3.10 or 3.11.**
Run the following commands within your python virtual environment:

```sh
pip install poetry

git clone https://github.com/continualist/esa_challenge.git
cd esa_challenge
# install runtime dependencies
poetry install
# install optional testing dependencies
poetry install --with test
poetry run pre-commit install
# build the Cython extension used by the segmentators
python spaceai/segmentators/setup.py build_ext --inplace
```

### Running the examples

To train a baseline model on the ESA dataset run:

```sh
poetry run python examples/esa_competition_experiment.py
```

You can then perform inference using:

```sh
poetry run python examples/labeller.py
```
