.DEFAULT_GOAL := check

clean:
	rm -rf docs/source/api/_autosummary

check:
	pre-commit run -a

isort:
	pycln --config=pyproject.toml spaceai
	isort spaceai

changelog:
	cz bump --changelog

install:
	python3 -m venv .buildenv
	.buildenv/bin/pip install poetry
	.buildenv/bin/poetry install
	rm -rf .buildenv
run:
	@KMP_WARNINGS=0 OMP_NESTED=FALSE python examples/esa_competition_experiment.py
