.PHONY: install smoke manifest preprocess baselines tune deep evaluate report test lint

install:
	python -m pip install -e ".[dev]"

smoke:
	python scripts/run_smoke_test.py

manifest:
	python scripts/00_make_manifest.py --config configs/m2_quick.yaml

preprocess:
	python scripts/01_preprocess.py --config configs/m2_quick.yaml

baselines:
	python scripts/02_train_baselines.py --config configs/m2_quick.yaml

tune:
	python scripts/03_tune_models.py --config configs/m2_quick.yaml

deep:
	python scripts/04_train_deep.py --config configs/m2_quick.yaml

evaluate:
	python scripts/05_evaluate.py --config configs/m2_quick.yaml

report:
	python scripts/06_make_report.py --config configs/m2_quick.yaml

test:
	pytest -q

lint:
	ruff check src scripts tests
