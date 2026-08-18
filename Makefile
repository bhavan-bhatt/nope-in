.PHONY: install data-smoke data features features-smoke regime regime-smoke train train-smoke eval eval-smoke validate test clean pre-commit

VENV = .venv/bin

install:
	python3 -m venv .venv
	$(VENV)/pip install -e ".[dev]" || $(VENV)/pip install -e . pytest mypy black isort pre-commit

pre-commit:
	$(VENV)/pre-commit install

data-smoke:
	$(VENV)/python scripts/00_smoke_test_data.py

data:
	$(VENV)/python scripts/01_build_all_data.py

features:
	$(VENV)/python -m nope_in.features.pipeline

features-smoke:
	$(VENV)/python scripts/02_features_smoke_test.py

regime:
	$(VENV)/python -m nope_in.regime.pipeline

regime-smoke:
	$(VENV)/python scripts/03_regime_smoke_test.py

train:
	$(VENV)/python -m nope_in.training.pipeline

train-smoke:
	$(VENV)/python scripts/04_train_smoke_test.py

eval:
	$(VENV)/python -m nope_in.evaluation.pipeline

eval-smoke:
	$(VENV)/python scripts/05_eval_smoke_test.py

validate:
	$(VENV)/python -m nope_in.data.validators.schema_validator

test:
	$(VENV)/pytest tests/ -v

clean:
	rm -rf data/raw/bhav_copy data/raw/ohlcv data/raw/vix
	rm -f data/processed/*.parquet
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
