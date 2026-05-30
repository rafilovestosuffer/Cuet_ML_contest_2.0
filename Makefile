.PHONY: help setup folds reproduce figures test clean
PY=PYTHONPATH=src python

help:
	@echo "setup     - pip install -r requirements.txt"
	@echo "folds     - build data/folds/folds_canonical.csv from raw train csv"
	@echo "reproduce - reproduce ensemble macro-F1 from artifacts/"
	@echo "test      - run pytest"

setup:
	pip install -r requirements.txt

folds:
	$(PY) -c "from disaster.data.folds import build_folds; build_folds('data/raw/Disaster_train.csv')"

reproduce:
	$(PY) scripts/reproduce_ensemble.py

test:
	$(PY) -m pytest tests/ -q

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
