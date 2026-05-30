.PHONY: help setup folds reproduce figures test submission stack clean

PY=PYTHONPATH=src python

help:
	@echo "setup        - pip install -r requirements.txt"
	@echo "folds        - rebuild data/folds/folds_canonical.csv from the host train CSV"
	@echo "reproduce    - reproduce ensemble macro-F1 from frozen OOF artifacts"
	@echo "figures      - generate all evaluation figures into results/figures/"
	@echo "test         - run pytest (fold integrity + F1 reproduction)"
	@echo "submission   - build submission.csv from frozen test arrays"
	@echo "stack        - rebuild the LightGBM stack OOF from the base OOF arrays"
	@echo "clean        - remove __pycache__"

setup:
	pip install -r requirements.txt

folds:
	$(PY) -c "from disaster.data.folds import build_folds; \
	          build_folds('data/raw/Disaster_train.csv')"

reproduce:
	$(PY) scripts/reproduce_ensemble.py

figures:
	$(PY) scripts/reproduce_ensemble.py --figures

test:
	$(PY) -m pytest tests/ -v

submission:
	$(PY) -m disaster.infer.predict \
	    --test-csv data/raw/Disaster_test.csv \
	    --sample-submission data/raw/sample_submission.csv \
	    --out results/tables/submission.csv \
	    --corrections results/tables/applied_corrections.csv

stack:
	$(PY) -c "from disaster.ensemble.stacking import train_stacker; \
	          import numpy as np, pandas as pd; \
	          y = pd.read_csv('data/folds/folds_canonical.csv', \
	              encoding='utf-8-sig')['label'].values; \
	          folds = pd.read_csv('data/folds/folds_canonical.csv', \
	              encoding='utf-8-sig')['fold'].values; \
	          names = ['banglabert_base','banglabert_multi','muril_large', \
	                   'eva02_large','fusion_eva_muril','pl_muril']; \
	          X = np.hstack([np.load(f'artifacts/oof_{n}.npy') for n in names]); \
	          oof, _ = train_stacker(X, y, folds); \
	          np.save('artifacts/oof_stack.npy', oof); \
	          print('Stack OOF saved.')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
