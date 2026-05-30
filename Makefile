.PHONY: help setup folds reproduce figures test submission train-text train-vision \
        train-fusion pseudo-label stack clean

PY=PYTHONPATH=src python

help:
	@echo "setup        - pip install -r requirements.txt"
	@echo "folds        - build data/folds/folds_canonical.csv from raw train CSV"
	@echo "reproduce    - reproduce ensemble macro-F1 from frozen OOF artifacts"
	@echo "figures      - generate all paper figures into results/figures/"
	@echo "test         - run pytest (fold integrity + F1 reproduction)"
	@echo "submission   - build submission.csv from test arrays (needs data/Test/test.csv)"
	@echo "train-text   - retrain text encoders (GPU required)"
	@echo "train-vision - retrain vision encoders (GPU required)"
	@echo "train-fusion - retrain fusion models (GPU required)"
	@echo "pseudo-label - select pseudo-labels + re-finetune PL-MuRIL"
	@echo "stack        - run LightGBM stacking meta-learner"
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
	    --test-csv data/Test/test.csv \
	    --out results/tables/submission.csv \
	    --corrections results/tables/applied_corrections.csv

train-text:
	$(PY) -m disaster.train.train_text --config configs/text_muril_large.yaml
	$(PY) -m disaster.train.train_text --config configs/text_banglabert_base.yaml
	$(PY) -m disaster.train.train_text --config configs/text_banglabert_multi.yaml

train-vision:
	$(PY) -m disaster.train.train_vision --config configs/vision_eva02_large.yaml

train-fusion:
	$(PY) -m disaster.train.train_fusion --config configs/fusion_eva02_muril.yaml

pseudo-label:
	$(PY) -m disaster.train.pseudo_label --config configs/pseudo_label_muril.yaml

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
