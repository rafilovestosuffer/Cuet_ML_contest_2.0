# Data

## Source & provenance

The dataset for this contest is a **custom multimodal Bengali disaster dataset
assembled by the contest host** specifically for the *Intra CUET ML Contest 2.0*.
It is a host-curated combination and is **proprietary to the organizers**.

- It is **not** a redistribution of any single public benchmark, and no public
  provenance (DOI, external split counts, etc.) is claimed here.
- The dataset — including the image–caption pairs, the category labels, and the
  canonical fold assignments derived from them — is **not committed to this
  repository** and is **not redistributed**.
- Obtain the data only from the official contest page.

## Expected local layout (not committed)

```
data/
├── raw/
│   ├── Disaster_train.csv      # columns: image_id, context, category
│   ├── Disaster_test.csv       # columns: image_id, context
│   ├── sample_submission.csv
│   ├── Train/                  # training images
│   └── Test/                   # test images
└── folds/folds_canonical.csv   # regenerated locally (see below) — NOT committed
```

## Regenerate the canonical 5-fold split

The split is deterministic (`StratifiedKFold`, shuffle, seed 42), so anyone with
the host-provided training CSV can reproduce `folds_canonical.csv` identically:

```bash
make folds
# or:
PYTHONPATH=src python -c "from disaster.data.folds import build_folds; build_folds('data/raw/Disaster_train.csv')"
```

Because the fold file embeds the host's captions and labels, it is treated as
host data and is git-ignored — regenerate it locally rather than expecting it in
the repo.
