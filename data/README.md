# Data

## Source & provenance (read this)
This competition uses the **BanglaCalamityMMD** benchmark:

> BanglaCalamityMMD: A Comprehensive Benchmark Dataset for Multimodal Disaster
> Identification in the Low-Resource Bangla Language. Mendeley Data,
> doi:10.17632/7dggbjn5sd.1 ; *Int. J. Disaster Risk Reduction*, 2025.

The published dataset has **7,903** instances across 8 categories, split into
**6,323 train / 790 test / 790 validation**. The contest exposes **6,323 train**
and **1,580 test** rows — the contest test set corresponds to the dataset's
**test + validation** splits combined. We report leakage-free 5-fold
cross-validation (OOF) on the training split as our primary metric.

The dataset is the property of its original authors under its own license and is
**not redistributed here**. Download it from Mendeley (or the contest page) and
place it as below.

## Expected layout (not committed)
```
data/
├── raw/
│   ├── Disaster_train.csv      # columns: image_id, context, category
│   ├── Disaster_test.csv       # columns: image_id, context
│   ├── sample_submission.csv
│   ├── Train/                  # training images (filename prefix = class)
│   └── Test/                   # test images (anonymized: test_0001.jpg ...)
└── folds/folds_canonical.csv   # committed: the frozen 5-fold split
```

## Build the folds
```
PYTHONPATH=src python -c "from disaster.data.folds import build_folds; build_folds('data/raw/Disaster_train.csv')"
```

## Note on filenames
Training image filenames embed the class (`drought_122.jpg`); test images are
anonymized. We do **not** use filenames as a feature at inference.
