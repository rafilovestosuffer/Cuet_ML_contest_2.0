"""Pytest configuration.

The host dataset and the frozen `.npy` arrays are not committed (the dataset is
proprietary to the contest organizers; the arrays are distributed via Release /
Git LFS). So tests skip gracefully when their inputs are absent:

  * fold file missing      → skip every test (all of them read it)
  * fold file present but
    artifacts missing       → skip only the `needs_artifacts` tests

On a full local checkout (folds regenerated via `make folds` + artifacts present)
the entire suite runs.
"""
import os

import pytest

_FOLDS_PRESENT = os.path.exists(
    os.path.join("data", "folds", "folds_canonical.csv"))
_ARTIFACTS_PRESENT = os.path.exists(
    os.path.join("artifacts", "oof_stack.npy"))


def pytest_collection_modifyitems(config, items):
    skip_no_data = pytest.mark.skip(
        reason="canonical folds absent — regenerate via `make folds` "
               "(host dataset is not redistributed; see data/README.md)")
    skip_no_artifacts = pytest.mark.skip(
        reason="frozen OOF artifacts absent — see artifacts/README.md "
               "(distribute via Git LFS / Release)")
    for item in items:
        if not _FOLDS_PRESENT:
            item.add_marker(skip_no_data)
        elif not _ARTIFACTS_PRESENT and item.get_closest_marker("needs_artifacts"):
            item.add_marker(skip_no_artifacts)
