"""Pytest configuration: skip artifact-dependent tests when the frozen `.npy`
arrays are not present (e.g. on a fresh clone or in CI without a Release/LFS
pull). Fold-integrity tests always run."""
import os

import pytest

_ARTIFACTS_PRESENT = os.path.exists(
    os.path.join("artifacts", "oof_stack.npy"))


def pytest_collection_modifyitems(config, items):
    if _ARTIFACTS_PRESENT:
        return
    skip = pytest.mark.skip(
        reason="frozen OOF artifacts absent — see artifacts/README.md "
               "(distribute via Git LFS / Release)")
    for item in items:
        if item.get_closest_marker("needs_artifacts"):
            item.add_marker(skip)
