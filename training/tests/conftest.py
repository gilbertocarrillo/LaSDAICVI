"""Shared test helpers."""
import pytest

from lasdaicvi_kws import data


def _dataset_root():
    try:
        return data.resolve_dataset_root()
    except FileNotFoundError:
        return None


DATASET_ROOT = _dataset_root()
needs_dataset = pytest.mark.skipif(
    DATASET_ROOT is None, reason="dataset not found (set LASDAICVI_ROOT)")
