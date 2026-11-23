"""T1: split loading, unknown/silence sampling and window placement."""
import hashlib
import random

import numpy as np
import pandas as pd
import pytest

from lasdaicvi_kws import data

from conftest import DATASET_ROOT as ROOT, needs_dataset

COMMANDS = ["si", "no", "verdadero", "falso", "detente", "siguiente",
            "aprender", "repite", "jugar", "continúa"]
UNKNOWN = ["pepe"] + [str(d) for d in range(10)]


def test_silence_count_uses_the_thesis_formula():
    assert data.silence_count(18991, 11) == 1727
    assert data.silence_count(1294, 11) == 118
    assert data.silence_count(1315, 11) == 120


def test_shuffle_permutation_matches_pandas_sample():
    df = pd.DataFrame({"a": range(50)})
    expected = df.sample(frac=1, random_state=931001).index.to_numpy()
    assert (data.shuffle_permutation(50, 931001) == expected).all()


def test_place_in_window_matches_the_original_rng_stream():
    waves = [np.ones(100, dtype=np.float32), np.ones(5000, dtype=np.float32)]
    X = data.place_in_window(waves, seed=7)
    rng = random.Random(7)
    assert X.shape == (2, data.CLIP_SAMPLES) and X.dtype == np.float32
    for wave, row in zip(waves, X):
        start = rng.randint(0, data.CLIP_SAMPLES - len(wave) - 1)
        assert row[start:start + len(wave)].sum() == len(wave)
        assert row.sum() == len(wave)


def test_place_in_window_is_recreated_per_call():
    waves = [np.ones(10, dtype=np.float32)] * 3
    assert (data.place_in_window(waves, 5) == data.place_in_window(waves, 5)).all()


@needs_dataset
@pytest.mark.parametrize("split, targets, unknown, stretched, total", [
    ("train", 2460, 253, 18991, 20718),
    ("validation", 1173, 121, None, 1412),
    ("test", 1194, 121, None, 1435),
])
def test_interaction_counts(split, targets, unknown, stretched, total):
    df = data.load_split(split, ROOT)
    sel = data.select_samples(df, COMMANDS, UNKNOWN, seed=931001)
    assert (sel["label"] == data.UNKNOWN_LABEL).sum() == unknown
    assert sel["label"].isin(COMMANDS).sum() == targets
    counts = data.sample_counts(sel, augment=stretched is not None)
    assert counts["selected"] == targets + unknown
    if stretched is not None:
        assert counts["stretched"] == stretched
    assert counts["total"] == total


@needs_dataset
@pytest.mark.parametrize("split, rows, sha256", [
    ("train", 14737, "f67cb1db5a3d05f8935a8420326eff35c9fb55bb79ac397a27b897db3879fd29"),
    ("validation", 7210, "4fda34140a9372a46866f6455ea7d4209036c78d0b759a851b1ee3ba16b865ed"),
    ("test", 7114, "b17b634b0d5b3ab3e2c94030042dd4bf14581ee6eefc5aed93763483779c91a9"),
])
def test_split_row_order_matches_the_thesis_split_builder(split, rows, sha256):
    """Row order drives the seeded unknown draw, so it must equal the thesis builder's.

    Golden hashes of the newline-joined `path` column, computed from the thesis' seeded split
    builder (grades ascending, then gender, `lasdaicvi.csv` order inside each group).
    """
    paths = data.load_split(split, ROOT)["path"].tolist()
    assert len(paths) == rows
    assert hashlib.sha256("\n".join(paths).encode()).hexdigest() == sha256


@needs_dataset
def test_pepe_thesis_unknown_bug_drops_label_78_and_everything_after_it():
    df = data.load_split("train", ROOT)
    pepe_unknown_thesis = ["si", "no", "verdadero", "falso", "detente", "siguiente",
                           "aprender", "repite", "jugar", "continúa",
                           "0", "1", "2", "3", "4", "5", "6", "78", "9"]
    fixed = [w for w in pepe_unknown_thesis if w != "78"]
    fixed.insert(fixed.index("9"), "8")
    fixed.insert(fixed.index("8"), "7")
    bugged = data.select_samples(df, ["pepe"], pepe_unknown_thesis, seed=931001)
    ok = data.select_samples(df, ["pepe"], fixed, seed=931001)
    assert "9" not in set(bugged["source_label"]) and "7" not in set(bugged["source_label"])
    assert {"7", "8", "9"} <= set(ok["source_label"])
    assert len(ok) > len(bugged)


def test_dataset_root_resolution(tmp_path, monkeypatch):
    (tmp_path / "splits").mkdir()
    (tmp_path / "lasdaicvi.csv").write_text("path\n")
    monkeypatch.delenv(data.DATASET_ENV_VAR, raising=False)
    assert data.resolve_dataset_root(tmp_path) == tmp_path  # explicit argument
    monkeypatch.setenv(data.DATASET_ENV_VAR, str(tmp_path))
    assert data.resolve_dataset_root() == tmp_path  # environment variable
    with pytest.raises(FileNotFoundError, match="--dataset-root"):
        data.resolve_dataset_root(tmp_path / "missing")
