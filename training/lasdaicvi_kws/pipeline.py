"""Feature building for one task: glue between `data`, `augment` and `features`.

The order of operations replays the thesis notebooks, because every step consumes a seeded
random stream and reordering would change the data:
select -> stretch (train, aug) -> place in 2 s window -> silence -> shuffle + noise
(train, aug) / shuffle + noisy copy (test) -> MFCC -> final shuffle.
Built features are cached as `.npz` so repeated training runs skip the audio work.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from . import augment, data, features
from .config import PKG_ROOT, TaskConfig

CACHE_DIR = PKG_ROOT / "cache"


@dataclass
class Features:
    """MFCC tensors (N, 61, 20, 1) and integer labels for every split, plus the class names."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_clean: np.ndarray
    y_clean: np.ndarray
    x_noisy: np.ndarray
    y_noisy: np.ndarray
    classes: list[str]

    def sizes(self) -> dict:
        """Number of clips per split."""
        return {"train": len(self.x_train), "val": len(self.x_val),
                "test_clean": len(self.x_clean), "test_noisy": len(self.x_noisy)}


def _log(msg: str) -> None:
    print(f"[data] {msg}", flush=True)


def _split_waves(split: str, task: TaskConfig, root: Path, seed: int):
    """Waveforms and labels of one split, unknown words relabelled."""
    selected = data.select_samples(data.load_split(split, root), task.labels, task.unknown, seed)
    waves = data.load_waves(selected, root)
    return waves, selected["label"].to_numpy(dtype=object)


def _window_and_silence(waves, labels, seed):
    """Place clips in the 2 s window, then append the silence clips."""
    return data.add_silence(data.place_in_window(waves, seed), labels)


def build_features(task: TaskConfig, aug: bool, thesis: bool, root: Path | None = None,
                   seed: int = data.SEED) -> Features:
    """Build all splits of `task` from the audio files.

    `aug` stretches and noises the training set; the test set always gets a clean and a noisy
    copy. `thesis` replays the notebooks' unrepeatable details (see `augment`).
    """
    root = data.resolve_dataset_root(root)
    t0 = time.time()
    if thesis:
        random.seed(seed)  # the notebooks drew the SNR from the unseeded global `random`; seed it once so runs repeat
    noise_root = root / "LaSDAICVI" / "_ruido_"

    # train
    waves, y = _split_waves("train", task, root, seed)
    _log(f"train: {len(waves)} selected clips")
    if aug:
        waves, y = augment.time_stretch_random(waves, y, augment.N_COPIES, *augment.STRETCH_RANGE, seed)
        _log(f"train: {len(waves)} after time stretch ({time.time() - t0:.0f}s)")
    X, y = _window_and_silence(waves, y, seed)
    del waves
    _log(f"train: {len(X)} with silence")
    if aug:
        perm = data.shuffle_permutation(len(y), seed)
        X, y = X[perm], y[perm]
        X = augment.add_background_noise(X, y, augment.load_noise(noise_root, sort=not thesis),
                                         seed, thesis)
    x_train = features.mfcc_features(X)
    del X
    perm = data.shuffle_permutation(len(y), seed)
    x_train, y_train = x_train[perm], y[perm]

    # validation
    waves, y = _split_waves("validation", task, root, seed)
    X, y = _window_and_silence(waves, y, seed)
    x_val = features.mfcc_features(X)
    perm = data.shuffle_permutation(len(y), seed)
    x_val, y_val = x_val[perm], y[perm]

    # test: clean and noisy copies share the same labels and the same final permutation
    waves, y = _split_waves("test", task, root, seed)
    X, y = _window_and_silence(waves, y, seed)
    perm = data.shuffle_permutation(len(y), seed)
    X, y = X[perm], y[perm]
    noisy = augment.add_background_noise(X, y, augment.load_noise(noise_root / "Test", sort=not thesis),
                                         seed, thesis)
    perm = data.shuffle_permutation(len(y), seed)
    x_clean, x_noisy = features.mfcc_features(X)[perm], features.mfcc_features(noisy)[perm]
    y_test = y[perm]
    _log(f"val {len(x_val)}, test {len(x_clean)} ({time.time() - t0:.0f}s)")

    encoder = LabelEncoder().fit(y_train)
    enc = encoder.transform
    return Features(x_train[..., None], enc(y_train), x_val[..., None], enc(y_val),
                    x_clean[..., None], enc(y_test), x_noisy[..., None], enc(y_test),
                    [str(c) for c in encoder.classes_])


def cache_path(task: str, aug: bool, thesis: bool, seed: int, cache_dir: Path = CACHE_DIR) -> Path:
    """Cache file of one feature set; the name encodes everything that changes the data."""
    return Path(cache_dir) / f"{task}_{'aug' if aug else 'base'}_{'thesis' if thesis else 'fixed'}_s{seed}.npz"


def load_or_build(task: TaskConfig, aug: bool, thesis: bool, root: Path | None = None,
                  seed: int = data.SEED, use_cache: bool = True,
                  cache_dir: Path = CACHE_DIR) -> Features:
    """`build_features` with an `.npz` cache keyed by task, mode and seed."""
    path = cache_path(task.name, aug, thesis, seed, cache_dir)
    if use_cache and path.exists():
        _log(f"loading cached features {path.name}")
        with np.load(path) as z:
            return Features(**{k: z[k] for k in z.files if k != "classes"},
                            classes=[str(c) for c in z["classes"]])
    feats = build_features(task, aug, thesis, root, seed)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in vars(feats).items() if k != "classes"}
        np.savez(path, classes=np.array(feats.classes), **arrays)
        _log(f"cached features to {path.name}")
    return feats
