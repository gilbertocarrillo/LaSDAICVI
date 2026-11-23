"""Augmentation: seeded time stretch (train) and background-noise mixing (train and noisy test).

Second stage of the pipeline. Each function replays the thesis' `data_augmentation.py`;
`thesis=True` keeps the one place where the notebooks were not repeatable (see
`add_background_noise`).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from .data import CLIP_SAMPLES, SEED, SILENCE_LABEL, STRETCH_COPIES, load_wave

N_COPIES = STRETCH_COPIES
STRETCH_RANGE = (0.85, 1.15)
NOISE_PROB = 0.8
SNR_RANGE_DB = (5.0, 30.0)


def _librosa_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Phase-vocoder time stretch (keyword-only `rate`, as modern librosa requires)."""
    import librosa
    return np.asarray(librosa.effects.time_stretch(y, rate=rate), dtype=np.float32)


def time_stretch_random(waves, labels, n_copies: int = N_COPIES, min_rate: float = STRETCH_RANGE[0],
                        max_rate: float = STRETCH_RANGE[1], seed: int = SEED, stretch_fn=None):
    """Originals first, then `n_copies` stretched copies per clip, row by row.

    Rates come from one `random.Random(seed)` stream, `uniform(min_rate, max_rate)` per copy,
    as in the original `time_stretch_random`. Returns (waves, labels).
    """
    stretch_fn = stretch_fn or _librosa_stretch
    rng = random.Random(seed)
    out_waves, out_labels = list(waves), list(labels)
    for wave, label in zip(waves, labels):
        for _ in range(n_copies):
            out_waves.append(np.asarray(stretch_fn(wave, rng.uniform(min_rate, max_rate)),
                                        dtype=np.float32))
            out_labels.append(label)
    return out_waves, np.asarray(out_labels, dtype=object)


def noise_files(directory: Path, sort: bool = True) -> list[Path]:
    """`.wav` files directly inside `directory`; `sort=False` keeps `os.listdir` order (thesis)."""
    names = [n for n in os.listdir(directory) if n.endswith(".wav")]
    if sort:
        names.sort()
    return [Path(directory) / n for n in names]


def load_noise(directory: Path, sort: bool = True) -> list[np.ndarray]:
    """Waveforms of every `.wav` directly inside `directory`."""
    files = noise_files(directory, sort)
    if not files:
        raise FileNotFoundError(f"no .wav noise files in {directory}")
    return [load_wave(f) for f in files]


def _rms(x: np.ndarray) -> float:
    """Root mean square in float64."""
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def add_background_noise(X: np.ndarray, labels, noises: list[np.ndarray], seed: int = SEED,
                         thesis: bool = False, prob: float = NOISE_PROB,
                         snr_db=SNR_RANGE_DB, silence_label: str = SILENCE_LABEL) -> np.ndarray:
    """Mix a random noise crop into each 2 s clip with probability `prob`; returns a new array.

    Per clip (in order): `rng.uniform(0, 1) < prob`, noise file `rng.randint`, crop start
    `rng.randint(0, len(noise) - CLIP - 1)`, SNR `uniform(*snr_db)`. The noise is scaled so
    that `rms(noise) = rms(clip) / 10^(snr/20)`; silence clips use the mean RMS of the
    non-silence clips. No clipping. `rng = Random(seed)`; the SNR comes from `rng` too, except
    in `thesis` mode where it comes from the module-level `random` (as in the original;
    seed it beforehand to make runs repeat).
    """
    labels = np.asarray(labels, dtype=object)
    rng = random.Random(seed)
    snr_source = random if thesis else rng
    non_silence = labels != silence_label
    mean_rms = float(np.mean([_rms(c) for c in X[non_silence]])) if non_silence.any() else 0.0
    out = X.astype(np.float32, copy=True)
    for i in range(len(out)):
        if not rng.uniform(0, 1) < prob:
            continue
        noise = noises[rng.randint(0, len(noises) - 1)]
        snr = snr_source.uniform(*snr_db)
        start = rng.randint(0, len(noise) - CLIP_SAMPLES - 1)
        crop = noise[start:start + CLIP_SAMPLES].astype(np.float64)
        crop_rms = _rms(crop)
        if crop_rms == 0.0:
            continue  # silent crop: nothing to mix (the original would divide by zero)
        signal_rms = mean_rms if labels[i] == silence_label else _rms(out[i])
        gain = signal_rms / (10 ** (snr / 20)) / crop_rms
        out[i] = (out[i].astype(np.float64) + crop * gain).astype(np.float32)
    return out
