"""Dataset construction: locating the dataset, split loading, unknown/silence samples and
2 s window placement.

First stage of the pipeline (`data` -> `augment` -> `features` -> `models` -> `evaluate`).
Everything here replays the `input_data.py` helper of the thesis notebooks, so that set
sizes, seeded draws and their order stay identical to the published results.
"""
from __future__ import annotations

import math
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SAMPLE_RATE = 16000
CLIP_SAMPLES = 2 * SAMPLE_RATE  # clips live inside a 2 s window
SEED = 931001
STRETCH_COPIES = 6

UNKNOWN_LABEL = "_desconocido_"
SILENCE_LABEL = "_silencio_"

DATASET_ENV_VAR = "LASDAICVI_ROOT"


# --------------------------------------------------------------------------- dataset root

def _is_dataset_root(path: Path) -> bool:
    return (path / "lasdaicvi.csv").is_file() and (path / "splits").is_dir()


def resolve_dataset_root(explicit: Path | str | None = None) -> Path:
    """Locate the dataset repository (the directory holding `lasdaicvi.csv` and `splits/`).

    Precedence: the `explicit` path (`--dataset-root`), the `LASDAICVI_ROOT` environment
    variable, then the first directory above this package that looks like the dataset. The
    last rule makes the code work out of the box when it lives in the dataset repo itself
    (as `training/`).

    Raises:
        FileNotFoundError: if no candidate is a dataset root.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    elif os.environ.get(DATASET_ENV_VAR):
        candidates.append(Path(os.environ[DATASET_ENV_VAR]).expanduser())
    else:
        candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if _is_dataset_root(candidate):
            return candidate
    raise FileNotFoundError(
        "LaSDAICVI dataset not found. Pass --dataset-root, set LASDAICVI_ROOT, or run from a "
        "checkout where this code sits inside the dataset repository (the directory with "
        "lasdaicvi.csv and splits/).")


# --------------------------------------------------------------------------- splits

def _natural_key(text: str) -> list:
    """Sort key that orders grade names numerically ("3" < "10")."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def load_split(split: str, root: Path | None = None) -> pd.DataFrame:
    """Read an official split CSV and restore the row order of the thesis split builder.

    The official CSVs list the same rows (same speakers) as the thesis' seeded split but in
    corpus order. The original builder emitted rows grouped by (grade, gender) and, within a
    group, in `lasdaicvi.csv` order. That order feeds the seeded shuffles of the unknown-word
    draw, so it is rebuilt here from the full corpus CSV.
    """
    root = resolve_dataset_root(root)
    df = pd.read_csv(root / "splits" / f"{split}.csv", dtype=str)
    full = pd.read_csv(root / "lasdaicvi.csv", dtype=str)
    position = pd.Series(np.arange(len(full)), index=full["path"])
    gender_rank = {g: i for i, g in enumerate(full["gender"].unique())}
    grade_rank = {g: i for i, g in enumerate(sorted(full["grade"].unique(), key=_natural_key))}
    order = pd.DataFrame({
        "grade": df["grade"].map(grade_rank),
        "gender": df["gender"].map(gender_rank),
        "row": position.loc[df["path"]].to_numpy(),
    })
    keys = order.sort_values(["grade", "gender", "row"], kind="stable").index
    return df.loc[keys].reset_index(drop=True)


def shuffle_permutation(n: int, seed: int) -> np.ndarray:
    """Row order produced by `DataFrame.sample(frac=1, random_state=seed)` on n rows."""
    return pd.Series(np.arange(n)).sample(frac=1, random_state=seed).to_numpy()


def _shuffle(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """`DataFrame.sample(frac=1)` with a fixed seed, the shuffle the notebooks use everywhere."""
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def select_samples(split_df: pd.DataFrame, targets: list[str], unknown: list[str],
                   seed: int = SEED) -> pd.DataFrame:
    """Target rows followed by the unknown rows, as `input_data.load_dataset` returns them.

    Unknown rows are relabelled `_desconocido_`; their original word stays in `source_label`.
    The per-word count is `ceil(ceil(N / nunique) / len(unknown))` and, as in the original,
    it is clipped to a word's pool size and *stays clipped* for the words that follow.
    """
    selected = split_df[split_df["label"].isin(targets)].reset_index(drop=True).copy()
    selected["source_label"] = selected["label"]
    if not unknown:
        return selected
    unknown_size = int(math.ceil(len(selected) * (1.0 / selected["label"].nunique())))
    n_unknown = int(math.ceil(unknown_size / len(unknown)))
    parts = [selected]
    for word in unknown:
        pool = split_df[split_df["label"] == word]
        if n_unknown > len(pool):
            n_unknown = len(pool)
        drawn = _shuffle(pool, seed).sample(n=n_unknown, random_state=seed).copy()
        drawn["source_label"] = drawn["label"]
        drawn["label"] = UNKNOWN_LABEL
        parts.append(drawn)
    return pd.concat(parts).reset_index(drop=True)


def silence_count(n_clips: int, n_labels: int) -> int:
    """Number of silence clips added to a set of `n_clips` with `n_labels` distinct labels."""
    return int(math.ceil(n_clips * (1.0 / n_labels)))


def sample_counts(selected: pd.DataFrame, augment: bool, copies: int = STRETCH_COPIES) -> dict:
    """Set sizes without touching audio: selected -> stretched -> plus silence."""
    n_selected = len(selected)
    stretched = n_selected * (1 + copies) if augment else n_selected
    silence = silence_count(stretched, selected["label"].nunique())
    return {"selected": n_selected, "stretched": stretched, "silence": silence,
            "total": stretched + silence}


# --------------------------------------------------------------------------- audio

def load_wave(path: Path) -> np.ndarray:
    """Mono float32 waveform at its native rate (all corpus files are 16 kHz)."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sr}")
    return y


def load_waves(selected: pd.DataFrame, root: Path | None = None) -> list[np.ndarray]:
    """Waveforms of the `path` column of `selected`, relative to the dataset's audio folder."""
    audio_root = resolve_dataset_root(root) / "LaSDAICVI"
    return [load_wave(audio_root / p) for p in selected["path"]]


def place_in_window(waves: list[np.ndarray], seed: int = SEED) -> np.ndarray:
    """Put every clip at a random offset inside a 2 s zero window, as the original does.

    One `random.Random(seed)` per call and `randint(0, CLIP - n - 1)` per clip (so a clip never
    ends on the very last sample). Clips of 2 s or longer would crash the original; here they
    are cropped to CLIP - 1 samples instead.
    """
    rng = random.Random(seed)
    out = np.zeros((len(waves), CLIP_SAMPLES), dtype=np.float32)
    for i, wave in enumerate(waves):
        wave = wave[: CLIP_SAMPLES - 1]
        start = rng.randint(0, CLIP_SAMPLES - len(wave) - 1)
        out[i, start:start + len(wave)] = wave
    return out


def add_silence(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Append all-zero clips labelled `_silencio_` (count: `silence_count`)."""
    n = silence_count(len(labels), len(set(labels)))
    X = np.concatenate([X, np.zeros((n, X.shape[1]), dtype=X.dtype)])
    labels = np.concatenate([np.asarray(labels, dtype=object),
                             np.array([SILENCE_LABEL] * n, dtype=object)])
    return X, labels
