"""Experiment definitions (tasks, labels, epochs) loaded from `configs/experiments.yaml`.

The yaml is the single source of truth for the 10 tasks x 3 architectures x base/augmented
grid that makes up the 60 rows of Table 4.1 of the thesis.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PKG_ROOT / "configs" / "experiments.yaml"
ARCHS = ["cnn", "gru", "crnn"]


@dataclass(frozen=True)
class TaskConfig:
    """One recognition task: target `labels`, `unknown` words and epochs per architecture."""

    name: str
    labels: list[str]
    unknown: list[str]
    epochs_by_arch: dict

    def epochs(self, arch: str, aug: bool) -> int:
        """Training epochs configured for `arch`, with or without augmentation."""
        return int(self.epochs_by_arch[arch]["aug" if aug else "base"])


@lru_cache(maxsize=None)
def _raw(path: str = str(CONFIG_PATH)) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def defaults() -> dict:
    """Pipeline hyper-parameters shared by all tasks (`defaults:` in the yaml)."""
    return dict(_raw()["defaults"])


def load_tasks() -> dict:
    """Raw task definitions keyed by task name."""
    return _raw()["tasks"]


def get_task(name: str, thesis: bool = False) -> TaskConfig:
    """Task definition; `thesis=True` swaps in the original (buggy) unknown list when one exists."""
    raw = load_tasks()[name]  # KeyError for an unknown task
    unknown = raw["unknown_thesis"] if thesis and "unknown_thesis" in raw else raw["unknown"]
    return TaskConfig(name, list(raw["labels"]), list(unknown), raw["epochs"])


def experiment_name(task: str, arch: str, aug: bool) -> str:
    """Canonical row name, e.g. `interaction_gru_aug`."""
    return f"{task}_{arch}_{'aug' if aug else 'base'}"


def experiment_rows() -> list[tuple[str, str, bool]]:
    """The 60 rows of the thesis' result tables: 10 tasks x 3 architectures x base/aug."""
    return [(t, a, aug) for t in load_tasks() for a in ARCHS for aug in (False, True)]
