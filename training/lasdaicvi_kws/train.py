"""Command-line entry point: train and evaluate one thesis experiment, or all 60.

    lasdaicvi-train --task interaction --arch gru --aug [--thesis]
    lasdaicvi-train --all [--thesis] [--skip-existing]

Default mode repairs the notebook bugs (seeded SNR, sorted noise files, pepe unknown words
'7' and '8'); `--thesis` replays them. Outputs go to `runs/<exp_name>[-thesis]/`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import config, data, evaluate, models, pipeline

PKG_ROOT = config.PKG_ROOT
DEVICES = ("auto", "cpu", "gpu")


def resolve_device(choice: str, arch: str) -> str:
    """TensorFlow device for an experiment.

    `auto` uses the CPU for every model, because tensorflow-metal 1.2.0 mis-trains them. On the
    same features, the GRU after 25 epochs reaches train/val 0.670/0.751 on the CPU (the thesis
    notebook: 0.670/0.754) but 0.522/0.710 on Metal, and a full Metal run stopped at 92.5%
    clean accuracy. The CNN after 10 epochs reaches 0.753/0.849 on the CPU (notebook:
    0.748/0.848) but 0.716/0.829 on Metal. `gpu` selects `/GPU:0` and is meant for CUDA
    machines (not validated here); do not use it with Apple Metal.
    """
    return "/GPU:0" if choice == "gpu" else "/CPU:0"


def run_dir_name(task: str, arch: str, aug: bool, thesis: bool, epochs: int | None) -> str:
    """Output folder name, e.g. `interaction_gru_aug-thesis`; `-eN` marks an epochs override."""
    name = config.experiment_name(task, arch, aug) + ("-thesis" if thesis else "")
    return f"{name}-e{epochs}" if epochs is not None else name  # a smoke run never clobbers a full one


def run_experiment(task_name: str, arch: str, aug: bool, thesis: bool = False,
                   epochs: int | None = None, out_root: Path = PKG_ROOT / "runs",
                   dataset_root: Path | None = None, seed: int = data.SEED,
                   use_cache: bool = True, plots: bool = False, device: str = "auto") -> dict:
    """Build features, train, pick the best checkpoint by `val_loss`, evaluate and save.

    Returns the metrics dict that is also written to `<out_root>/<run>/metrics.json`.
    """
    import tensorflow as tf

    target = resolve_device(device, arch)
    with tf.device(target):
        return _run(task_name, arch, aug, thesis, epochs, out_root, dataset_root, seed,
                    use_cache, plots, target)


def _run(task_name, arch, aug, thesis, epochs, out_root, dataset_root, seed, use_cache, plots,
         device_name) -> dict:
    """Body of `run_experiment`, executed inside the chosen `tf.device` scope."""
    import keras
    import pandas as pd

    task = config.get_task(task_name, thesis)
    defaults = config.defaults()
    configured_epochs = task.epochs(arch, aug)
    override = epochs if epochs is not None and epochs != configured_epochs else None
    epochs = epochs or configured_epochs
    exp_name = config.experiment_name(task_name, arch, aug) + ("-thesis" if thesis else "")
    run_dir = Path(out_root) / run_dir_name(task_name, arch, aug, thesis, override)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"== {exp_name}: {epochs} epochs on {device_name} -> {run_dir}", flush=True)

    keras.utils.set_random_seed(seed)  # python `random`, numpy and TensorFlow
    feats = pipeline.load_or_build(task, aug, thesis, dataset_root, seed, use_cache)
    sizes = feats.sizes()
    print("split sizes: " + ", ".join(f"{k}={v}" for k, v in sizes.items()), flush=True)
    n_classes = len(feats.classes)

    keras.backend.clear_session()
    model = models.build_model(arch, n_classes, feats.x_train.shape[1:])
    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=defaults["learning_rate"]),
                  metrics=["accuracy"])
    checkpoint = run_dir / "best.keras"
    started = time.time()
    history = model.fit(
        feats.x_train, np.eye(n_classes)[feats.y_train], epochs=epochs,
        validation_data=(feats.x_val, np.eye(n_classes)[feats.y_val]),
        batch_size=defaults["batch_size"],
        callbacks=[keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", mode="min",
                                                   save_best_only=True)],
        verbose=2)
    elapsed = time.time() - started
    pd.DataFrame(history.history).to_csv(run_dir / "history.csv", index_label="epoch")

    best = keras.models.load_model(checkpoint)
    predict = lambda x: best.predict(x, batch_size=512, verbose=0)
    train_acc = evaluate.accuracy(feats.y_train, predict(feats.x_train))
    val_acc = evaluate.accuracy(feats.y_val, predict(feats.x_val))
    clean = evaluate.evaluate_predictions(feats.y_clean, predict(feats.x_clean), feats.classes)
    noisy = evaluate.evaluate_predictions(feats.y_noisy, predict(feats.x_noisy), feats.classes)
    if plots:
        evaluate.save_plots(clean, feats.classes, run_dir, "clean")
        evaluate.save_plots(noisy, feats.classes, run_dir, "noisy")

    metrics = {
        "experiment": exp_name, "task": task_name, "arch": arch, "aug": aug, "thesis": thesis,
        "seed": seed, "device": device_name, "epochs": epochs, "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "classes": feats.classes, "split_sizes": sizes, "params": int(best.count_params()),
        "train_seconds": round(elapsed, 1),
        "train_accuracy": train_acc, "val_accuracy": val_acc,
        "clean_accuracy": clean["accuracy"], "noisy_accuracy": noisy["accuracy"],
        "clean": clean, "noisy": noisy,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"train {train_acc:.4f} | val {val_acc:.4f} | clean {clean['accuracy']:.4f} "
          f"(DET area {clean['det_area_micro']:.4f}) | noisy {noisy['accuracy']:.4f} "
          f"(DET area {noisy['det_area_micro']:.4f})", flush=True)
    print(f"wrote {run_dir / 'metrics.json'}", flush=True)
    return metrics


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the command line; requires `--task` and `--arch`, or `--all`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=list(config.load_tasks()))
    p.add_argument("--arch", choices=config.ARCHS)
    p.add_argument("--aug", action="store_true", help="time-stretch + background noise on train")
    p.add_argument("--thesis", action="store_true", help="replay the original notebook behaviour")
    p.add_argument("--all", action="store_true", help="run the 60 thesis rows")
    p.add_argument("--skip-existing", action="store_true", help="with --all: skip rows that have metrics.json")
    p.add_argument("--epochs", type=int, help="override the configured epochs (run dir gets an -eN suffix)")
    p.add_argument("--out", type=Path, default=PKG_ROOT / "runs")
    p.add_argument("--dataset-root", type=Path,
                   help="dataset repository (default: $LASDAICVI_ROOT, else the enclosing checkout)")
    p.add_argument("--seed", type=int, default=data.SEED)
    p.add_argument("--no-cache", action="store_true", help="do not read or write cached features")
    p.add_argument("--plots", action="store_true", help="save confusion-matrix PNGs")
    p.add_argument("--device", choices=DEVICES, default="auto",
                   help="auto = CPU (default). gpu = /GPU:0, for CUDA machines; do not use Apple Metal, "
                        "tensorflow-metal mis-trains these models")
    args = p.parse_args(argv)
    if not args.all and not (args.task and args.arch):
        p.error("give --task and --arch, or --all")
    return args


def main(argv=None) -> None:
    """CLI entry point (`lasdaicvi-train`)."""
    args = parse_args(argv)
    rows = config.experiment_rows() if args.all else [(args.task, args.arch, args.aug)]
    for task, arch, aug in rows:
        run = args.out / run_dir_name(task, arch, aug, args.thesis, None)
        if args.all and args.skip_existing and (run / "metrics.json").exists():
            print(f"skip {run.name} (done)")
            continue
        run_experiment(task, arch, aug, args.thesis, args.epochs, args.out, args.dataset_root,
                       args.seed, not args.no_cache, args.plots, args.device)


if __name__ == "__main__":
    main()
