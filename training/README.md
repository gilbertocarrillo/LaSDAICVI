# LaSDAICVI training code

Training code of the thesis *Diseño de interacción humano-robot para el aprendizaje de las
tablas de multiplicación e implementación de los modelos de reconocimiento del habla*
(Gilberto Carrillo, Universidad de los Andes, 2021,
[DOI 10.13140/RG.2.2.27502.95042](https://doi.org/10.13140/RG.2.2.27502.95042)).

It trains the keyword-spotting models of Table 4.1 on the LaSDAICVI dataset: **10 tasks x 3
architectures (CNN, GRU, CRNN) x with/without augmentation = 60 experiments**. One package
and one command replace the original per-experiment notebooks.

## Requirements and install

Python 3.12. The code reads the dataset from the repository it lives in, so cloning the
repository gets both the code and the data (about 900 MB of audio, stored in plain git):

```bash
git clone https://github.com/gilbertocarrillo/LaSDAICVI.git
cd LaSDAICVI/training
```

The script finds these by itself in the repository root: `lasdaicvi.csv` (one row per recording),
`splits/` (the official train/validation/test speaker split) and `LaSDAICVI/` (the audio, with the
background-noise clips in `LaSDAICVI/_ruido_/`). Nothing needs to be downloaded or moved.

Other copies of the dataset:

- **Zenodo** archives each GitHub release as a zip with the same layout. Unzip it and run from the
  `training/` folder inside it (releases from v1.1.0 on). For v1.0.0, which predates this code,
  pass `--dataset-root` pointing to the unzipped folder.
- **Hugging Face** stores the recordings as Parquet files (`audio`, `label`, `speaker_id`, `gender`,
  `age`, `grade`, ...) without `lasdaicvi.csv` and without the `_ruido_/` noise clips, so this code
  cannot read it directly. Use the GitHub or Zenodo copy to train.

```bash
uv sync            # creates .venv with TensorFlow 2.18 and the other dependencies
# without uv:  python3.12 -m venv .venv && . .venv/bin/activate && pip install -e .
```

If the code lives elsewhere, point it to the dataset with `--dataset-root PATH` or
`export LASDAICVI_ROOT=PATH`. Resolution order: flag, environment variable, the first parent
directory containing `lasdaicvi.csv`.

## Quickstart

```bash
# one row: task x architecture, with augmentation
uv run lasdaicvi-train --task interaction --arch gru --aug

# the same without augmentation
uv run lasdaicvi-train --task interaction --arch gru

# all 60 rows (very long); finished rows are skipped when you resume
uv run lasdaicvi-train --all --skip-existing
```

`python -m lasdaicvi_kws.train` is equivalent to `lasdaicvi-train`. Useful flags: `--thesis`
(see below), `--epochs N` (quick tests; the run folder gets an `-eN` suffix), `--plots`
(confusion-matrix PNGs), `--no-cache`, `--out DIR`, `--seed`.

## Tasks

Each task is defined in [`configs/experiments.yaml`](configs/experiments.yaml). Every task adds
two extra classes: `_desconocido_` (other words) and `_silencio_`.

| Task | Recognises | Classes |
|---|---|---|
| `table2` ... `table9` | the ten results of one multiplication table (`table2`: 2, 4, ... 20; `table9`: 9, 18, ... 90) | 12 |
| `interaction` | the ten commands `si no verdadero falso detente siguiente aprender repite jugar continúa` | 12 |
| `pepe` | the robot's wake word `pepe`, against the ten commands and the digits 0–9 as unknown words | 3 |

Epochs are 300, except 400 for the base CRNN of `table4`, `table5`, `table8`, `table9` and
`pepe`, and the base GRU of `table9`.

## Pipeline in brief

1. **Split.** The official `splits/{train,validation,test}.csv` (by speaker, so no child appears in
   two sets).
2. **Unknown and silence.** Words outside the task are sampled into `_desconocido_`; all-zero
   clips are added as `_silencio_`. Both counts follow the thesis formulas.
3. **Time stretch** (training set, augmented runs). Six stretched copies per clip, rate in
   [0.85, 1.15].
4. **Placement.** Each clip goes at a random offset inside a 2 s window.
5. **Background noise** (augmented training set and the noisy test set). With probability 0.8,
   a crop of a background recording is mixed in at an SNR drawn from [5, 30] dB: the 8 clips in
   `_ruido_/` for training and the 4 clips in `_ruido_/Test/` for the noisy test set, so no noise
   recording is shared. The test set is always evaluated clean and noisy.
6. **MFCC.** 20 coefficients, 40 mel bands, 1024-sample windows, hop 512, giving a 61 x 20 input
   (TensorFlow raw ops, as in the thesis).
7. **Models.** Google-style CNN (2.77 M parameters), GRU (0.59 M) and CRNN (0.47 M) for 12 classes.
8. **Training.** Adam, learning rate 1e-5, batch 128, seed 931001. The checkpoint with the
   lowest validation loss is kept.
9. **Metrics.** Accuracy, classification report, confusion matrix, micro/macro ROC AUC, and the
   thesis "AUC", which is the area under the DET curve (`auc(fpr, fnr)` from
   `sklearn.metrics.det_curve`), where lower is better. It is not a ROC AUC.

## Default mode and `--thesis`

The default mode fixes defects of the original notebooks. `--thesis` replays them so the
published numbers can be reproduced.

| | default | `--thesis` |
|---|---|---|
| Noise SNR draw | seeded, repeatable | module-level `random`, seeded once with the seed |
| Noise file order | sorted by name | `os.listdir` order |
| `pepe` unknown words | includes `'7'`, `'8'` and `'9'` | the notebook's typo `'78'`: that word has no samples, so `'7'`, `'8'` and `'9'` never reach the unknown class |

The notebooks also built the noisy-test labels from the clean ones. Both test sets are shuffled
with the same seed, so the labels are identical and nothing changes; this code simply uses the
noisy set's own labels in both modes.

Everything else is shared by both modes and was kept identical to the thesis: sample counts,
seeded draws and their order, MFCC, models and the training setup.

## Validation against the thesis

Five of the 60 rows were retrained from scratch on CPU with `--thesis` and compared with
Table 4.1 of the thesis (accuracy %, clean / noisy test):

| Row | Table 4.1 | This code | Command |
|---|---|---|---|
| interaction, GRU, aug | 98.33 / 96.86 | 98.54 / 97.35 | `lasdaicvi-train --task interaction --arch gru --aug --thesis` |
| interaction, GRU, base | 94.63 / 43.07 | 94.08 / 44.18 | `lasdaicvi-train --task interaction --arch gru --thesis` |
| interaction, CNN, base | 89.62 / 40.35 | 89.69 / 40.70 | `lasdaicvi-train --task interaction --arch cnn --thesis` |
| table2, CRNN, aug | 97.86 / 96.33 | 97.99 / 95.44 | `lasdaicvi-train --task table2 --arch crnn --aug --thesis` |
| pepe, GRU, aug | 99.38 / 97.82 | 97.82 / 95.64 | `lasdaicvi-train --task pepe --arch gru --aug --thesis` |

Clean accuracy is within 0.6 points on four rows. Noisy accuracy differs by up to 1.1 points,
and `pepe` is the furthest off (1.6 clean, 2.2 noisy). Retraining is not bit-exact (see
[Hardware](#hardware)), and `pepe` has only 321 test clips, so its gap is 5 clips clean and 7
noisy. The original `pepe` notebooks also vary between runs (99.38% and 98.75% clean).

The default mode was trained in full on the first row too (`lasdaicvi-train --task interaction
--arch gru --aug`): 97.91 / 95.75. It draws different noise (seeded SNR, sorted noise files),
including for the noisy test set, so its numbers are close to but not directly comparable with
Table 4.1.

The thesis' own trained models were also checked: loading each of the 60 original weight files
into `models.build_model` and scoring the test arrays saved with it reproduces the clean and
noisy accuracy of Table 4.1 exactly for all 60, so the models, features and metrics are the same.
Those 60 models are published on Hugging Face:
[gilbertocp93/LaSDAICVI-models](https://huggingface.co/gilbertocp93/LaSDAICVI-models).

## Hardware

The thesis models were trained in 2021 on **Google Colab with a GPU runtime**, with Python 3.7
and the TensorFlow and librosa releases of the time. This code runs on a current stack
(TensorFlow 2.18, Keras 3, librosa 0.10+) and defaults to the CPU. Different hardware, library
versions, random weight initialisation and the unseeded noise draw of the original are why
retrained models land close to Table 4.1 rather than matching it to the decimal.

Train on the **CPU**; this is the default (`--device auto`). On a laptop CPU a full run takes
from minutes (models without augmentation) to hours (augmented runs, the CNN being the slowest),
so `--all` is a long job; use `--epochs 1` for a quick check that everything runs.

Apple's `tensorflow-metal` GPU plugin (1.2.0) **mis-trains** these models, so it is not
supported. On identical features:

- the GRU after 25 epochs reaches train/val 0.670/0.751 on CPU (thesis notebook: 0.670/0.754)
  but 0.522/0.710 on Metal, and a full Metal run ended at 92.5% clean accuracy against 98.5% on
  CPU;
- the CNN after 10 epochs reaches 0.753/0.849 on CPU (thesis notebook: 0.748/0.848) but
  0.716/0.829 on Metal.

`--device gpu` selects `/GPU:0` and is meant for CUDA machines, the setup the thesis used. It was
not validated with this code.

## Outputs

```text
runs/<task>_<arch>_<aug|base>[-thesis][-eN]/
├── best.keras      # checkpoint with the lowest validation loss
├── history.csv     # per-epoch loss and accuracy
├── metrics.json    # train/val/clean/noisy accuracy, ROC and DET areas, report, confusion matrix
└── confusion_*.png # with --plots
cache/              # preprocessed features, reused by later runs of the same task and mode
```

## Tests

```bash
uv run --extra dev pytest -q
```

Tests that need the dataset are skipped when it is not found.

## Citation

Please cite the dataset and the thesis; see [`CITATION.cff`](../CITATION.cff) in the repository
root.

## License

Same as the dataset repository: [CC BY-NC 4.0](../LICENSE).
