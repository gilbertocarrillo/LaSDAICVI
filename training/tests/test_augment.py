"""T3: time stretch and background noise (fixed and thesis modes)."""
import random

import numpy as np
import pytest

from lasdaicvi_kws import augment

SILENCE = "_silencio_"


def rms(x):
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def test_stretch_keeps_originals_first_then_six_copies_per_clip():
    waves = [np.ones(100, np.float32), np.ones(200, np.float32)]
    seen = []

    def fake(y, rate):
        seen.append(rate)
        return y[: int(len(y) / rate)]

    out, src = augment.time_stretch_random(waves, ["a", "b"], 6, 0.85, 1.15, 931001, stretch_fn=fake)
    assert len(out) == 14 and list(src) == ["a", "b"] + ["a"] * 6 + ["b"] * 6
    rng = random.Random(931001)
    assert seen == [rng.uniform(0.85, 1.15) for _ in range(12)]


def test_stretch_uses_modern_librosa_signature():
    y = np.random.default_rng(0).uniform(-1, 1, 8000).astype(np.float32)
    out, _ = augment.time_stretch_random([y], ["a"], 2, 0.85, 1.15, 1)
    assert len(out) == 3 and all(o.dtype == np.float32 for o in out)


def make_clips(n_speech=20, n_silence=5, seed=0):
    r = np.random.default_rng(seed)
    X = np.zeros((n_speech + n_silence, 32000), np.float32)
    X[:n_speech] = r.uniform(-0.3, 0.3, (n_speech, 32000))
    labels = np.array(["a"] * n_speech + [SILENCE] * n_silence)
    return X, labels


def noise_bank(seed=1):
    r = np.random.default_rng(seed)
    return [r.uniform(-0.5, 0.5, 100000).astype(np.float32),
            r.uniform(-0.2, 0.2, 90000).astype(np.float32)]


@pytest.mark.parametrize("thesis", [False, True])
def test_snr_formula_and_silence_uses_mean_rms(thesis):
    X, labels = make_clips()
    mean_rms = np.mean([rms(c) for c in X[labels != SILENCE]])
    out = augment.add_background_noise(X, labels, noise_bank(), seed=3, thesis=thesis,
                                       prob=1.0, snr_db=(10.0, 10.0))
    for clean, mixed, lab in zip(X, out, labels):
        added = mixed.astype(np.float64) - clean.astype(np.float64)
        reference = mean_rms if lab == SILENCE else rms(clean)
        assert rms(added) / reference == pytest.approx(10 ** (-10 / 20), rel=1e-3)


def test_no_clipping_is_applied():
    X = np.full((1, 32000), 0.99, np.float32)
    bank = [np.full(50000, 0.5, np.float32)]
    out = augment.add_background_noise(X, np.array(["a"]), bank, seed=1, prob=1.0,
                                       snr_db=(5.0, 5.0))
    assert out.max() > 1.0


def test_probability_zero_returns_input_unchanged_and_input_is_not_mutated():
    X, labels = make_clips()
    before = X.copy()
    out = augment.add_background_noise(X, labels, noise_bank(), seed=3, prob=0.0)
    assert (out == before).all() and (X == before).all()


def test_fixed_mode_is_deterministic_and_seeded():
    X, labels = make_clips()
    a = augment.add_background_noise(X, labels, noise_bank(), seed=3)
    b = augment.add_background_noise(X, labels, noise_bank(), seed=3)
    c = augment.add_background_noise(X, labels, noise_bank(), seed=4)
    assert (a == b).all() and not (a == c).all()
    assert 0 < (a != X).any(axis=1).sum() < len(X)  # about 80 percent mixed


def test_fixed_mode_draws_snr_from_the_seeded_rng_not_the_global_one():
    X, labels = make_clips()
    random.seed(1)
    a = augment.add_background_noise(X, labels, noise_bank(), seed=3, thesis=False)
    random.seed(999)
    b = augment.add_background_noise(X, labels, noise_bank(), seed=3, thesis=False)
    assert (a == b).all()


def test_thesis_mode_draws_snr_from_the_global_random_module():
    X, labels = make_clips()
    random.seed(931001)
    a = augment.add_background_noise(X, labels, noise_bank(), seed=3, thesis=True)
    state_after = random.getstate()
    random.seed(931001)
    b = augment.add_background_noise(X, labels, noise_bank(), seed=3, thesis=True)
    random.seed(5)
    c = augment.add_background_noise(X, labels, noise_bank(), seed=3, thesis=True)
    assert (a == b).all() and not (a == c).all()
    random.seed(931001)
    assert random.getstate() != state_after  # the global stream was consumed


def test_noise_loader_sorted_vs_listdir(tmp_path):
    import soundfile as sf
    for name in ["b.wav", "a.wav", "c.wav"]:
        sf.write(tmp_path / name, np.zeros(100, np.float32), 16000)
    (tmp_path / "Test").mkdir()
    assert augment.noise_files(tmp_path, sort=True) == [tmp_path / n for n in ["a.wav", "b.wav", "c.wav"]]
    assert sorted(augment.noise_files(tmp_path, sort=False)) == augment.noise_files(tmp_path, sort=True)
