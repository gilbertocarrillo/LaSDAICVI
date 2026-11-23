"""MFCC features: shape and agreement with the thesis' implementation."""
import numpy as np

from lasdaicvi_kws import features


def _tone_mix() -> np.ndarray:
    """Deterministic 2 s test signal: two tones plus an amplitude-modulated third."""
    t = np.arange(32000) / 16000
    return (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1250 * t + 1)
            + 0.1 * np.sin(2 * np.pi * 3100 * t) * np.sin(2 * np.pi * 3 * t)).astype(np.float32)


def test_mfcc_shape():
    x = np.stack([_tone_mix()] * 3)
    assert features.mfcc_features(x).shape == (3, 61, 20)


def test_mfcc_matches_golden_values_of_the_thesis_implementation():
    # Golden values: the thesis' `speech_features.mfcc` (same TensorFlow raw ops) applied to
    # `_tone_mix()`; frames 0, 10, 30 and 60, first four coefficients, plus whole-array sums.
    golden = np.array([
        [-45.5120, 16.8677, -4.5715, 2.3587],
        [-46.6925, 17.9636, -4.7973, 2.2652],
        [-48.3522, 19.0140, -5.8703, 1.6382],
        [-44.5250, 16.5350, -4.8191, 3.0378]])
    out = features.mfcc_features(np.stack([_tone_mix(), _tone_mix()]), batch_size=1)
    np.testing.assert_allclose(out[0][[0, 10, 30, 60]][:, :4], golden, atol=1e-3)
    np.testing.assert_allclose(out[1], out[0])  # batching does not change a clip's features
    assert abs(out[0].sum() - (-3215.698)) < 0.5
    assert abs(np.abs(out[0]).sum() - 8771.916) < 0.5
