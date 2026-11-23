"""MFCC features, third stage of the pipeline.

The thesis computes MFCCs with TensorFlow's raw spectrogram/MFCC ops rather than librosa, so
the same ops are used here: it is what the trained models expect (the two implementations
differ numerically). A 2 s clip gives a (61, 20) matrix.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .data import SAMPLE_RATE

FRAME_SIZE = 1024   # samples per analysis window
FRAME_STEP = 512    # hop between windows, in samples
UPPER_HZ = 8000
LOWER_HZ = 0
N_MELS = 40
N_MFCC = 20


def mfcc_batch(waves: np.ndarray) -> np.ndarray:
    """(B, samples) float waveforms -> (B, frames, N_MFCC) MFCCs.

    `AudioSpectrogram` treats the second axis of its input as channels, so a batch is passed
    transposed; each channel is processed independently, hence the result equals the original
    one-clip-at-a-time call.
    """
    signal = tf.constant(np.ascontiguousarray(np.asarray(waves, dtype=np.float32).T))
    spectrogram = tf.raw_ops.AudioSpectrogram(
        input=signal, window_size=FRAME_SIZE, stride=FRAME_STEP, magnitude_squared=True)
    mfcc = tf.raw_ops.Mfcc(
        spectrogram=spectrogram, sample_rate=SAMPLE_RATE,
        upper_frequency_limit=UPPER_HZ, lower_frequency_limit=LOWER_HZ,
        filterbank_channel_count=N_MELS, dct_coefficient_count=N_MFCC)
    return mfcc.numpy()


def mfcc_features(X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """MFCCs for many clips, computed in batches; (N, 61, 20) float32 for 2 s clips."""
    return np.concatenate([mfcc_batch(X[i:i + batch_size])
                           for i in range(0, len(X), batch_size)]).astype(np.float32)
