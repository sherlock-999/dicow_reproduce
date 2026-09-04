"""Frame-level voice activity helpers for clean source tracks."""

import numpy as np


def energy_vad(source: np.ndarray, win_s: float = 0.025, rel_threshold: float = 0.03,
               sr: int = 16000) -> np.ndarray:
    """Relative-energy VAD for a clean source waveform.

    The threshold is relative to the signal's own 95th percentile, so volume
    perturbation does not change the resulting activity.
    """
    win = max(1, int(win_s * sr))
    source = np.asarray(source, dtype=np.float64)
    env = np.convolve(np.abs(source), np.ones(win, dtype=np.float64) / win, mode="same")
    thr = rel_threshold * max(float(np.percentile(env, 95)), 1e-8)
    return (env > thr).astype(np.float32)
