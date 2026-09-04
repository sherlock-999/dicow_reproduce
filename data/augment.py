"""The three training-time augmentations used by the DiCoW reproduction.

- STNO mask augs: src/data/collators.py:50-139 (gaussian noise + soft segment relabel)
- MUSAN additive noise: src/data/augmentations.py:382-429 (RandomBackgroundNoise)
All preserve the STNO simplex property (channels >= 0, sum to 1 per frame).
"""

import os
import pathlib
import random

import torch


def add_gaussian_noise_and_rescale(
    prob_mask: torch.Tensor,
    variance: float = 0.05,
    fraction: float = 0.5,
) -> torch.Tensor:
    """Perturb each batch item with probability ``fraction`` and renormalize."""

    if prob_mask.ndim != 3:
        raise ValueError("prob_mask must have shape [batch, channels, frames]")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0, 1]")
    if variance < 0:
        raise ValueError("variance must be non-negative")

    batch, channels, frames = prob_mask.shape
    selected = torch.rand(batch, device=prob_mask.device) < fraction
    num_noisy = int(selected.sum().item())
    if num_noisy == 0 or variance == 0:
        return prob_mask

    noisy = prob_mask.clone()
    noisy[selected] += torch.randn(
        (num_noisy, channels, frames),
        device=prob_mask.device,
        dtype=prob_mask.dtype,
    ) * (variance**0.5)
    min_values = torch.clamp(noisy[selected].amin(dim=1, keepdim=True), max=0)
    noisy[selected] -= min_values
    noisy[selected] /= noisy[selected].sum(dim=1, keepdim=True).clamp_min(1e-8)
    return noisy


def soft_segment_augmentation(stno_mask: torch.Tensor, change_prob: float = 0.2,
                              min_seg_len: int = 5, max_seg_len: int = 20):
    """Softly relabel random time segments of a (B, C, T) STNO mask toward another class."""
    if stno_mask.ndim != 3 or stno_mask.shape[1] != 4:
        raise ValueError("stno_mask must have shape [batch, 4, frames]")
    if not 0 <= change_prob <= 1:
        raise ValueError("change_prob must lie in [0, 1]")
    if min_seg_len <= 0 or max_seg_len < min_seg_len:
        raise ValueError("segment lengths must satisfy 0 < min <= max")
    B, C, T = stno_mask.shape
    out = stno_mask.clone()
    for b in range(B):
        pos = 0
        while pos < T:
            seg_len = torch.randint(min_seg_len, max_seg_len + 1, (1,)).item()
            end = min(pos + seg_len, T)
            if torch.rand(1).item() < change_prob:
                segment = out[b, :, pos:end]
                dominant = segment.mean(dim=1).argmax().item()
                candidates = [c for c in range(C) if c != dominant]
                target_class = candidates[torch.randint(0, len(candidates), (1,)).item()]
                target_dist = torch.zeros_like(segment)
                target_dist[target_class, :] = 1.0
                softness = torch.rand(1).item()
                new_segment = (1 - softness) * segment + softness * target_dist
                out[b, :, pos:end] = new_segment / new_segment.sum(dim=0, keepdim=True)
            pos = end
    return out

class RandomBackgroundNoise:
    """MUSAN additive noise at random SNR in [min_snr_db, max_snr_db]."""

    def __init__(self, sample_rate: int, noise_dir: str, min_snr_db: int = 0, max_snr_db: int = 15):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        if not os.path.exists(noise_dir):
            raise IOError(f"Noise directory `{noise_dir}` does not exist")
        self.noise_files_list = list(pathlib.Path(noise_dir).glob("**/*.wav"))
        if len(self.noise_files_list) == 0:
            raise IOError(f"No .wav file found in `{noise_dir}`")

    def __call__(self, audio_data: torch.Tensor) -> torch.Tensor:
        import torchaudio
        import torchaudio.functional as taf

        noise, sr = torchaudio.load(str(random.choice(self.noise_files_list)))
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            noise = taf.resample(noise, sr, self.sample_rate)
        n_audio, n_noise = audio_data.shape[-1], noise.shape[-1]
        if n_noise > n_audio:
            off = random.randint(0, n_noise - n_audio)
            noise = noise[..., off:off + n_audio]
        elif n_noise < n_audio:
            repeats = (n_audio + n_noise - 1) // n_noise
            noise = noise.repeat(1, repeats)[..., :n_audio]

        # L2 norm is an amplitude quantity, hence dB is converted with /20.
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        amplitude_ratio = 10 ** (snr_db / 20)
        scale = audio_data.norm(p=2) / (
            amplitude_ratio * noise.norm(p=2).clamp(min=1e-8)
        )
        noise = noise.squeeze(0).to(device=audio_data.device, dtype=audio_data.dtype)
        return audio_data + scale * noise
