"""Small Lhotse data pipeline for the Hugging Face DiCoW model.

Every input cut represents one training example and identifies its target
speaker either with an ``_tsidxN`` ID suffix (created by
``data.export_ts_cuts``) or with ``cut.custom["target_speaker"]``.

This follows Lhotse's batch-transform pattern: a sampler gives ``__getitem__``
a CutSet and the dataset returns the tensors consumed by ``model.DiCoW``::

    input_features  [batch, 128, 3000]
    attention_mask  [batch, 3000]
    stno_mask       [batch, 4, 1500]
    labels          [batch, tokens]

Training examples are fixed to Whisper's 30-second window. Long recordings
are handled at inference time by ``DiCoW.generate`` and should be cut into
<=30-second examples before training.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from lhotse import CutSet
from lhotse.dataset import AudioSamples, DynamicBucketingSampler, DynamicCutSampler
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from model.stno import lhotse_cut_to_stno
from txt_norm import get_text_norm

from .augment import (
    RandomBackgroundNoise,
    add_gaussian_noise_and_rescale,
    soft_segment_augmentation,
)


SAMPLE_RATE = 16_000
WINDOW_SECONDS = 30
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
STNO_FRAME_HZ = 50
STNO_FRAMES = WINDOW_SECONDS * STNO_FRAME_HZ
_TARGET_INDEX = re.compile(r"_tsidx(\d+)$")


@dataclass(frozen=True)
class AugmentationConfig:
    """The augmentation settings reported in the reproduction README."""

    stno_gaussian_probability: float = 0.75
    stno_gaussian_variance: float = 0.2
    stno_segment_probability: float = 0.3
    stno_segment_change_probability: float = 0.1
    stno_segment_min_frames: int = 5
    stno_segment_max_frames: int = 20
    musan_probability: float = 0.3
    musan_min_snr_db: int = 0
    musan_max_snr_db: int = 15


def speakers_in_cut(cut) -> list[str]:
    """Return the deterministic speaker ordering used by manifest expansion."""

    return sorted({s.speaker for s in cut.supervisions if s.speaker is not None})


def target_speaker_from_cut(cut) -> str:
    """Resolve the target speaker encoded in a target-expanded Lhotse cut."""

    speakers = speakers_in_cut(cut)
    if not speakers:
        raise ValueError(f"cut {cut.id!r} has no speakers")

    match = _TARGET_INDEX.search(cut.id)
    if match is not None:
        index = int(match.group(1))
        if index >= len(speakers):
            raise ValueError(
                f"cut {cut.id!r} requests target index {index}, but only "
                f"{len(speakers)} speakers are present"
            )
        return speakers[index]

    custom = getattr(cut, "custom", None) or {}
    target = custom.get("target_speaker") or custom.get("target_spk")
    if target is not None:
        if target not in speakers:
            raise ValueError(
                f"target speaker {target!r} is not present in cut {cut.id!r}"
            )
        return str(target)

    raise ValueError(
        f"cut {cut.id!r} has no target speaker; run data.export_ts_cuts or "
        "set cut.custom['target_speaker']"
    )


def target_text(cut, target_speaker: str, normalize: Callable[[str], str]) -> str:
    """Join only the target speaker's supervision text, in temporal order."""

    supervisions = sorted(
        (s for s in cut.supervisions if s.speaker == target_speaker),
        key=lambda s: s.start,
    )
    parts = [normalize(s.text or "").strip() for s in supervisions]
    return " ".join(part for part in parts if part).strip()


class TargetTextFilter:
    """Keep target-expanded cuts with non-empty normalized target text."""

    def __init__(self, text_normalizer: str = "whisper_nsf") -> None:
        self.normalize = get_text_norm(text_normalizer)

    def __call__(self, cut) -> bool:
        target = target_speaker_from_cut(cut)
        return bool(target_text(cut, target, self.normalize))


class DiCoWDataset(torch.utils.data.Dataset):
    """Turn a Lhotse CutSet batch into a Hugging Face DiCoW training batch."""

    def __init__(
        self,
        processor: WhisperProcessor,
        *,
        is_train: bool = False,
        text_normalizer: str = "whisper_nsf",
        augmentation: AugmentationConfig | None = None,
        musan_dir: str | Path | None = None,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.processor.tokenizer.set_prefix_tokens(
            language="english",
            task="transcribe",
            predict_timestamps=False,
        )
        self.is_train = is_train
        self.normalize = get_text_norm(text_normalizer)
        self.augmentation = augmentation or AugmentationConfig()
        self.return_metadata = return_metadata
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.musan = None
        if is_train and musan_dir is not None:
            self.musan = RandomBackgroundNoise(
                sample_rate=SAMPLE_RATE,
                noise_dir=str(musan_dir),
                min_snr_db=self.augmentation.musan_min_snr_db,
                max_snr_db=self.augmentation.musan_max_snr_db,
            )

    def __getitem__(self, cuts: CutSet) -> dict[str, object]:
        if not isinstance(cuts, CutSet):
            items = cuts if isinstance(cuts, Sequence) else [cuts]
            cuts = CutSet.from_cuts(items)

        too_long = [cut.id for cut in cuts if cut.duration > WINDOW_SECONDS + 1e-3]
        if too_long:
            raise ValueError(
                "training cuts must be at most 30 seconds; found "
                + ", ".join(repr(cut_id) for cut_id in too_long[:3])
            )

        audio, audio_lengths, cuts = self.load_audio(cuts)
        cut_list = list(cuts)
        waveforms = []
        targets = []
        texts = []
        masks = []

        for row, (cut, length) in enumerate(zip(cut_list, audio_lengths.tolist())):
            if cut.sampling_rate != SAMPLE_RATE:
                raise ValueError(
                    f"cut {cut.id!r} has sample rate {cut.sampling_rate}; "
                    f"resample manifests to {SAMPLE_RATE} Hz"
                )

            waveform = audio[row, : int(length)].to(torch.float32)
            if (
                self.musan is not None
                and torch.rand(()).item() < self.augmentation.musan_probability
            ):
                waveform = self.musan(waveform)

            target = target_speaker_from_cut(cut)
            text = target_text(cut, target, self.normalize)
            if not text:
                raise ValueError(
                    f"target speaker {target!r} has no transcript in cut {cut.id!r}"
                )

            waveforms.append(waveform.cpu().numpy())
            targets.append(target)
            texts.append(text)
            masks.append(
                lhotse_cut_to_stno(
                    cut,
                    target_speaker=target,
                    duration=WINDOW_SECONDS,
                    frame_hz=STNO_FRAME_HZ,
                )
            )

        features = self.processor.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            max_length=WINDOW_SAMPLES,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        stno_mask = torch.stack(masks)
        if stno_mask.shape[1:] != (4, STNO_FRAMES):
            raise RuntimeError(f"unexpected STNO shape: {tuple(stno_mask.shape)}")

        if self.is_train:
            cfg = self.augmentation
            if (
                cfg.stno_segment_probability > 0
                and torch.rand(()).item() < cfg.stno_segment_probability
            ):
                stno_mask = soft_segment_augmentation(
                    stno_mask,
                    change_prob=cfg.stno_segment_change_probability,
                    min_seg_len=cfg.stno_segment_min_frames,
                    max_seg_len=cfg.stno_segment_max_frames,
                )
            stno_mask = add_gaussian_noise_and_rescale(
                stno_mask,
                variance=cfg.stno_gaussian_variance,
                fraction=cfg.stno_gaussian_probability,
            )

        tokenized = self.processor.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.masked_fill(
            tokenized.attention_mask.ne(1), -100
        )

        batch: dict[str, object] = {
            "input_features": features.input_features,
            "attention_mask": features.attention_mask,
            "stno_mask": stno_mask,
            "labels": labels,
        }
        if self.return_metadata:
            batch.update(
                cut_ids=[cut.id for cut in cut_list],
                target_speakers=targets,
                texts=texts,
            )
        return batch


def load_weighted_cutsets(
    manifest_paths: Sequence[str | Path],
    weights: Sequence[float],
    *,
    seed: int = 0,
    infinite: bool = True,
    min_cut_duration: float | None = None,
    max_cut_duration: float | None = None,
) -> CutSet:
    """Lazily mix target-expanded manifests using the requested corpus weights."""

    if len(manifest_paths) != len(weights) or not manifest_paths:
        raise ValueError("manifest_paths and weights must have the same non-zero length")
    target_filter = TargetTextFilter()
    cutsets = [
        CutSet.from_jsonl_lazy(path).filter(target_filter)
        for path in manifest_paths
    ]
    if min_cut_duration is not None or max_cut_duration is not None:
        minimum = 0.0 if min_cut_duration is None else min_cut_duration
        maximum = math.inf if max_cut_duration is None else max_cut_duration
        cutsets = [
            cuts.filter(lambda cut: minimum <= cut.duration <= maximum)
            for cuts in cutsets
        ]
    if infinite:
        return CutSet.infinite_mux(*cutsets, weights=list(weights), seed=seed)
    return CutSet.mux(*cutsets, weights=list(weights), seed=seed)


def make_dataloader(
    cuts: CutSet,
    dataset: DiCoWDataset,
    *,
    max_duration: float,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 0,
    use_bucketing: bool = True,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Construct a rank-sharded Lhotse sampler/DataLoader pair.

    Keep this loader outside ``Accelerator.prepare``. Accelerate rebuilds
    loaders with ``batch_size=None`` and replaces their Lhotse sampler with a
    ``SequentialSampler``, which incorrectly requires ``len(dataset)``.
    """

    sampler_type = DynamicBucketingSampler if use_bucketing else DynamicCutSampler
    sampler = sampler_type(
        cuts,
        max_duration=max_duration,
        shuffle=shuffle,
        seed=seed,
        world_size=world_size,
        rank=rank,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
    )
