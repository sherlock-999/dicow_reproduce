"""Convert oracle diarization into the STNO mask used by DiCoW.

STNO means:

    S = silence
    T = target speaker only
    N = one or more non-target speakers, without the target
    O = overlap between the target and at least one non-target speaker

The returned channel order is always ``[S, T, N, O]``.  Whisper encodes audio
at 50 frames per second.  We keep the mask for the complete recording here;
the decoding loop selects the matching 1500-frame mask for each 30-second
Whisper window.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping, Sequence

import torch


STNO_CHANNELS = ("silence", "target", "non_target", "overlap")


@dataclass(frozen=True)
class DiarizationSegment:
    """One oracle diarization segment, with times measured in seconds."""

    speaker: str
    start: float
    end: float


def speaker_activity_to_stno(
    speaker_activity: torch.Tensor,
    target_index: int,
) -> torch.Tensor:
    """Convert per-speaker activity ``[speakers, frames]`` to ``[4, frames]``.

    Activity may be binary or soft, but every value must lie in ``[0, 1]``.
    The four output channels form a probability simplex at every frame.
    """
    activity = torch.as_tensor(speaker_activity, dtype=torch.float32)
    if activity.ndim != 2 or activity.shape[0] == 0:
        raise ValueError("speaker_activity must have shape [speakers, frames]")
    if not 0 <= target_index < activity.shape[0]:
        raise IndexError(
            f"target_index={target_index} is invalid for "
            f"{activity.shape[0]} speakers"
        )
    if torch.any((activity < 0) | (activity > 1)):
        raise ValueError("speaker activity values must lie in [0, 1]")

    target = activity[target_index]
    other_indices = [i for i in range(activity.shape[0]) if i != target_index]

    if other_indices:
        # Probability that every non-target speaker is inactive.
        no_other = torch.prod(1.0 - activity[other_indices], dim=0)
    else:
        no_other = torch.ones_like(target)

    silence = (1.0 - target) * no_other
    target_only = target * no_other
    non_target = (1.0 - target) * (1.0 - no_other)
    overlap = target * (1.0 - no_other)

    return torch.stack((silence, target_only, non_target, overlap), dim=0)


def diarization_to_stno(
    segments: Sequence[DiarizationSegment | Mapping[str, Any]],
    target_speaker: str,
    duration: float,
    frame_hz: int = 50,
) -> torch.Tensor:
    """Convert a complete diarization result to a ``[4, frames]`` mask.

    Each item can be a :class:`DiarizationSegment` or a mapping containing
    ``speaker``, ``start``, and ``end``.  Partial-frame activity is represented
    by the fraction of the frame covered by a segment.  ``duration`` must be
    the complete audio duration, so long recordings are never silently cut.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if frame_hz <= 0:
        raise ValueError("frame_hz must be positive")

    parsed = [_as_segment(segment) for segment in segments]
    speakers = sorted({segment.speaker for segment in parsed} | {target_speaker})
    speaker_to_index = {speaker: index for index, speaker in enumerate(speakers)}

    num_frames = round(duration * frame_hz)
    activity = torch.zeros((len(speakers), num_frames), dtype=torch.float32)

    for segment in parsed:
        start = max(0.0, min(float(duration), segment.start))
        end = max(0.0, min(float(duration), segment.end))
        if end <= start:
            continue

        first_frame = max(0, int(start * frame_hz))
        last_frame = min(num_frames, ceil(end * frame_hz))
        speaker_index = speaker_to_index[segment.speaker]

        for frame in range(first_frame, last_frame):
            frame_start = frame / frame_hz
            frame_end = (frame + 1) / frame_hz
            covered = max(0.0, min(end, frame_end) - max(start, frame_start))
            occupancy = min(1.0, covered * frame_hz)
            old_occupancy = float(activity[speaker_index, frame])
            activity[speaker_index, frame] = max(old_occupancy, occupancy)

    return speaker_activity_to_stno(
        activity,
        target_index=speaker_to_index[target_speaker],
    )


def lhotse_cut_to_stno(
    cut: Any,
    target_speaker: str,
    duration: float | None = None,
    frame_hz: int = 50,
) -> torch.Tensor:
    """Build a complete oracle STNO mask directly from a Lhotse cut.

    This reads the cut's supervision intervals and therefore supports both
    Lhotse ``MonoCut`` and ``MixedCut`` objects without allocating a large
    sample-level diarization matrix.  By default the result covers the complete
    cut.  An explicit ``duration`` may be used for already-windowed training
    examples; shorter cuts are right-padded as pure silence.
    """
    if duration is None:
        duration = float(cut.duration)
    if duration <= 0:
        raise ValueError("duration must be positive")

    segments = [
        DiarizationSegment(
            speaker=supervision.speaker,
            start=float(supervision.start),
            end=float(supervision.end),
        )
        for supervision in cut.supervisions
    ]
    return diarization_to_stno(
        segments=segments,
        target_speaker=target_speaker,
        duration=duration,
        frame_hz=frame_hz,
    )


def slice_stno_mask(
    stno_mask: torch.Tensor,
    start_frame: int,
    num_frames: int = 1500,
) -> torch.Tensor:
    """Select one decoding window and pad its tail with pure silence.

    Args:
        stno_mask: Full mask shaped ``[4, total_frames]`` or
            ``[batch, 4, total_frames]``.
        start_frame: Window start on the 50 Hz STNO time axis.
        num_frames: Window length.  Whisper's default is 1500 frames (30 s).
    """
    mask = torch.as_tensor(stno_mask)
    unbatched = mask.ndim == 2
    if unbatched:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[1] != 4:
        raise ValueError(
            "stno_mask must have shape [4, frames] or [batch, 4, frames]"
        )
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    window = mask[..., start_frame : start_frame + num_frames]
    missing_frames = num_frames - window.shape[-1]

    if missing_frames > 0:
        silence_padding = torch.zeros(
            (mask.shape[0], 4, missing_frames),
            device=mask.device,
            dtype=mask.dtype,
        )
        silence_padding[:, 0] = 1
        window = torch.cat((window, silence_padding), dim=-1)

    return window.squeeze(0) if unbatched else window


def _as_segment(
    segment: DiarizationSegment | Mapping[str, Any],
) -> DiarizationSegment:
    if isinstance(segment, DiarizationSegment):
        return segment
    try:
        return DiarizationSegment(
            speaker=str(segment["speaker"]),
            start=float(segment["start"]),
            end=float(segment["end"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "each diarization segment needs speaker, start, and end"
        ) from error
