"""Compatibility imports for the single Whisper STNO implementation.

STNO conversion belongs to ``model.stno`` because the 50 Hz alignment is a
property of Whisper. This module prevents a second NeMo-specific implementation
from drifting away from the mask used by the model.
"""

from model.stno import (  # noqa: F401
    STNO_CHANNELS,
    DiarizationSegment,
    diarization_to_stno,
    lhotse_cut_to_stno,
    slice_stno_mask,
    speaker_activity_to_stno,
)

__all__ = [
    "STNO_CHANNELS",
    "DiarizationSegment",
    "diarization_to_stno",
    "lhotse_cut_to_stno",
    "slice_stno_mask",
    "speaker_activity_to_stno",
]
