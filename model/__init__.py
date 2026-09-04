"""Whisper large-v3-turbo with per-encoder-layer FDDT conditioning."""

from .DiCoW import DiCoW, DiCoWEncoder, DiCoWForConditionalGeneration
from .FDDT import FDDT

__all__ = ["DiCoW", "DiCoWEncoder", "DiCoWForConditionalGeneration", "FDDT"]
