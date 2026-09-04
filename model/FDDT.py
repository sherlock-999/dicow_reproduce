"""Feature-wise Diarization-Dependent Transformation (FDDT).

The STNO mask has four channels in this order:

    silence, target speaker, non-target speaker, overlap

Each channel selects a learned diagonal affine transformation.  A diagonal
transformation is deliberately used here: it is the simplest form described
by DiCoW and adds only ``2 * 4 * d_model`` parameters per encoder layer.
"""

from __future__ import annotations

import torch
from torch import nn


class DiagonalAffine(nn.Module):
    """Apply an independent scale and bias to every hidden feature."""

    def __init__(self, size: int, initial_scale: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((size,), float(initial_scale)))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.weight + self.bias


class FDDT(nn.Module):
    """Mix four diagonal affine transforms according to an STNO mask."""

    def __init__(self, d_model: int, non_target_scale: float = 1.0) -> None:
        super().__init__()

        # Target speech and overlap initially pass through unchanged.  Silence
        # and non-target speech may optionally be attenuated at initialization.
        self.silence = DiagonalAffine(d_model, non_target_scale)
        self.target = DiagonalAffine(d_model, 1.0)
        self.non_target = DiagonalAffine(d_model, non_target_scale)
        self.overlap = DiagonalAffine(d_model, 1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        stno_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Encoder states shaped ``[batch, frames, d_model]``.
            stno_mask: STNO probabilities shaped ``[batch, 4, frames]``.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, frames, d_model], "
                f"got {tuple(hidden_states.shape)}"
            )
        if stno_mask.ndim != 3 or stno_mask.shape[1] != 4:
            raise ValueError(
                "stno_mask must have shape [batch, 4, frames], "
                f"got {tuple(stno_mask.shape)}"
            )
        if stno_mask.shape[0] != hidden_states.shape[0]:
            raise ValueError("STNO batch size does not match hidden states")
        if stno_mask.shape[2] != hidden_states.shape[1]:
            raise ValueError("STNO frame count does not match hidden states")

        mask = stno_mask.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).unsqueeze(-1)

        return (
            self.silence(hidden_states) * mask[:, 0]
            + self.target(hidden_states) * mask[:, 1]
            + self.non_target(hidden_states) * mask[:, 2]
            + self.overlap(hidden_states) * mask[:, 3]
        )

