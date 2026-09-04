"""A minimal DiCoW implementation built on Hugging Face Whisper.

The only architectural change is one diagonal FDDT immediately before every
Whisper encoder transformer layer.  The decoder and the ordinary Whisper
sequence-to-sequence loss are unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperEncoder

try:
    from .FDDT import FDDT
    from .stno import slice_stno_mask
except ImportError:  # Allows running this file directly during development.
    from FDDT import FDDT
    from stno import slice_stno_mask


class DiCoWEncoder(WhisperEncoder):
    """Whisper encoder with an FDDT before every transformer block."""

    def __init__(self, config) -> None:
        super().__init__(config)

        # The first FDDT starts by attenuating silence and non-target speech.
        # Every later FDDT starts as the identity and learns from training.
        initial_scale = getattr(config, "fddt_initial_non_target_scale", 0.5)
        self.fddts = torch.nn.ModuleList(
            [
                FDDT(
                    config.d_model,
                    non_target_scale=initial_scale if index == 0 else 1.0,
                )
                for index in range(len(self.layers))
            ]
        )

        self._stno_mask: Optional[torch.Tensor] = None
        self._fddt_hooks = [
            layer.register_forward_pre_hook(self._make_fddt_hook(index))
            for index, layer in enumerate(self.layers)
        ]

    def _make_fddt_hook(self, layer_index: int):
        def apply_fddt(_module, inputs):
            hidden_states = inputs[0]
            if self._stno_mask is None:
                raise RuntimeError("Set stno_mask before running the DiCoW encoder")

            transformed = self.fddts[layer_index](
                hidden_states,
                self._stno_mask,
            )
            return (transformed, *inputs[1:])

        return apply_fddt

    def set_stno_mask(self, stno_mask: torch.Tensor) -> None:
        """Store the mask used by the per-layer forward hooks."""
        if stno_mask.ndim == 2:
            stno_mask = stno_mask.unsqueeze(0)
        if stno_mask.ndim != 3 or stno_mask.shape[1] != 4:
            raise ValueError(
                "stno_mask must have shape [batch, 4, frames] or [4, frames]"
            )
        self._stno_mask = stno_mask

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        stno_mask=None,
    ):
        if stno_mask is not None:
            self.set_stno_mask(stno_mask)
        if self._stno_mask is None:
            raise ValueError("DiCoWEncoder requires an stno_mask")

        try:
            return super().forward(
                input_features=input_features,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        finally:
            # Never accidentally reuse a mask from a previous batch.
            self._stno_mask = None


class DiCoWForConditionalGeneration(WhisperForConditionalGeneration):
    """Whisper with a diarization-conditioned encoder and frozen decoder."""

    def __init__(self, config) -> None:
        if not hasattr(config, "fddt_initial_non_target_scale"):
            config.fddt_initial_non_target_scale = 0.5

        super().__init__(config)
        self.model.encoder = DiCoWEncoder(config)
        self.config.architectures = [self.__class__.__name__]
        self.freeze_for_encoder_finetuning()

    def freeze_for_encoder_finetuning(self) -> None:
        """Train the Whisper encoder and FDDTs; freeze decoder and LM head."""
        self.requires_grad_(False)
        self.model.encoder.requires_grad_(True)

        # Whisper positional embeddings are fixed in the original model.
        self.model.encoder.embed_positions.requires_grad_(False)

    def forward(
        self,
        input_features=None,
        stno_mask=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        encoder_outputs = kwargs.get("encoder_outputs")

        if encoder_outputs is None:
            if stno_mask is None:
                raise ValueError("DiCoW requires stno_mask when encoding audio")
            self.model.encoder.set_stno_mask(stno_mask)

        return super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, input_features=None, stno_mask=None, **kwargs):
        """Decode short or long audio using one full-recording STNO mask.

        Whisper owns the long-form loop and its timestamp-driven seek.  The
        override below slices the matching STNO window before every encoder
        call, so audio and diarization always advance together.
        """
        if stno_mask is None:
            raise ValueError("DiCoW.generate requires a full-recording stno_mask")
        if stno_mask.ndim == 2:
            stno_mask = stno_mask.unsqueeze(0)
        if stno_mask.ndim != 3 or stno_mask.shape[1] != 4:
            raise ValueError(
                "stno_mask must have shape [batch, 4, frames] or [4, frames]"
            )

        # All reproduction datasets are English.  Setting this explicitly
        # skips Whisper's separate language-detection encoder call.
        kwargs.setdefault("language", "en")
        kwargs.setdefault("task", "transcribe")

        return super().generate(
            input_features=input_features,
            stno_mask=stno_mask,
            **kwargs,
        )

    def generate_with_fallback(
        self,
        segment_input,
        decoder_input_ids,
        cur_bsz,
        seek,
        batch_idx_map,
        temperatures,
        generation_config,
        logits_processor,
        stopping_criteria,
        prefix_allowed_tokens_fn,
        synced_gpus,
        return_token_timestamps,
        do_condition_on_prev_tokens,
        is_shortform,
        batch_size,
        attention_mask,
        kwargs,
    ):
        """Attach the STNO window matching Whisper's current audio window."""
        kwargs = dict(kwargs)
        full_stno_mask = kwargs.get("stno_mask")
        if full_stno_mask is None:
            raise ValueError("Long-form DiCoW decoding requires an stno_mask")

        # Whisper's seek is measured in log-Mel frames (100 Hz).  Dividing by
        # the encoder convolution stride maps it to STNO/encoder frames (50 Hz).
        input_stride = (
            self.model.encoder.conv1.stride[0]
            * self.model.encoder.conv2.stride[0]
        )
        stno_windows = []
        for current_index in range(cur_bsz):
            original_index = batch_idx_map[current_index]
            stno_start = int(seek[original_index].item()) // input_stride
            stno_windows.append(
                slice_stno_mask(
                    full_stno_mask[original_index : original_index + 1],
                    start_frame=stno_start,
                    num_frames=self.config.max_source_positions,
                )
            )
        kwargs["stno_mask"] = torch.cat(stno_windows, dim=0)

        if attention_mask is not None:
            attention_mask = attention_mask[batch_idx_map]

        return super().generate_with_fallback(
            segment_input=segment_input,
            decoder_input_ids=decoder_input_ids,
            cur_bsz=cur_bsz,
            seek=seek,
            batch_idx_map=batch_idx_map,
            temperatures=temperatures,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            return_token_timestamps=return_token_timestamps,
            do_condition_on_prev_tokens=do_condition_on_prev_tokens,
            is_shortform=is_shortform,
            batch_size=batch_size,
            attention_mask=attention_mask,
            kwargs=kwargs,
        )


# Short name used by the reproduction scripts.
DiCoW = DiCoWForConditionalGeneration
