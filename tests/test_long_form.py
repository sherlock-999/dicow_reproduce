"""End-to-end structural test for DiCoW long-form decoding.

This uses a tiny random Whisper model so the test is fast on CPU.  The feature
extractor, 30-second Whisper loop, dynamic seek, STNO slicing, and real audio
dimensions are identical to the production model.
"""

from pathlib import Path

import torch
from lhotse import CutSet
from lhotse.utils import fastcopy
from transformers import GenerationConfig, WhisperConfig, WhisperFeatureExtractor

from model.DiCoW import DiCoW
from model.stno import lhotse_cut_to_stno, slice_stno_mask


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "whisper-large-v3-turbo"
MANIFEST = (
    ROOT
    / "test_data"
    / "long_form"
    / "notsofar1_MTG_32009_354s.jsonl"
)
AUDIO = ROOT / "test_data" / "long_form" / "notsofar1_MTG_32009_354s.wav"


def load_portable_cut():
    """Replace the creator machine's manifest path with this checkout's WAV."""

    cut = next(iter(CutSet.from_file(MANIFEST)))
    sources = [fastcopy(source, source=str(AUDIO)) for source in cut.recording.sources]
    recording = fastcopy(cut.recording, sources=sources)
    return fastcopy(cut, recording=recording)


class TracedDiCoW(DiCoW):
    """Record the STNO seek chosen by Whisper at each decoding iteration."""

    def __init__(self, config):
        super().__init__(config)
        self.stno_starts = []

    def generate_with_fallback(self, *args, **kwargs):
        seek = kwargs["seek"]
        batch_idx_map = kwargs["batch_idx_map"]
        input_stride = (
            self.model.encoder.conv1.stride[0]
            * self.model.encoder.conv2.stride[0]
        )
        self.stno_starts.extend(
            int(seek[index].item()) // input_stride for index in batch_idx_map
        )
        return super().generate_with_fallback(*args, **kwargs)


def make_tiny_model() -> TracedDiCoW:
    config = WhisperConfig.from_pretrained(MODEL_DIR)
    config.d_model = 32
    config.encoder_layers = 1
    config.decoder_layers = 1
    config.encoder_attention_heads = 4
    config.decoder_attention_heads = 4
    config.encoder_ffn_dim = 64
    config.decoder_ffn_dim = 64

    model = TracedDiCoW(config).eval()
    model.generation_config = GenerationConfig.from_pretrained(MODEL_DIR)
    return model


def test_long_audio_uses_matching_stno_windows():
    torch.manual_seed(0)

    cut = load_portable_cut()
    audio = cut.load_audio()[0]
    full_stno = lhotse_cut_to_stno(cut, target_speaker="Ron").unsqueeze(0)

    extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    inputs = extractor(
        audio,
        sampling_rate=cut.sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=False,
        padding="longest",
    )

    assert cut.duration > 30
    assert inputs.input_features.shape[-1] > 3000
    assert full_stno.shape[-1] > 1500

    model = make_tiny_model()
    observed_windows = []
    original_set_stno_mask = model.model.encoder.set_stno_mask

    def record_stno_window(stno_mask):
        observed_windows.append(stno_mask.detach().cpu().clone())
        original_set_stno_mask(stno_mask)

    model.model.encoder.set_stno_mask = record_stno_window

    with torch.no_grad():
        model.generate(
            input_features=inputs.input_features,
            attention_mask=inputs.attention_mask,
            stno_mask=full_stno,
            return_timestamps=True,
            max_new_tokens=4,
        )

    assert len(model.stno_starts) > 1
    assert model.stno_starts == sorted(model.stno_starts)
    assert model.stno_starts[-1] > 1500
    assert len(observed_windows) == len(model.stno_starts)

    for start, observed in zip(model.stno_starts, observed_windows):
        expected = slice_stno_mask(full_stno, start_frame=start)
        torch.testing.assert_close(observed, expected)
