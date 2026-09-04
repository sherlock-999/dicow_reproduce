"""Integration checks for Lhotse -> Whisper -> DiCoW training."""

from pathlib import Path

import torch
from lhotse import CutSet
from lhotse.utils import fastcopy
from transformers import WhisperConfig, WhisperProcessor

from data.augment import add_gaussian_noise_and_rescale, soft_segment_augmentation
from data.dataset import DiCoWDataset, speakers_in_cut
from model.DiCoW import DiCoW
from train.train import parse_args


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "whisper-large-v3-turbo"
MANIFEST = ROOT / "test_data" / "long_form" / "notsofar1_MTG_32009_354s.jsonl"
AUDIO = ROOT / "test_data" / "long_form" / "notsofar1_MTG_32009_354s.wav"


def training_cut():
    recording = next(iter(CutSet.from_file(MANIFEST)))
    sources = [
        fastcopy(source, source=str(AUDIO))
        for source in recording.recording.sources
    ]
    portable_recording = fastcopy(recording.recording, sources=sources)
    recording = fastcopy(recording, recording=portable_recording)
    cut = recording.truncate(
        offset=45,
        duration=30,
        keep_excessive_supervisions=False,
    )
    assert speakers_in_cut(cut)
    return fastcopy(cut, id=f"{cut.id}_tsidx0")


def tiny_model() -> DiCoW:
    config = WhisperConfig.from_pretrained(MODEL_DIR)
    config.d_model = 32
    config.encoder_layers = 1
    config.decoder_layers = 1
    config.encoder_attention_heads = 4
    config.decoder_attention_heads = 4
    config.encoder_ffn_dim = 64
    config.decoder_ffn_dim = 64
    return DiCoW(config)


def test_lhotse_batch_matches_dicow_forward():
    processor = WhisperProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    dataset = DiCoWDataset(processor, is_train=False, return_metadata=True)
    batch = dataset[CutSet.from_cuts([training_cut()])]

    assert batch["input_features"].shape == (1, 128, 3000)
    assert batch["attention_mask"].shape == (1, 3000)
    assert batch["stno_mask"].shape == (1, 4, 1500)
    torch.testing.assert_close(
        batch["stno_mask"].sum(dim=1),
        torch.ones((1, 1500)),
    )
    assert batch["texts"]

    tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    model = tiny_model().train()
    result = model(**tensors)
    assert torch.isfinite(result.loss)
    result.loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.model.encoder.parameters()
        if parameter.requires_grad
    )
    assert all(parameter.grad is None for parameter in model.model.decoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.model.decoder.parameters())


def test_stno_augmentations_keep_probability_simplex_for_batch_size_one():
    mask = torch.zeros((1, 4, 100))
    mask[:, 0] = 1
    gaussian = add_gaussian_noise_and_rescale(mask, variance=0.2, fraction=1.0)
    segmented = soft_segment_augmentation(
        gaussian,
        change_prob=1.0,
        min_seg_len=5,
        max_seg_len=10,
    )

    assert torch.all(segmented >= 0)
    torch.testing.assert_close(segmented.sum(dim=1), torch.ones((1, 100)))


def test_training_yaml_is_executable_and_cli_can_override_it():
    args = parse_args(
        [
            "--config",
            str(ROOT / "configs" / "debug.yaml"),
            "--max-steps",
            "3",
        ]
    )

    assert args.train_manifest == ["manifests/notsofar1_train_30s_ts.jsonl.gz"]
    assert args.weights == [1]
    assert args.max_steps == 3
    assert args.mixed_precision == "no"
    assert args.stno_gaussian_probability == 0.0
