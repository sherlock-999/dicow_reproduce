"""Integration checks for Lhotse -> Whisper -> DiCoW training."""

from pathlib import Path

import torch
from lhotse import CutSet, MonoCut, SupervisionSegment
from lhotse.utils import fastcopy
from transformers import WhisperConfig, WhisperProcessor

from data.augment import add_gaussian_noise_and_rescale, soft_segment_augmentation
from data.dataset import DiCoWDataset, speakers_in_cut
from data.export_ts_cuts import expanded_cuts
from model.DiCoW import DiCoW
from train.train import (
    build_optimizer_and_scheduler,
    configure_trainable_parameters,
    parse_args,
    ranked_checkpoints,
    word_errors,
)


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


def test_target_expansion_skips_speakers_without_transcripts():
    cut = MonoCut(
        id="example",
        start=0.0,
        duration=2.0,
        channel=0,
        supervisions=[
            SupervisionSegment(
                id="empty-a",
                recording_id="recording",
                start=0.0,
                duration=0.5,
                channel=0,
                speaker="a",
                text="",
            ),
            SupervisionSegment(
                id="spoken-b",
                recording_id="recording",
                start=0.5,
                duration=1.0,
                channel=0,
                speaker="b",
                text="hello",
            ),
        ],
    )

    expanded = list(expanded_cuts(CutSet.from_cuts([cut])))

    assert [item.id for item in expanded] == ["example_tsidx1"]


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


def test_dicon_v1_training_protocol_is_encoded_in_yaml():
    args = parse_args(["--config", str(ROOT / "configs" / "dicow_v1.yaml")])

    assert args.weights == [6, 6, 1, 1, 1, 1]
    assert args.validation_manifest.endswith("notsofar1_dev1_30s_ts.jsonl.gz")
    assert args.max_steps == 40_000
    assert args.max_duration == 200
    assert args.validation_max_duration == 200
    assert args.encoder_learning_rate == 5e-5
    assert args.fddt_learning_rate == 5e-4
    assert args.fddt_only_steps == args.warmup_steps == 2_000
    assert args.eval_steps == 1_000
    assert args.max_eval_batches == 60
    assert args.save_top_k == 2


def test_word_error_and_top_k_checkpoint_selection():
    assert word_errors("one two three", "one four three five") == (2, 3)

    checkpoints = [
        {"path": "step-1000", "step": 1000, "val_wer": 0.30},
        {"path": "step-2000", "step": 2000, "val_wer": 0.20},
    ]
    ranked, retained = ranked_checkpoints(
        checkpoints,
        {"path": "step-3000", "step": 3000, "val_wer": 0.10},
        save_top_k=2,
    )

    assert retained
    assert [item["path"] for item in ranked] == ["step-3000", "step-2000"]


def test_encoder_learning_rate_is_delayed_while_fddt_warms_up():
    model = tiny_model()
    configure_trainable_parameters(model)
    args = parse_args(
        [
            "--config",
            str(ROOT / "configs" / "debug.yaml"),
            "--max-steps",
            "10",
            "--warmup-steps",
            "2",
            "--fddt-only-steps",
            "3",
        ]
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

    assert scheduler.get_last_lr() == [0.0, 0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [args.fddt_learning_rate / 2, 0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [args.fddt_learning_rate, 0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[1] == 0.0
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[1] == args.encoder_learning_rate / 2
