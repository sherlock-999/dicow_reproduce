"""Fine-tune the Whisper encoder and FDDT layers for DiCoW.

This is intentionally a plain training loop. The decoder and output projection
remain frozen; only the Whisper encoder and its per-layer FDDTs are optimized.

Example with the six target-expanded training manifests::

    accelerate launch -m train.train \
        --train-manifest notsofar1_train_30s_ts.jsonl.gz \
                         ami-sdm_train_30s_ts.jsonl.gz \
                         libri2mix_100_noisy_30s_ts.jsonl.gz \
                         libri2mix_360_noisy_30s_ts.jsonl.gz \
                         libri3mix_360_noisy_30s_ts.jsonl.gz \
                         librispeechmix_train_3mix_ts.jsonl.gz \
        --weights 6 6 1 1 1 1 \
        --validation-manifest dev_30s_ts.jsonl.gz \
        --output-dir outputs/dicow

The optimization, validation, and checkpoint defaults match DiCoN v1. Test
sets are intentionally kept out of training and are evaluated separately.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from lhotse import CutSet
from transformers import WhisperProcessor

from data.dataset import (
    AugmentationConfig,
    DiCoWDataset,
    load_weighted_cutsets,
    make_dataloader,
)
from model.DiCoW import DiCoW
from txt_norm import get_text_norm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="YAML configuration. Explicit command-line arguments override it.",
    )
    parser.add_argument("--model", default="whisper-large-v3-turbo")
    parser.add_argument("--train-manifest", nargs="+", default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=[6, 6, 1, 1, 1, 1])
    parser.add_argument("--validation-manifest", default=None)
    parser.add_argument("--musan-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None, help="Path to a saved checkpoint directory")

    parser.add_argument("--max-steps", type=int, default=40_000)
    parser.add_argument("--max-duration", type=float, default=200.0,
                        help="Maximum summed audio seconds in one micro-batch")
    parser.add_argument("--min-cut-duration", type=float, default=0.4)
    parser.add_argument("--max-cut-duration", type=float, default=31.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--encoder-learning-rate", type=float, default=5e-5)
    parser.add_argument("--fddt-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--fddt-only-steps", type=int, default=2_000,
                        help="Optional initial steps that update only FDDTs")
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--max-eval-batches", type=int, default=60)
    parser.add_argument("--save-top-k", type=int, default=2)
    parser.add_argument("--validation-max-duration", type=float, default=200.0)
    parser.add_argument("--validation-num-workers", type=int, default=4)
    parser.add_argument("--stno-gaussian-probability", type=float, default=0.75)
    parser.add_argument("--stno-gaussian-variance", type=float, default=0.2)
    parser.add_argument("--stno-segment-probability", type=float, default=0.3)
    parser.add_argument("--stno-segment-change-probability", type=float, default=0.1)
    parser.add_argument("--stno-segment-min-frames", type=int, default=5)
    parser.add_argument("--stno-segment-max-frames", type=int, default=20)
    parser.add_argument("--musan-probability", type=float, default=0.3)
    parser.add_argument("--musan-min-snr-db", type=int, default=0)
    parser.add_argument("--musan-max-snr-db", type=int, default=15)
    return parser


def flatten_config(mapping: dict) -> dict:
    """Flatten readable YAML sections into argparse destination names."""

    flattened = {}
    for key, value in mapping.items():
        key = key.replace("-", "_")
        if isinstance(value, dict):
            nested = flatten_config(value)
            duplicate = set(flattened).intersection(nested)
            if duplicate:
                raise ValueError(f"duplicate YAML option(s): {sorted(duplicate)}")
            flattened.update(nested)
        else:
            flattened[key] = value
    return flattened


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config")
    known, _ = config_probe.parse_known_args(argv)

    if known.config:
        config_path = Path(known.config)
        loaded = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(loaded, dict):
            parser.error("the YAML root must be a mapping")
        defaults = flatten_config(loaded)
        valid_options = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - valid_options)
        if unknown:
            parser.error(f"unknown YAML option(s): {', '.join(unknown)}")
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    if not args.train_manifest:
        parser.error("provide --train-manifest or data.train_manifest in YAML")
    if not args.output_dir:
        parser.error("provide --output-dir or output.output_dir in YAML")
    if args.mixed_precision not in {"no", "fp16", "bf16"}:
        parser.error("mixed_precision must be one of: no, fp16, bf16")
    if args.save_top_k < 1:
        parser.error("save_top_k must be at least 1")
    return args


def configure_trainable_parameters(model: DiCoW) -> None:
    """Expose the encoder to DDP while keeping the decoder frozen."""

    model.requires_grad_(False)
    model.model.encoder.requires_grad_(True)
    model.model.encoder.embed_positions.requires_grad_(False)


def enter_train_mode(model, accelerator: Accelerator) -> None:
    """Train the encoder while leaving the frozen decoder deterministic."""

    model.train()
    accelerator.unwrap_model(model).model.decoder.eval()


def verify_frozen_decoder(model: DiCoW) -> None:
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("model.encoder.")
    ]
    if unexpected:
        raise RuntimeError(f"non-encoder parameters are trainable: {unexpected[:5]}")


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return Levenshtein word errors and reference word count."""

    ref = reference.split()
    hyp = hypothesis.split()
    previous = list(range(len(hyp) + 1))
    for ref_word in ref:
        current = [previous[0] + 1]
        for index, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[index] + 1,
                    previous[index - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1], len(ref)


def build_optimizer_and_scheduler(model: DiCoW, args: argparse.Namespace):
    """Match DiCoN v1's FDDT/encoder groups and delayed cosine schedule."""

    fddt_parameters = list(model.model.encoder.fddts.parameters())
    fddt_ids = {id(parameter) for parameter in fddt_parameters}
    encoder_parameters = [
        parameter
        for parameter in model.model.encoder.parameters()
        if parameter.requires_grad and id(parameter) not in fddt_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": fddt_parameters,
                "lr": args.fddt_learning_rate,
                "weight_decay": args.weight_decay,
                "name": "fddt",
            },
            {
                "params": encoder_parameters,
                "lr": args.encoder_learning_rate,
                "weight_decay": args.weight_decay,
                "name": "encoder",
            },
        ],
        betas=(0.9, 0.98),
    )

    def cosine(step: int, start: int) -> float:
        if step < start:
            return 0.0
        if step < start + args.warmup_steps:
            return (step - start) / max(args.warmup_steps, 1)
        progress = (step - start - args.warmup_steps) / max(
            args.max_steps - start - args.warmup_steps, 1
        )
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(progress, 1.0))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            lambda step: cosine(step, 0),
            lambda step: cosine(step, args.fddt_only_steps),
        ],
    )
    return optimizer, scheduler


def move_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Keep metadata out of model.forward and move only tensor fields."""

    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


@torch.no_grad()
def validation_metrics(
    model,
    loader,
    accelerator: Accelerator,
    processor: WhisperProcessor,
    max_batches: int,
) -> dict[str, float]:
    """Decode the development cuts and aggregate loss and WER over all GPUs."""

    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    normalize = get_text_norm("whisper_nsf")
    unwrapped = accelerator.unwrap_model(model)
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        references = batch["texts"]
        tensors = move_batch(batch, accelerator.device)
        count = tensors["input_features"].shape[0]
        loss = model(**tensors).loss
        token_ids = unwrapped.generate(
            input_features=tensors["input_features"],
            attention_mask=tensors["attention_mask"],
            stno_mask=tensors["stno_mask"],
            language="en",
            task="transcribe",
            use_cache=True,
        )
        hypotheses = [normalize(text) for text in processor.batch_decode(
            token_ids, skip_special_tokens=True
        )]
        errors = words = 0
        for reference, hypothesis in zip(references, hypotheses):
            utterance_errors, utterance_words = word_errors(reference, hypothesis)
            errors += utterance_errors
            words += utterance_words
        totals += torch.tensor(
            [loss.item() * count, count, errors, words],
            dtype=torch.float64,
            device=accelerator.device,
        )

    totals = accelerator.reduce(totals, reduction="sum")
    enter_train_mode(model, accelerator)
    return {
        "loss": float((totals[0] / totals[1].clamp_min(1)).item()),
        "wer": float((totals[2] / totals[3].clamp_min(1)).item()),
        "word_errors": int(totals[2].item()),
        "reference_words": int(totals[3].item()),
    }


def save_checkpoint(
    accelerator: Accelerator,
    model: DiCoW,
    processor: WhisperProcessor,
    checkpoint: Path,
    step: int,
    best_validation_wer: float,
    top_checkpoints: list[dict],
) -> Path:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and checkpoint.exists():
        shutil.rmtree(checkpoint)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        checkpoint,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        processor.save_pretrained(checkpoint)
        (checkpoint / "training_state.json").write_text(
            json.dumps(
                {
                    "global_step": step,
                    "best_validation_wer": (
                        best_validation_wer
                        if math.isfinite(best_validation_wer)
                        else None
                    ),
                    "top_checkpoints": top_checkpoints,
                },
                indent=2,
            )
            + "\n"
        )
    accelerator.save_state(checkpoint / "accelerator_state")
    accelerator.wait_for_everyone()
    return checkpoint


def ranked_checkpoints(
    checkpoints: list[dict], candidate: dict, save_top_k: int
) -> tuple[list[dict], bool]:
    """Return the lowest-WER checkpoints and whether the candidate belongs."""

    ranked = sorted(
        [*checkpoints, candidate],
        key=lambda item: (float(item["val_wer"]), int(item["step"])),
    )[:save_top_k]
    return ranked, any(item["path"] == candidate["path"] for item in ranked)


def remove_checkpoints(
    accelerator: Accelerator, output_dir: Path, checkpoint_names: set[str]
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        for name in checkpoint_names:
            path = output_dir / name
            if path.exists():
                shutil.rmtree(path)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    if len(args.train_manifest) != len(args.weights):
        raise ValueError("--train-manifest and --weights must have the same length")
    if args.fddt_only_steps < 0 or args.fddt_only_steps > args.max_steps:
        raise ValueError("--fddt-only-steps must lie between 0 and --max-steps")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "arguments.json").write_text(
            json.dumps(vars(args), indent=2) + "\n"
        )

    load_path = args.resume or args.model
    processor = WhisperProcessor.from_pretrained(load_path, local_files_only=True)
    model = DiCoW.from_pretrained(load_path, local_files_only=True)
    model.config.use_cache = False

    initial_step = 0
    best_validation_wer = math.inf
    top_checkpoints: list[dict] = []
    if args.resume:
        state_path = Path(args.resume) / "training_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            initial_step = int(state["global_step"])
            stored_best = state.get("best_validation_wer")
            best_validation_wer = (
                float(stored_best) if stored_best is not None else math.inf
            )
            top_checkpoints = list(state.get("top_checkpoints", []))

    fddt_only = initial_step < args.fddt_only_steps
    configure_trainable_parameters(model)
    verify_frozen_decoder(model)

    train_cuts = load_weighted_cutsets(
        args.train_manifest,
        args.weights,
        seed=args.seed,
        infinite=True,
        min_cut_duration=args.min_cut_duration,
        max_cut_duration=args.max_cut_duration,
    )
    train_dataset = DiCoWDataset(
        processor,
        is_train=True,
        augmentation=AugmentationConfig(
            stno_gaussian_probability=args.stno_gaussian_probability,
            stno_gaussian_variance=args.stno_gaussian_variance,
            stno_segment_probability=args.stno_segment_probability,
            stno_segment_change_probability=args.stno_segment_change_probability,
            stno_segment_min_frames=args.stno_segment_min_frames,
            stno_segment_max_frames=args.stno_segment_max_frames,
            musan_probability=args.musan_probability,
            musan_min_snr_db=args.musan_min_snr_db,
            musan_max_snr_db=args.musan_max_snr_db,
        ),
        musan_dir=args.musan_dir,
    )
    train_loader = make_dataloader(
        train_cuts,
        train_dataset,
        max_duration=args.max_duration,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
    )

    validation_loader = None
    if args.validation_manifest:
        validation_cuts = CutSet.from_jsonl_lazy(args.validation_manifest).filter(
            lambda cut: args.min_cut_duration <= cut.duration <= args.max_cut_duration
        )
        validation_dataset = DiCoWDataset(
            processor,
            is_train=False,
            return_metadata=True,
        )
        validation_loader = make_dataloader(
            validation_cuts,
            validation_dataset,
            max_duration=args.validation_max_duration,
            num_workers=args.validation_num_workers,
            shuffle=False,
            seed=args.seed,
            use_bucketing=False,
        )

    optimizer_parameters = [
        parameter
        for parameter in model.model.encoder.parameters()
        if parameter.requires_grad
    ]
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

    if validation_loader is None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    else:
        model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, validation_loader, scheduler
        )
    if args.resume:
        accelerator.load_state(Path(args.resume) / "accelerator_state")

    accelerator.print(
        f"Starting at step {initial_step}; "
        f"trainable parameters={trainable_parameter_count(model):,}; "
        f"stage={'FDDT only' if fddt_only else 'encoder + FDDT'}"
    )
    enter_train_mode(model, accelerator)
    optimizer.zero_grad(set_to_none=True)
    global_step = initial_step
    running_loss = 0.0

    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            if fddt_only and global_step >= args.fddt_only_steps:
                fddt_only = False
                accelerator.print(
                    f"Step {global_step}: enabling Whisper encoder + FDDT training"
                )

            batch = move_batch(batch, accelerator.device)
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        optimizer_parameters, args.max_grad_norm
                    )
                    finite = torch.isfinite(grad_norm).to(torch.int32)
                    finite = accelerator.reduce(finite, reduction="min")
                    if not bool(finite.item()):
                        for parameter in optimizer_parameters:
                            if parameter.grad is not None:
                                parameter.grad.zero_()
                        accelerator.print(
                            f"WARNING step={global_step}: non-finite gradients; "
                            "zeroed this update to protect AdamW state"
                        )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().item())
            if not accelerator.sync_gradients:
                continue

            global_step += 1
            if global_step % args.log_steps == 0:
                mean_loss = running_loss / (args.log_steps * args.gradient_accumulation_steps)
                accelerator.print(
                    f"step={global_step} loss={mean_loss:.4f} "
                    f"fddt_lr={scheduler.get_last_lr()[0]:.3e} "
                    f"encoder_lr={scheduler.get_last_lr()[1]:.3e}"
                )
                running_loss = 0.0

            if validation_loader is not None and global_step % args.eval_steps == 0:
                metrics = validation_metrics(
                    model,
                    validation_loader,
                    accelerator,
                    processor,
                    args.max_eval_batches,
                )
                accelerator.print(
                    f"step={global_step} validation_loss={metrics['loss']:.4f} "
                    f"val_wer={metrics['wer']:.4f} "
                    f"({metrics['word_errors']}/{metrics['reference_words']})"
                )
                best_validation_wer = min(best_validation_wer, metrics["wer"])
                checkpoint_name = (
                    f"checkpoint-step={global_step}-val_wer={metrics['wer']:.4f}"
                )
                candidate = {
                    "path": checkpoint_name,
                    "step": global_step,
                    "val_wer": metrics["wer"],
                }
                previous_names = {item["path"] for item in top_checkpoints}
                top_checkpoints, candidate_is_top = ranked_checkpoints(
                    top_checkpoints, candidate, args.save_top_k
                )
                retained_names = {item["path"] for item in top_checkpoints}

                if candidate_is_top:
                    checkpoint = save_checkpoint(
                        accelerator,
                        model,
                        processor,
                        output_dir / checkpoint_name,
                        global_step,
                        best_validation_wer,
                        top_checkpoints,
                    )
                    accelerator.print(f"Saved top-{args.save_top_k} checkpoint {checkpoint}")

                remove_checkpoints(
                    accelerator,
                    output_dir,
                    previous_names - retained_names,
                )
                last_checkpoint = save_checkpoint(
                    accelerator,
                    model,
                    processor,
                    output_dir / "checkpoint-last",
                    global_step,
                    best_validation_wer,
                    top_checkpoints,
                )
                accelerator.print(f"Updated {last_checkpoint}")

    # DiCoN keeps `last` even when the final step is not a validation boundary.
    checkpoint = save_checkpoint(
        accelerator,
        model,
        processor,
        output_dir / "checkpoint-last",
        global_step,
        best_validation_wer,
        top_checkpoints,
    )
    accelerator.print(f"Saved final state to {checkpoint}")


if __name__ == "__main__":
    main()
