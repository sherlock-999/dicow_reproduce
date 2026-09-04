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

The defaults below are practical starting values, not yet claimed to be the
paper's exact optimization recipe. Record the final chosen command alongside
the resulting checkpoint for a reproducible experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from lhotse import CutSet
from transformers import WhisperProcessor, get_linear_schedule_with_warmup

from data.dataset import (
    AugmentationConfig,
    DiCoWDataset,
    load_weighted_cutsets,
    make_dataloader,
)
from model.DiCoW import DiCoW


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

    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--max-duration", type=float, default=120.0,
                        help="Maximum summed audio seconds in one micro-batch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--fddt-only-steps", type=int, default=0,
                        help="Optional initial steps that update only FDDTs")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=1_000)
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
    return args


def configure_trainable_parameters(model: DiCoW) -> None:
    """Expose the encoder to DDP while keeping the decoder frozen."""

    model.requires_grad_(False)
    model.model.encoder.requires_grad_(True)
    model.model.encoder.embed_positions.requires_grad_(False)


def clear_non_fddt_gradients(model: DiCoW) -> None:
    """Make an optional warm-up update FDDTs without changing the DDP graph."""

    for name, parameter in model.model.encoder.named_parameters():
        if not name.startswith("fddts."):
            parameter.grad = None


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


def move_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Keep metadata out of model.forward and move only tensor fields."""

    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


@torch.no_grad()
def validation_loss(model, loader, accelerator: Accelerator, max_batches: int) -> float:
    model.eval()
    total_loss = torch.zeros((), device=accelerator.device)
    total_examples = torch.zeros((), device=accelerator.device)
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = move_batch(batch, accelerator.device)
        count = batch["input_features"].shape[0]
        loss = model(**batch).loss
        total_loss += loss * count
        total_examples += count

    totals = accelerator.reduce(
        torch.stack((total_loss, total_examples)), reduction="sum"
    )
    enter_train_mode(model, accelerator)
    return float((totals[0] / totals[1].clamp_min(1)).item())


def save_checkpoint(
    accelerator: Accelerator,
    model: DiCoW,
    processor: WhisperProcessor,
    output_dir: Path,
    step: int,
    best_validation_loss: float,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step}"
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
                    "best_validation_loss": best_validation_loss,
                },
                indent=2,
            )
            + "\n"
        )
    accelerator.save_state(checkpoint / "accelerator_state")
    return checkpoint


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
    best_validation_loss = math.inf
    if args.resume:
        state_path = Path(args.resume) / "training_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            initial_step = int(state["global_step"])
            best_validation_loss = float(state.get("best_validation_loss", math.inf))

    fddt_only = initial_step < args.fddt_only_steps
    configure_trainable_parameters(model)
    verify_frozen_decoder(model)

    train_cuts = load_weighted_cutsets(
        args.train_manifest,
        args.weights,
        seed=args.seed,
        infinite=True,
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
        validation_cuts = CutSet.from_jsonl_lazy(args.validation_manifest)
        validation_dataset = DiCoWDataset(processor, is_train=False)
        validation_loader = make_dataloader(
            validation_cuts,
            validation_dataset,
            max_duration=args.max_duration,
            num_workers=args.num_workers,
            shuffle=False,
            seed=args.seed,
        )

    # Include the whole encoder in the optimizer even during an optional
    # FDDT-only warm-up. Its gradients are enabled when that stage ends.
    optimizer_parameters = list(model.model.encoder.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

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
                if fddt_only:
                    clear_non_fddt_gradients(accelerator.unwrap_model(model))
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(optimizer_parameters, args.max_grad_norm)
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
                    f"lr={scheduler.get_last_lr()[0]:.3e}"
                )
                running_loss = 0.0

            if validation_loader is not None and global_step % args.eval_steps == 0:
                value = validation_loss(
                    model, validation_loader, accelerator, args.max_eval_batches
                )
                accelerator.print(f"step={global_step} validation_loss={value:.4f}")
                best_validation_loss = min(best_validation_loss, value)

            if global_step % args.save_steps == 0:
                checkpoint = save_checkpoint(
                    accelerator,
                    model,
                    processor,
                    output_dir,
                    global_step,
                    best_validation_loss,
                )
                accelerator.print(f"Saved {checkpoint}")

    if global_step % args.save_steps != 0:
        checkpoint = save_checkpoint(
            accelerator,
            model,
            processor,
            output_dir,
            global_step,
            best_validation_loss,
        )
        accelerator.print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
