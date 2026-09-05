"""Locate the first non-finite value in a DiCoW forward pass.

Run this on one GPU with ``qsub scripts/diagnose_nan.pbs``. The script uses
the first batch selected by ``configs/debug.yaml`` and compares ordinary
Whisper, DiCoW, and DiCoW with identity FDDTs.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from data.dataset import DiCoWDataset, load_weighted_cutsets, make_dataloader
from model.DiCoW import DiCoW


MODEL_PATH = "whisper-large-v3-turbo"
MANIFEST = "manifests/notsofar1_train_30s_ts.jsonl.gz"


def tensors_in(value):
    """Yield tensors contained in common model-output structures."""

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from tensors_in(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from tensors_in(item)


def describe(name: str, value) -> bool:
    """Print tensor ranges and return whether every value is finite."""

    tensors = list(tensors_in(value))
    if not tensors:
        return True

    all_finite = True
    for index, tensor in enumerate(tensors):
        if not tensor.is_floating_point():
            continue
        finite = torch.isfinite(tensor)
        all_finite &= bool(finite.all().item())
        finite_values = tensor[finite]
        suffix = f"[{index}]" if len(tensors) > 1 else ""
        if finite_values.numel():
            value_range = (
                f"min={finite_values.min().item():.6g} "
                f"max={finite_values.max().item():.6g}"
            )
        else:
            value_range = "no finite values"
        print(
            f"{name}{suffix}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"nan={torch.isnan(tensor).sum().item()} "
            f"inf={torch.isinf(tensor).sum().item()} {value_range}",
            flush=True,
        )
    return all_finite


class ForwardTrace:
    """Attach concise forward hooks to the modified Whisper path."""

    def __init__(self, model: DiCoW) -> None:
        self.handles = []
        self.first_nonfinite = None
        encoder = model.model.encoder
        self._add("encoder.conv1", encoder.conv1)
        self._add("encoder.conv2", encoder.conv2)
        for index, (fddt, layer) in enumerate(zip(encoder.fddts, encoder.layers)):
            self._add(f"encoder.fddt.{index}", fddt)
            self._add(f"encoder.layer.{index}", layer)
        self._add("encoder.layer_norm", encoder.layer_norm)
        for index, layer in enumerate(model.model.decoder.layers):
            self._add(f"decoder.layer.{index}", layer)
        self._add("proj_out", model.proj_out)

    def _add(self, name, module) -> None:
        def hook(_module, _inputs, output):
            if not describe(name, output) and self.first_nonfinite is None:
                self.first_nonfinite = name
                print(f"FIRST OBSERVED NON-FINITE MODULE: {name}", flush=True)

        self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def identity_fddts(model: DiCoW) -> None:
    """Make every STNO branch an identity affine transformation."""

    with torch.no_grad():
        for fddt in model.model.encoder.fddts:
            for branch in (
                fddt.silence,
                fddt.target,
                fddt.non_target,
                fddt.overlap,
            ):
                branch.weight.fill_(1.0)
                branch.bias.zero_()


def first_batch(processor: WhisperProcessor) -> dict[str, torch.Tensor]:
    cuts = load_weighted_cutsets(
        [MANIFEST],
        [1],
        seed=0,
        infinite=True,
        min_cut_duration=0.4,
        max_cut_duration=31.0,
    )
    dataset = DiCoWDataset(processor, is_train=False, return_metadata=True)
    loader = make_dataloader(
        cuts,
        dataset,
        max_duration=30.0,
        num_workers=0,
        shuffle=True,
        seed=0,
    )
    batch = next(iter(loader))
    print(f"cut_ids={batch['cut_ids']}")
    print(f"texts={batch['texts']}")
    labels = batch["labels"]
    print(
        "valid_label_tokens=",
        labels.ne(-100).sum(dim=1).tolist(),
        flush=True,
    )
    tensors = {
        key: value.cuda()
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    describe("input_features", tensors["input_features"])
    describe("stno_mask", tensors["stno_mask"])
    return tensors


def run_whisper(batch: dict[str, torch.Tensor]) -> None:
    print("\n=== 1. ORIGINAL WHISPER FP32 ===", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH, local_files_only=True
    ).cuda().eval()
    model.config.use_cache = False
    with torch.no_grad():
        output = model(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
    describe("whisper.logits", output.logits)
    describe("whisper.loss", output.loss)
    del output, model
    gc.collect()
    torch.cuda.empty_cache()


def run_dicow(batch: dict[str, torch.Tensor], *, bypass_fddt: bool) -> None:
    title = "IDENTITY FDDT" if bypass_fddt else "INITIALIZED FDDT"
    print(f"\n=== DiCoW FP32: {title} ===", flush=True)
    model = DiCoW.from_pretrained(MODEL_PATH, local_files_only=True).cuda().eval()
    model.config.use_cache = False
    if bypass_fddt:
        identity_fddts(model)
    trace = ForwardTrace(model)
    try:
        with torch.no_grad():
            output = model(**batch)
        describe("dicow.logits", output.logits)
        describe("dicow.loss", output.loss)
    finally:
        trace.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires one CUDA GPU")
    torch.manual_seed(0)
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    batch = first_batch(processor)
    run_whisper(batch)
    run_dicow(batch, bypass_fddt=False)
    run_dicow(batch, bypass_fddt=True)


if __name__ == "__main__":
    main()
