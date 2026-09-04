"""Evaluate the Hugging Face DiCoW model with oracle diarization.

Each Lhotse cut is one session. The script decodes the complete recording once
per reference speaker. ``DiCoW.generate`` performs Whisper's long-form seek
loop and slices the matching part of the full-recording STNO mask.

Example::

    python -m eval.run_eval \
        --cutset manifests/ami_sdm_dev.jsonl.gz \
        --checkpoint outputs/checkpoint-best \
        --output outputs/ami_sdm_dev.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from lhotse import CutSet
from transformers import WhisperProcessor

from model.DiCoW import DiCoW
from model.stno import lhotse_cut_to_stno
from txt_norm import get_text_norm

from .scoring import aggregate, score_session
from .seglst import (
    normalize_hyp_segments,
    reference_segments,
    whisper_segments_to_segments,
)


def load_mono_audio(cut):
    audio = cut.load_audio()
    if audio.shape[0] != 1:
        audio = audio.mean(axis=0, keepdims=True)
    return audio[0]


def decode_target(model, processor, cut, target_speaker: str, device: torch.device, max_new_tokens=None):
    """Decode one target from a short or arbitrarily long session."""

    audio = load_mono_audio(cut)
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=cut.sampling_rate,
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt",
    )
    full_stno = lhotse_cut_to_stno(cut, target_speaker).unsqueeze(0)
    generation_args = {
        "input_features": inputs.input_features.to(device),
        "attention_mask": inputs.attention_mask.to(device),
        "stno_mask": full_stno.to(device),
        "return_timestamps": True,
        "return_segments": True,
    }
    if max_new_tokens is not None:
        generation_args["max_new_tokens"] = max_new_tokens

    with torch.inference_mode():
        result = model.generate(**generation_args)

    return whisper_segments_to_segments(
        result["segments"][0],
        session_id=cut.id,
        speaker=target_speaker,
        decode=processor.tokenizer.decode,
    )


def evaluate_cut(model, processor, cut, device, text_norm, max_new_tokens=None):
    """Decode every oracle target speaker, then score the complete session."""

    speakers = sorted({s.speaker for s in cut.supervisions if s.speaker is not None})
    references = reference_segments(cut, cut.id, text_norm)
    hypotheses = []
    started = time.perf_counter()
    for speaker in speakers:
        hypotheses.extend(
            decode_target(
                model,
                processor,
                cut,
                speaker,
                device,
                max_new_tokens=max_new_tokens,
            )
        )
    hypotheses = normalize_hyp_segments(hypotheses, text_norm)
    elapsed = time.perf_counter() - started
    scores = score_session(references, hypotheses)
    scores.update(
        duration=float(cut.duration),
        decode_seconds=elapsed,
        rtf=elapsed / float(cut.duration),
        reference_segments=references,
        hypothesis_segments=hypotheses,
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--processor",
        default=None,
        help="Processor directory; defaults to --checkpoint.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cuts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--text-norm", default="whisper_nsf")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor_path = args.processor or args.checkpoint
    processor = WhisperProcessor.from_pretrained(processor_path, local_files_only=True)
    model = DiCoW.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    text_norm = get_text_norm(args.text_norm)

    results = []
    cuts = CutSet.from_file(args.cutset)
    for index, cut in enumerate(cuts):
        if args.max_cuts is not None and index >= args.max_cuts:
            break
        result = evaluate_cut(
            model,
            processor,
            cut,
            device,
            text_norm,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(result)
        print(
            f"[{index + 1}] {cut.id}: "
            f"tcpWER={result['tcp_wer']:.3f}, cpWER={result['cp_wer']:.3f}, "
            f"RTF={result['rtf']:.3f}",
            flush=True,
        )

    summary = aggregate(results)
    payload = {"summary": summary, "sessions": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
