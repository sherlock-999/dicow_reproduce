"""One-time export: expand DiCoW cutsets into per-(cut, target-speaker) cutsets.

Each output cut is the same audio window referenced once per speaker, with the
target speaker's INDEX (into the sorted speaker list, which the dataset
recomputes identically) encoded as an id suffix `_tsidx{N}`. The id suffix is
used instead of cut.custom because MixedCut (LibriMix cutsets) has no custom
field in Lhotse 1.32. Dataset weighting is performed by
``data.dataset.load_weighted_cutsets``.

Usage (CPU only, no GPU needed):
    python -m data.export_ts_cuts \
        --input <source>/ami-sdm_cutset_train_30s.jsonl.gz \
        --output manifests/ami-sdm_train_30s_ts.jsonl.gz --tag ami
"""

import argparse
from pathlib import Path

from lhotse import CutSet
from lhotse.utils import fastcopy


def expanded_cuts(cuts: CutSet):
    """Yield non-empty target-speaker cuts without materializing them."""

    for cut in cuts:
        speakers = sorted(
            {s.speaker for s in cut.supervisions if s.speaker is not None}
        )
        if not speakers:
            continue
        for idx, speaker in enumerate(speakers):
            has_transcript = any(
                (supervision.text or "").strip()
                for supervision in cut.supervisions
                if supervision.speaker == speaker
            )
            if not has_transcript:
                continue
            yield fastcopy(cut, id=f"{cut.id}_tsidx{idx}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    cuts = CutSet.from_jsonl_lazy(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with CutSet.open_writer(args.output, overwrite=True) as writer:
        for cut in expanded_cuts(cuts):
            writer.write(cut)
            count += 1
    print(f"[{args.tag}] {args.input} -> {args.output}: {count} target-speaker cuts")


if __name__ == "__main__":
    main()
