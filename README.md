# Straightforward DiCoW reproduction

This project reproduces the core DiCoW idea on top of
`openai/whisper-large-v3-turbo`:

- add one diagonal FDDT immediately before every Whisper encoder transformer
  layer;
- condition each FDDT with the oracle diarization mask `[S, T, N, O]`;
- fine-tune the Whisper encoder and FDDTs;
- keep the Whisper decoder and output projection frozen;
- use Whisper's timestamp-driven seek loop for recordings longer than 30
  seconds, while moving the STNO mask with the audio window.

`S`, `T`, `N`, and `O` mean silence, target-only speech, non-target-only
speech, and overlap involving the target speaker.

## Repository layout

```text
configs/             full and smoke-test training configurations
data/                target expansion, Lhotse batching, STNO augmentation
eval/                oracle long-form decoding and meeteval scoring
model/               FDDT, STNO conversion, and the DiCoW model
train/train.py       encoder+FDDT training loop
tests/               data, gradients, and long-form seek tests
whisper-large-v3-turbo/
                     local Hugging Face checkpoint and processor
```

The downloaded Whisper checkpoint, generated manifests, datasets, and training
outputs are intentionally excluded from Git. The long-form WAV under
`test_data/` is intentionally versioned because it is the structural seek test
fixture.

## 1. Environment setup

The tested environment uses Python 3.10, PyTorch 2.6.0, Transformers 4.57.6,
and Lhotse 1.32.2. Create it with CUDA 12.4 wheels:

```bash
./setup.sh
conda activate dicow-reproduce
```

Choose a different environment name or a CPU-only PyTorch installation with:

```bash
./setup.sh my-dicow-env cu124
./setup.sh my-dicow-env cpu
```

The CPU environment is suitable for tests, not large-v3-turbo training.
Dependencies other than PyTorch are pinned in `requirements.txt`.

Verify the installation:

```bash
python -m pytest -q tests
```

## 2. Prepare the source manifests

Use `mt-asr-data-prep` to download and prepare the corpora. AMI and NOTSOFAR-1
training sessions are converted into cuts no longer than 30 seconds by
`pre_segment_using_alignments.py`.

That script does not blindly retain the complete text when a boundary crosses
an utterance. It uses word timestamps to keep only words fully contained in
the current audio window, and revisits unfinished/overlapping supervisions in
the following window. Consequently, a sentence may be divided, but every
individual audio-text pair remains aligned.

LibriMix preparation is different: it retains already-created mixtures shorter
than 30 seconds. LibriSpeechMix is synthesized from source cuts that fit in the
30-second limit.

Expected training corpora and sampling weights:

| Target-expanded manifest | Corpus | Weight |
|---|---|---:|
| `notsofar1_train_30s_ts.jsonl.gz` | NOTSOFAR-1 SDM | 6 |
| `ami-sdm_train_30s_ts.jsonl.gz` | AMI SDM | 6 |
| `libri2mix_100_noisy_30s_ts.jsonl.gz` | Libri2Mix train-100 noisy | 1 |
| `libri2mix_360_noisy_30s_ts.jsonl.gz` | Libri2Mix train-360 noisy | 1 |
| `libri3mix_360_noisy_30s_ts.jsonl.gz` | Libri3Mix train-360 noisy | 1 |
| `librispeechmix_train_3mix_ts.jsonl.gz` | LibriSpeechMix, up to 3 speakers | 1 |

## 3. Expand each cut by target speaker

The 30-second source manifests still describe all speakers together. Convert
each cut into one row per possible target speaker:

```bash
mkdir -p manifests

python -m data.export_ts_cuts \
    --input /path/to/notsofar1_train_cutset_30s.jsonl.gz \
    --output manifests/notsofar1_train_30s_ts.jsonl.gz \
    --tag notsofar1

python -m data.export_ts_cuts \
    --input /path/to/ami-sdm_cutset_train_30s.jsonl.gz \
    --output manifests/ami-sdm_train_30s_ts.jsonl.gz \
    --tag ami
```

Repeat the same command for the four synthetic manifests, using the output
names listed in `configs/dicow_v1.yaml`.

The exporter adds `_tsidxN` to each cut ID. It does not copy audio. At training
time, both the exporter and dataset sort speaker IDs identically, so `_tsidx0`
always identifies the same target.

A validation-loss manifest must also contain cuts no longer than 30 seconds
and be target-expanded. Full-session manifests are used by `eval/run_eval.py`,
not by the training loss loader.

## 4. Check configuration paths

The main configuration is [configs/dicow_v1.yaml](configs/dicow_v1.yaml). Paths are
interpreted relative to the directory where training is launched, so run from
this repository root or replace them with absolute paths.

Important groups in the YAML:

- `data`: manifests, `6:6:1:1:1:1` weights, MUSAN, and batch duration;
- `optimization`: steps, learning rate, warm-up, accumulation, and precision;
- `augmentation`: all reported STNO and MUSAN probabilities;
- `monitoring`: logging, validation, and checkpoint intervals;
- `output`: output directory and optional resume checkpoint.

The optimizer defaults are explicit starting values. They are not claimed to
be the paper's exact recipe until the original learning rate, batch size,
warm-up, and number of updates have been confirmed.

## 5. Smoke test training

After producing at least the NOTSOFAR target-expanded manifest, edit its path
in `configs/debug.yaml` and run:

```bash
accelerate launch -m train.train --config configs/debug.yaml
```

This runs ten updates with one 30-second micro-batch and no augmentation. It is
intended to catch path, memory, and forward/backward errors before a full run.

## 6. Full training

```bash
accelerate launch -m train.train --config configs/dicow_v1.yaml
```

Explicit command-line options override YAML values. For example:

```bash
accelerate launch -m train.train \
    --config configs/dicow_v1.yaml \
    --max-duration 60 \
    --gradient-accumulation-steps 4 \
    --mixed-precision fp16
```

`max_duration` is the sum of audio durations in one GPU micro-batch. With
30-second cuts, `120` is approximately four full-length examples. Reduce it if
large-v3-turbo does not fit in GPU memory, then use gradient accumulation to
recover the desired effective batch size.

The training batch is:

```text
input_features  [B, 128, 3000]   Whisper log-Mel features
attention_mask  [B, 3000]        valid audio frames
stno_mask       [B, 4, 1500]     oracle/augmented S-T-N-O mask
labels          [B, U]           target-speaker Whisper tokens
```

Training uses teacher forcing and token cross-entropy. The frozen decoder still
passes gradients back to the encoder; it is simply excluded from optimization.
Only `model.encoder.*` can be updated. The optional `fddt_only_steps` setting
can make the first updates affect only FDDTs and defaults to zero.

### Resume

Each checkpoint stores model and processor files plus Accelerate optimizer,
scheduler, scaler, and random state:

```bash
accelerate launch -m train.train \
    --config configs/dicow_v1.yaml \
    --resume outputs/dicow-v1-large-v3-turbo/checkpoint-10000
```

You may instead set `output.resume` in the YAML.

## 7. Long-form oracle evaluation

Training always uses prepared cuts no longer than 30 seconds. Evaluation may
use complete meeting recordings.

```bash
python -m eval.run_eval \
    --cutset /path/to/full_session_dev_cutset.jsonl.gz \
    --checkpoint outputs/dicow-large-v3-turbo/checkpoint-N \
    --output outputs/notsofar-dev-scores.json
```

Evaluation performs one decode per oracle target speaker and reports tcpWER,
cpWER, and real-time factor. For a long recording, Whisper repeatedly encodes
at most 30 seconds:

```text
audio features: [seek : seek + 3000]       100 Hz
STNO mask:      [seek/2 : seek/2 + 1500]    50 Hz
```

The seek value comes from Whisper's predicted timestamp segments. DiCoW uses
the same seek to select the matching part of the full-recording STNO mask. The
last STNO window is padded with pure silence rather than cropping the recording.

All current reproduction data is English, so generation explicitly uses
`language="en"` and `task="transcribe"`; no language-detection pass is needed.

## 8. Tests

```bash
python -m pytest -q tests
```

The tests verify:

- real Lhotse audio becomes `[128, 3000]` features and `[4, 1500]` STNO;
- STNO channels form a probability distribution at every frame;
- a DiCoW loss can backpropagate into the encoder;
- decoder parameters remain frozen and receive no gradients;
- a 354.95-second recording advances audio and STNO with matching seek values.
