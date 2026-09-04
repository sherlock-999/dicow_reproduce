# DiCoW reproduction

- Backbone: `openai/whisper-large-v3-turbo`.
- Goal: reproduce DiCoW with a small, readable Hugging Face codebase.
- Training protocol: aligned with the DiCoN `train` branch.

## Model

| Component | Behavior |
|---|---|
| Whisper encoder | Fine-tuned |
| FDDT | Inserted before every encoder transformer layer and fine-tuned |
| Whisper decoder | Frozen |
| Output projection | Frozen |
| Diarization input | Oracle STNO mask |

| STNO channel | Meaning |
|---|---|
| `S` | Silence |
| `T` | Target speaker only |
| `N` | Non-target speaker only |
| `O` | Target speaker overlapping another speaker |

## Repository

| Path | Purpose |
|---|---|
| `model/DiCoW.py` | Whisper model with FDDT and long-form decoding |
| `model/FDDT.py` | Diagonal FDDT layer |
| `model/stno.py` | Oracle diarization to STNO conversion |
| `data/export_ts_cuts.py` | Expands each cut into one example per target speaker |
| `data/dataset.py` | Lhotse loading, STNO creation, tokenization, and batching |
| `train/train.py` | Multi-GPU training, validation, and checkpoints |
| `eval/run_eval.py` | Full-session oracle decoding and WER scoring |
| `configs/dicow_v1.yaml` | Main training configuration |
| `configs/debug.yaml` | Ten-step smoke test |

## 1. Setup

- Tested with Python 3.10, PyTorch 2.6.0, Transformers 4.57.6, and Lhotse 1.32.2.
- CUDA 12.4 environment:

```bash
./setup.sh
conda activate dicow-reproduce
```

- CPU-only environment for tests:

```bash
./setup.sh dicow-reproduce cpu
conda activate dicow-reproduce
```

- Verify the installation:

```bash
python -m pytest -q tests
```

## 2. Data

### Experiment splits

| Role | Data | Input form |
|---|---|---|
| Training | NOTSOFAR-1, AMI, Libri2Mix, Libri3Mix, LibriSpeechMix | Target-expanded cuts of at most 30 seconds |
| Validation | NOTSOFAR-1 `dev1` | Target-expanded cuts of at most 30 seconds |
| Test | AMI test or another evaluation set | Full meeting sessions |

- Test data is never read by `train/train.py`.
- Select checkpoints using validation WER only.
- Run full-session testing separately with `eval/run_eval.py`.

### Training mixture

| Manifest | Weight |
|---|---:|
| `notsofar1_train_30s_ts.jsonl.gz` | 6 |
| `ami-sdm_train_30s_ts.jsonl.gz` | 6 |
| `libri2mix_100_noisy_30s_ts.jsonl.gz` | 1 |
| `libri2mix_360_noisy_30s_ts.jsonl.gz` | 1 |
| `libri3mix_360_noisy_30s_ts.jsonl.gz` | 1 |
| `librispeechmix_train_3mix_ts.jsonl.gz` | 1 |

### Creating 30-second cuts

- Use `mt-asr-data-prep` to prepare the corpora.
- Use alignment-aware segmentation for AMI and NOTSOFAR-1.
- Word timestamps determine which words belong to each audio window.
- A sentence may be split, but text outside the window is not added to its target.
- LibriMix and LibriSpeechMix examples already fit within the 30-second limit.

## 3. Target-speaker expansion

- One source cut may contain several speakers.
- DiCoW needs a separate training example for every possible target speaker.
- Create the target-expanded manifests:

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

### Concrete example

| Output cut | Audio | Target | Training text |
|---|---|---|---|
| `meeting_001_tsidx0` | Same 30-second meeting window | Speaker A | `let us start` |
| `meeting_001_tsidx1` | Same 30-second meeting window | Speaker B | `I agree` |

- The loader converts these into:
  - `(audio, STNO for Speaker A) -> "let us start"`.
  - `(audio, STNO for Speaker B) -> "I agree"`.
- `_tsidxN` is an index into the cut's sorted speaker list.
- `_tsidx0` is not a global speaker identity across meetings.
- The exporter references the same audio; it does not copy it.
- Target-expand NOTSOFAR-1 `dev1` in the same way and save:
  - `manifests/notsofar1_dev1_30s_ts.jsonl.gz`.

## 4. Training configuration

- Main config: `configs/dicow_v1.yaml`.
- Paths are relative to the repository root.

| Setting | Value |
|---|---:|
| GPUs | 8 |
| Steps | 40,000 |
| Batch duration | 200 seconds per GPU |
| Precision | BF16 |
| Gradient clipping | 1.0 |
| AdamW betas | `(0.9, 0.98)` |
| Weight decay | 0.001 |
| FDDT learning rate | `5e-4` |
| Encoder learning rate | `5e-5` |
| Warm-up | 2,000 steps |
| Encoder delay | 2,000 steps |
| Minimum LR ratio | 0.05 |

- Learning-rate behavior:
  - Steps `0–1999`: FDDT trains; encoder learning rate is zero.
  - Step `2000` onward: encoder warms up; encoder and FDDT train together.
  - Both learning rates use cosine decay.
- Trainable data flow:

| Input | Shape | Purpose |
|---|---|---|
| `input_features` | `[B, 128, 3000]` | Whisper log-Mel features |
| `attention_mask` | `[B, 3000]` | Valid audio frames |
| `stno_mask` | `[B, 4, 1500]` | Oracle or augmented speaker condition |
| `labels` | `[B, U]` | Target-speaker tokens |

- Training uses teacher forcing and token cross-entropy.
- Gradients pass through the frozen decoder into the encoder.
- Only encoder and FDDT parameters are updated.

### Smoke test

- Update the manifest path in `configs/debug.yaml`.
- Run:

```bash
accelerate launch -m train.train --config configs/debug.yaml
```

### Eight-GPU training

```bash
accelerate launch --multi_gpu --num_processes 8 \
    -m train.train --config configs/dicow_v1.yaml
```

- GPU count is controlled by `--num_processes`, not the YAML.
- Reduce `--max-duration` if the model does not fit in memory.
- Use gradient accumulation if reducing per-GPU batch duration.

## 5. Validation and checkpoints

| Setting | Value |
|---|---|
| Validation data | NOTSOFAR-1 `dev1` |
| Frequency | Every 1,000 optimizer steps |
| Limit | 60 batches |
| Shuffle | No |
| Duration bucketing | No |
| Selection metric | Normalized `val_wer` |
| Selection direction | Lower is better |
| Retention | Best 2 plus `checkpoint-last` |

- Validation loss is logged for diagnosis.
- Decoded WER selects checkpoints.
- Example output:

```text
checkpoint-step=12000-val_wer=0.1842/
checkpoint-step=15000-val_wer=0.1798/
checkpoint-last/
```

- A new top-two checkpoint removes the previous worst one.
- `checkpoint-last` contains the latest model, optimizer, scheduler, scaler, RNG, and step state.
- Resume with:

```bash
accelerate launch -m train.train \
    --config configs/dicow_v1.yaml \
    --resume outputs/dicow-v1-large-v3-turbo/checkpoint-last
```

## 6. Full-session evaluation

- Use the best stable validation-WER checkpoint.
- Example AMI test command:

```bash
python -m eval.run_eval \
    --cutset /path/to/ami-sdm_cutset_test.jsonl.gz \
    --checkpoint outputs/dicow-v1-large-v3-turbo/checkpoint-step=N-val_wer=W \
    --output outputs/ami-test-scores.json
```

- Oracle evaluation:
  - Reads the session's reference speaker IDs.
  - Builds one full-session STNO mask per target speaker.
  - Decodes the full recording once per target speaker.
  - Reports tcpWER, cpWER, and real-time factor.
- Speaker scoring:
  - Target expansion chooses the speaker during training.
  - Oracle STNO chooses the speaker during inference.
  - cpWER finds the best hypothesis-to-reference speaker permutation.

### Audio longer than 30 seconds

- Whisper decodes repeated windows of at most 30 seconds.
- Timestamp tokens determine the next safe `seek` position.
- DiCoW advances audio and STNO together:

| Signal | Window |
|---|---|
| Whisper features at 100 Hz | `[seek : seek + 3000]` |
| STNO frames at 50 Hz | `[seek / 2 : seek / 2 + 1500]` |

- The final short STNO window is padded with silence.
- Audio after 30 seconds is not cropped.
- English is fixed with `language="en"` and `task="transcribe"`.

## 7. Tests

```bash
python -m pytest -q tests
```

- Tests cover:
  - Lhotse audio and STNO shapes.
  - STNO probability constraints.
  - Encoder gradients and frozen decoder parameters.
  - DiCoN-compatible configuration and learning-rate staging.
  - Top-two checkpoint ranking.
  - Matching audio/STNO seek on a 354.95-second recording.
