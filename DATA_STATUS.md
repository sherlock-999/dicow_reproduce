# NSCC data status

> Snapshot: 2026-09-05 18:58 SGT

## Locations

| Item | NSCC path |
|---|---|
| Scratch root | `/scratch/users/ntu/d230009` |
| Audio and corpus assets | `/scratch/users/ntu/d230009/data` |
| Lhotse manifests | `/scratch/users/ntu/d230009/manifests` |
| Preparation logs | `/scratch/users/ntu/d230009/logs` |
| DiCoW repository | `/scratch/users/ntu/d230009/project/DiCoW_exp/dicow_reproduce` |

## What the files mean

| Item | Contains audio? | Meaning |
|---|---:|---|
| Dataset directory under `data/` | Yes | Downloaded or extracted corpus files |
| `*_recordings.jsonl.gz` | No | Audio paths and recording metadata |
| `*_supervisions.jsonl.gz` | No | Speaker, timing, and transcript annotations |
| `*_cutset.jsonl.gz` | No | Recording and supervision metadata joined as Lhotse cuts |
| `*_30s.jsonl.gz` | No | Training cuts shorter than or equal to 30 seconds; construction is corpus-specific |
| `*_30s_ts.jsonl.gz` | No | One target-speaker training example per cut and speaker |

- Manifest files reference audio stored under `data/`; they do not contain audio.
- A full cutset is not necessarily limited to 30 seconds.
- A `30s` cutset is duration-limited but is not necessarily target-expanded.
- AMI and NOTSOFAR use alignment-aware segmentation; LibriMix filters out cuts that are already too long.
- A `_ts` manifest is the final form consumed by the current DiCoW training configuration.

## Corpus status

| Corpus | Data size | Raw/audio assets | Full manifests | 30-second training cutset | Target-expanded `_ts` | Status |
|---|---:|---:|---:|---:|---:|---|
| AMI | 22 GB | Present | Present | Present | Present | Target expansion complete |
| NOTSOFAR-1 | 7.7 GB | Present | Present | Present | Present | Target expansion complete |
| LibriSpeech | Not refreshed | Present | Present | Not applicable | Not applicable | Source preparation complete |
| LibriMix | 38 GB | Present | Present | Present | Present | Complete |
| LibriSpeechMix | Not refreshed | Present | Present | Present | Present | Complete |
| AliMeeting | Not refreshed | Present | Present | Not created | Not created | Source preparation complete; optional for `dicow_v1.yaml` |
| MUSAN | 12 GB extracted | Present | Not applicable | Not applicable | Not applicable | Complete |
| WHAM noise | 53 GB including archive and extracted data | Present | Present | Not applicable | Not applicable | Complete; used by noisy LibriMix |

## Available AMI manifests

| Microphone | Split | Full cutset | 30-second training cutset | Intended use |
|---|---|---:|---:|---|
| SDM | Train | Present | Present: 12,786 cuts | DiCoW training after target expansion |
| SDM | Dev | Present | Not created | Optional evaluation |
| SDM | Test | Present | Not needed | Full-session final evaluation |
| IHM mix | Train | Present | Present | Not used by `dicow_v1.yaml` |
| IHM mix | Dev | Present | Not created | Not used by `dicow_v1.yaml` |
| IHM mix | Test | Present | Not needed | Optional evaluation |

- Required AMI training source:
  - `/scratch/users/ntu/d230009/manifests/ami/ami-sdm_cutset_train_30s.jsonl.gz`
- Required AMI test source:
  - `/scratch/users/ntu/d230009/manifests/ami/ami-sdm_cutset_test.jsonl.gz`

## Available NOTSOFAR-1 manifests

| Split | Full cutset | 30-second cutset | Intended use |
|---|---:|---:|---|
| Train | Present | Present: 4,591 cuts | DiCoW training after target expansion |
| `dev1` | Present: 177 full-session cuts | Present: 2,341 cuts | Target-expanded: 11,314 validation examples |
| Eval-small with GT | Present | Missing | Optional full-session oracle evaluation |

- Required NOTSOFAR-1 training source:
  - `/scratch/users/ntu/d230009/manifests/notsofar1/notsofar1_sdm_train_set_240825.1_train_cutset_30s.jsonl.gz`
- Current `dev1` source:
  - `/scratch/users/ntu/d230009/manifests/notsofar1/notsofar1_sdm_dev_set_240825.1_dev1_cutset.jsonl.gz`

## LibriSpeech and mixture status

| Item | Current status | Next event |
|---|---|---|
| LibriSpeech raw audio | Extracted | None |
| LibriSpeech recordings, supervisions, and cutsets | Complete for all seven standard splits | None |
| WHAM noise for LibriMix | Downloaded and extracted; `tr`, `cv`, and `tt` data are present | None |
| LibriMix manifests | Complete for 2- and 3-speaker clean/noisy train, dev, and test mixtures | None |
| Libri2Mix clean-100 noisy 30s | 13,900 source cuts; 27,800 target examples | Complete |
| Libri2Mix clean-360 noisy 30s | 50,800 source cuts; 101,600 target examples | Complete |
| Libri3Mix clean-360 noisy 30s | 33,900 source cuts; 101,700 target examples | Complete |
| LibriSpeechMix evaluation manifests | Complete for 1-, 2-, and 3-speaker dev/test mixtures | None |
| LibriSpeechMix custom train manifest | 100,000 source cuts; 211,700 target examples; maximum 29.9999 seconds | Complete |

## Available AliMeeting manifests

| Split | Recordings | Supervisions | Full cutset | Intended use |
|---|---:|---:|---:|---|
| Train | Present | Present | Present | Optional; not used by `dicow_v1.yaml` |
| Eval | Present | Present | Present | Optional evaluation |
| Test | Present | Present | Present | Optional evaluation |

- AliMeeting preparation is complete.
- No AliMeeting 30-second or target-expanded manifests have been created.

## Files required by `dicow_v1.yaml`

| Required manifest | Current status | Required action |
|---|---|---|
| `notsofar1_train_30s_ts.jsonl.gz` | Present: 21,984 target examples | Complete |
| `ami-sdm_train_30s_ts.jsonl.gz` | Present: 32,617 target examples | Complete |
| `libri2mix_100_noisy_30s_ts.jsonl.gz` | Present: 27,800 target examples | Complete |
| `libri2mix_360_noisy_30s_ts.jsonl.gz` | Present: 101,600 target examples | Complete |
| `libri3mix_360_noisy_30s_ts.jsonl.gz` | Present: 101,700 target examples | Complete |
| `librispeechmix_train_3mix_ts.jsonl.gz` | Present: 211,700 target examples | Complete |
| `notsofar1_dev1_30s_ts.jsonl.gz` | Present: 11,314 target examples | Complete |

- Current number of required `_ts` manifests present: `7 / 7`.
- All configured training and validation manifests are ready.

## Preparation jobs

| Job | Status |
|---|---|
| LibriSpeech | Complete |
| LibriSpeechMix | Complete |
| MUSAN | Complete |
| LibriMix | Complete |
| Target-speaker expansion | Complete for all seven configured manifests |
| Whisper large-v3-turbo | Download complete and validated |

- The PID stored in `data_prep.pid` is stale.
- No data-preparation or target-expansion process is currently running.
- All seven target-expanded manifests passed `gzip -t` and target-index validation.
- Audio loading was verified from each newly generated manifest.
- The exporter skipped nine NOTSOFAR `dev1` targets whose source annotations had empty text and no word alignments; the data loader rejects empty transcripts.
- A five-corpus loader smoke test produced `[5, 128, 3000]` Whisper features and `[5, 4, 1500]` STNO masks with non-empty target text.
- AMI and NOTSOFAR logs report successful completion.
- Their logs also show a non-fatal `pyannote_audio-3.1.1-nspkg.pth` startup warning.
- Older LibriSpeech logs contain `lhotse: command not found`; the job was relaunched with the new environment and subsequently completed.

## Environment readiness

| Item | Status |
|---|---|
| Conda environment | Ready at `/home/users/ntu/d230009/miniconda3/envs/dicow-reproduce` |
| Location requirement | Satisfied: environment is outside `/scratch` |
| Replaced environments | `exp_ts_dicow` and `transformer` removed as requested |
| Core imports | Passed: PyTorch, Transformers, Accelerate, Lhotse, and MeetEval |
| Loudness normalization | `pyloudnorm==0.1.1` installed and LibriSpeechMix audio loading verified |
| Whisper checkpoint | Ready at `whisper-large-v3-turbo/`; processor, config, and 587 weight tensors validated locally |
| Repository tests | Not rerun after download; the checkpoint-dependent tests are now unblocked, but the long-form test fixture is still absent |

## Remaining steps before training

1. Run the end-to-end training smoke test.
2. Add the missing long-form test fixture if the complete test suite is required.
