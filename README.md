# audio8Training

Fine-tuning and serving code for **Audio8 TTS 0.1B** (`Edge0-Audio8-TTS-Preview-0.1b`) on Indian English, with
voice cloning from a short reference clip.

This repo holds the scripts, configs, logs and small audio samples. It does **not** hold the model weights,
checkpoints, ONNX exports, the encoded training data or the Python environment (see `.gitignore`). The official
model code lives in [Audio8-AI/Audio8_TTS](https://github.com/Audio8-AI/Audio8_TTS); clone it next to this
repo as `Audio8_TTS/`, because the training script imports its trainer.

## Layout

| Path | What it is |
| --- | --- |
| `finetune/prep_stream.py` | Data pipeline: streams the source dataset, filters it with Whisper, splits each clip into reference + target, encodes both with the Audio8 codec |
| `finetune/run_prep.sh` | Starts supervised prep workers and the Hub sync in the background |
| `finetune/sync_hub.py` | Uploads finished encoded shards to a private Hub dataset |
| `finetune/train_sft.py` | Fine-tuning script (wraps the official Audio8 trainer) |
| `finetune/run_train.sh` | Training launcher with the hyperparameters used |
| `finetune/eval_ab.py` | A/B test: fine-tuned vs base model, same sentences and seeds, scored by Whisper WER |
| `finetune/runs/` | Run configs, tokenizer files, logs and `trainer_state.json` (weights excluded) |
| `audio8_tts.py` | Command-line synthesis, optionally cloning a voice from `--ref-audio` |
| `clone_compare.sh`, `compare_clones.py` | Re-generate a clone set with a checkpoint and compare it to the base model (WER, duration) |
| `band_energy.py`, `hf_bursts.py` | Audio checks: energy per frequency band, short high-frequency bursts |
| `simran_stream/` | Low-latency streaming TTS server on CPU (ONNX INT8) |
| `bench_onnx.py`, `register_onnx_voices.py` | ONNX runtime benchmarks and voice registration |
| `dataset_samples/`, `out*/` | Listening samples: source clips, codec round-trips, clones from each checkpoint |

## Training pipeline

```
source dataset (Hub parquet) ──► prep_stream.py ──► encoded/<shard>.parquet ──► train_sft.py ──► runs/<name>/
                                  Whisper filter         reference + target           official Audio8
                                  split + verify         codec codes [10, T]          DualAR trainer
```

### 1. Data preparation (`finetune/prep_stream.py`)

Source: `MeghanaKap/IndianEnglishCleanYoutubeData`. Shards are downloaded one at a time (the next one prefetches
in a background thread) and deleted after processing, so disk use stays small. Each shard goes through:

1. **Decode and clean.** Drop exact duplicates (MD5 of the audio bytes), undecodable files, and clips outside
   4–25 s. Audio is mixed to mono and resampled to 16 kHz.
2. **Check the transcript against the audio.** Whisper large-v3-turbo re-transcribes every clip with token
   timestamps. A clip is kept only if the dataset transcript and Whisper agree (WER ≤ 0.10 after lower-casing,
   removing punctuation and fillers like "uh"/"um") and the speaking rate is 1.0–5.5 words/s.
3. **Split into reference + target.** The source has no speaker labels, so each clip supplies its own
   same-speaker reference: the first sentences become the reference (≥ 3 s), the rest the target (≥ 2 s). The
   split goes at a sentence end (`.`, `?`, `!`) with a pause of at least 0.25 s, between 25% and 75% of the
   clip, choosing the longest pause.
4. **Cut carefully and verify.** The cut is placed 0.15 s after the last reference word's final token.
   Whisper token timestamps run late, so cutting at the next word's timestamp clipped the first target word in
   about 80% of clips; this rule kept both boundary words intact in 24 of 27 test splits (vs 14 of 27). Both
   halves are then transcribed again, and the clip is dropped unless the last reference word and the first
   target word are still there and the target still matches (WER ≤ 0.10).
5. **Encode.** Reference and target are encoded separately with the Audio8 codec (16 kHz → 44.1 kHz,
   10 codebooks × T frames, values < 4096), stored as `int16`.
6. **Write.** `encoded/<shard>.parquet` (zstd) plus `done/<shard>.json` with per-shard filter counts. Finished
   shards are skipped on restart, so the pipeline is resumable.

Each output row holds `ref_text`, `ref_codes` [10, ref_n_frames], `text`, `codes` [10, n_frames], the source
and Whisper transcripts, the WER and the split point.

Implementation details that matter:

- **The codec runs in fp32.** bf16 encodes disagreed with fp32 on about 21% of the semantic codes.
- **Fixed input shapes.** Codec inputs are padded to multiples of 5 s (exact, because the encoder is causal) and
  `MIOPEN_FIND_MODE=FAST` is set, so ROCm does not re-tune convolutions for every new length.
- **Whisper encoder run separately** before `generate()`, so asking for token timestamps does not keep all 32
  encoder attention maps in memory.
- **Crash handling.** A GPU error can leave the HIP context unusable, so a failing worker exits with code 3 and
  `run_prep.sh` restarts it in a fresh process. A shard that fails 3 times is skipped. Workers stop when free
  disk drops below 30 GB.

Result on 188 source shards: 113,591 clips (235.5 h) in, **7,723 clips (30.9 h) kept**. The main losses were
transcript mismatch (37,286 clips), duration (32,803), no usable split point (21,365) and duplicates (13,193).

```bash
cd finetune
WORKERS=1 ./run_prep.sh            # supervised workers + Hub sync every 15 min; logs in data/logs/
```

### 2. Fine-tuning (`finetune/train_sft.py`)

The script reuses the official trainer from `Audio8_TTS/audio8_tts_sft.py` (`Audio8TTSTrainer`, the DualAR
slow/fast losses, branch freezing, resume logic) and adds three things:

- **`ParquetSFTDataset`.** Reads the encoded parquet shards, deduplicates clips across shards by audio MD5 and
  holds out about 2% of shards for evaluation (shards whose name hash % 50 == 0, at most 400 eval clips).
  Every example is built with the official `build_sft_example` and conditioned on its own reference
  (`reference_prob` 1.0), so training matches voice-cloning use: reference text + reference codes → target
  text → target codes.
- **`CompactHeadTrainer`.** The 0.1B checkpoint's slow head (`semantic_output`) only has
  `codebook_size + 1` classes (semantic codes, then EOS), while the official trainer scores it against
  full-vocabulary token IDs. Those out-of-range labels fault the GPU, so labels are remapped: semantic ID →
  ID − `semantic_begin_id`, EOS → `codebook_size`, everything else ignored.
- **Two losses**, combined with the trainer's slow/fast loss weights:
  - *slow loss*: cross-entropy of the slow (temporal) model's next semantic code per frame, plus EOS;
  - *fast loss*: cross-entropy of the fast model's 10 codebooks for each supervised frame, given the slow
    model's hidden state.

  It also logs `slow_accuracy`, `fast_accuracy`, `eos_accuracy` and the number of supervised frames.

Hyperparameters (`run_train.sh` defaults, all overridable by environment variable):

| Setting | Value |
| --- | --- |
| Precision | bf16, gradient checkpointing on |
| Batch | 4 per device × 8 accumulation = 32 clips per step |
| Learning rate | 1e-5, cosine schedule, 3% warm-up, no weight decay |
| Gradient clipping | 1.0 |
| Max sequence length | 2,048 tokens |
| Trained parts | slow AR always; fast AR unless `FREEZE_FAST_AR=true` |
| Loss weights | slow 1.0, fast 1.0 (`SLOW_LOSS_WEIGHT`, `FAST_LOSS_WEIGHT`) |
| Eval / checkpoint | every 250 steps, keep the last 4; TensorBoard logs |
| Resume | automatic from the newest checkpoint in the run folder |

```bash
cd finetune
RUN=pilot_full MAX_STEPS=3000 ./run_train.sh                              # slow + fast AR
FREEZE_FAST_AR=true FAST_LOSS_WEIGHT=0 RUN=pilot_slow ./run_train.sh      # slow AR only
```

### 3. Runs so far

| Run | Setup | Steps | Eval loss | Slow loss (first → last) | Fast loss (first → last) |
| --- | --- | --- | --- | --- | --- |
| `smoke` | pipeline check | 15 | — | 5.91 → 5.07 | 8.49 → 7.28 |
| `pilot_v2` | slow + fast AR | 600 (2.5 epochs) | 9.853 → 9.793 | 6.42 → 4.28 | 9.05 → 5.43 |
| `pilot_full` | slow + fast AR | 500 of 1,000 | 9.849 → 9.806 | 6.47 → 4.24 | 9.32 → 5.40 |
| `pilot_slow` | slow AR only, fast frozen | 360 of 480 | 4.391 → 4.366 | 6.36 → 4.33 | — (frozen) |

Each step took about 6.5–7 s. Eval losses are not comparable between `pilot_slow` and the others, because the
fast loss is switched off there. Full logs are in `finetune/runs/<run>/trainer_state.json` and `train.log`.

### 4. Evaluation

- `finetune/eval_ab.py runs/<run>` synthesizes the same Indian English sentences with the base and the
  fine-tuned model (same reference voice, same seeds), writes WAVs for listening and scores intelligibility with
  Whisper WER.
- `clone_compare.sh <checkpoint> <out_dir>` regenerates the clone set (voices simran, sana, rani; same
  sentences, seed 0); `compare_clones.py` compares two folders by WER and duration. The outputs are in
  `out_clone*/`, one folder per checkpoint.

## Serving: `simran_stream/`

A CPU streaming server for the ONNX INT8 export of the 0.1B model, built on the official `ArkTtsRuntime`:

- **Voice prefix cache:** the part of the prompt shared by every request for a voice (system text, reference
  transcript and codes) is run once and its slow-AR state saved, so a request only processes its own text.
- **Stateful streaming codec** (`export_stream_codec.py`): the decoder is split into a post-transformer step
  with a K/V cache and a conv decoder with 8 frames of context, so each chunk decodes only new frames. The
  output matches a full decode.
- **Pipelining and guard rails:** frame generation and audio decoding run in parallel threads; long text is
  split into sentences, each with a frame budget so a missing EOS cannot run away.
- `server.py` serves a pool of engines over HTTP and streams 44.1 kHz PCM or WAV; `bench_server.py` load-tests
  it and counts playback stalls.

```bash
ENGINES=6 AR_THREADS=2 CODEC_THREADS=2 \
  ../Audio8_TTS/onnx_runtime_0_1b_int8/.venv/bin/python -m uvicorn server:app --port 8030
curl -N localhost:8030/v1/tts -H 'content-type: application/json' \
  -d '{"text": "Your EMI is due on the fifth.", "format": "wav"}' > out.wav
```

## Setup

```bash
git clone https://github.com/Audio8-AI/Audio8_TTS.git      # next to this repo's files
python -m venv .venv && source .venv/bin/activate
pip install -r Audio8_TTS/requirements-train.txt
export HF_TOKEN=hf_...                                      # needed for the source dataset and Hub sync
```

Scripts expect the base model at `~/models/Edge0-Audio8-TTS-Preview-0.1b` (change `AUDIO8_MODEL` in
`prep_stream.py` and `--model_name_or_path` in `train_sft.py` otherwise).
