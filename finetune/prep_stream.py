"""Stream MeghanaKap/IndianEnglishCleanYoutubeData into Audio8 training encodings.

Per source shard (downloaded one at a time, deleted afterwards):
  1. decode clips, drop exact duplicates and clips outside [MIN_DUR, MAX_DUR]
  2. re-transcribe with Whisper (with word timestamps); keep a clip only when the
     dataset transcript agrees with the audio (WER <= --max-wer after
     normalisation) and the speaking rate is plausible
  3. split the clip at a sentence boundary with a pause into reference (first
     sentences) + target (rest), cutting just after the last reference word, then
     re-transcribe both parts and drop clips whose boundary words got clipped.
     The source has no speaker labels, so each clip supplies its own same-speaker
     reference and every example matches voice-cloning use
  4. encode reference and target separately with the Audio8 codec
     (fp32, 16 kHz -> 44.1 kHz -> [10, T] codes)
  5. write encoded/<shard>.parquet and done/<shard>.json (resumable)

Run several workers on one GPU with --worker/--num-workers; sync_hub.py uploads
finished parquet files to the Hub.

    source ~/TTS/.venv/bin/activate   # provides HF_TOKEN
    python prep_stream.py --worker 0 --num-workers 2
"""

from __future__ import annotations

import os

# MIOpen (ROCm) otherwise runs an exhaustive kernel search for every new conv
# input shape; FAST mode plus the fixed bucketed shapes below keeps encoding fast.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio.functional as AF
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModel, WhisperForConditionalGeneration, WhisperProcessor

SOURCE_REPO = "MeghanaKap/IndianEnglishCleanYoutubeData"
AUDIO8_MODEL = str(Path.home() / "models/Edge0-Audio8-TTS-Preview-0.1b")
WHISPER = "openai/whisper-large-v3-turbo"
ROOT = Path(__file__).resolve().parent / "data"
SR = 16000
MIN_DUR, MAX_DUR = 4.0, 25.0
MIN_REF, MIN_TARGET = 3.0, 2.0  # seconds
SPLIT_RANGE = (0.25, 0.75)  # split point as a fraction of the clip
MIN_PAUSE = 0.25  # seconds between the last reference word and the first target word
CUT_AFTER_WORD = 0.15  # cut this long after the last reference word's final token
# Whisper token timestamps land late, so cutting at the next word's timestamp
# clipped the first target word in ~80% of clips; this rule kept both boundary
# words intact in 24/27 test splits (vs 14/27), and every split is re-verified.
MAX_ATTEMPTS = 3  # give up on a source shard after this many failures
BUCKET_SECONDS = 5  # codec input lengths are padded to multiples of this (exact: the encoder is causal)
MIN_WPS, MAX_WPS = 1.0, 5.5
FILLERS = {"uh", "um", "umm", "uhm", "hmm", "hm", "ah", "er", "erm", "mm", "mhm", "huh"}

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("shard", pa.string()),
    ("audio_id", pa.string()),
    ("source_text", pa.string()),  # dataset transcript of the whole clip
    ("asr_text", pa.string()),  # Whisper transcript of the whole clip
    ("wer", pa.float32()),  # dataset vs Whisper, normalised
    ("duration", pa.float32()),
    ("split_sec", pa.float32()),
    ("ref_text", pa.string()),
    ("ref_n_frames", pa.int32()),
    ("ref_codes", pa.list_(pa.int16())),  # row-major [10, ref_n_frames]
    ("text", pa.string()),  # target transcript
    ("n_frames", pa.int32()),
    ("codes", pa.list_(pa.int16())),  # row-major [10, n_frames]
    ("audio_md5", pa.string()),
])


def normalize_words(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower()).replace("'", "")
    return [w for w in text.split() if w not in FILLERS]


def wer(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 1.0 if hypothesis else 0.0
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_word in enumerate(hypothesis, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word))
        previous = current
    return previous[-1] / len(reference)


def choose_split(words: list[tuple[str, float, float]], duration: float) -> int | None:
    """Index of the first target word: a sentence start after the longest pause."""
    best, best_gap = None, 0.0
    for i in range(1, len(words)):
        previous, current = words[i - 1], words[i]
        if previous[0][-1:] not in ".?!":
            continue
        cut = previous[2] + CUT_AFTER_WORD
        gap = current[1] - previous[2]
        if gap < MIN_PAUSE or not (SPLIT_RANGE[0] * duration <= cut <= SPLIT_RANGE[1] * duration):
            continue
        if cut < MIN_REF or duration - cut < MIN_TARGET:
            continue
        if gap > best_gap:
            best, best_gap = i, gap
    return best


class Encoders:
    def __init__(self, device: str, whisper_batch: int, codec_batch: int):
        self.device = device
        self.whisper_batch, self.codec_batch = whisper_batch, codec_batch
        self.wproc = WhisperProcessor.from_pretrained(WHISPER)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER, dtype=torch.float16).to(device).eval()
        self.eos = self.wproc.tokenizer.eos_token_id
        # fp32 on purpose: bf16 codec encodes disagree with fp32 on ~21% of semantic codes.
        audio8 = AutoModel.from_pretrained(AUDIO8_MODEL, trust_remote_code=True, dtype=torch.float32).eval()
        self.codec = audio8.load_codec(device=device, dtype=torch.float32)
        self.codec_rate = int(audio8.config.codec_sample_rate)
        self.num_codebooks = int(audio8.config.num_codebooks)
        del audio8  # only the codec is needed

    def _words(self, tokens: list[int], times: list[float]) -> list[tuple[str, float, float]]:
        """(word, first token time, last token time) from Whisper token timestamps."""
        words: list[tuple[str, float, float]] = []
        current, first, last = "", None, None
        for token, t in zip(tokens, times):
            if token >= self.eos:  # special / timestamp tokens
                continue
            piece = self.wproc.tokenizer.decode([token])
            if piece.startswith(" ") and current.strip():
                words.append((current.strip(), first, last))
                current, first = "", None
            current += piece
            first = t if first is None else first
            last = t
        if current.strip():
            words.append((current.strip(), first, last))
        return words

    @torch.inference_mode()
    def transcribe_plain(self, waves: list[np.ndarray]) -> list[str]:
        order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
        out = [""] * len(waves)
        for start in range(0, len(order), 2 * self.whisper_batch):
            idx = order[start : start + 2 * self.whisper_batch]
            features = self.wproc([waves[i] for i in idx], sampling_rate=SR, return_tensors="pt").input_features
            ids = self.whisper.generate(features.to(self.device, torch.float16), language="en", task="transcribe")
            for i, text in zip(idx, self.wproc.batch_decode(ids, skip_special_tokens=True)):
                out[i] = text.strip()
        return out

    @torch.inference_mode()
    def transcribe(self, waves: list[np.ndarray]) -> list[tuple[str, list[tuple[str, float, float]]]]:
        order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
        out: list = [None] * len(waves)
        for start in range(0, len(order), self.whisper_batch):
            idx = order[start : start + self.whisper_batch]
            inputs = self.wproc([waves[i] for i in idx], sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
            features = inputs.input_features.to(self.device, torch.float16)
            # Encode separately: token timestamps make generate() request attention maps,
            # which would otherwise also materialise all 32 encoder self-attention maps.
            encoded = self.whisper.model.encoder(features)
            result = self.whisper.generate(
                features,
                encoder_outputs=encoded,
                attention_mask=inputs.attention_mask.to(self.device),
                language="en", task="transcribe",
                return_token_timestamps=True, return_dict_in_generate=True,
            )
            sequences, times = result["sequences"], result["token_timestamps"]
            for k, i in enumerate(idx):
                text = self.wproc.decode(sequences[k], skip_special_tokens=True).strip()
                out[i] = (text, self._words(sequences[k].tolist(), times[k].tolist()))
        return out

    @torch.inference_mode()
    def encode(self, waves: list[np.ndarray]) -> list[np.ndarray]:
        order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
        out: list[np.ndarray] = [None] * len(waves)  # type: ignore[list-item]
        bucket = BUCKET_SECONDS * self.codec_rate
        for start in range(0, len(order), self.codec_batch):
            idx = order[start : start + self.codec_batch]
            resampled = [AF.resample(torch.from_numpy(waves[i]).to(self.device), SR, self.codec_rate) for i in idx]
            lengths = torch.tensor([w.numel() for w in resampled], device=self.device)
            # Fixed [codec_batch, 1, k * bucket] shapes: few distinct conv shapes for MIOpen.
            padded = -(-int(lengths.max()) // bucket) * bucket
            batch = torch.zeros(self.codec_batch, 1, padded, device=self.device)
            for k, w in enumerate(resampled):
                batch[k, 0, : w.numel()] = w
            lengths = torch.cat([lengths, lengths.new_full((self.codec_batch - len(idx),), padded)])
            codes, code_lengths = self.codec.encode(batch, lengths)
            for k, i in enumerate(idx):
                values = codes[k, :, : int(code_lengths[k])].cpu().numpy()
                assert values.shape[0] == self.num_codebooks and values.min() >= 0 and values.max() < 4096
                out[i] = values.astype(np.int16)
        return out


def process_shard(enc: Encoders, path: Path, shard: str, max_wer: float) -> tuple[pa.Table, dict]:
    t0 = time.time()
    rows = pq.read_table(path).to_pylist()
    stats = {"shard": shard, "rows": len(rows), "duplicates": 0, "bad_duration": 0, "decode_errors": 0,
             "bad_wer": 0, "bad_rate": 0, "no_split": 0, "bad_cut": 0, "kept": 0, "hours_in": 0.0, "hours_kept": 0.0}
    seen: set[str] = set()
    clips = []
    for index, row in enumerate(rows):
        data = row["audio"]["bytes"]
        md5 = hashlib.md5(data).hexdigest()
        if md5 in seen:
            stats["duplicates"] += 1
            continue
        seen.add(md5)
        try:
            wave, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        except Exception:
            stats["decode_errors"] += 1
            continue
        wave = wave.mean(axis=1)
        if rate != SR:
            wave = AF.resample(torch.from_numpy(wave), rate, SR).numpy()
        duration = len(wave) / SR
        stats["hours_in"] += duration / 3600
        if not MIN_DUR <= duration <= MAX_DUR or not np.isfinite(wave).all():
            stats["bad_duration"] += 1
            continue
        clips.append({"index": index, "row": row, "wave": np.ascontiguousarray(wave), "duration": duration, "md5": md5})

    asr = enc.transcribe([c["wave"] for c in clips])
    candidates = []
    for clip, (hyp, words) in zip(clips, asr):
        ref_words = normalize_words(clip["row"]["text"])
        clip["asr"], clip["wer"] = hyp, wer(ref_words, normalize_words(hyp))
        wps = len(ref_words) / clip["duration"]
        if clip["wer"] > max_wer or len(ref_words) < 4:
            stats["bad_wer"] += 1
            continue
        if not MIN_WPS <= wps <= MAX_WPS:
            stats["bad_rate"] += 1
            continue
        split = choose_split(words, clip["duration"])
        if split is None:
            stats["no_split"] += 1
            continue
        cut = int(round((words[split - 1][2] + CUT_AFTER_WORD) * SR))
        clip["split_sec"] = cut / SR
        clip["ref_text"] = " ".join(w for w, _, _ in words[:split])
        clip["text"] = " ".join(w for w, _, _ in words[split:])
        clip["ref_wave"], clip["target_wave"] = clip["wave"][:cut], clip["wave"][cut:]
        candidates.append(clip)

    # Verify the cut: both halves must still contain their boundary words.
    halves = enc.transcribe_plain([c["ref_wave"] for c in candidates] + [c["target_wave"] for c in candidates])
    kept = []
    for k, clip in enumerate(candidates):
        heard_ref, heard_target = normalize_words(halves[k]), normalize_words(halves[len(candidates) + k])
        ref_words, target_words = normalize_words(clip["ref_text"]), normalize_words(clip["text"])
        intact = (heard_ref and ref_words and heard_ref[-1] == ref_words[-1]
                  and heard_target and target_words and heard_target[0] == target_words[0]
                  and wer(target_words, heard_target) <= max_wer)
        if intact:
            kept.append(clip)
        else:
            stats["bad_cut"] += 1

    ref_codes = enc.encode([c["ref_wave"] for c in kept]) if kept else []
    target_codes = enc.encode([c["target_wave"] for c in kept]) if kept else []
    columns = {name: [] for name in SCHEMA.names}
    for clip, ref, target in zip(kept, ref_codes, target_codes):
        columns["id"].append(f"{shard}-{clip['index']:05d}")
        columns["shard"].append(shard)
        columns["audio_id"].append(str(clip["row"].get("audio_id", "")))
        columns["source_text"].append(" ".join(clip["row"]["text"].split()))
        columns["asr_text"].append(clip["asr"])
        columns["wer"].append(clip["wer"])
        columns["duration"].append(clip["duration"])
        columns["split_sec"].append(clip["split_sec"])
        columns["ref_text"].append(clip["ref_text"])
        columns["ref_n_frames"].append(ref.shape[1])
        columns["ref_codes"].append(ref.reshape(-1))
        columns["text"].append(clip["text"])
        columns["n_frames"].append(target.shape[1])
        columns["codes"].append(target.reshape(-1))
        columns["audio_md5"].append(clip["md5"])
        stats["hours_kept"] += clip["duration"] / 3600
    stats["kept"] = len(kept)
    stats["seconds"] = round(time.time() - t0, 1)
    return pa.table(columns, schema=SCHEMA), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--max-shards", type=int, default=None, help="stop after this many shards (this worker)")
    ap.add_argument("--max-wer", type=float, default=0.10)
    ap.add_argument("--whisper-batch", type=int, default=16)  # timestamps keep cross-attention maps
    ap.add_argument("--codec-batch", type=int, default=8)
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--gpu-mem-fraction", type=float, default=None, help="cap this worker's share of GPU memory")
    args = ap.parse_args()

    if args.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    token = os.environ["HF_TOKEN"]
    for sub in ("encoded", "done", "tmp", "logs"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in HfApi(token=token).list_repo_files(SOURCE_REPO, repo_type="dataset") if f.endswith(".parquet"))
    random.Random(0).shuffle(files)  # diverse sources early, stable across restarts
    mine = [f for k, f in enumerate(files) if k % args.num_workers == args.worker]
    failed_dir = ROOT / "failed"
    failed_dir.mkdir(exist_ok=True)

    def attempts(name: str) -> int:
        marker = failed_dir / (Path(name).stem + ".count")
        return int(marker.read_text()) if marker.exists() else 0

    todo = [f for f in mine
            if not (ROOT / "done" / (Path(f).stem + ".json")).exists() and attempts(f) < MAX_ATTEMPTS][: args.max_shards]
    print(f"[worker {args.worker}] {len(mine)} shards assigned, {len(todo)} to do", flush=True)

    tmp = ROOT / "tmp" / f"w{args.worker}"
    shutil.rmtree(tmp, ignore_errors=True)

    def fetch(name: str) -> Path:
        for attempt in range(5):
            try:
                return Path(hf_hub_download(SOURCE_REPO, name, repo_type="dataset", token=token, local_dir=tmp))
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("unreachable")

    enc = Encoders("cuda", args.whisper_batch, args.codec_batch)
    pool = ThreadPoolExecutor(1)
    pending = pool.submit(fetch, todo[0]) if todo else None
    for k, name in enumerate(todo):
        shard = Path(name).stem
        if shutil.disk_usage(ROOT).free / 1e9 < args.min_free_gb:
            raise SystemExit(f"less than {args.min_free_gb} GB free disk; stopping")
        path = None
        try:
            future, pending = pending, None
            path = future.result()
            pending = pool.submit(fetch, todo[k + 1]) if k + 1 < len(todo) else None
            table, stats = process_shard(enc, path, shard, args.max_wer)
            out = ROOT / "encoded" / f"{shard}.parquet"
            pq.write_table(table, out.with_suffix(".tmp"), compression="zstd")
            out.with_suffix(".tmp").replace(out)
            (ROOT / "done" / f"{shard}.json").write_text(json.dumps(stats) + "\n")
            print(f"[worker {args.worker}] {k + 1}/{len(todo)} {shard}: kept {stats['kept']}/{stats['rows']} "
                  f"({stats['hours_kept']:.2f}/{stats['hours_in']:.2f} h) dup={stats['duplicates']} "
                  f"dur={stats['bad_duration']} wer={stats['bad_wer']} rate={stats['bad_rate']} "
                  f"no_split={stats['no_split']} bad_cut={stats['bad_cut']} in {stats['seconds']}s", flush=True)
        except Exception:
            print(f"[worker {args.worker}] FAILED {shard}\n{traceback.format_exc()}", flush=True)
            marker = failed_dir / f"{shard}.count"
            marker.write_text(str(attempts(name) + 1))
            # A GPU error (OOM on the shared card, MIOpen failure) can leave the HIP
            # context unusable, after which every later shard fails. Exit so the
            # supervisor in run_prep.sh restarts with a fresh process.
            if pending is not None:
                pending.cancel()
            shutil.rmtree(tmp, ignore_errors=True)
            raise SystemExit(3)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)  # never leave raw shards behind
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
