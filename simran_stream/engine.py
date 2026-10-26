"""Low-latency streaming engine for Audio8 0.1B ONNX INT8 on CPU.

On top of the official ``ArkTtsRuntime`` (slow/fast AR graphs, sampling):

1. Voice prefix cache: everything in the prompt before the target text is the
   same for every request with a voice (system text, reference transcript,
   reference codec frames). Its slow-AR state (attention KV + Mamba conv/SSM)
   is computed once, saved as ``voices/<voice>/prefix_cache.npz`` and reused,
   so a request only prefills its own text tokens.
2. Stateful streaming codec (see ``export_stream_codec.py``): each chunk runs
   the post transformer on new frames only (K/V cache) and the conv decoder on
   the new frames plus 8 frames of context. Output matches a full decode.
3. Pipelining: autoregressive frame generation runs in a producer thread while
   the caller's thread decodes audio, so the two overlap.
4. Guard rails for autoregressive drift: per-sentence generation (long text is
   split), and a frame budget proportional to text length so a sentence that
   fails to emit EOS cannot run away.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
RUNTIME_DIR = HERE.parent / "Audio8_TTS/onnx_runtime_0_1b_int8"
sys.path.insert(0, str(RUNTIME_DIR))
from arktts_runtime.runtime import ArkTtsRuntime, _sample  # noqa: E402

ARTIFACTS = HERE / "artifacts"
PREFIX_CACHE = "prefix_cache.npz"
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")


def _session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Split on sentence punctuation; hard-wrap very long sentences on commas/spaces."""
    out: list[str] = []
    for sentence in _SENTENCE_RE.split(" ".join(text.split())):
        while len(sentence) > max_chars:
            cut = max(sentence.rfind(", ", 0, max_chars), sentence.rfind(" ", 0, max_chars))
            cut = cut if cut > 0 else max_chars
            out.append(sentence[: cut + 1].strip())
            sentence = sentence[cut + 1 :].strip()
        if sentence:
            out.append(sentence)
    return out


class StreamingCodec:
    """Stateful wrapper around codec_post_step.onnx + codec_conv.onnx."""

    def __init__(self, post: ort.InferenceSession, conv: ort.InferenceSession, meta: dict):
        self.post, self.conv, self.meta = post, conv, meta
        shape = (meta["layers"], 1, meta["n_kv"], meta["window"] - 1, meta["head_dim"])
        self.cache_k = np.zeros(shape, np.float32)
        self.cache_v = np.zeros(shape, np.float32)
        self.position = 0
        self.history = np.zeros((1, 1024, 0), np.float32)

    def push(self, frames: np.ndarray) -> np.ndarray:
        """frames: [10, n] codes -> float32 audio for exactly those n frames."""
        if self.position + frames.shape[1] > self.meta["max_positions"]:
            raise ValueError("utterance exceeds streaming codec max positions")
        post, self.cache_k, self.cache_v = self.post.run(None, {
            "codes": frames[np.newaxis].astype(np.int64, copy=False),
            "start": np.asarray([self.position], np.int64),
            "cache_k": self.cache_k,
            "cache_v": self.cache_v,
        })
        self.position += frames.shape[1]
        window = np.concatenate([self.history, post], axis=2)
        context = window.shape[2] - post.shape[2]
        audio = self.conv.run(None, {"post": window})[0][0]
        self.history = window[:, :, -self.meta["conv_context"] :]
        return audio[context * self.meta["hop"] :]


class StreamingEngine:
    def __init__(self, ar_threads: int = 4, codec_threads: int = 4, voices_dir: Path | None = None):
        self.rt = ArkTtsRuntime(RUNTIME_DIR / "model", voices_dir or RUNTIME_DIR / "voices", threads=ar_threads)
        self.codec_meta = json.loads((ARTIFACTS / "stream_codec.json").read_text())
        self.codec_post = _session(ARTIFACTS / "codec_post_step.onnx", codec_threads)
        self.codec_conv = _session(ARTIFACTS / "codec_conv.onnx", codec_threads)
        self.sample_rate = int(self.codec_meta["sample_rate"])
        self._prefix: dict[str, dict] = {}
        self._prefix_lock = threading.Lock()
        m = self.rt.manifest
        self._begin, self._stop = int(m["semantic_begin_id"]), int(m["im_end_id"])
        self._codebooks, self._codebook_size = int(m["num_codebooks"]), int(m["codebook_size"])
        self._max_seq = int(m["max_seq_len"])

    # ---- voice prefix cache -------------------------------------------------

    def _voice_digest(self, codes: np.ndarray, meta: dict) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(codes, dtype=np.int64).tobytes())
        h.update(str(meta["reference_text"]).encode())
        h.update(str(self.rt.manifest["model_fingerprint"]).encode())
        return h.hexdigest()

    def _prefix_columns(self, codes: np.ndarray, meta: dict) -> np.ndarray:
        # The target text is the first thing that differs between two prompts.
        a = self.rt.prompt_builder.build("A", meta["reference_text"], codes)
        b = self.rt.prompt_builder.build("Z", meta["reference_text"], codes)
        n = min(a.shape[2], b.shape[2])
        length = int(np.argmax((a[0, :, :n] != b[0, :, :n]).any(axis=0)))
        return a[:, :, :length]

    def prefix(self, voice: str) -> dict:
        """Return (and lazily build/load) the cached slow-AR state for a voice."""
        with self._prefix_lock:
            if voice in self._prefix:
                return self._prefix[voice]
            codes, meta = self.rt.voices.load(voice)
            digest = self._voice_digest(codes, meta)
            path = self.rt.voices.root / voice / PREFIX_CACHE
            entry = None
            if path.is_file():
                data = np.load(path, allow_pickle=False)
                if str(data["digest"]) == digest:
                    entry = {k: data[k] for k in ("columns", "cache_keys", "cache_values", "conv_states", "ssm_states")}
            if entry is None:
                t = time.time()
                columns = self._prefix_columns(codes, meta)
                state = self.rt._empty_slow_state()
                for position in range(columns.shape[2]):
                    self.rt._slow_step(columns[:, :, position : position + 1], position, state)
                length = columns.shape[2]
                entry = {
                    "columns": columns,
                    "cache_keys": state["cache_keys"][:, :, :, :length].copy(),
                    "cache_values": state["cache_values"][:, :, :, :length].copy(),
                    "conv_states": state["conv_states"],
                    "ssm_states": state["ssm_states"],
                }
                tmp = path.with_suffix(".tmp.npz")
                np.savez(tmp, digest=np.asarray(digest), **entry)
                tmp.replace(path)
                print(f"[engine] built prefix cache for {voice!r}: {length} positions in {time.time() - t:.1f}s -> {path}", flush=True)
            entry["meta"] = meta
            entry["codes"] = codes
            self._prefix[voice] = entry
            return entry

    def _state_from_prefix(self, entry: dict) -> dict[str, np.ndarray]:
        state = self.rt._empty_slow_state()
        length = entry["columns"].shape[2]
        state["cache_keys"][:, :, :, :length] = entry["cache_keys"]
        state["cache_values"][:, :, :, :length] = entry["cache_values"]
        state["conv_states"] = entry["conv_states"].copy()
        state["ssm_states"] = entry["ssm_states"].copy()
        return state

    # ---- autoregressive generation -------------------------------------------

    def iter_frames(
        self,
        text: str,
        voice: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int | None = None,
        max_frames: int | None = None,
        stop_event: threading.Event | None = None,
        stats: dict | None = None,
    ) -> Iterator[np.ndarray]:
        """Same sampling as ArkTtsRuntime.iter_codes, starting from the cached prefix."""
        entry = self.prefix(voice)
        prompt = self.rt.prompt_builder.build(text, entry["meta"]["reference_text"], entry["codes"])
        cached = entry["columns"].shape[2]
        if not np.array_equal(prompt[:, :, :cached], entry["columns"]):
            raise RuntimeError("prompt prefix does not match cached voice prefix")
        prompt_len = prompt.shape[2]
        budget = max_frames if max_frames is not None else 40 + 3 * len(text)
        budget = min(budget, self._max_seq - prompt_len)
        rng = np.random.default_rng(seed)

        t0 = time.time()
        state = self._state_from_prefix(entry)
        logits = hidden = None
        for position in range(cached, prompt_len):
            logits, hidden = self.rt._slow_step(prompt[:, :, position : position + 1], position, state)
        if stats is not None:
            stats["prefill_tokens"] = prompt_len - cached
            stats["prefill_s"] = time.time() - t0

        previous: list[int] = []
        for step in range(budget):
            if stop_event is not None and stop_event.is_set():
                return
            semantic = self.rt._sample_semantic(logits, previous, temperature, top_p, top_k, rng)
            if semantic == self._stop:
                return
            previous.append(semantic)
            previous = previous[-10:]
            caches = self.rt._empty_fast_caches()
            self.rt._fast_step(hidden, 0, True, 0, caches)
            token = semantic - self._begin
            frame = [token]
            for fast_position in range(1, self._codebooks):
                fast_logits = self.rt._fast_step(hidden, token, False, fast_position, caches)
                token = _sample(fast_logits, temperature, top_p, top_k, rng)
                frame.append(token)
            yield np.asarray(frame, dtype=np.int64)
            if step + 1 >= budget:
                if stats is not None:
                    stats["hit_frame_budget"] = True
                return
            column = np.concatenate([[semantic], frame]).reshape(1, -1, 1)
            logits, hidden = self.rt._slow_step(column, prompt_len + step, state)

    # ---- streaming -------------------------------------------------------------

    def stream(
        self,
        text: str,
        voice: str = "simran",
        first_chunk: int = 4,
        chunk: int = 12,
        max_chunk: int = 24,
        sentence_pause: float = 0.12,
        seed: int | None = None,
        stats: dict | None = None,
        **sampling,
    ) -> Iterator[np.ndarray]:
        """Yield float32 44.1 kHz audio chunks as soon as they are decoded.

        Chunks start small for a fast first byte and grow (``chunk`` up to
        ``max_chunk`` frames) once playback is ahead, which amortises the
        codec's 8-frame context.
        """
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("text must not be empty")
        frames_q: queue.Queue = queue.Queue(maxsize=256)
        stop = threading.Event()
        gen_stats: list[dict] = []

        def produce():
            try:
                for index, sentence in enumerate(sentences):
                    s = {"sentence": sentence}
                    gen_stats.append(s)
                    sentence_seed = None if seed is None else seed + index
                    for frame in self.iter_frames(sentence, voice, seed=sentence_seed, stop_event=stop, stats=s, **sampling):
                        frames_q.put(frame)
                    frames_q.put("eos")
            except BaseException as exc:  # surface errors in the consumer
                frames_q.put(exc)
            finally:
                frames_q.put(None)

        producer = threading.Thread(target=produce, daemon=True)
        t0 = time.time()
        producer.start()
        pause = np.zeros(int(sentence_pause * self.sample_rate), np.float32)
        pending: list[np.ndarray] = []
        size = first_chunk
        first_audio = None
        audio_samples = 0
        try:
            codec = StreamingCodec(self.codec_post, self.codec_conv, self.codec_meta)
            done = False
            while not done:
                item = frames_q.get()
                flush = False
                if item is None:
                    done = flush = True
                elif isinstance(item, BaseException):
                    raise item
                elif isinstance(item, str):  # end of sentence
                    flush = True
                else:
                    pending.append(item)
                    flush = len(pending) >= size
                if flush and pending:
                    audio = codec.push(np.stack(pending, axis=1))
                    pending = []
                    if first_audio is None:
                        first_audio = time.time() - t0
                    audio_samples += audio.size
                    yield audio
                    size = chunk if size < chunk else min(max_chunk, size * 2)
                if isinstance(item, str) and sentence_pause > 0 and not done:
                    audio_samples += pause.size
                    yield pause
        finally:
            stop.set()
            while producer.is_alive():
                try:
                    frames_q.get_nowait()
                except queue.Empty:
                    producer.join(timeout=0.05)
            if stats is not None:
                total = time.time() - t0
                stats.update({
                    "first_audio_s": first_audio,
                    "total_s": total,
                    "audio_s": audio_samples / self.sample_rate,
                    "rtf": total / max(audio_samples / self.sample_rate, 1e-6),
                    "sentences": gen_stats,
                })

    def synthesize(self, text: str, voice: str = "simran", **kwargs) -> np.ndarray:
        return np.concatenate(list(self.stream(text, voice, **kwargs)))
