"""Streaming TTS server for Audio8 0.1B (ONNX INT8, CPU) with a pool of engines.

Each engine owns its ONNX sessions and handles one request at a time; requests
queue for a free engine. Audio streams as 16-bit PCM (or a WAV header + PCM)
at 44.1 kHz mono as soon as each chunk is decoded.

    ENGINES=6 AR_THREADS=2 CODEC_THREADS=2 \
      ../Audio8_TTS/onnx_runtime_0_1b_int8/.venv/bin/python -m uvicorn server:app --port 8030

    curl -N localhost:8030/v1/tts -H 'content-type: application/json' \
      -d '{"text": "Your EMI is due on the fifth.", "format": "wav"}' > out.wav
"""

from __future__ import annotations

import os
import queue
import struct
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine import StreamingEngine

ENGINES = int(os.environ.get("ENGINES", "4"))
AR_THREADS = int(os.environ.get("AR_THREADS", "2"))
CODEC_THREADS = int(os.environ.get("CODEC_THREADS", "2"))
DEFAULT_VOICE = os.environ.get("VOICE", "simran")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "1000"))


class Pool:
    def __init__(self, size: int):
        self.engines: queue.Queue[StreamingEngine] = queue.Queue()
        self.size = size
        self.lock = threading.Lock()
        self.waiting = 0
        self.served = 0
        self.first_audio = deque(maxlen=500)
        self.queue_wait = deque(maxlen=500)
        self.rtf = deque(maxlen=500)
        for index in range(size):
            engine = StreamingEngine(ar_threads=AR_THREADS, codec_threads=CODEC_THREADS)
            engine.prefix(DEFAULT_VOICE)
            engine.synthesize("Warm up.", DEFAULT_VOICE, seed=0)  # first-run graph/kernel setup
            self.engines.put(engine)
            print(f"[server] engine {index + 1}/{size} ready", flush=True)

    def acquire(self) -> tuple[StreamingEngine, float]:
        start = time.time()
        with self.lock:
            self.waiting += 1
        try:
            return self.engines.get(), time.time() - start
        finally:
            with self.lock:
                self.waiting -= 1

    def release(self, engine: StreamingEngine, stats: dict, waited: float) -> None:
        self.engines.put(engine)
        with self.lock:
            self.served += 1
            self.queue_wait.append(waited)
            if stats.get("first_audio_s") is not None:
                self.first_audio.append(stats["first_audio_s"] + waited)
                self.rtf.append(stats["rtf"])


pool: Pool | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pool
    pool = Pool(ENGINES)
    yield


app = FastAPI(title="Audio8 Simran streaming TTS", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = DEFAULT_VOICE
    format: str = Field("pcm", pattern="^(pcm|wav)$")
    seed: int | None = None
    temperature: float = Field(0.7, gt=0, le=1.5)
    top_p: float = Field(0.9, gt=0, le=1.0)
    top_k: int = Field(50, ge=1, le=4096)


class OpenAISpeechRequest(BaseModel):
    model: str = "audio8-0.1b"
    input: str = Field(..., min_length=1)
    voice: str = DEFAULT_VOICE
    response_format: str = Field("wav", pattern="^(pcm|wav)$")


def wav_header(sample_rate: int) -> bytes:
    # Streaming WAV: sizes set to the maximum since the length is unknown up front.
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16) +
            b"data" + struct.pack("<I", 0xFFFFFFFF))


def audio_stream(text: str, voice: str, fmt: str, seed=None, **sampling):
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"text longer than {MAX_CHARS} characters")
    assert pool is not None

    def generate():
        engine, waited = pool.acquire()
        stats: dict = {}
        try:
            if fmt == "wav":
                yield wav_header(engine.sample_rate)
            for chunk in engine.stream(text, voice, seed=seed, stats=stats, **sampling):
                yield (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        finally:
            pool.release(engine, stats, waited)

    media = "audio/wav" if fmt == "wav" else "audio/L16;rate=44100;channels=1"
    return StreamingResponse(generate(), media_type=media)


@app.post("/v1/tts")
def tts(request: TTSRequest):
    return audio_stream(request.text, request.voice, request.format, seed=request.seed,
                        temperature=request.temperature, top_p=request.top_p, top_k=request.top_k)


@app.post("/v1/audio/speech")
def openai_speech(request: OpenAISpeechRequest):
    return audio_stream(request.input, request.voice, request.response_format)


@app.get("/health")
def health():
    return {"ok": pool is not None, "engines": ENGINES}


@app.get("/metrics")
def metrics():
    assert pool is not None
    pct = lambda values, q: round(float(np.percentile(values, q)), 3) if values else None  # noqa: E731
    with pool.lock:
        return {
            "engines": pool.size,
            "busy": pool.size - pool.engines.qsize(),
            "waiting": pool.waiting,
            "served": pool.served,
            "first_audio_s_p50": pct(list(pool.first_audio), 50),
            "first_audio_s_p95": pct(list(pool.first_audio), 95),
            "queue_wait_s_p95": pct(list(pool.queue_wait), 95),
            "rtf_p50": pct(list(pool.rtf), 50),
            "config": {"ar_threads": AR_THREADS, "codec_threads": CODEC_THREADS},
        }
