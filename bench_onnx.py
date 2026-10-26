"""Benchmark the Audio8 0.1B ONNX INT8 runtime on CPU.

    Audio8_TTS/onnx_runtime_0_1b_int8/.venv/bin/python bench_onnx.py samples
    Audio8_TTS/onnx_runtime_0_1b_int8/.venv/bin/python bench_onnx.py threads
    Audio8_TTS/onnx_runtime_0_1b_int8/.venv/bin/python bench_onnx.py concurrency
"""

import multiprocessing as mp
import sys
import time
from pathlib import Path

import soundfile as sf

RUNTIME = Path(__file__).resolve().parent / "Audio8_TTS/onnx_runtime_0_1b_int8"
sys.path.insert(0, str(RUNTIME))
from arktts_runtime.runtime import ArkTtsRuntime  # noqa: E402

SR = 44100
SENTENCES = [
    "Hello, thank you for calling Kapture support. How can I help you today?",
    "Your EMI of twelve thousand rupees is due on the fifth of every month.",
    "I'm sorry for the inconvenience. Let me check that for you right away.",
]
BENCH_TEXT = SENTENCES[1]


def load(threads):
    return ArkTtsRuntime(RUNTIME / "model", RUNTIME / "voices", threads=threads)


def timed(rt, text, voice, seed=0):
    t = time.time()
    first = None
    chunks = []
    for event in rt.stream(text=text, voice=voice, seed=seed, max_new_tokens=512):
        if event["type"] != "audio_chunk":
            continue
        if first is None:
            first = time.time() - t
        chunks.append(event["audio"])
    return time.time() - t, first, chunks


def samples():
    import numpy as np

    rt = load(8)
    out = Path(__file__).resolve().parent / "out_onnx"
    for voice in ["default", "simran", "sana", "rani"]:
        (out / voice).mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(SENTENCES):
            wav, _ = rt.synthesize(text=text, voice=voice, seed=0, max_new_tokens=512)
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            sf.write(out / voice / f"{i:03d}.wav", wav, SR)
            print(f"{voice}/{i:03d}.wav {len(wav) / SR:.2f}s", flush=True)


def threads():
    for n in [2, 4, 8, 12, 24]:
        rt = load(n)
        timed(rt, "Warm up.", "simran")
        el, first, chunks = timed(rt, BENCH_TEXT, "simran")
        audio = sum(c.size for c in chunks) / SR
        print(f"threads={n:2d} wall={el:5.2f}s audio={audio:4.2f}s RTF={el / audio:.3f} first_chunk={first:.2f}s", flush=True)


def _worker(args):
    threads, barrier = args
    rt = load(threads)
    timed(rt, "Warm up.", "simran")
    barrier.wait()
    el, first, chunks = timed(rt, BENCH_TEXT, "simran")
    return el, first, sum(c.size for c in chunks) / SR


def concurrency():
    for procs, th in [(1, 8), (2, 8), (3, 8), (4, 6), (6, 4), (8, 3), (12, 2)]:
        with mp.Manager() as m:
            barrier = m.Barrier(procs)
            with mp.Pool(procs) as pool:
                res = pool.map(_worker, [(th, barrier)] * procs)
        rtfs = [el / a for el, _, a in res]
        firsts = [f for _, f, _ in res]
        print(f"procs={procs:2d} x threads={th}: worst_RTF={max(rtfs):.3f} mean_RTF={sum(rtfs) / procs:.3f} "
              f"worst_first_chunk={max(firsts):.2f}s", flush=True)


if __name__ == "__main__":
    {"samples": samples, "threads": threads, "concurrency": concurrency}[sys.argv[1]]()
