"""Check the cached-prefix path against the official runtime, then time streaming."""
import sys, time
import numpy as np
import soundfile as sf
from engine import StreamingEngine

TEXT = "Your EMI of twelve thousand rupees is due on the fifth of every month."
eng = StreamingEngine(ar_threads=8, codec_threads=8)
eng.prefix("simran")  # build/load cache

ref = np.stack(list(eng.rt.iter_codes(text=TEXT, voice="simran", seed=0, max_new_tokens=512)), 1)
t = time.time(); got = np.stack(list(eng.iter_frames(TEXT, "simran", seed=0, max_frames=512)), 1)
print(f"cached prefix frames identical to official runtime: {ref.shape == got.shape and np.array_equal(ref, got)} ({got.shape[1]} frames, {time.time()-t:.2f}s)")

long_text = ("Hello, thank you for calling Kapture support. Your EMI of twelve thousand rupees is due on the fifth of every month. "
             "I'm sorry for the inconvenience. Let me check that for you right away.")
for run in range(2):
    stats = {}
    chunks = list(eng.stream(long_text, "simran", seed=0, stats=stats))
    print(f"run {run}: first_audio={stats['first_audio_s']:.2f}s total={stats['total_s']:.2f}s audio={stats['audio_s']:.2f}s RTF={stats['rtf']:.3f} chunks={len(chunks)}")
    for s in stats["sentences"]:
        print(f"   prefill {s.get('prefill_tokens')} tok {s.get('prefill_s', 0):.2f}s budget_hit={s.get('hit_frame_budget', False)} | {s['sentence']}")
sf.write("simran_stream_test.wav", np.concatenate(chunks), eng.sample_rate)
