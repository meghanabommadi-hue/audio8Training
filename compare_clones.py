"""Whisper-transcribe two clone folders (same voices/files) side by side: WER and duration."""
import re, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF
from transformers import WhisperForConditionalGeneration, WhisperProcessor

TEXTS = ["Hello, thank you for calling Kapture support.", "How can I help you today?",
         "Your EMI of twelve thousand rupees is due on the fifth of every month.",
         "I'm sorry for the inconvenience.", "Let me check that for you right away."]
words = lambda t: re.sub(r"[^a-z0-9 ]+", " ", t.lower().replace("kapture", "capture")).split()

def wer(r, h):
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        n = [i] + [0] * len(h)
        for j, b in enumerate(h, 1):
            n[j] = min(d[j] + 1, n[j - 1] + 1, d[j - 1] + (a != b))
        d = n
    return d[-1] / len(r)

proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo", dtype=torch.float16).cuda().eval()
def asr(path):
    w, sr = sf.read(path, dtype="float32")
    f = proc(AF.resample(torch.from_numpy(w), sr, 16000).numpy(), sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
    with torch.inference_mode():
        return proc.batch_decode(model.generate(f, language="en"), skip_special_tokens=True)[0].strip(), len(w) / sr

dirs = sys.argv[1:]
totals = {d: [] for d in dirs}
for voice in ["simran", "sana", "rani"]:
    for i, text in enumerate(TEXTS):
        row = []
        for d in dirs:
            heard, dur = asr(Path(d) / voice / f"{i:03d}.wav")
            e = wer(words(text), words(heard)); totals[d].append(e)
            row.append(f"{dur:5.1f}s WER {e:.2f} {heard[:45]!r}")
        print(f"{voice:6s} {i}: " + "  ||  ".join(row))
for d in dirs:
    print(f"{d}: mean WER {np.mean(totals[d]):.3f}")
