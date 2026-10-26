"""Share of spectral energy per band (speech-active frames only) for groups of WAV files."""
import sys, glob
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF

BANDS = [(0, 4000), (4000, 8000), (8000, 12000), (12000, 22050)]

def profile(path):
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != 44100:
        w = AF.resample(torch.from_numpy(w), sr, 44100).numpy()
    spec = np.abs(torch.stft(torch.from_numpy(w), 2048, 512, window=torch.hann_window(2048), return_complex=True).numpy()) ** 2
    freqs = np.linspace(0, 22050, spec.shape[0])
    frame_energy = spec.sum(0)
    active = frame_energy > np.percentile(frame_energy, 30)  # skip silences
    total = spec[:, active].sum()
    return [spec[(freqs >= lo) & (freqs < hi)][:, active].sum() / total for lo, hi in BANDS]

for group in sys.argv[1:]:
    name, pattern = group.split("=", 1)
    files = [f for f in sorted(glob.glob(pattern, recursive=True)) if "combined" not in f]
    if not files:
        continue
    p = np.mean([profile(f) for f in files], axis=0) * 100
    print(f"{name:28s} n={len(files):2d}  0-4k {p[0]:5.1f}%  4-8k {p[1]:5.2f}%  8-12k {p[2]:5.2f}%  12k+ {p[3]:5.2f}%")
