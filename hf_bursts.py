"""Find short high-frequency bursts: per file, the worst frames' share of energy above 4 kHz."""
import sys, glob
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF

def frames(path):
    w, sr = sf.read(path, dtype="float32", always_2d=True); w = w.mean(1)
    if sr != 44100: w = AF.resample(torch.from_numpy(w), sr, 44100).numpy()
    spec = np.abs(torch.stft(torch.from_numpy(w), 2048, 512, window=torch.hann_window(2048), return_complex=True).numpy()) ** 2
    f = np.linspace(0, 22050, spec.shape[0]); e = spec.sum(0)
    act = e > np.percentile(e, 30)
    hf = spec[f >= 4000][:, act].sum(0) / e[act]
    loud = 10 * np.log10(e[act] / np.median(e[act]))
    return hf, loud, len(w) / 44100

for d in sys.argv[1:]:
    files = [f for f in sorted(glob.glob(f"{d}/*/*.wav")) if "combined" not in f]
    if not files: print(d, "(no files yet)"); continue
    print(f"== {d}")
    for f in files:
        hf, loud, dur = frames(f)
        print(f"  {f.split('/',1)[1]:18s} {dur:5.1f}s  HF share p50 {np.median(hf)*100:5.2f}%  p99 {np.percentile(hf,99)*100:5.1f}%  max {hf.max()*100:5.1f}%  frames>10% HF: {(hf>0.10).sum():3d}  peak loudness +{loud.max():4.1f} dB")
