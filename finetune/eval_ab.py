"""A/B a fine-tuned checkpoint against the base model on Indian English sentences.

Synthesizes every sentence with Simran's reference using each model (same seeds),
writes WAVs for listening, and scores intelligibility with Whisper (WER).

    source ~/TTS/.venv/bin/activate
    python eval_ab.py runs/pilot            # compares against the base model
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoModel, AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor

BASE = str(Path.home() / "models/Edge0-Audio8-TTS-Preview-0.1b")
REF_AUDIO = str(Path.home() / "FlowTTS/sample_files/simran.wav")
REF_TEXT = ("Hello, I am talking about finance. Your loan account number ending in 3456, "
            "the total due amount is rupees 22,500.")
SENTENCES = [
    "Hello, thank you for calling Kapture support. How can I help you today?",
    "Your EMI of twelve thousand rupees is due on the fifth of every month.",
    "Please keep your PAN card and Aadhaar card ready for the KYC verification.",
    "I have raised a ticket for you, and our team will call you back within twenty four hours.",
    "You can pay using UPI, net banking, or a debit card through our mobile app.",
    "Mister Venkatesh, your policy renewal from Star Health is pending since last week.",
    "We have a branch in Koramangala, Bangalore, and another one in Andheri, Mumbai.",
    "Kindly do the needful and revert back to us at the earliest.",
    "Your order from Lenskart has been dispatched and will reach Hyderabad by Thursday.",
    "I completely understand your concern, ma'am. Let me check the status for you.",
]
SEEDS = [0, 1]


def words(text):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def wer(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1] / max(len(ref), 1)


@torch.inference_mode()
def synthesize(model_path, out_dir: Path):
    # Trainer checkpoints do not include the processor; the tokenizer is unchanged by fine-tuning.
    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    model.load_codec(device="cuda", dtype=torch.float32)
    for seed in SEEDS:
        # One sentence at a time keeps peak GPU memory low on the shared card.
        for i, sentence in enumerate(SENTENCES):
            torch.manual_seed(seed * 1000 + i)
            inputs = processor(text=[sentence], reference_audio=[REF_AUDIO],
                               reference_text=[REF_TEXT], return_tensors="pt")
            output = model.generate(**{k: v.cuda() for k, v in inputs.items()}, max_new_tokens=512, temperature=0.7,
                                    top_p=0.9, top_k=50, do_sample=True, return_dict_in_generate=True)
            waves, lengths = model.decode_audio(output.codes)
            path = out_dir / f"s{i:02d}_seed{seed}.wav"
            sf.write(path, waves[0, : int(lengths[0])].float().cpu().numpy(), model.config.codec_sample_rate)
            paths.append((i, path))
    del model
    torch.cuda.empty_cache()
    return paths


@torch.inference_mode()
def score(paths, asr):
    proc, model = asr
    results = []
    for i, path in paths:
        wave, rate = sf.read(path, dtype="float32")
        wave = AF.resample(torch.from_numpy(wave), rate, 16000).numpy()
        feats = proc(wave, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
        hyp = proc.batch_decode(model.generate(feats, language="en"), skip_special_tokens=True)[0].strip()
        results.append({"sentence": i, "file": path.name, "wer": wer(words(SENTENCES[i]), words(hyp)),
                        "duration": len(wave) / 16000, "asr": hyp})
    return results


def main():
    tuned = sys.argv[1]
    out = Path(tuned) / "ab_eval"
    asr_id = "openai/whisper-large-v3-turbo"
    asr = (WhisperProcessor.from_pretrained(asr_id),
           WhisperForConditionalGeneration.from_pretrained(asr_id, dtype=torch.float16).cuda().eval())
    report = {}
    for name, path in [("base", BASE), ("tuned", tuned)]:
        report[name] = score(synthesize(path, out / name), asr)
        mean = np.mean([r["wer"] for r in report[name]])
        print(f"{name}: mean WER {mean:.3f}  ({len(report[name])} clips) -> {out / name}")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    for i, sentence in enumerate(SENTENCES):
        b = np.mean([r["wer"] for r in report["base"] if r["sentence"] == i])
        t = np.mean([r["wer"] for r in report["tuned"] if r["sentence"] == i])
        print(f"  s{i:02d} base {b:.2f} tuned {t:.2f} | {sentence}")


if __name__ == "__main__":
    main()
