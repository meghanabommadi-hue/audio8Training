"""English TTS with Edge0/Audio8-TTS-Preview-0.1b.

Examples:
    python audio8_tts.py "Hello there." "How can I help you today?"
    python audio8_tts.py -f sentences.txt --out-dir out
    python audio8_tts.py "Hi." --ref-audio ref.wav --ref-text "Transcript of ref.wav."
    python audio8_tts.py "Hi." --ref-audio ref.wav          # transcript via Whisper
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL = "/home/meghna/models/Edge0-Audio8-TTS-Preview-0.1b"


class Audio8TTS:
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        # Fine-tuning checkpoints have no processor files; the tokenizer is unchanged.
        self.processor = AutoProcessor.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=dtype)
            .eval()
            .to(self.device)
        )
        self.sample_rate = self.model.config.codec_sample_rate

    @torch.inference_mode()
    def synthesize(
        self,
        texts: list[str],
        ref_audio: str | None = None,
        ref_text: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> list[np.ndarray]:
        """Synthesize a batch of sentences; returns one float32 waveform per sentence."""
        kwargs = {}
        if ref_audio:
            if not ref_text:
                raise ValueError("--ref-text is required with --ref-audio")
            kwargs = {
                "reference_audio": [ref_audio] * len(texts),
                "reference_text": [ref_text] * len(texts),
            }
        inputs = self.processor(text=texts, return_tensors="pt", **kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            return_dict_in_generate=True,
        )
        waveforms, lengths = self.model.decode_audio(output.codes)
        return [waveforms[i, : int(lengths[i])].float().cpu().numpy() for i in range(len(texts))]


def transcribe(audio_path: str, model_id: str = "openai/whisper-large-v3-turbo") -> str:
    """Whisper transcript of a reference clip, used when --ref-text is not given."""
    import torchaudio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device).eval()
    wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torchaudio.functional.resample(torch.from_numpy(wav.mean(axis=1)), sr, 16000).numpy()
    features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
    with torch.inference_mode():
        ids = model.generate(features, language="en", task="transcribe")
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    del model
    torch.cuda.empty_cache()
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="English sentences (each arg may hold several)")
    ap.add_argument("-f", "--file", help="Text file, one or more sentences per line")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Default: cuda if available")
    ap.add_argument("--ref-audio", help="Reference wav for zero-shot voice cloning")
    ap.add_argument("--ref-text", help="Exact transcript of --ref-audio (auto-transcribed with Whisper if omitted)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=512, help="~21.5 frames per second of audio")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pause", type=float, default=0.3, help="Silence (s) between sentences in combined.wav")
    args = ap.parse_args()

    raw = list(args.text)
    if args.file:
        raw += Path(args.file).read_text(encoding="utf-8").splitlines()
    sentences = [s for chunk in raw for s in split_sentences(chunk)]
    if not sentences:
        ap.error("no input text")

    if args.ref_audio and not args.ref_text:
        args.ref_text = transcribe(args.ref_audio)
        print(f"Reference transcript: {args.ref_text!r}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    t0 = time.time()
    tts = Audio8TTS(args.model, device=args.device)
    print(f"Loaded model on {tts.device} in {time.time() - t0:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for start in range(0, len(sentences), args.batch_size):
        batch = sentences[start : start + args.batch_size]
        t = time.time()
        wavs = tts.synthesize(
            batch,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        elapsed = time.time() - t
        audio_secs = sum(len(w) for w in wavs) / tts.sample_rate
        print(f"batch {start // args.batch_size}: {len(batch)} sentences, {audio_secs:.1f}s audio "
              f"in {elapsed:.2f}s (RTF {elapsed / max(audio_secs, 1e-6):.3f})")
        for i, (sentence, wav) in enumerate(zip(batch, wavs), start=start):
            path = out_dir / f"{i:03d}.wav"
            sf.write(path, wav, tts.sample_rate)
            print(f"  {path}  {len(wav) / tts.sample_rate:.2f}s  {sentence!r}")
            clips.append(wav)

    gap = np.zeros(int(args.pause * tts.sample_rate), dtype=np.float32)
    combined = np.concatenate([x for c in clips for x in (c, gap)][:-1])
    sf.write(out_dir / "combined.wav", combined, tts.sample_rate)
    print(f"  {out_dir / 'combined.wav'}  {len(combined) / tts.sample_rate:.2f}s")


if __name__ == "__main__":
    main()
