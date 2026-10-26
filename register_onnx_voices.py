"""Register FlowTTS Indian English reference voices with the Audio8 0.1B ONNX runtime.

Run with the ONNX runtime venv:
    Audio8_TTS/onnx_runtime_0_1b_int8/.venv/bin/python register_onnx_voices.py
"""

import json
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent / "Audio8_TTS/onnx_runtime_0_1b_int8"
sys.path.insert(0, str(RUNTIME))
from arktts_runtime.registration import VoiceRegistration  # noqa: E402

SAMPLES = Path.home() / "FlowTTS/sample_files"
# Transcripts produced by whisper-large-v3-turbo.
VOICES = {
    "simran": "Hello, I am talking about finance. Your loan account number ending in 3456, "
              "the total due amount is rupees 22,500.",
    "sana": "My dad is a lawyer. They were very scared of him. Obviously, everybody in the colony "
            "was scared of him. So, yeah, I think that was my baddie origin story. But I think it was Disney.",
    "rani": "came here, I got to know this, this modulation, this fluctuation in voice by you only. So I started",
}

model_dir = RUNTIME / "model"
manifest = json.loads((model_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
reg = VoiceRegistration(model_dir / "registration", RUNTIME / "voices", str(manifest["model_fingerprint"]))
for name, text in VOICES.items():
    wav = SAMPLES / f"{name}.wav"
    result = reg.register(wav.read_bytes(), wav.name, text, name, overwrite=True)
    print(name, {k: v for k, v in result.items() if k != "reference_text"})
