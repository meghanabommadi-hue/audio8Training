#!/usr/bin/env bash
# Re-generate the earlier clone set (out_clone/: simran, sana, rani; same sentences, seed 0)
# with a fine-tuned checkpoint.   ./clone_compare.sh finetune/runs/pilot_v2/checkpoint-150 out_clone_ckpt150
set -euo pipefail
cd "$(dirname "$0")"
MODEL="$1"; OUT="$2"
for v in simran sana rani; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python audio8_tts.py --model "$MODEL" --seed 0 --out-dir "$OUT/$v" \
    --ref-audio "$HOME/FlowTTS/sample_files/$v.wav" \
    "Hello, thank you for calling Kapture support. How can I help you today?" \
    "Your EMI of twelve thousand rupees is due on the fifth of every month." \
    "I'm sorry for the inconvenience. Let me check that for you right away." 2>&1 | grep -E "out_clone|Error" || true
done
