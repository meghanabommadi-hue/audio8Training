#!/usr/bin/env bash
# Pilot fine-tune of Audio8 0.1B on the encoded Indian English data (resumable).
#   RUN=pilot_full MAX_STEPS=3000 ./run_train.sh
#   FREEZE_FAST_AR=true FAST_LOSS_WEIGHT=0 RUN=pilot_slow ./run_train.sh
set -euo pipefail
cd "$(dirname "$0")"
source "$HOME/TTS/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
RUN="${RUN:-pilot_full}"
mkdir -p runs/"$RUN"
setsid nohup python train_sft.py \
  --output_dir "runs/$RUN" \
  --freeze_fast_ar "${FREEZE_FAST_AR:-false}" \
  --fast_loss_weight "${FAST_LOSS_WEIGHT:-1.0}" \
  --slow_loss_weight "${SLOW_LOSS_WEIGHT:-1.0}" \
  --max_steps "${MAX_STEPS:-3000}" \
  --per_device_train_batch_size "${BATCH_SIZE:-4}" \
  --per_device_eval_batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate "${LR:-1e-5}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --eval_strategy steps --eval_steps "${EVAL_STEPS:-250}" \
  --save_strategy steps --save_steps "${EVAL_STEPS:-250}" --save_total_limit 4 \
  --logging_steps 10 \
  --dataloader_num_workers 4 \
  --report_to tensorboard \
  --resume_mode auto \
  >> "runs/$RUN/train.log" 2>&1 &
echo "started runs/$RUN (log: runs/$RUN/train.log)"
