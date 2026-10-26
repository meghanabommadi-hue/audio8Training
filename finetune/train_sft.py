"""Fine-tune Audio8 0.1B on encoded Indian English shards (parquet from prep_stream.py).

Reuses the official trainer (Audio8_TTS/audio8_tts_sft.py: DualAR slow/fast
losses, branch freezing) with a parquet-backed dataset that

* deduplicates clips across shards and holds out ~2% of shards for eval,
* conditions every example on its own reference: prep_stream.py splits each clip
  at a sentence boundary into reference (first part) + target (rest), so every
  example matches voice-cloning use.

    source ~/TTS/.venv/bin/activate
    python train_sft.py --output_dir runs/pilot --max_steps 3000 ...
"""

from __future__ import annotations

import hashlib
import logging
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoProcessor, HfArgumentParser

REPO = Path(__file__).resolve().parents[1] / "Audio8_TTS"
sys.path.insert(0, str(REPO))
from audio8_tts_data import build_sft_example, collate_sft_examples  # noqa: E402
from audio8_tts_sft import (  # noqa: E402
    Audio8TTSTrainer,
    Audio8TTSTrainingArguments,
    actual_model,
    fast_codebook_logits,
    resolve_resume_checkpoint,
    sanitize_config_for_save,
    set_trainable_modules,
)

LOGGER = logging.getLogger("train_sft")
COLUMNS = ["id", "shard", "text", "n_frames", "codes", "ref_text", "ref_n_frames", "ref_codes", "audio_md5"]


@dataclass
class Args:
    model_name_or_path: str = field(default=str(Path.home() / "models/Edge0-Audio8-TTS-Preview-0.1b"))
    data_dir: str = field(default=str(Path(__file__).resolve().parent / "data/encoded"))
    max_length: int = field(default=2048)
    freeze_slow_ar: bool = field(default=False)
    freeze_fast_ar: bool = field(default=True)
    eval_shard_mod: int = field(default=50, metadata={"help": "shards with hash % mod == 0 are held out"})
    max_eval_samples: int = field(default=400)
    reference_prob: float = field(default=1.0, metadata={"help": "share of examples trained with their reference"})


def is_eval_shard(shard: str, mod: int) -> bool:
    return int(hashlib.md5(shard.encode()).hexdigest(), 16) % mod == 0


def load_rows(data_dir: Path, eval_mod: int):
    train, evals, seen = [], [], set()
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no encoded shards in {data_dir}")
    for path in files:
        table = pq.read_table(path, columns=COLUMNS).to_pydict()
        for k in range(len(table["id"])):
            md5 = table["audio_md5"][k]
            if md5 in seen:
                continue
            seen.add(md5)
            row = {
                "id": table["id"][k],
                "text": table["text"][k],
                "codes": np.asarray(table["codes"][k], dtype=np.int16).reshape(10, int(table["n_frames"][k])),
                "ref_text": table["ref_text"][k],
                "ref_codes": np.asarray(table["ref_codes"][k], dtype=np.int16).reshape(10, int(table["ref_n_frames"][k])),
            }
            (evals if is_eval_shard(table["shard"][k], eval_mod) else train).append(row)
    return train, evals, len(files)


class ParquetSFTDataset(Dataset):
    def __init__(self, rows, *, tokenizer, config, args: Args, train: bool):
        self.rows, self.tokenizer, self.config, self.args, self.train = rows, tokenizer, config, args, train
        LOGGER.info("%s dataset: %d clips", "train" if train else "eval", len(rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rng = random if self.train else random.Random(index)
        use_ref = rng.random() < self.args.reference_prob
        return build_sft_example(
            tokenizer=self.tokenizer, config=self.config, text=row["text"],
            target_codes=torch.from_numpy(row["codes"].astype(np.int64)),
            reference_codes=torch.from_numpy(row["ref_codes"].astype(np.int64)) if use_ref else None,
            reference_text=row["ref_text"] if use_ref else None,
            max_length=self.args.max_length,
        )


class CompactHeadTrainer(Audio8TTSTrainer):
    """The official trainer, adapted to the 0.1B checkpoint's compact slow head.

    The repo trainer scores slow logits against full-vocabulary token IDs, but
    the 0.1B model's ``semantic_output`` only has ``codebook_size + 1`` classes
    (semantic codes, then EOS). Out-of-range labels fault the GPU, so slow labels
    are remapped: semantic ID -> ID - semantic_begin_id, EOS -> codebook_size;
    other supervised text tokens (the newline after EOS) are ignored.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs, return_dict=True)
        core = actual_model(model)
        cfg = core.config
        row0 = labels[:, 0]
        semantic_mask = row0.ge(int(cfg.semantic_begin_id)) & row0.le(int(cfg.semantic_end_id))
        if not semantic_mask.any():
            raise ValueError("batch contains no supervised semantic frames")
        slow_labels = torch.full_like(row0, -100)
        slow_labels[semantic_mask] = row0[semantic_mask] - int(cfg.semantic_begin_id)
        slow_labels[row0.eq(int(cfg.eos_token_id))] = int(cfg.codebook_size)
        logits = outputs.logits.float()
        slow_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), slow_labels.reshape(-1), ignore_index=-100)

        slow_hidden = outputs.hidden_states[semantic_mask]
        if bool(getattr(cfg, "norm_fastlayer_input", False)):
            slow_hidden = core.norm(slow_hidden)
        codebooks = labels[:, 1:].permute(0, 2, 1)[semantic_mask]
        codebook_logits = fast_codebook_logits(core, slow_hidden, codebooks).float()
        fast_loss = F.cross_entropy(codebook_logits.reshape(-1, codebook_logits.shape[-1]), codebooks.reshape(-1))
        loss = self.slow_loss_weight * slow_loss + self.fast_loss_weight * fast_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: slow={float(slow_loss)}, fast={float(fast_loss)}")

        with torch.no_grad():
            valid = slow_labels.ne(-100)
            eos = slow_labels.eq(int(cfg.codebook_size))
            self._record_metrics({
                "slow_loss": slow_loss,
                "fast_loss": fast_loss,
                "slow_accuracy": logits[valid].argmax(-1).eq(slow_labels[valid]).float().mean(),
                "eos_accuracy": logits[eos].argmax(-1).eq(slow_labels[eos]).float().mean(),
                "fast_accuracy": codebook_logits.argmax(-1).eq(codebooks).float().mean(),
                "semantic_frames": loss.new_tensor(float(semantic_mask.sum())),
            })
        if return_outputs:
            return loss, {"logits": outputs.logits, "codebook_logits": codebook_logits}
        return loss


def main():
    parser = HfArgumentParser((Args, Audio8TTSTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level=logging.INFO)
    training_args.remove_unused_columns = False

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, dtype=dtype)
    model.config.use_gradient_checkpointing = bool(training_args.gradient_checkpointing)
    set_trainable_modules(model, freeze_slow=args.freeze_slow_ar, freeze_fast=args.freeze_fast_ar)

    train_rows, eval_rows, files = load_rows(Path(args.data_dir), args.eval_shard_mod)
    LOGGER.info("loaded %d train / %d eval clips from %d shard files", len(train_rows), len(eval_rows), files)
    eval_rows = eval_rows[: args.max_eval_samples]
    kwargs = dict(tokenizer=processor.tokenizer, config=model.config, args=args)
    train_dataset = ParquetSFTDataset(train_rows, train=True, **kwargs)
    eval_dataset = ParquetSFTDataset(eval_rows, train=False, **kwargs) if eval_rows else None

    trainer = CompactHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate_sft_examples, pad_token_id=int(model.config.pad_token_id)),
        slow_loss_weight=training_args.slow_loss_weight,
        fast_loss_weight=training_args.fast_loss_weight,
    )
    if eval_dataset is not None:
        LOGGER.info("baseline eval: %s", trainer.evaluate())
    trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(training_args.output_dir, training_args.resume_mode))
    trainer.save_state()
    final = actual_model(trainer.model)
    final.gradient_checkpointing_disable()
    final.config.use_gradient_checkpointing = False
    sanitize_config_for_save(final.config)
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    LOGGER.info("saved %s", training_args.output_dir)


if __name__ == "__main__":
    main()
