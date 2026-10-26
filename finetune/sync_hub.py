"""Upload finished encoded shards from prep_stream.py to a private Hub dataset.

Batches new files into one commit per run to stay under Hub commit limits and
keeps a README with the schema and aggregate filtering stats.

    source ~/TTS/.venv/bin/activate
    python sync_hub.py --loop 900     # every 15 minutes
    python sync_hub.py                # once
"""

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "MeghanaKap/IndianEnglish-Audio8-0.1b-codes"
ROOT = Path(__file__).resolve().parent / "data"
MAX_FILES_PER_COMMIT = 100

README = """---
license: other
task_categories: [text-to-speech]
language: [en]
---
# Indian English → Audio8 0.1B codec encodings

Filtered and encoded from `MeghanaKap/IndianEnglishCleanYoutubeData` with
`finetune/prep_stream.py`.

* Kept only clips whose dataset transcript matches a Whisper large-v3-turbo
  transcript (WER ≤ 10% after lower-casing, removing punctuation and fillers),
  4–25 s long, 1.0–5.5 words/s, exact duplicates removed.
* Each clip is split at a sentence boundary with a ≥0.25 s pause (25–75% into the clip)
  into a reference part (≥ 3 s) and a target part (≥ 2 s); clips whose boundary
  words are clipped by the cut (checked by re-transcribing both parts) are dropped. Each part is encoded
  separately, so every example matches voice-cloning use (same-speaker reference
  + target); the source has no speaker labels.
* Codes: Audio8 codec indices from the fp32 encoder, 16 kHz audio resampled to
  44.1 kHz, flattened row-major `[10, n_frames]` (int16, 0–4095, ~21.5 frames/s).
  Texts of the two parts come from Whisper.

| column | type |
|---|---|
| id, shard, audio_id, source_text, asr_text, ref_text, text, audio_md5 | string |
| wer, duration, split_sec | float32 |
| ref_n_frames, n_frames | int32 |
| ref_codes, codes | list<int16> |

## Progress
{stats}
"""


def stats_table() -> str:
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "done").glob("*.json"))]
    if not rows:
        return "No shards processed yet."
    total = lambda key: sum(r[key] for r in rows)  # noqa: E731
    return (
        f"* source shards processed: {len(rows)} / 1381\n"
        f"* clips kept: {total('kept'):,} / {total('rows'):,}\n"
        f"* hours kept: {total('hours_kept'):.1f} / {total('hours_in'):.1f}\n"
        f"* rejected — transcript mismatch: {total('bad_wer'):,}, duplicates: {total('duplicates'):,}, "
        f"duration: {total('bad_duration'):,}, speaking rate: {total('bad_rate'):,}, "
        f"no split point: {sum(r.get('no_split', 0) for r in rows):,}, "
        f"clipped boundary word: {sum(r.get('bad_cut', 0) for r in rows):,}\n"
        f"* updated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n"
    )


def sync(api: HfApi) -> int:
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    remote = set(api.list_repo_files(REPO, repo_type="dataset"))
    local = sorted((ROOT / "encoded").glob("*.parquet"))
    new = [p for p in local if f"data/{p.name}" not in remote][:MAX_FILES_PER_COMMIT]
    if not new:
        return 0
    operations = [CommitOperationAdd(f"data/{p.name}", str(p)) for p in new]
    operations.append(CommitOperationAdd("README.md", README.replace("{stats}", stats_table()).encode()))
    api.create_commit(REPO, operations, repo_type="dataset", commit_message=f"Add {len(new)} encoded shards")
    return len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between syncs; 0 = run once")
    args = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    while True:
        try:
            count = sync(api)
            print(f"{time.strftime('%H:%M:%S')} uploaded {count} shard files to {REPO}", flush=True)
        except Exception as exc:  # keep looping through transient Hub errors
            print(f"{time.strftime('%H:%M:%S')} sync failed: {exc}", flush=True)
            count = 0
        if not args.loop:
            break
        # Drain a backlog quickly, otherwise wait.
        time.sleep(30 if count >= MAX_FILES_PER_COMMIT else args.loop)


if __name__ == "__main__":
    main()
