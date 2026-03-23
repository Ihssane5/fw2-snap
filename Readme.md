# snapsift

Extract snapshot-specific data from FineWeb-2 and push it to the HuggingFace Hub.

![snapsift overview](image.png)

---

## The problem

FineWeb organizes data **per snapshot** — each Common Crawl crawl lives in its own folder, making chronological access trivial. FineWeb-2 changed this: it deduplicates globally per language, so rows from all 96 snapshots (2013–2024) are mixed together inside every shard file. There is no folder to point at for "only 2024 data".

This makes FineWeb-2 harder to work with for researchers who need a specific snapshot — especially for the 1,000+ languages it covers, many of them low-resource, where understanding *when* data was collected matters as much as *how much* exists.

The only option without a tool like this is to stream the entire language subset row by row, filtering as you go — downloading hundreds of GB to keep a small fraction, with no way to resume if your session dies.

---

## Methodology

Each FineWeb-2 shard is a parquet file that mixes rows from multiple snapshots:

```
shard_00.parquet
  row 0    →  CC-MAIN-2024-10
  row 1    →  CC-MAIN-2024-10
  row 2    →  CC-MAIN-2019-35
  row 3    →  CC-MAIN-2013-20
  row 4    →  CC-MAIN-2024-10
  ...
```

**Phase 1 — indexing.** fw2-snap reads only the `dump` column from each shard using Parquet column projection — the `text` column (95%+ of file size) is never downloaded. It makes a single pass over the dump values, detects contiguous runs of the same snapshot, and records each run as a `start` / `end` row range:

```json
"shard_00.parquet": {
  "total_rows": 114688,
  "dumps": {
    "CC-MAIN-2024-10": [{"start": 0,  "end": 38400}],
    "CC-MAIN-2019-35": [{"start": 38400, "end": 76800}],
    "CC-MAIN-2013-20": [{"start": 76800, "end": 114688}]
  }
}
```

A snapshot that appears in one contiguous block produces a single range. One that is scattered across the shard produces a few ranges. Either way the index stays compact — two integers per run instead of one integer per row. The index is written to disk after every shard so a crashed session resumes exactly where it stopped.

**Phase 2 — extraction.** Given the index, fw2-snap skips every shard that doesn't contain the target snapshot. For shards that do, it adds a temporary row-number column and filters to only the stored ranges — no full scan, no row-by-row comparison.

**Phase 3 — export.** The extracted parquet is pushed directly to a HuggingFace dataset repo with a dataset card generated automatically.

---

## Install

```bash
git clone https://github.com/your-username/fw2-snap

pip install -r requirements.txt
```

## Usage

Before running, add these variables to your `.env` file:

```env
HF_TOKEN=hf_xxx
HF_USERNAME=your-hf-username
```

`HF_TOKEN` must be a Hugging Face **write** token so upload/export can create and push dataset files.

### Script mode

Use [main.py](main.py) when you want a fixed configuration inside the script (easy for quick edits and reruns).

```bash
python main.py
```

### CLI mode

Use [main_arg.py](main_arg.py) when you want to pass the snapshot/language from the command line instead.

Constructor-style minimal example:

```bash
python main_arg.py --language ary_Arab --target-dump CC-MAIN-2023-40 --max-rows 50000
```

