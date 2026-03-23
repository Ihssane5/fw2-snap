import json, time
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


class ParquetExtractor:
    def __init__(self, index_path, hf_token=None):
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        with open(self.index_path) as f:
            self.index = json.load(f)
        self._is_hf    = not self.index["meta"]["source"].startswith("/")
        self._hf_token = hf_token
        self._hf_fs    = HfFileSystem(token=hf_token) if self._is_hf else None

    def extract(self, value, output_path, max_rows=None, columns=None, resume=True):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = output_path.with_suffix(".ckpt.json")

        relevant = [
            (s, e["dumps"][value])
            for s, e in self.index["shards"].items()
            if value in e["dumps"]
        ]
        if not relevant:
            print(f"No shards for '{value}'. Run .summary() for available values.")
            return

        done_shards = (
            set(json.load(open(checkpoint)))
            if resume and checkpoint.exists()
            else set()
        )
        total_avail = sum(
            sum(r["end"] - r["start"] for r in ranges) for _, ranges in relevant
        )
        target = min(max_rows, total_avail) if max_rows else total_avail

        print(f"Value           : '{value}'")
        print(f"Relevant shards : {len(relevant)}")
        print(f"Rows available  : {total_avail:,}")
        print(f"Rows to extract : {target:,}")
        print(f"Output          : {output_path}\n")

        writer = None
        rows_written = 0

        for i, (shard_path, ranges) in enumerate(relevant):
            if shard_path in done_shards:
                rows_written += sum(r["end"] - r["start"] for r in ranges)
                print(f"[{i+1}/{len(relevant)}] skipped (checkpoint)")
                continue

            if max_rows and rows_written >= max_rows:
                break

            ranges_to_fetch = ranges
            if max_rows:
                remaining = max_rows - rows_written
                out = []
                total = 0
                for r in ranges:
                    size = r["end"] - r["start"]
                    if total + size <= remaining:
                        out.append(r)
                        total += size
                    else:
                        leftover = remaining - total
                        if leftover > 0:
                            out.append({"start": r["start"], "end": r["start"] + leftover})
                        break
                ranges_to_fetch = out

            if not ranges_to_fetch:
                continue

            row_count = sum(r["end"] - r["start"] for r in ranges_to_fetch)
            parts = shard_path.split("/")
            short = "/".join(parts[-3:]) if len(parts) > 3 else shard_path
            print(f"[{i+1}/{len(relevant)}] {short} — {row_count:,} rows ...", end=" ", flush=True)
            t0 = time.time()

            try:
                uri = shard_path
                conditions = [
                    (pl.col("_row_idx") >= r["start"]) & (pl.col("_row_idx") < r["end"])
                    for r in ranges_to_fetch
                ]
                mask = conditions[0]
                for cond in conditions[1:]:
                    mask = mask | cond

                # ── FIX 3: pass token via storage_options ─────────────────
                storage_opts = {"token": self._hf_token} if self._is_hf else {}

                scan = (
                    pl.scan_parquet(uri, storage_options=storage_opts)
                    .with_row_index("_row_idx")
                )
                if columns:
                    scan = scan.select(
                        ["_row_idx"] + [c for c in columns if c != "_row_idx"]
                    )

                batch = scan.filter(mask).drop("_row_idx").collect()

                if writer is None:
                    writer = pq.ParquetWriter(
                        str(output_path), batch.to_arrow().schema, compression="zstd"
                    )
                writer.write_table(batch.to_arrow())
                rows_written += len(batch)
                done_shards.add(shard_path)

                # atomic checkpoint write
                tmp = checkpoint.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(list(done_shards), f)
                tmp.replace(checkpoint)

                print(f"done ({len(batch):,} rows, {time.time()-t0:.1f}s) — total: {rows_written:,}")

            except Exception as e:
                print(f"FAILED  {e}")
                continue

        if writer:
            writer.close()
        if checkpoint.exists():
            checkpoint.unlink()

        print(f"\nExtraction complete — {rows_written:,} rows -> {output_path}")