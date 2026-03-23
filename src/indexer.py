
import json, time
from datetime import datetime
from pathlib import Path
import polars as pl
from huggingface_hub import HfFileSystem

class ParquetIndexer:
    def __init__(self, source, column, index_path, language, hf_token, glob_pattern="**/*.parquet"):
        self.source = source
        self.column = column
        self.index_path = Path(index_path)
        self.language = language
        self.glob_pattern = glob_pattern
        self._is_hf = not source.startswith("/") and not source.startswith(".")
        self.hf_token = hf_token
        if self._is_hf:
            self._hf_fs    = HfFileSystem(token=hf_token) if self._is_hf else None
        else:
            self._hf_fs = None

    def build(self, resume=True):
        index = self._load_existing() if resume and self.index_path.exists() else self._empty_index()
        all_shards = self._list_shards()
        done_shards = set(index["shards"].keys())
        pending = [s for s in all_shards if s not in done_shards]
        print(f"Total shards    : {len(all_shards)}")
        print(f"Already indexed : {len(done_shards)}")
        print(f"Remaining       : {len(pending)}")
        index["meta"]["total_shards"] = len(all_shards)
        for i, shard_path in enumerate(pending):
            print(f"shard_path{shard_path}")
            parts = shard_path.split("/")
            short = "/".join(parts[-3:]) if len(parts) > 3 else shard_path
            print(f"[{i+1}/{len(pending)}] {short} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                shard_path = f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/{self.language}/train/{short.split("/")[-1]}"
                entry = self._index_one_shard(shard_path)
                index["shards"][shard_path] = entry
                layout = "contiguous" if entry["contiguous"] else "interleaved"
                print(f"{entry['total_rows']:,} rows | {len(entry['dumps'])} dumps | {layout} | {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"FAILED — {e}")
                continue
            index["meta"]["completed_shards"] = len(index["shards"])
            index["meta"]["last_updated"] = datetime.utcnow().isoformat()
            self._save(index)
        all_cont = all(v["contiguous"] for v in index["shards"].values())
        index["meta"]["layout"] = "contiguous" if all_cont else "interleaved"
        self._save(index)
        print(f"Done. Index -> {self.index_path}")
        return index

    def summary(self):
        if not self.index_path.exists():
            print("No index found."); return
        index = self._load_existing()
        totals = {}
        for entry in index["shards"].values():
            for dump, ranges in entry["dumps"].items():
                totals[dump] = totals.get(dump, 0) + sum(r["end"]-r["start"] for r in ranges)
        print(f"Index summary")
        print(f"  source  : {index['meta']['source']}")
        print(f"  column  : {index['meta']['column']}")
        print(f"  shards  : {index['meta']['completed_shards']} / {index['meta']['total_shards']}")
        print(f"  layout  : {index['meta'].get('layout', 'unknown')}")
        print(f" {'dump value':<35} {'rows':>12}")
        print("  " + "-" * 49)
        for dump, count in sorted(totals.items(), key=lambda x: -x[1]):
            print(f"  {dump:<35} {count:>12,}")

    def _index_one_shard(self, shard_path):
        series = pl.scan_parquet(shard_path).select(self.column).collect()[self.column]
        return _build_ranges(series)

    def _list_shards(self):
        if self._is_hf:
            return self._hf_fs.glob(f"datasets/{self.source}/{self.glob_pattern}")
        base = Path(self.source)
        return sorted(str(p) for p in base.glob(self.glob_pattern))

    def _empty_index(self):
        return {"meta": {"source": self.source, "column": self.column,
            "built_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "total_shards": 0, "completed_shards": 0, "layout": "unknown"}, "shards": {}}

    def _load_existing(self):
        with open(self.index_path) as f: return json.load(f)

    def _save(self, index):
        tmp = self.index_path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f: json.dump(index, f, indent=2)
        tmp.replace(self.index_path)
        

def _build_ranges(series):
    values = series.to_list()
    total_rows = len(values)
    if total_rows == 0: return {"total_rows": 0, "contiguous": True, "dumps": {}}
    dump_ranges = {}
    current_val = values[0]
    current_start = 0
    for idx in range(1, total_rows):
        if values[idx] != current_val:
            dump_ranges.setdefault(current_val, []).append({"start": current_start, "end": idx})
            current_val = values[idx]
            current_start = idx
    dump_ranges.setdefault(current_val, []).append({"start": current_start, "end": total_rows})
    contiguous = all(len(r) == 1 for r in dump_ranges.values())
    return {"total_rows": total_rows, "contiguous": contiguous, "dumps": dump_ranges}
