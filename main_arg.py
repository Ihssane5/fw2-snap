from src.indexer import ParquetIndexer
from src.extractor import ParquetExtractor
from src.exporter import HFExporter
from dotenv import load_dotenv
import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FineWeb2 indexing, extraction, and export from CLI arguments.")
    parser.add_argument("--language", default="ary_Arab", help="Language subset to process.")
    parser.add_argument("--target-dump", default="CC-MAIN-2023-40", help="Dump value to extract.")
    parser.add_argument("--max-rows", type=int, default=50_000, help="Maximum rows to extract.")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    username = os.getenv("HF_USERNAME")
    hf_token = os.getenv("HF_TOKEN")

    args = parse_args()
    dataset = "HuggingFaceFW/fineweb-2"
    language = args.language
    glob_pattern = f"data/{language}/train/*.parquet"
    target_dump = args.target_dump
    max_rows = args.max_rows
    repo_name = f"fineweb2-{language}-{target_dump.replace('/', '_')}"

    index_dir = Path("index")
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(index_dir / f"index_{language}.json")
    output_dir = Path("parquet")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hf_token:
        raise ValueError("HF_TOKEN is missing. Add it to your .env file.")

    indexer = ParquetIndexer(
        source=dataset,
        column="dump",
        index_path=index_path,
        language=language,
        hf_token=hf_token,
        glob_pattern=glob_pattern,
    )

    indexer.build(resume=True)

    output_file = f"{output_dir}/{target_dump.replace('/', '_')}.parquet"
    extractor = ParquetExtractor(index_path, hf_token=hf_token)
    extractor.extract(
        value=target_dump,
        output_path=output_file,
        max_rows=max_rows,
        columns=["text", "id", "dump"],
        resume=True,
    )

    exporter = HFExporter(hf_token=hf_token, username=username)
    exporter.export_file(
        local_path=output_file,
        repo_name=repo_name,
        private=False,
    )
    exporter.write_dataset_card(
        repo_name=repo_name,
        language=language,
        source=dataset,
        dump_values=[target_dump],
    )

    """
    python main_arg.py --language ary_Arab --target-dump CC-MAIN-2023-40 --max-rows 50000
    """
