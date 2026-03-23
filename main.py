from indexer import ParquetIndexer
from extractor import ParquetExtractor
from exporter import HFExporter
from dotenv import load_dotenv
import os
from pathlib import Path


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    DATASET    = 'HuggingFaceFW/fineweb-2'
    LANGUAGE   = 'abn_Latn'   
    GLOB       = f'data/{LANGUAGE}/train/*.parquet'
    TARGET_DUMP = 'CC-MAIN-2017-39'
    MAX_ROWS    = 50
    index_dir = Path("index")
    parquet_dir = Path("parquet")
    index_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    INDEX_PATH = str(index_dir / f'index_{LANGUAGE}.json')
    indexer = ParquetIndexer(
        source       = DATASET,
        column       = 'dump',
        index_path   = INDEX_PATH,
        language = LANGUAGE,
        hf_token=hf_token,
        glob_pattern = GLOB,
    )

    indexer.build(resume=True)
    output_file = f"{parquet_dir}/{TARGET_DUMP.replace('/', '_')}.parquet"
    extractor = ParquetExtractor(INDEX_PATH, hf_token=hf_token)
    extractor.extract(
        value       = TARGET_DUMP,
        output_path = output_file,
        max_rows    = MAX_ROWS,
        columns     = ['text', 'id', 'dump'],
        resume      = True,
    )
    exporter = HFExporter(hf_token=hf_token, username=username)
    exporter.export_file(
        local_path = output_file,
        repo_name  = f"fineweb2-{LANGUAGE}-{TARGET_DUMP.replace('/', '_')}",
        private    = False,
    )

    exporter.write_dataset_card(
        repo_name   = f"fineweb2-{LANGUAGE}-{TARGET_DUMP.replace('/', '_')}",
        language    = LANGUAGE,
        source      = DATASET,
        dump_values = [TARGET_DUMP],
    )
    
