from pathlib import Path
from huggingface_hub import HfApi, create_repo, RepoCard


class HFExporter:
    def __init__(self, hf_token: str, username: str):
        """
        Args:
            hf_token:  HuggingFace token (needs write access)
            username:  your HuggingFace username or org name
        """
        self.token    = hf_token
        self.username = username
        self.api      = HfApi(token=hf_token)

    # ------------------------------------------------------------------ #
    #  public API                                                          #
    # ------------------------------------------------------------------ #

    def export_file(
        self,
        local_path:     str,
        repo_name:      str,
        remote_path:    str | None = None,
        private:        bool = False,
        commit_message: str | None = None,
    ) -> str:
        """
        Upload a single parquet file to a HuggingFace dataset repo.

        Args:
            local_path:     path to the local .parquet file
            repo_name:      repo name (without username), e.g. "fineweb2-arb-CC2024"
            remote_path:    path inside the repo, e.g. "data/CC-MAIN-2024.parquet"
                            defaults to  data/{filename}
            private:        whether to make the repo private
            commit_message: custom commit message

        Returns:
            full repo URL
        """
        local_path  = Path(local_path)
        repo_id     = f"{self.username}/{repo_name}"
        remote_path = remote_path or f"data/{local_path.name}"

        self._ensure_repo(repo_id, private)

        print(f"Uploading {local_path.name} → {repo_id}/{remote_path} ...", end=" ", flush=True)
        self.api.upload_file(
            path_or_fileobj = str(local_path),
            path_in_repo    = remote_path,
            repo_id         = repo_id,
            repo_type       = "dataset",
            commit_message  = commit_message or f"Add {local_path.name}",
        )
        url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"done\n→ {url}/blob/main/{remote_path}")
        return url

    def export_directory(
        self,
        local_dir:      str,
        repo_name:      str,
        remote_dir:     str = "data",
        glob:           str = "*.parquet",
        private:        bool = False,
        resume:         bool = True,
        commit_message: str | None = None,
    ) -> str:
        """
        Upload all parquet files in a directory, one by one.
        Skips files already present in the repo when resume=True.

        Args:
            local_dir:   directory containing .parquet files
            repo_name:   repo name (without username)
            remote_dir:  folder inside the repo to upload into
            glob:        file pattern to match, default *.parquet
            private:     whether to make the repo private
            resume:      skip files already in the repo
            commit_message: applied to every upload

        Returns:
            full repo URL
        """
        local_dir = Path(local_dir)
        repo_id   = f"{self.username}/{repo_name}"
        files     = sorted(local_dir.glob(glob))

        if not files:
            print(f"No files matching '{glob}' found in {local_dir}")
            return ""

        self._ensure_repo(repo_id, private)

        existing = set()
        if resume:
            existing = set(self.api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
            if existing:
                print(f"Found {len(existing)} existing file(s) in repo — will skip them")

        print(f"\nUploading {len(files)} file(s) to {repo_id}/{remote_dir}/\n")

        uploaded = 0
        for i, local_file in enumerate(files):
            remote_path = f"{remote_dir}/{local_file.name}"

            if remote_path in existing:
                print(f"[{i+1}/{len(files)}] {local_file.name} — skipped (already uploaded)")
                continue

            print(f"[{i+1}/{len(files)}] {local_file.name} ({_size(local_file)}) ...", end=" ", flush=True)
            try:
                self.api.upload_file(
                    path_or_fileobj = str(local_file),
                    path_in_repo    = remote_path,
                    repo_id         = repo_id,
                    repo_type       = "dataset",
                    commit_message  = commit_message or f"Add {local_file.name}",
                )
                print("done")
                uploaded += 1
            except Exception as e:
                print(f"FAILED — {e}")
                continue

        url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"\n{uploaded} file(s) uploaded → {url}")
        return url

    def write_dataset_card(
        self,
        repo_name:   str,
        language:    str,
        source:      str = "HuggingFaceFW/fineweb-2",
        dump_values: list[str] | None = None,
        extra_notes: str = "",
    ):
        """
        Write a minimal README.md (dataset card) to the repo.
        Call this after uploading files so the repo has a description.

        Args:
            repo_name:   repo name (without username)
            language:    language code, e.g. "arb_Arab"
            source:      upstream dataset
            dump_values: list of dump values included, e.g. ["CC-MAIN-2024-10"]
            extra_notes: any extra text to append to the card
        """
        repo_id = f"{self.username}/{repo_name}"

        dumps_section = ""
        if dump_values:
            dumps_list    = "\n".join(f"- `{d}`" for d in sorted(dump_values))
            dumps_section = f"\n## Snapshots included\n\n{dumps_list}\n"

        content = f"""---
            language:
            - {language}
            license: odc-by
            source_datasets:
            - {source}
            tags:
            - fineweb2
            - web-crawl
            - text
            ---

            # {repo_name}

            Extracted from [{source}](https://huggingface.co/datasets/{source}) using [snapsift](https://github.com/{self.username}/snapsift).

            **Language:** `{language}`  
            **Source:** `{source}`
            {dumps_section}
            ## Usage

            ```python
            import polars as pl
            df = pl.read_parquet("hf://datasets/{repo_id}/data/*.parquet")
            ```
            {extra_notes}
            """
        self.api.upload_file(
            path_or_fileobj = content.encode(),
            path_in_repo    = "README.md",
            repo_id         = repo_id,
            repo_type       = "dataset",
            commit_message  = "Add dataset card",
        )
        print(f"Dataset card written → https://huggingface.co/datasets/{repo_id}")

    def list_repo_files(self, repo_name: str) -> list[str]:
        """List all files currently in the repo."""
        repo_id = f"{self.username}/{repo_name}"
        files   = list(self.api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        print(f"\nFiles in {repo_id}:")
        for f in files:
            print(f"  {f}")
        return files

    # ------------------------------------------------------------------ #
    #  internals                                                           #
    # ------------------------------------------------------------------ #

    def _ensure_repo(self, repo_id: str, private: bool):
        """Create the repo if it doesn't exist yet."""
        create_repo(
            repo_id   = repo_id,
            repo_type = "dataset",
            token     = self.token,
            private   = private,
            exist_ok  = True,
        )
        print(f"Repo: https://huggingface.co/datasets/{repo_id}")


# ------------------------------------------------------------------ #
#  helpers                                                            #
# ------------------------------------------------------------------ #

def _size(path: Path) -> str:
    """Human-readable file size."""
    b = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"