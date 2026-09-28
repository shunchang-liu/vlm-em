"""Maintainer tool: upload the staged repositories to the Hugging Face Hub.

    hf auth login
    python release/upload.py /path/to/staging [--user shunchang-liu]

The dataset repository is public. The model repository is gated with manual approval, so
every download request is reviewed by the maintainer.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("staging", type=Path)
    ap.add_argument("--user", default="shunchang-liu")
    a = ap.parse_args()
    api = HfApi()
    data_repo, model_repo = f"{a.user}/multimodal-em-data", f"{a.user}/multimodal-em-models"

    api.create_repo(data_repo, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(repo_id=data_repo, repo_type="dataset", folder_path=a.staging / "data")

    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    api.update_repo_settings(model_repo, gated="manual")
    api.upload_large_folder(repo_id=model_repo, repo_type="model", folder_path=a.staging / "models")
    print(f"https://huggingface.co/datasets/{data_repo}\nhttps://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
