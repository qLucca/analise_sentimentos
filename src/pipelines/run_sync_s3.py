from __future__ import annotations

import os
from pathlib import Path

import boto3

from src.utils.logger import setup_logging
from src.utils.paths import BRONZE_DIR, OUTPUTS_DIR, OUTPUT_DIR, PROCESSED_DIR, RAW_DIR


SYNC_DIRECTORIES = {
    RAW_DIR: "raw",
    BRONZE_DIR: "bronze",
    PROCESSED_DIR: "processed",
    OUTPUT_DIR: "data-output",
    OUTPUTS_DIR: "outputs",
}


def _iter_files(directory: Path):
    for path in directory.rglob("*"):
        if path.is_file():
            yield path


def run() -> None:
    setup_logging()

    bucket_name = os.getenv("S3_BUCKET")
    if not bucket_name:
        raise ValueError("Defina S3_BUCKET antes de executar o sync para o S3.")

    prefix = os.getenv("S3_KEY_PREFIX", "nubank-sentiment").strip("/")
    client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "sa-east-1"))

    for local_dir, remote_dir in SYNC_DIRECTORIES.items():
        for file_path in _iter_files(local_dir):
            relative_path = file_path.relative_to(local_dir).as_posix()
            key = f"{prefix}/{remote_dir}/{relative_path}"
            client.upload_file(str(file_path), bucket_name, key)


if __name__ == "__main__":
    run()
