from __future__ import annotations

from io import BytesIO
import os

import pandas as pd


class S3ArtifactStore:
    def __init__(
        self,
        bucket_name: str | None = None,
        key_prefix: str | None = None,
        client=None,
    ) -> None:
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET", "")
        self.key_prefix = (key_prefix or os.getenv("S3_KEY_PREFIX", "")).strip("/")
        if client is not None:
            self.client = client
        else:
            import boto3

            self.client = boto3.client(
                "s3",
                region_name=os.getenv("AWS_REGION", "sa-east-1"),
            )

    def _build_key(self, key: str) -> str:
        normalized_key = key.strip("/")
        if not self.key_prefix:
            return normalized_key
        return f"{self.key_prefix}/{normalized_key}"

    def list_keys(self, prefix: str = "") -> list[str]:
        response = self.client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=self._build_key(prefix),
        )
        return [item["Key"] for item in response.get("Contents", [])]

    def read_bytes(self, key: str) -> bytes:
        response = self.client.get_object(
            Bucket=self.bucket_name,
            Key=self._build_key(key),
        )
        return response["Body"].read()

    def read_csv(self, key: str, **kwargs) -> pd.DataFrame:
        payload = self.read_bytes(key)
        return pd.read_csv(BytesIO(payload), **kwargs)
