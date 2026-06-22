from io import BytesIO

import pandas as pd

from src.storage.s3 import S3ArtifactStore


class FakeS3Client:
    def __init__(self):
        self.requested = []

    def list_objects_v2(self, Bucket, Prefix):
        self.requested.append(("list", Bucket, Prefix))
        return {
            "Contents": [
                {"Key": "analytics/outputs/model_comparison_summary.csv"},
                {"Key": "analytics/outputs/figures/sentiment.png"},
            ]
        }

    def get_object(self, Bucket, Key):
        self.requested.append(("get", Bucket, Key))
        body = BytesIO(b"modelo,accuracy\nBERTimbau,0.85\n")
        return {"Body": body}


def test_s3_artifact_store_lists_keys_with_prefix():
    client = FakeS3Client()
    store = S3ArtifactStore(
        bucket_name="nubank-cloud-bucket",
        key_prefix="analytics",
        client=client,
    )

    keys = store.list_keys("outputs")

    assert keys == [
        "analytics/outputs/model_comparison_summary.csv",
        "analytics/outputs/figures/sentiment.png",
    ]
    assert client.requested[0] == ("list", "nubank-cloud-bucket", "analytics/outputs")


def test_s3_artifact_store_reads_csv():
    client = FakeS3Client()
    store = S3ArtifactStore(
        bucket_name="nubank-cloud-bucket",
        key_prefix="analytics",
        client=client,
    )

    dataframe = store.read_csv("outputs/model_comparison_summary.csv")

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.to_dict(orient="records") == [{"modelo": "BERTimbau", "accuracy": 0.85}]
    assert client.requested[-1] == (
        "get",
        "nubank-cloud-bucket",
        "analytics/outputs/model_comparison_summary.csv",
    )
