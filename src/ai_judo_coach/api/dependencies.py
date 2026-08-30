"""Dependencies used by the FastAPI control plane."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

import boto3
import modal
from botocore.config import Config


if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


_BUCKET_NAME_ENVIRONMENT_VARIABLE: Final[str] = (
    "AI_JUDO_COACH_S3_BUCKET"
)
_AWS_REGION_ENVIRONMENT_VARIABLE: Final[str] = (
    "AWS_DEFAULT_REGION"
)
_JOB_STORE_NAME: Final[str] = (
    "ai-judo-coach-jobs-dev"
)


def get_s3_client() -> S3Client:
    """
    Return an S3 client configured for the application's AWS region.

    Boto3 obtains credentials from the local AWS profile during local
    development and from environment variables injected by Modal when
    deployed.
    """

    region_name = os.environ[
        _AWS_REGION_ENVIRONMENT_VARIABLE
    ]

    return boto3.client(
        "s3",
        region_name=region_name,
        endpoint_url=(
            f"https://s3.{region_name}.amazonaws.com"
        ),
        config=Config(
            signature_version="s3v4",
            s3={
                "addressing_style": "virtual",
            },
        ),
    )


def get_bucket_name() -> str:
    """Return the configured private S3 bucket name."""

    return os.environ[
        _BUCKET_NAME_ENVIRONMENT_VARIABLE
    ]


def get_job_store() -> modal.Dict:
    """Return the persistent Modal dictionary containing job records."""

    return modal.Dict.from_name(
        _JOB_STORE_NAME,
        create_if_missing=True,
    )


def get_pipeline_worker() -> modal.Function:
    """Return the deployed Modal pipeline worker."""

    return modal.Function.from_name(
        "ai-judo-coach",
        "process_video_job",
    )


    