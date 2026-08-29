import os
from pathlib import Path
from uuid import uuid4

import boto3
import httpx
import pytest
from botocore.config import Config

from ai_judo_coach.storage import (
    check_input_video_exists,
    create_presigned_generated_clip_download_url,
    create_presigned_upload_post,
    download_input_video,
    upload_generated_clip,
)


pytestmark = pytest.mark.integration

_RUN_REAL_S3_INTEGRATION_TESTS = (
    os.getenv(
        "AI_JUDO_COACH_RUN_S3_INTEGRATION_TESTS"
    )
    == "1"
)


@pytest.mark.skipif(
    not _RUN_REAL_S3_INTEGRATION_TESTS,
    reason="Real S3 integration tests were not enabled",
)
def test_s3_video_storage_round_trip(
    tmp_path: Path,
) -> None:
    """
    Exercise the complete storage flow against the development bucket.

    This test is deliberately opt-in because it requires real AWS
    credentials and creates temporary objects in S3.
    """

    bucket_name = os.getenv(
        "AI_JUDO_COACH_S3_BUCKET"
    )

    if not bucket_name:
        pytest.fail(
            "AI_JUDO_COACH_S3_BUCKET must be set when "
            "real S3 integration tests are enabled"
        )

    profile_name = os.getenv(
        "AWS_PROFILE",
        "ai-judo-coach-dev",
    )

    region_name = os.getenv(
        "AWS_DEFAULT_REGION",
        os.getenv(
            "AWS_REGION",
            "eu-west-2",
        ),
    )

    session = boto3.Session(
        profile_name=profile_name,
        region_name=region_name,
    )

    s3_client = session.client(
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

    job_id = f"storage-smoke-{uuid4()}"

    input_object_key = (
        f"jobs/{job_id}/input/source.mp4"
    )
    generated_clip_object_key = (
        f"jobs/{job_id}/generated-clips/"
        "attempt_000.mp4"
    )

    input_video_content = (
        b"AI Judo Coach S3 input storage smoke test"
    )
    generated_clip_content = (
        b"AI Judo Coach S3 generated clip smoke test"
    )

    try:
        assert not check_input_video_exists(
            s3_client=s3_client,
            bucket_name=bucket_name,
            job_id=job_id,
        )

        presigned_upload = (
            create_presigned_upload_post(
                s3_client=s3_client,
                bucket_name=bucket_name,
                job_id=job_id,
            )
        )

        assert (
            presigned_upload["object_key"]
            == input_object_key
        )

        upload_response = httpx.post(
            presigned_upload["url"],
            data=presigned_upload["fields"],
            files={
                "file": (
                    "source.mp4",
                    input_video_content,
                    "video/mp4",
                ),
            },
            timeout=60.0,
        )

        assert upload_response.status_code in {
            200,
            201,
            204,
        }, upload_response.text

        assert check_input_video_exists(
            s3_client=s3_client,
            bucket_name=bucket_name,
            job_id=job_id,
        )

        downloaded_input_path = (
            tmp_path
            / "downloaded"
            / "input"
            / "source.mp4"
        )

        returned_download_path = (
            download_input_video(
                s3_client=s3_client,
                bucket_name=bucket_name,
                job_id=job_id,
                local_destination_path=str(
                    downloaded_input_path
                ),
            )
        )

        assert returned_download_path == str(
            downloaded_input_path
        )
        assert downloaded_input_path.is_file()
        assert (
            downloaded_input_path.read_bytes()
            == input_video_content
        )

        local_generated_clip_path = (
            tmp_path
            / "generated_clips"
            / "attempt_000.mp4"
        )

        local_generated_clip_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        local_generated_clip_path.write_bytes(
            generated_clip_content
        )

        returned_generated_clip_key = (
            upload_generated_clip(
                s3_client=s3_client,
                bucket_name=bucket_name,
                job_id=job_id,
                clip_id="0",
                local_clip_path=str(
                    local_generated_clip_path
                ),
            )
        )

        assert (
            returned_generated_clip_key
            == generated_clip_object_key
        )

        generated_clip_metadata = (
            s3_client.head_object(
                Bucket=bucket_name,
                Key=returned_generated_clip_key,
            )
        )

        assert (
            generated_clip_metadata["ContentType"]
            == "video/mp4"
        )

        presigned_download_url = (
            create_presigned_generated_clip_download_url(
                s3_client=s3_client,
                bucket_name=bucket_name,
                generated_clip_object_key=(
                    returned_generated_clip_key
                ),
            )
        )

        download_response = httpx.get(
            presigned_download_url,
            timeout=60.0,
        )

        assert (
            download_response.status_code == 200
        ), download_response.text

        assert (
            download_response.content
            == generated_clip_content
        )
    finally:
        s3_client.delete_objects(
            Bucket=bucket_name,
            Delete={
                "Objects": [
                    {
                        "Key": input_object_key,
                    },
                    {
                        "Key": (
                            generated_clip_object_key
                        ),
                    },
                ],
                "Quiet": True,
            },
        )
