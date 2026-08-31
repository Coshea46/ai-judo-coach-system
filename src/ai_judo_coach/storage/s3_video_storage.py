"""S3 storage operations for input videos and generated attempt clips."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Final,
    TypedDict,
)

from botocore.exceptions import ClientError


if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


# config variables for s3
# shouldn't go in config.py because they are solely for s3
_JOBS_PREFIX: Final[str] = "jobs"
_INPUT_PREFIX: Final[str] = "input"
_INPUT_VIDEO_FILENAME: Final[str] = "source.mp4"
_GENERATED_CLIPS_PREFIX: Final[str] = "generated-clips"
_GENERATED_CLIP_FILENAME_PATTERN: Final[str] = (
    "attempt_{clip_id:03d}.mp4"
)

_VIDEO_CONTENT_TYPE: Final[str] = "video/mp4"
_DEFAULT_PRESIGNED_EXPIRY_SECONDS: Final[int] = 3600
_MAX_INPUT_VIDEO_SIZE_BYTES: Final[int] = 2 * 1024**3
_MIN_INPUT_VIDEO_SIZE_BYTES: Final[int] = 1

_EXAMPLES_PREFIX: Final[str] = "examples"
_EXAMPLE_VIDEO_FILENAMES: Final[
    dict[str, str]
] = {
    "full": "example_judo_match.mp4",
    "short": "shorter_example_match.mp4",
}



class PresignedBrowserUpload(TypedDict):
    """Information required to upload a video directly to S3."""

    object_key: str
    url: str
    fields: dict[str, str]


def create_presigned_upload_post(
    s3_client: S3Client,
    bucket_name: str,
    job_id: str,
) -> PresignedBrowserUpload:
    """
    Generate the presigned POST information required for the frontend
    to upload an input video directly to S3.
    """

    object_key = _construct_input_video_s3_path(
        job_id=job_id,
    )

    response = s3_client.generate_presigned_post(
        Bucket=bucket_name,
        Key=object_key,
        Fields={
            "Content-Type": _VIDEO_CONTENT_TYPE,
        },
        Conditions=[
            [
                "content-length-range",
                _MIN_INPUT_VIDEO_SIZE_BYTES,
                _MAX_INPUT_VIDEO_SIZE_BYTES,
            ],
            {
                "Content-Type": _VIDEO_CONTENT_TYPE,
            },
        ],
        ExpiresIn=_DEFAULT_PRESIGNED_EXPIRY_SECONDS,
    )

    return PresignedBrowserUpload(
        object_key=object_key,
        url=response["url"],
        fields=response["fields"],
    )


def copy_example_video_to_job_input(
    s3_client: S3Client,
    bucket_name: str,
    job_id: str,
    example: str = "full",
) -> str:
    """
    Copy the example video to a new job's input object key.

    The copy occurs entirely within S3 and does not expose the private
    example object to the browser.
    """

    try:
        example_filename = (
            _EXAMPLE_VIDEO_FILENAMES[example]
        )
    except KeyError as error:
        raise ValueError(
            f"Unknown example video: {example!r}"
        ) from error

    source_object_key = str(
        PurePosixPath(
            _EXAMPLES_PREFIX,
            example_filename,
        )
    )

    destination_object_key = (
        _construct_input_video_s3_path(
            job_id=job_id,
        )
    )

    s3_client.copy_object(
        CopySource={
            "Bucket": bucket_name,
            "Key": source_object_key,
        },
        Bucket=bucket_name,
        Key=destination_object_key,
        ContentType=_VIDEO_CONTENT_TYPE,
        MetadataDirective="REPLACE",
    )

    return destination_object_key



def create_presigned_example_video_preview_url(
    s3_client: S3Client,
    bucket_name: str,
    example: str,
) -> str:
    """
    Create a temporary browser URL for an allowlisted example video.

    Callers select a public example identifier rather than supplying
    an S3 filename or object key.
    """

    try:
        example_filename = (
            _EXAMPLE_VIDEO_FILENAMES[example]
        )
    except KeyError as error:
        raise ValueError(
            f"Unknown example video: {example!r}"
        ) from error

    example_object_key = str(
        PurePosixPath(
            _EXAMPLES_PREFIX,
            example_filename,
        )
    )

    return s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": bucket_name,
            "Key": example_object_key,
        },
        ExpiresIn=(
            _DEFAULT_PRESIGNED_EXPIRY_SECONDS
        ),
    )




def check_input_video_exists(
    s3_client: S3Client,
    bucket_name: str,
    job_id: str,
) -> bool:
    """
    Return whether the input video for a given job exists in S3.

    Only definite missing-object responses result in False. Other AWS
    errors, including permission and credential errors, are propagated.
    """

    object_key = _construct_input_video_s3_path(
        job_id=job_id,
    )

    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key=object_key,
        )
    except ClientError as error:
        error_code = str(
            error.response.get(
                "Error",
                {},
            ).get(
                "Code",
                "",
            )
        )

        if error_code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False

        raise

    return True


def download_input_video(
    s3_client: S3Client,
    bucket_name: str,
    job_id: str,
    local_destination_path: str,
) -> str:
    """
    Download a job's input video to a caller-supplied local path.

    The caller owns the temporary workspace and its cleanup.
    """

    object_key = _construct_input_video_s3_path(
        job_id=job_id,
    )

    destination_path = Path(
        local_destination_path
    )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    s3_client.download_file(
        Bucket=bucket_name,
        Key=object_key,
        Filename=str(destination_path),
    )

    return str(destination_path)


def upload_generated_clip(
    s3_client: S3Client,
    bucket_name: str,
    job_id: str,
    clip_id: str,
    local_clip_path: str,
) -> str:
    """
    Upload one generated clip and return its stable S3 object key.
    """

    source_path = Path(
        local_clip_path
    )

    if not source_path.is_file():
        raise FileNotFoundError(
            "Generated clip does not exist: "
            f"{source_path}"
        )

    object_key = _construct_generated_clip_s3_path(
        job_id=job_id,
        generated_clip_id=clip_id,
    )

    s3_client.upload_file(
        Filename=str(source_path),
        Bucket=bucket_name,
        Key=object_key,
        ExtraArgs={
            "ContentType": _VIDEO_CONTENT_TYPE,
        },
    )

    return object_key


def create_presigned_generated_clip_download_url(
    s3_client: S3Client,
    bucket_name: str,
    generated_clip_object_key: str,
) -> str:
    """
    Create a temporary download URL for a generated clip in S3.

    The supplied object key must come from a trusted backend result,
    such as the value returned by upload_generated_clip().
    """

    if not generated_clip_object_key:
        raise ValueError(
            "generated_clip_object_key must not be empty"
        )

    return s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": bucket_name,
            "Key": generated_clip_object_key,
        },
        ExpiresIn=_DEFAULT_PRESIGNED_EXPIRY_SECONDS,
    )


def _construct_input_video_s3_path(
    job_id: str,
) -> str:
    """
    Construct the S3 object key for a job's input video.
    """

    _validate_job_id(
        job_id=job_id,
    )

    generated_path = PurePosixPath(
        _JOBS_PREFIX,
        job_id,
        _INPUT_PREFIX,
        _INPUT_VIDEO_FILENAME,
    )

    return str(generated_path)


def _construct_generated_clip_s3_path(
    job_id: str,
    generated_clip_id: str,
) -> str:
    """
    Construct the S3 object key for one generated attempt clip.
    """

    _validate_job_id(
        job_id=job_id,
    )

    numeric_clip_id = _validate_clip_id(
        clip_id=generated_clip_id,
    )

    generated_clip_filename = (
        _GENERATED_CLIP_FILENAME_PATTERN.format(
            clip_id=numeric_clip_id,
        )
    )

    generated_path = PurePosixPath(
        _JOBS_PREFIX,
        job_id,
        _GENERATED_CLIPS_PREFIX,
        generated_clip_filename,
    )

    return str(generated_path)


def _validate_job_id(
    job_id: str,
) -> None:
    """Validate that a job ID is safe to use as one S3 key component."""

    if (
        not job_id
        or not job_id.strip()
        or job_id in {".", ".."}
        or "/" in job_id
        or "\\" in job_id
    ):
        raise ValueError(
            "job_id must be a non-empty S3 key component"
        )


def _validate_clip_id(
    clip_id: str,
) -> int:
    """Validate and return a non-negative numeric clip ID."""

    if not clip_id or not clip_id.isdigit():
        raise ValueError(
            "clip_id must contain a non-negative numeric value"
        )

    return int(clip_id)
