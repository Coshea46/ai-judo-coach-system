from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from ai_judo_coach.storage import (
    check_input_video_exists,
    create_presigned_generated_clip_download_url,
    create_presigned_upload_post,
    download_input_video,
    upload_generated_clip,
)


def _create_client_error(
    error_code: str,
    operation_name: str,
) -> ClientError:
    """Create a Botocore client error for an S3 operation."""

    return ClientError(
        error_response={
            "Error": {
                "Code": error_code,
                "Message": "Test S3 error",
            },
        },
        operation_name=operation_name,
    )


def test_create_presigned_upload_post_returns_upload_information(
    mocker,
) -> None:
    s3_client = mocker.Mock()

    generated_fields = {
        "Content-Type": "video/mp4",
        "key": "jobs/job-123/input/source.mp4",
        "policy": "signed-policy",
        "x-amz-algorithm": "AWS4-HMAC-SHA256",
        "x-amz-credential": "test-credential",
        "x-amz-date": "test-date",
        "x-amz-signature": "test-signature",
    }

    s3_client.generate_presigned_post.return_value = {
        "url": "https://test-bucket.s3.amazonaws.com",
        "fields": generated_fields,
    }

    result = create_presigned_upload_post(
        s3_client=s3_client,
        bucket_name="test-bucket",
        job_id="job-123",
    )

    assert result == {
        "object_key": "jobs/job-123/input/source.mp4",
        "url": "https://test-bucket.s3.amazonaws.com",
        "fields": generated_fields,
    }

    s3_client.generate_presigned_post.assert_called_once_with(
        Bucket="test-bucket",
        Key="jobs/job-123/input/source.mp4",
        Fields={
            "Content-Type": "video/mp4",
        },
        Conditions=[
            [
                "content-length-range",
                1,
                2 * 1024**3,
            ],
            {
                "Content-Type": "video/mp4",
            },
        ],
        ExpiresIn=3600,
    )


@pytest.mark.parametrize(
    "invalid_job_id",
    [
        "",
        " ",
        ".",
        "..",
        "/",
        "\\",
        "job/123",
        "job\\123",
    ],
)
def test_create_presigned_upload_post_rejects_invalid_job_id(
    mocker,
    invalid_job_id: str,
) -> None:
    s3_client = mocker.Mock()

    with pytest.raises(
        ValueError,
        match="job_id must be a non-empty S3 key component",
    ):
        create_presigned_upload_post(
            s3_client=s3_client,
            bucket_name="test-bucket",
            job_id=invalid_job_id,
        )

    s3_client.generate_presigned_post.assert_not_called()


def test_create_presigned_upload_post_propagates_s3_errors(
    mocker,
) -> None:
    s3_client = mocker.Mock()

    client_error = _create_client_error(
        error_code="AccessDenied",
        operation_name="GeneratePresignedPost",
    )

    s3_client.generate_presigned_post.side_effect = (
        client_error
    )

    with pytest.raises(ClientError) as exception_info:
        create_presigned_upload_post(
            s3_client=s3_client,
            bucket_name="test-bucket",
            job_id="job-123",
        )

    assert exception_info.value is client_error


def test_check_input_video_exists_returns_true_when_object_exists(
    mocker,
) -> None:
    s3_client = mocker.Mock()

    s3_client.head_object.return_value = {
        "ContentLength": 1024,
        "ContentType": "video/mp4",
    }

    result = check_input_video_exists(
        s3_client=s3_client,
        bucket_name="test-bucket",
        job_id="job-123",
    )

    assert result is True

    s3_client.head_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="jobs/job-123/input/source.mp4",
    )


@pytest.mark.parametrize(
    "error_code",
    [
        "404",
        "NoSuchKey",
        "NotFound",
    ],
)
def test_check_input_video_exists_returns_false_when_object_is_missing(
    mocker,
    error_code: str,
) -> None:
    s3_client = mocker.Mock()

    s3_client.head_object.side_effect = (
        _create_client_error(
            error_code=error_code,
            operation_name="HeadObject",
        )
    )

    result = check_input_video_exists(
        s3_client=s3_client,
        bucket_name="test-bucket",
        job_id="job-123",
    )

    assert result is False

    s3_client.head_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="jobs/job-123/input/source.mp4",
    )


def test_check_input_video_exists_propagates_permission_errors(
    mocker,
) -> None:
    s3_client = mocker.Mock()

    client_error = _create_client_error(
        error_code="AccessDenied",
        operation_name="HeadObject",
    )

    s3_client.head_object.side_effect = (
        client_error
    )

    with pytest.raises(ClientError) as exception_info:
        check_input_video_exists(
            s3_client=s3_client,
            bucket_name="test-bucket",
            job_id="job-123",
        )

    assert exception_info.value is client_error


def test_download_input_video_creates_parent_directory_and_downloads(
    tmp_path: Path,
    mocker,
) -> None:
    s3_client = mocker.Mock()

    local_destination_path = (
        tmp_path
        / "input"
        / "source.mp4"
    )

    result = download_input_video(
        s3_client=s3_client,
        bucket_name="test-bucket",
        job_id="job-123",
        local_destination_path=str(
            local_destination_path
        ),
    )

    assert local_destination_path.parent.is_dir()
    assert result == str(local_destination_path)

    s3_client.download_file.assert_called_once_with(
        Bucket="test-bucket",
        Key="jobs/job-123/input/source.mp4",
        Filename=str(local_destination_path),
    )


def test_download_input_video_propagates_s3_errors(
    tmp_path: Path,
    mocker,
) -> None:
    s3_client = mocker.Mock()

    client_error = _create_client_error(
        error_code="AccessDenied",
        operation_name="GetObject",
    )

    s3_client.download_file.side_effect = (
        client_error
    )

    local_destination_path = (
        tmp_path
        / "input"
        / "source.mp4"
    )

    with pytest.raises(ClientError) as exception_info:
        download_input_video(
            s3_client=s3_client,
            bucket_name="test-bucket",
            job_id="job-123",
            local_destination_path=str(
                local_destination_path
            ),
        )

    assert exception_info.value is client_error


@pytest.mark.parametrize(
    (
        "clip_id",
        "expected_filename",
    ),
    [
        ("0", "attempt_000.mp4"),
        ("7", "attempt_007.mp4"),
        ("42", "attempt_042.mp4"),
        ("123", "attempt_123.mp4"),
    ],
)
def test_upload_generated_clip_uploads_to_backend_controlled_key(
    tmp_path: Path,
    mocker,
    clip_id: str,
    expected_filename: str,
) -> None:
    s3_client = mocker.Mock()

    local_clip_path = (
        tmp_path
        / expected_filename
    )
    local_clip_path.touch()

    result = upload_generated_clip(
        s3_client=s3_client,
        bucket_name="test-bucket",
        job_id="job-123",
        clip_id=clip_id,
        local_clip_path=str(local_clip_path),
    )

    expected_object_key = (
        "jobs/job-123/generated-clips/"
        f"{expected_filename}"
    )

    assert result == expected_object_key

    s3_client.upload_file.assert_called_once_with(
        Filename=str(local_clip_path),
        Bucket="test-bucket",
        Key=expected_object_key,
        ExtraArgs={
            "ContentType": "video/mp4",
        },
    )


def test_upload_generated_clip_raises_when_local_file_is_missing(
    tmp_path: Path,
    mocker,
) -> None:
    s3_client = mocker.Mock()

    missing_clip_path = (
        tmp_path
        / "attempt_000.mp4"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Generated clip does not exist",
    ):
        upload_generated_clip(
            s3_client=s3_client,
            bucket_name="test-bucket",
            job_id="job-123",
            clip_id="0",
            local_clip_path=str(
                missing_clip_path
            ),
        )

    s3_client.upload_file.assert_not_called()


@pytest.mark.parametrize(
    "invalid_clip_id",
    [
        "",
        "-1",
        "throw",
        "1.5",
        " 7 ",
    ],
)
def test_upload_generated_clip_rejects_invalid_clip_id(
    tmp_path: Path,
    mocker,
    invalid_clip_id: str,
) -> None:
    s3_client = mocker.Mock()

    local_clip_path = (
        tmp_path
        / "attempt.mp4"
    )
    local_clip_path.touch()

    with pytest.raises(
        ValueError,
        match=(
            "clip_id must contain a non-negative "
            "numeric value"
        ),
    ):
        upload_generated_clip(
            s3_client=s3_client,
            bucket_name="test-bucket",
            job_id="job-123",
            clip_id=invalid_clip_id,
            local_clip_path=str(
                local_clip_path
            ),
        )

    s3_client.upload_file.assert_not_called()


def test_upload_generated_clip_propagates_s3_errors(
    tmp_path: Path,
    mocker,
) -> None:
    s3_client = mocker.Mock()

    local_clip_path = (
        tmp_path
        / "attempt_000.mp4"
    )
    local_clip_path.touch()

    client_error = _create_client_error(
        error_code="AccessDenied",
        operation_name="PutObject",
    )

    s3_client.upload_file.side_effect = (
        client_error
    )

    with pytest.raises(ClientError) as exception_info:
        upload_generated_clip(
            s3_client=s3_client,
            bucket_name="test-bucket",
            job_id="job-123",
            clip_id="0",
            local_clip_path=str(
                local_clip_path
            ),
        )

    assert exception_info.value is client_error


def test_create_presigned_generated_clip_download_url_returns_url(
    mocker,
) -> None:
    s3_client = mocker.Mock()

    generated_clip_object_key = (
        "jobs/job-123/generated-clips/"
        "attempt_000.mp4"
    )

    expected_url = (
        "https://test-bucket.s3.amazonaws.com/"
        "jobs/job-123/generated-clips/"
        "attempt_000.mp4?signed=true"
    )

    s3_client.generate_presigned_url.return_value = (
        expected_url
    )

    result = (
        create_presigned_generated_clip_download_url(
            s3_client=s3_client,
            bucket_name="test-bucket",
            generated_clip_object_key=(
                generated_clip_object_key
            ),
        )
    )

    assert result == expected_url

    s3_client.generate_presigned_url.assert_called_once_with(
        ClientMethod="get_object",
        Params={
            "Bucket": "test-bucket",
            "Key": generated_clip_object_key,
        },
        ExpiresIn=3600,
    )


def test_create_presigned_generated_clip_download_url_rejects_empty_key(
    mocker,
) -> None:
    s3_client = mocker.Mock()

    with pytest.raises(
        ValueError,
        match=(
            "generated_clip_object_key must not be empty"
        ),
    ):
        create_presigned_generated_clip_download_url(
            s3_client=s3_client,
            bucket_name="test-bucket",
            generated_clip_object_key="",
        )

    s3_client.generate_presigned_url.assert_not_called()
