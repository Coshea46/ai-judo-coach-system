"""Modal deployment configuration and distributed video processing."""

from __future__ import annotations

import math
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Final

import modal


_PROJECT_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[3]
    if modal.is_local()
    else Path("/app")
)

_CLASSIFIER_WHEEL_PATH: Final[Path] = (
    _PROJECT_ROOT
    / "wheels"
    / "v1_clip_classification_model-0.1.0-py3-none-any.whl"
)

_SOURCE_DIRECTORY: Final[Path] = (
    _PROJECT_ROOT
    / "src"
)

_WEIGHTS_DIRECTORY: Final[Path] = (
    _PROJECT_ROOT
    / "weights"
)

_LOCAL_INPUT_PREFIX: Final[str] = "input"
_LOCAL_INPUT_FILENAME: Final[str] = "source.mp4"

_LOCAL_CLEANSED_PREFIX: Final[str] = "input_cleanse"
_LOCAL_CLEANSED_FILENAME: Final[str] = "cleansed_input.mp4"

_PROCESSING_PREFIX: Final[str] = "processing"
_PROCESSING_CLEANSED_FILENAME: Final[str] = (
    "cleansed_input.mp4"
)

_DISTRIBUTED_GROUP_COUNT: Final[int] = 4

_NVENC_VIDEO_OUTPUT_OPTIONS: Final[
    dict[str, object]
] = {
    "vcodec": "h264_nvenc",
    "preset": "p4",
    "tune": "lossless",
    "rc": "constqp",
    "qp": 0,
    "pix_fmt": "yuv420p",
}


app = modal.App(
    "ai-judo-coach"
)

aws_secret = modal.Secret.from_name(
    "ai-judo-coach-aws-dev"
)


pipeline_image = (
    modal.Image
    .debian_slim(
        python_version="3.12"
    )
    .apt_install(
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
    )
    .pip_install(
        "boto3==1.43.66",
        "decord==0.6.0",
        "ffmpeg-python==0.2.0",
        "numpy==2.5.1",
        "opencv-python==5.0.0.93",
        "polars==1.43.2",
        "torch==2.13.0",
        "ultralytics==8.4.126",
        "lap==0.5.13",
    )
    .pip_install(
        "pydantic>=2.10,<3.0",
    )
    .add_local_file(
        local_path=str(
            _CLASSIFIER_WHEEL_PATH
        ),
        remote_path=(
            "/app/wheels/"
            "v1_clip_classification_model-0.1.0-"
            "py3-none-any.whl"
        ),
        copy=True,
    )
    .run_commands(
        "python -m pip install "
        "/app/wheels/"
        "v1_clip_classification_model-0.1.0-"
        "py3-none-any.whl"
    )
    .add_local_dir(
        local_path=str(
            _SOURCE_DIRECTORY
        ),
        remote_path="/app/src",
        copy=True,
    )
    .add_local_dir(
        local_path=str(
            _WEIGHTS_DIRECTORY
        ),
        remote_path="/app/weights",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/app/src",
        }
    )
)


api_image = (
    modal.Image
    .debian_slim(
        python_version="3.12"
    )
    .pip_install(
        "boto3==1.43.66",
        "fastapi>=0.115,<1.0",
        "pydantic>=2.10,<3.0",
    )
    .add_local_dir(
        local_path=str(
            _SOURCE_DIRECTORY
        ),
        remote_path="/app/src",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/app/src",
        }
    )
)


@app.function(
    image=pipeline_image,
    gpu="T4",
    memory=12288,
    secrets=[aws_secret],
    timeout=1800,
)
def cleanse_video_job(
    job_id: str,
    cleansed_object_key: str,
) -> dict[str, object]:
    """
    Download and cleanse one input video using T4 NVENC.

    Audio removal and frame-rate normalisation remain two distinct
    stages. The cleansed video is uploaded for distributed workers.
    """

    from ai_judo_coach.storage import (
        download_input_video,
    )
    from ai_judo_coach.video import (
        cleanse_input_video,
    )

    worker_started_at = perf_counter()

    s3_client, bucket_name = _create_s3_client()

    with TemporaryDirectory(
        prefix="ai-judo-cleanse-",
    ) as temporary_directory:
        temporary_path = Path(
            temporary_directory
        )

        local_input_video_path = (
            _construct_local_input_video_path(
                temporary_job_directory=temporary_path,
            )
        )

        download_started_at = perf_counter()

        download_input_video(
            s3_client=s3_client,
            bucket_name=bucket_name,
            job_id=job_id,
            local_destination_path=str(
                local_input_video_path
            ),
        )

        download_seconds = (
            perf_counter() - download_started_at
        )

        cleanse_started_at = perf_counter()

        (
            cleansed_video_path_string,
            cleansed_video_duration,
        ) = cleanse_input_video(
            input_video_path=str(
                local_input_video_path
            ),
            output_directory=str(
                temporary_path
            ),
            video_output_options=dict(
                _NVENC_VIDEO_OUTPUT_OPTIONS
            ),
        )

        cleanse_seconds = (
            perf_counter() - cleanse_started_at
        )

        cleansed_video_path = Path(
            cleansed_video_path_string
        )

        if not cleansed_video_path.is_file():
            raise RuntimeError(
                "Cleansing completed without creating "
                f"{cleansed_video_path}"
            )

        upload_started_at = perf_counter()

        s3_client.upload_file(
            Filename=str(
                cleansed_video_path
            ),
            Bucket=bucket_name,
            Key=cleansed_object_key,
            ExtraArgs={
                "ContentType": "video/mp4",
            },
        )

        upload_seconds = (
            perf_counter() - upload_started_at
        )

        worker_total_seconds = (
            perf_counter() - worker_started_at
        )

        return {
            "cleansed_video_duration": (
                float(cleansed_video_duration)
            ),
            "timings": {
                "download_input_seconds": (
                    download_seconds
                ),
                "two_stage_cleanse_seconds": (
                    cleanse_seconds
                ),
                "upload_cleansed_seconds": (
                    upload_seconds
                ),
                "cleanse_worker_total_seconds": (
                    worker_total_seconds
                ),
            },
        }


@app.function(
    image=pipeline_image,
    gpu="T4",
    memory=12288,
    secrets=[aws_secret],
    timeout=3600,
)
def process_window_group_job(
    cleansed_object_key: str,
    group_index: int,
    serialised_work_items: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    """
    Process one contiguous classifier-window group on one T4.

    Each worker downloads the cleansed video once and returns only
    ordered primitive classification results and stage timings.
    """

    from ai_judo_coach.pipeline.distributed_stages import (
        process_window_group,
    )

    if group_index < 0:
        raise ValueError(
            "group_index must not be negative"
        )

    if not serialised_work_items:
        raise ValueError(
            "A distributed window group must not be empty"
        )

    worker_started_at = perf_counter()

    s3_client, bucket_name = _create_s3_client()

    with TemporaryDirectory(
        prefix=f"ai-judo-group-{group_index}-",
    ) as temporary_directory:
        temporary_path = Path(
            temporary_directory
        )

        cleansed_video_path = (
            _construct_local_cleansed_video_path(
                temporary_job_directory=temporary_path,
            )
        )

        cleansed_video_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        download_started_at = perf_counter()

        s3_client.download_file(
            Bucket=bucket_name,
            Key=cleansed_object_key,
            Filename=str(
                cleansed_video_path
            ),
        )

        download_seconds = (
            perf_counter() - download_started_at
        )

        work_items = [
            _deserialise_window_work_item(
                serialised_work_item
            )
            for serialised_work_item
            in serialised_work_items
        ]

        stage_timings: dict[
            str,
            float,
        ] = {}

        ordered_results = process_window_group(
            cleansed_video_path=str(
                cleansed_video_path
            ),
            work_items=work_items,
            stage_timings=stage_timings,
        )

        if len(ordered_results) != len(
            work_items
        ):
            raise RuntimeError(
                "Distributed worker returned an unexpected "
                "number of window results"
            )

        group_worker_total_seconds = (
            perf_counter() - worker_started_at
        )

        return {
            "group_index": group_index,
            "results": [
                _serialise_clip_processing_result(
                    result
                )
                for result in ordered_results
            ],
            "timings": {
                "download_cleansed_seconds": (
                    download_seconds
                ),
                **stage_timings,
                "group_worker_total_seconds": (
                    group_worker_total_seconds
                ),
            },
        }


@app.function(
    image=pipeline_image,
    secrets=[aws_secret],
    timeout=3600,
)
def process_video_job(
    job_id: str,
) -> dict[str, object]:
    """
    Coordinate cleansing, distributed inference and clip generation.

    Four contiguous window groups are processed concurrently when the
    video contains at least four classifier windows.
    """

    from ai_judo_coach.pipeline.distributed_stages import (
        build_window_work_items,
        finalise_window_results,
    )
    from ai_judo_coach.storage import (
        upload_generated_clip,
    )

    coordinator_started_at = perf_counter()

    _validate_job_id_component(
        job_id=job_id,
    )

    cleansed_object_key = (
        _construct_processing_cleansed_s3_path(
            job_id=job_id,
        )
    )

    s3_client, bucket_name = _create_s3_client()

    cleanup_error: Exception | None = None

    try:
        cleanse_call = cleanse_video_job.spawn(
            job_id=job_id,
            cleansed_object_key=(
                cleansed_object_key
            ),
        )

        cleanse_result = cleanse_call.get()

        cleansed_video_duration = float(
            cleanse_result[
                "cleansed_video_duration"
            ]
        )

        if (
            not math.isfinite(
                cleansed_video_duration
            )
            or cleansed_video_duration <= 0.0
        ):
            raise RuntimeError(
                "Cleanse worker returned an invalid "
                "video duration"
            )

        with TemporaryDirectory(
            prefix="ai-judo-coordinator-",
        ) as temporary_directory:
            temporary_path = Path(
                temporary_directory
            )

            cleansed_video_path = (
                _construct_local_cleansed_video_path(
                    temporary_job_directory=(
                        temporary_path
                    ),
                )
            )

            cleansed_video_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            coordinator_download_started_at = (
                perf_counter()
            )

            s3_client.download_file(
                Bucket=bucket_name,
                Key=cleansed_object_key,
                Filename=str(
                    cleansed_video_path
                ),
            )

            coordinator_download_seconds = (
                perf_counter()
                - coordinator_download_started_at
            )

            work_item_build_started_at = (
                perf_counter()
            )

            work_items = build_window_work_items(
                cleansed_video_path=str(
                    cleansed_video_path
                ),
            )

            work_item_build_seconds = (
                perf_counter()
                - work_item_build_started_at
            )

            if not work_items:
                raise RuntimeError(
                    "The cleansed video did not produce "
                    "any classifier windows"
                )

            window_groups = (
                _partition_contiguous_work_items(
                    work_items=work_items,
                    maximum_group_count=(
                        _DISTRIBUTED_GROUP_COUNT
                    ),
                )
            )

            fanout_started_at = perf_counter()

            group_calls = [
                process_window_group_job.spawn(
                    cleansed_object_key=(
                        cleansed_object_key
                    ),
                    group_index=group_index,
                    serialised_work_items=[
                        _serialise_window_work_item(
                            work_item
                        )
                        for work_item in group
                    ],
                )
                for group_index, group in enumerate(
                    window_groups
                )
            ]

            group_responses = [
                group_call.get()
                for group_call in group_calls
            ]

            fanout_wall_seconds = (
                perf_counter() - fanout_started_at
            )

            ordered_group_responses = (
                _validate_and_order_group_responses(
                    group_responses=group_responses,
                    expected_group_count=len(
                        window_groups
                    ),
                )
            )

            ordered_results = [
                _deserialise_clip_processing_result(
                    serialised_result
                )
                for group_response
                in ordered_group_responses
                for serialised_result
                in _get_serialised_results(
                    group_response
                )
            ]

            if len(ordered_results) != len(
                work_items
            ):
                raise RuntimeError(
                    "Distributed processing returned "
                    f"{len(ordered_results)} results for "
                    f"{len(work_items)} windows"
                )

            finalisation_started_at = (
                perf_counter()
            )

            generated_clips = (
                finalise_window_results(
                    cleansed_video_path=str(
                        cleansed_video_path
                    ),
                    cleansed_video_duration=(
                        cleansed_video_duration
                    ),
                    temporary_output_directory=str(
                        temporary_path
                    ),
                    work_items=work_items,
                    ordered_results=(
                        ordered_results
                    ),
                )
            )

            finalisation_seconds = (
                perf_counter()
                - finalisation_started_at
            )

            upload_started_at = perf_counter()

            uploaded_clips: list[
                dict[str, str | float]
            ] = []

            for generated_clip in generated_clips:
                object_key = (
                    upload_generated_clip(
                        s3_client=s3_client,
                        bucket_name=bucket_name,
                        job_id=job_id,
                        clip_id=(
                            generated_clip.clip_id
                        ),
                        local_clip_path=(
                            generated_clip.file_path
                        ),
                    )
                )

                uploaded_clips.append(
                    {
                        "clip_id": (
                            generated_clip.clip_id
                        ),
                        "start_time_seconds": (
                            generated_clip
                            .start_time_seconds
                        ),
                        "end_time_seconds": (
                            generated_clip
                            .end_time_seconds
                        ),
                        "object_key": object_key,
                    }
                )

            upload_seconds = (
                perf_counter() - upload_started_at
            )

            coordinator_total_seconds = (
                perf_counter()
                - coordinator_started_at
            )

            result = {
                "job_id": job_id,
                "clips": uploaded_clips,
                "timings": {
                    "cleanse_worker": (
                        cleanse_result.get(
                            "timings",
                            {},
                        )
                    ),
                    "coordinator_download_cleansed_seconds": (
                        coordinator_download_seconds
                    ),
                    "work_item_build_seconds": (
                        work_item_build_seconds
                    ),
                    "distributed_group_count": len(
                        window_groups
                    ),
                    "distributed_window_count": len(
                        work_items
                    ),
                    "fanout_wall_seconds": (
                        fanout_wall_seconds
                    ),
                    "group_workers": [
                        {
                            "group_index": (
                                group_response[
                                    "group_index"
                                ]
                            ),
                            "timings": (
                                group_response.get(
                                    "timings",
                                    {},
                                )
                            ),
                        }
                        for group_response
                        in ordered_group_responses
                    ],
                    "finalisation_seconds": (
                        finalisation_seconds
                    ),
                    "upload_generated_clips_seconds": (
                        upload_seconds
                    ),
                    "coordinator_total_seconds": (
                        coordinator_total_seconds
                    ),
                },
            }

            print(
                json.dumps(
                    {
                        "event": (
                            "process_video_job_completed"
                        ),
                        "job_id": job_id,
                        "window_results": [
                            {
                                "clip_id": (
                                    window_result.clip_id
                                ),
                                "contains_throw_attempt": (
                                    window_result
                                    .contains_throw_attempt
                                ),
                                "attempt_probability": (
                                    window_result
                                    .attempt_probability
                                ),
                                "predicted_class_name": (
                                    window_result
                                    .predicted_class_name
                                ),
                            }
                            for window_result
                            in ordered_results
                        ],
                        "timings": result["timings"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


            return result

    finally:
        try:
            s3_client.delete_object(
                Bucket=bucket_name,
                Key=cleansed_object_key,
            )
        except Exception as error:
            cleanup_error = error

        if cleanup_error is not None:
            print(
                "Unable to delete temporary cleansed "
                f"object {cleansed_object_key}: "
                f"{cleanup_error}",
                flush=True,
            )



@app.function(
    image=api_image,
    secrets=[aws_secret],
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI control plane through Modal."""

    from ai_judo_coach.main import app as api

    return api


def _create_s3_client():
    """Create the configured S3 client and return its bucket."""

    import os

    import boto3
    from botocore.config import Config

    region_name = os.environ[
        "AWS_DEFAULT_REGION"
    ]
    bucket_name = os.environ[
        "AI_JUDO_COACH_S3_BUCKET"
    ]

    s3_client = boto3.client(
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

    return s3_client, bucket_name


def _partition_contiguous_work_items(
    work_items: list,
    maximum_group_count: int,
) -> list[list]:
    """Partition ordered work into balanced contiguous groups."""

    if maximum_group_count <= 0:
        raise ValueError(
            "maximum_group_count must be greater than zero"
        )

    if not work_items:
        return []

    group_count = min(
        maximum_group_count,
        len(work_items),
    )

    base_group_size, larger_group_count = divmod(
        len(work_items),
        group_count,
    )

    groups: list[list] = []
    next_item_index = 0

    for group_index in range(
        group_count
    ):
        group_size = (
            base_group_size
            + (
                1
                if group_index
                < larger_group_count
                else 0
            )
        )

        group_end_index = (
            next_item_index
            + group_size
        )

        group = work_items[
            next_item_index:group_end_index
        ]

        if not group:
            raise RuntimeError(
                "Distributed partitioning produced "
                "an empty group"
            )

        groups.append(group)
        next_item_index = group_end_index

    flattened_items = [
        work_item
        for group in groups
        for work_item in group
    ]

    if flattened_items != work_items:
        raise RuntimeError(
            "Distributed partitioning changed work-item "
            "order or coverage"
        )

    return groups


def _serialise_window_work_item(
    work_item,
) -> dict[str, object]:
    """Convert one internal window work item to primitives."""

    window, absolute_frame_indices = work_item

    return {
        "start_time": float(
            window.start_time
        ),
        "end_time": float(
            window.end_time
        ),
        "window_id": int(
            window.window_id
        ),
        "absolute_frame_indices": [
            int(frame_index)
            for frame_index
            in absolute_frame_indices
        ],
    }


def _deserialise_window_work_item(
    serialised_work_item: dict[str, object],
):
    """Reconstruct one internal window work item."""

    from ai_judo_coach.schemas.internal import (
        InitialClipWindow,
    )

    start_time = serialised_work_item.get(
        "start_time"
    )
    end_time = serialised_work_item.get(
        "end_time"
    )
    window_id = serialised_work_item.get(
        "window_id"
    )
    absolute_frame_indices = (
        serialised_work_item.get(
            "absolute_frame_indices"
        )
    )

    if (
        isinstance(start_time, bool)
        or not isinstance(
            start_time,
            (int, float),
        )
    ):
        raise TypeError(
            "Window start_time must be numeric"
        )

    if (
        isinstance(end_time, bool)
        or not isinstance(
            end_time,
            (int, float),
        )
    ):
        raise TypeError(
            "Window end_time must be numeric"
        )

    if (
        isinstance(window_id, bool)
        or not isinstance(window_id, int)
    ):
        raise TypeError(
            "Window window_id must be an integer"
        )

    if not isinstance(
        absolute_frame_indices,
        list,
    ):
        raise TypeError(
            "absolute_frame_indices must be a list"
        )

    if any(
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        for frame_index
        in absolute_frame_indices
    ):
        raise TypeError(
            "Every absolute frame index must be an integer"
        )

    return (
        InitialClipWindow(
            start_time=float(
                start_time
            ),
            end_time=float(
                end_time
            ),
            window_id=window_id,
        ),
        list(absolute_frame_indices),
    )


def _serialise_clip_processing_result(
    result,
) -> dict[str, object]:
    """Convert one clip-processing result to primitives."""

    return {
        "clip_id": result.clip_id,
        "contains_throw_attempt": (
            result.contains_throw_attempt
        ),
        "attempt_probability": float(
            result.attempt_probability
        ),
        "predicted_class_name": (
            result.predicted_class_name
        ),
    }


def _deserialise_clip_processing_result(
    serialised_result: dict[str, object],
):
    """Reconstruct one clip-processing result."""

    from ai_judo_coach.schemas.internal import (
        ClipProcessingResult,
    )

    clip_id = serialised_result.get(
        "clip_id"
    )
    contains_throw_attempt = (
        serialised_result.get(
            "contains_throw_attempt"
        )
    )
    attempt_probability = (
        serialised_result.get(
            "attempt_probability"
        )
    )
    predicted_class_name = (
        serialised_result.get(
            "predicted_class_name"
        )
    )

    if not isinstance(
        clip_id,
        str,
    ) or not clip_id:
        raise TypeError(
            "clip_id must be a non-empty string"
        )

    if not isinstance(
        contains_throw_attempt,
        bool,
    ):
        raise TypeError(
            "contains_throw_attempt must be boolean"
        )

    if (
        isinstance(attempt_probability, bool)
        or not isinstance(
            attempt_probability,
            (int, float),
        )
    ):
        raise TypeError(
            "attempt_probability must be numeric"
        )

    probability = float(
        attempt_probability
    )

    if not math.isfinite(
        probability
    ):
        raise ValueError(
            "attempt_probability must be finite"
        )

    if not isinstance(
        predicted_class_name,
        str,
    ) or not predicted_class_name:
        raise TypeError(
            "predicted_class_name must be a "
            "non-empty string"
        )

    return ClipProcessingResult(
        clip_id=clip_id,
        contains_throw_attempt=(
            contains_throw_attempt
        ),
        attempt_probability=probability,
        predicted_class_name=(
            predicted_class_name
        ),
    )


def _validate_and_order_group_responses(
    group_responses: list[
        dict[str, object]
    ],
    expected_group_count: int,
) -> list[dict[str, object]]:
    """Validate worker coverage and order responses by group."""

    if len(group_responses) != (
        expected_group_count
    ):
        raise RuntimeError(
            "Expected "
            f"{expected_group_count} group responses, "
            f"received {len(group_responses)}"
        )

    responses_by_group: dict[
        int,
        dict[str, object],
    ] = {}

    for response in group_responses:
        group_index = response.get(
            "group_index"
        )

        if (
            isinstance(group_index, bool)
            or not isinstance(group_index, int)
        ):
            raise TypeError(
                "Worker group_index must be an integer"
            )

        if group_index in responses_by_group:
            raise RuntimeError(
                "Received duplicate response for "
                f"group {group_index}"
            )

        _get_serialised_results(
            response
        )

        responses_by_group[
            group_index
        ] = response

    expected_indices = list(
        range(expected_group_count)
    )
    actual_indices = sorted(
        responses_by_group
    )

    if actual_indices != expected_indices:
        raise RuntimeError(
            "Distributed group coverage changed: "
            f"{actual_indices} != {expected_indices}"
        )

    return [
        responses_by_group[group_index]
        for group_index in expected_indices
    ]


def _get_serialised_results(
    group_response: dict[str, object],
) -> list[dict[str, object]]:
    """Validate and return one worker's result list."""

    results = group_response.get(
        "results"
    )

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "Worker results must be a list"
        )

    if any(
        not isinstance(result, dict)
        for result in results
    ):
        raise TypeError(
            "Every worker result must be a dictionary"
        )

    return results


def _construct_local_input_video_path(
    temporary_job_directory: Path,
) -> Path:
    """Construct the local downloaded source-video path."""

    return (
        temporary_job_directory
        / _LOCAL_INPUT_PREFIX
        / _LOCAL_INPUT_FILENAME
    )


def _construct_local_cleansed_video_path(
    temporary_job_directory: Path,
) -> Path:
    """Construct the local cleansed-video path."""

    return (
        temporary_job_directory
        / _LOCAL_CLEANSED_PREFIX
        / _LOCAL_CLEANSED_FILENAME
    )


def _construct_processing_cleansed_s3_path(
    job_id: str,
) -> str:
    """Construct the temporary cleansed-video S3 key."""

    _validate_job_id_component(
        job_id=job_id,
    )

    return str(
        PurePosixPath(
            "jobs",
            job_id,
            _PROCESSING_PREFIX,
            _PROCESSING_CLEANSED_FILENAME,
        )
    )


def _validate_job_id_component(
    job_id: str,
) -> None:
    """Validate that a job ID is one safe S3 key component."""

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


@app.local_entrypoint()
def main(
    job_id: str,
) -> None:
    """Invoke the remote coordinator and print its result."""

    result = process_video_job.remote(
        job_id
    )

    print(result)
