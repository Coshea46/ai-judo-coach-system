"""Modal deployment configuration and video-processing worker."""

from pathlib import Path
from tempfile import TemporaryDirectory
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


# Modal configs
app = modal.App(
    "ai-judo-coach"
)

aws_secret = modal.Secret.from_name(
    "ai-judo-coach-aws-dev"
)

# Image used by the Modal worker that runs the processing pipeline.
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


@app.function(
    image=pipeline_image,
    gpu="T4",
    secrets=[aws_secret],
    timeout=3600,
)
def process_video_job(
    job_id: str,
) -> dict[str, object]:
    """
    Download and process one video job, then upload generated clips.

    The worker creates and cleans up its own temporary local workspace.

    Preconditions:
        The job's input video exists at its backend-controlled S3 key.
    """

    import os

    import boto3
    from botocore.config import Config

    from ai_judo_coach.pipeline.orchestrator import (
        run_pipeline,
    )
    from ai_judo_coach.storage import (
        download_input_video,
        upload_generated_clip,
    )

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

    with TemporaryDirectory(
        prefix="ai-judo-coach-job-",
    ) as temporary_directory:
        temporary_job_directory = Path(
            temporary_directory
        )

        local_input_video_path = (
            _construct_local_input_video_path(
                temporary_job_directory=(
                    temporary_job_directory
                ),
            )
        )

        download_input_video(
            s3_client=s3_client,
            bucket_name=bucket_name,
            job_id=job_id,
            local_destination_path=str(
                local_input_video_path
            ),
        )

        generated_clips = run_pipeline(
            input_video_path=str(
                local_input_video_path
            ),
            temporary_output_directory=str(
                temporary_job_directory
            ),
        )

        uploaded_clips: list[
            dict[str, str | float]
        ] = []

        for generated_clip in generated_clips:
            object_key = upload_generated_clip(
                s3_client=s3_client,
                bucket_name=bucket_name,
                job_id=job_id,
                clip_id=generated_clip.clip_id,
                local_clip_path=(
                    generated_clip.file_path
                ),
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

        return {
            "job_id": job_id,
            "clips": uploaded_clips,
        }


def _construct_local_input_video_path(
    temporary_job_directory: Path,
) -> Path:
    """Construct the local path for a downloaded input video."""

    return (
        temporary_job_directory
        / _LOCAL_INPUT_PREFIX
        / _LOCAL_INPUT_FILENAME
    )


@app.local_entrypoint()
def main(
    job_id: str,
) -> None:
    """Invoke the remote pipeline worker and print its result."""

    result = process_video_job.remote(
        job_id
    )

    print(result)
