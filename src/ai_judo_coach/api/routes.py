import uuid
from typing import Annotated, Any, Literal

import modal
from fastapi import(
    APIRouter,
    status,
    Depends,
    HTTPException
)
from fastapi.responses import RedirectResponse

from ai_judo_coach.storage import(
    check_input_video_exists,
    copy_example_video_to_job_input,
    create_presigned_example_video_preview_url,
    create_presigned_generated_clip_download_url,
    create_presigned_upload_post
)
from ai_judo_coach.api.dependencies import (
    get_bucket_name,
    get_job_store,
    get_s3_client,
    get_pipeline_worker
)
from ai_judo_coach.schemas.api import (
    CreateJobResponse,
    HealthResponse,
    JobStatusResponse,
    SubmitJobResponse,
)



router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    response_model=HealthResponse,
)
async def health() -> dict[str, str]:
    """
    Endpoint allowing frontend
    to check api health
    """
    return {"status": "ok"}



@router.post(
    "/jobs",
    status_code=status.HTTP_201_CREATED,
    response_model=CreateJobResponse,
)
def create_job(
    s3_client: Annotated[Any, Depends(get_s3_client)],
    bucket_name: Annotated[str, Depends(get_bucket_name)],
    job_store: Annotated[modal.Dict, Depends(get_job_store)]
) -> dict[str, object]:
    """
    Creates a job on the modal
    server and hands and return
    its browser upload instructions
    """

    # create uuid for job
    job_id = str(uuid.uuid4())

    presigned_browser_upload = create_presigned_upload_post(
        s3_client=s3_client,
        bucket_name=bucket_name,
        job_id=job_id
    )

    # add the job to the dictionary of jobs stored by the modal server
    # use put() function as job_store should be a modal dict (not standard py dict)
    job_store.put(
        job_id,
        {
            "status": "awaiting_upload",
            "modal_call_id": None,
            "clips": None,
            "error": None,
        },
        skip_if_exists=True,
    )

    return {
        "job_id": job_id,
        "status": "awaiting_upload",
        "upload": {
            "url": (
                presigned_browser_upload["url"]
            ),
            "fields": (
                presigned_browser_upload["fields"]
            ),
        },
    }



@router.post(
    "/jobs/{job_id}/submit",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SubmitJobResponse,
)
def submit_job(
    job_id: str,
    s3_client: Annotated[Any, Depends(get_s3_client)],
    bucket_name: Annotated[str, Depends(get_bucket_name)],
    job_store: Annotated[modal.Dict, Depends(get_job_store)],
    modal_pipeline_worker: Annotated[Any, Depends(get_pipeline_worker)]
) -> dict[str, str]:
    """
    Submit an uploaded video job for asynchronous processing.
    """

    # read job record from job store
    job_record = job_store.get(
        job_id
    )

    if job_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job does not exist",
        )

    if job_record.get("status") != "awaiting_upload":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job has already been submitted",
        )

    input_video_exists = check_input_video_exists(
        s3_client=s3_client,
        bucket_name=bucket_name,
        job_id=job_id,
    )

    if not input_video_exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The input video has not been uploaded",
        )

    function_call = modal_pipeline_worker.spawn(
        job_id
    )

    job_store.put(
        job_id,
        {
            "status": "processing",
            "modal_call_id": function_call.object_id,
            "clips": None,
            "error": None,
        },
    )

    return {
        "job_id": job_id,
        "status": "processing",
    }




@router.get(
    "/jobs/{job_id}",
    status_code=status.HTTP_200_OK,
    response_model=JobStatusResponse,
)
def read_job_status(
    job_id: str,
    s3_client: Annotated[Any, Depends(get_s3_client)],
    bucket_name: Annotated[str, Depends(get_bucket_name)],
    job_store: Annotated[modal.Dict, Depends(get_job_store)],
) -> dict[str, object]:
    """
    Return the current status and available results for a video job.
    """

    job_record = job_store.get(
        job_id
    )

    if job_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job does not exist",
        )

    job_status = job_record.get(
        "status"
    )

    if job_status == "awaiting_upload":
        return {
            "job_id": job_id,
            "status": "awaiting_upload",
            "clips": [],
            "error": None,
        }

    if job_status == "processing":
        modal_call_id = job_record.get(
            "modal_call_id"
        )

        if not isinstance(
            modal_call_id,
            str,
        ) or not modal_call_id:
            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    "Processing job does not have "
                    "a Modal call ID"
                ),
            )

        function_call = (
            modal.FunctionCall.from_id(
                modal_call_id
            )
        )

        try:
            worker_result = function_call.get(
                timeout=0,
            )
        except TimeoutError:
            return {
                "job_id": job_id,
                "status": "processing",
                "clips": [],
                "error": None,
            }
        except Exception:
            job_record = {
                "status": "failed",
                "modal_call_id": modal_call_id,
                "clips": None,
                "error": "Video processing failed",
            }

            job_store.put(
                job_id,
                job_record,
            )
        else:
            job_record = {
                "status": "completed",
                "modal_call_id": modal_call_id,
                "clips": worker_result["clips"],
                "error": None,
            }

            job_store.put(
                job_id,
                job_record,
            )

    job_status = job_record.get(
        "status"
    )

    if job_status == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "clips": [],
            "error": job_record.get(
                "error"
            ),
        }

    if job_status != "completed":
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Job has an invalid status",
        )

    downloadable_clips: list[
        dict[str, object]
    ] = []

    for clip in job_record.get(
        "clips",
        [],
    ):
        download_url = (
            create_presigned_generated_clip_download_url(
                s3_client=s3_client,
                bucket_name=bucket_name,
                generated_clip_object_key=(
                    clip["object_key"]
                ),
            )
        )

        downloadable_clips.append(
            {
                "clip_id": clip["clip_id"],
                "start_time_seconds": (
                    clip["start_time_seconds"]
                ),
                "end_time_seconds": (
                    clip["end_time_seconds"]
                ),
                "download_url": download_url,
            }
        )

    return {
        "job_id": job_id,
        "status": "completed",
        "clips": downloadable_clips,
        "error": None,
    }




@router.get(
    "/examples/{example}/preview",
    response_class=RedirectResponse,
    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
)
def preview_example_video(
    example: Literal[
        "full",
        "short",
    ],
    s3_client: Annotated[
        Any,
        Depends(get_s3_client),
    ],
    bucket_name: Annotated[
        str,
        Depends(get_bucket_name),
    ],
) -> RedirectResponse:
    """
    Redirect the browser to a temporary URL for an example video.
    """

    preview_url = (
        create_presigned_example_video_preview_url(
            s3_client=s3_client,
            bucket_name=bucket_name,
            example=example,
        )
    )

    return RedirectResponse(
        url=preview_url,
        status_code=(
            status.HTTP_307_TEMPORARY_REDIRECT
        ),
        headers={
            # Do not cache a redirect containing an expiring signature.
            "Cache-Control": "no-store",
        },
    )





@router.post(
    "/examples",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SubmitJobResponse,
)
def create_example_job(
    s3_client: Annotated[
        Any,
        Depends(get_s3_client),
    ],
    bucket_name: Annotated[
        str,
        Depends(get_bucket_name),
    ],
    job_store: Annotated[
        modal.Dict,
        Depends(get_job_store),
    ],
    modal_pipeline_worker: Annotated[
        Any,
        Depends(get_pipeline_worker),
    ],
    example: Literal[
        "full",
        "short",
    ] = "full",
) -> dict[str, str]:
    """
    Create and submit a processing job using the example video.
    """

    job_id = str(uuid.uuid4())

    copy_example_video_to_job_input(
        s3_client=s3_client,
        bucket_name=bucket_name,
        job_id=job_id,
        example=example,
    )

    job_store.put(
        job_id,
        {
            "status": "awaiting_upload",
            "modal_call_id": None,
            "clips": None,
            "error": None,
        },
        skip_if_exists=True,
    )

    function_call = modal_pipeline_worker.spawn(
        job_id
    )

    job_store.put(
        job_id,
        {
            "status": "processing",
            "modal_call_id": function_call.object_id,
            "clips": None,
            "error": None,
        },
    )

    return {
        "job_id": job_id,
        "status": "processing",
    }
