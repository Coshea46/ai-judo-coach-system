"""Response models returned by the FastAPI control plane."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """API health response."""

    status: Literal["ok"]


class BrowserUploadResponse(BaseModel):
    """Browser instructions for uploading a job's input video."""

    url: str
    fields: dict[str, str]


class CreateJobResponse(BaseModel):
    """Response returned when a new job is created."""

    job_id: str
    status: Literal["awaiting_upload"]
    upload: BrowserUploadResponse


class SubmitJobResponse(BaseModel):
    """Response returned when a job is submitted for processing."""

    job_id: str
    status: Literal["processing"]


class GeneratedClipResponse(BaseModel):
    """A generated clip available for download."""

    clip_id: str
    start_time_seconds: float
    end_time_seconds: float
    download_url: str


class JobStatusResponse(BaseModel):
    """Current status and available results for a video job."""

    job_id: str
    status: Literal[
        "awaiting_upload",
        "processing",
        "completed",
        "failed",
    ]
    clips: list[GeneratedClipResponse]
    error: str | None
