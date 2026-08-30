"""Unit tests for the FastAPI control-plane routes."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import ai_judo_coach.api.routes as routes
from ai_judo_coach.api.dependencies import (
    get_bucket_name,
    get_job_store,
    get_pipeline_worker,
    get_s3_client,
)
from ai_judo_coach.main import app


class FakeJobStore:
    """In-memory replacement for the persistent Modal Dict."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        """Return a stored job record."""

        return self.records.get(key)

    def put(
        self,
        key: str,
        value: dict[str, Any],
        *,
        skip_if_exists: bool = False,
    ) -> None:
        """Store a job record."""

        if skip_if_exists and key in self.records:
            return

        self.records[key] = value


class FakeSpawnedFunctionCall:
    """Function call returned when the fake worker is spawned."""

    def __init__(self, object_id: str) -> None:
        self.object_id = object_id


class FakePipelineWorker:
    """Replacement for the deployed Modal pipeline worker."""

    def __init__(self, call_id: str = "fc-test-call-id") -> None:
        self.call_id = call_id
        self.spawned_job_ids: list[str] = []

    def spawn(self, job_id: str) -> FakeSpawnedFunctionCall:
        """Record a spawned job and return its call ID."""

        self.spawned_job_ids.append(job_id)

        return FakeSpawnedFunctionCall(
            object_id=self.call_id,
        )


class FakeRecoveredFunctionCall:
    """Replacement for a recovered Modal FunctionCall."""

    def __init__(
        self,
        *,
        result: object | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.result = result
        self.exception = exception
        self.get_timeouts: list[int] = []

    def get(self, *, timeout: int) -> object:
        """Return the configured result or raise the configured exception."""

        self.get_timeouts.append(timeout)

        if self.exception is not None:
            raise self.exception

        return self.result


def _install_recovered_function_call(
    monkeypatch: pytest.MonkeyPatch,
    recovered_function_call: FakeRecoveredFunctionCall,
) -> list[str]:
    """Install a fake implementation of Modal FunctionCall recovery."""

    requested_call_ids: list[str] = []

    class FakeFunctionCall:
        @staticmethod
        def from_id(call_id: str) -> FakeRecoveredFunctionCall:
            requested_call_ids.append(call_id)
            return recovered_function_call

    monkeypatch.setattr(
        routes.modal,
        "FunctionCall",
        FakeFunctionCall,
    )

    return requested_call_ids


@pytest.fixture
def s3_client() -> object:
    """Return an opaque fake S3 client."""

    return object()


@pytest.fixture
def bucket_name() -> str:
    """Return the test bucket name."""

    return "test-video-bucket"


@pytest.fixture
def job_store() -> FakeJobStore:
    """Return an empty fake job store."""

    return FakeJobStore()


@pytest.fixture
def pipeline_worker() -> FakePipelineWorker:
    """Return a fake Modal pipeline worker."""

    return FakePipelineWorker()


@pytest.fixture
def client(
    s3_client: object,
    bucket_name: str,
    job_store: FakeJobStore,
    pipeline_worker: FakePipelineWorker,
) -> Iterator[TestClient]:
    """Return an API client with all external dependencies overridden."""

    app.dependency_overrides[get_s3_client] = lambda: s3_client
    app.dependency_overrides[get_bucket_name] = lambda: bucket_name
    app.dependency_overrides[get_job_store] = lambda: job_store
    app.dependency_overrides[get_pipeline_worker] = (
        lambda: pipeline_worker
    )

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_health_returns_ok(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_create_job_returns_upload_instructions_and_stores_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    s3_client: object,
    bucket_name: str,
    job_store: FakeJobStore,
) -> None:
    job_id = "784112e9-7af6-46d7-9f4f-58b99195dcf5"

    monkeypatch.setattr(
        routes.uuid,
        "uuid4",
        lambda: job_id,
    )

    def fake_create_presigned_upload_post(
        *,
        s3_client: object,
        bucket_name: str,
        job_id: str,
    ) -> dict[str, object]:
        assert s3_client is not None
        assert bucket_name == "test-video-bucket"
        assert job_id == "784112e9-7af6-46d7-9f4f-58b99195dcf5"

        return {
            "object_key": (
                "jobs/"
                "784112e9-7af6-46d7-9f4f-58b99195dcf5/"
                "input/source.mp4"
            ),
            "url": "https://test-video-bucket.example.com",
            "fields": {
                "key": (
                    "jobs/"
                    "784112e9-7af6-46d7-9f4f-58b99195dcf5/"
                    "input/source.mp4"
                ),
                "Content-Type": "video/mp4",
            },
        }

    monkeypatch.setattr(
        routes,
        "create_presigned_upload_post",
        fake_create_presigned_upload_post,
    )

    response = client.post("/jobs")

    assert response.status_code == 201
    assert response.json() == {
        "job_id": job_id,
        "status": "awaiting_upload",
        "upload": {
            "url": "https://test-video-bucket.example.com",
            "fields": {
                "key": (
                    "jobs/"
                    "784112e9-7af6-46d7-9f4f-58b99195dcf5/"
                    "input/source.mp4"
                ),
                "Content-Type": "video/mp4",
            },
        },
    }

    assert job_store.records[job_id] == {
        "status": "awaiting_upload",
        "modal_call_id": None,
        "clips": None,
        "error": None,
    }


def test_submit_job_returns_not_found_for_unknown_job(
    client: TestClient,
) -> None:
    response = client.post("/jobs/unknown-job/submit")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job does not exist",
    }


def test_submit_job_rejects_already_submitted_job(
    client: TestClient,
    job_store: FakeJobStore,
) -> None:
    job_store.records["job-1"] = {
        "status": "processing",
        "modal_call_id": "fc-existing",
        "clips": None,
        "error": None,
    }

    response = client.post("/jobs/job-1/submit")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Job has already been submitted",
    }


def test_submit_job_rejects_missing_input_video(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    job_store: FakeJobStore,
) -> None:
    job_store.records["job-1"] = {
        "status": "awaiting_upload",
        "modal_call_id": None,
        "clips": None,
        "error": None,
    }

    monkeypatch.setattr(
        routes,
        "check_input_video_exists",
        lambda **_: False,
    )

    response = client.post("/jobs/job-1/submit")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "The input video has not been uploaded",
    }

    assert job_store.records["job-1"]["status"] == "awaiting_upload"


def test_submit_job_spawns_worker_and_updates_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    job_store: FakeJobStore,
    pipeline_worker: FakePipelineWorker,
) -> None:
    job_store.records["job-1"] = {
        "status": "awaiting_upload",
        "modal_call_id": None,
        "clips": None,
        "error": None,
    }

    monkeypatch.setattr(
        routes,
        "check_input_video_exists",
        lambda **_: True,
    )

    response = client.post("/jobs/job-1/submit")

    assert response.status_code == 202
    assert response.json() == {
        "job_id": "job-1",
        "status": "processing",
    }

    assert pipeline_worker.spawned_job_ids == [
        "job-1",
    ]

    assert job_store.records["job-1"] == {
        "status": "processing",
        "modal_call_id": "fc-test-call-id",
        "clips": None,
        "error": None,
    }


def test_read_job_status_returns_not_found_for_unknown_job(
    client: TestClient,
) -> None:
    response = client.get("/jobs/unknown-job")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job does not exist",
    }


def test_read_job_status_returns_awaiting_upload_without_polling_modal(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    job_store: FakeJobStore,
) -> None:
    job_store.records["job-1"] = {
        "status": "awaiting_upload",
        "modal_call_id": None,
        "clips": None,
        "error": None,
    }

    class UnexpectedFunctionCall:
        @staticmethod
        def from_id(call_id: str) -> None:
            raise AssertionError(
                f"Modal should not be polled for call {call_id}"
            )

    monkeypatch.setattr(
        routes.modal,
        "FunctionCall",
        UnexpectedFunctionCall,
    )

    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-1",
        "status": "awaiting_upload",
        "clips": [],
        "error": None,
    }


def test_read_job_status_returns_processing_when_worker_is_running(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    job_store: FakeJobStore,
) -> None:
    processing_record = {
        "status": "processing",
        "modal_call_id": "fc-running",
        "clips": None,
        "error": None,
    }
    job_store.records["job-1"] = processing_record

    recovered_function_call = FakeRecoveredFunctionCall(
        exception=TimeoutError(),
    )
    requested_call_ids = _install_recovered_function_call(
        monkeypatch,
        recovered_function_call,
    )

    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-1",
        "status": "processing",
        "clips": [],
        "error": None,
    }

    assert requested_call_ids == [
        "fc-running",
    ]
    assert recovered_function_call.get_timeouts == [
        0,
    ]
    assert job_store.records["job-1"] == processing_record


def test_read_job_status_caches_completed_result_and_returns_download_urls(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    job_store: FakeJobStore,
) -> None:
    job_store.records["job-1"] = {
        "status": "processing",
        "modal_call_id": "fc-completed",
        "clips": None,
        "error": None,
    }

    stable_clips = [
        {
            "clip_id": "attempt_000",
            "start_time_seconds": 12.5,
            "end_time_seconds": 20.0,
            "object_key": (
                "jobs/job-1/generated-clips/attempt_000.mp4"
            ),
        },
    ]

    recovered_function_call = FakeRecoveredFunctionCall(
        result={
            "job_id": "job-1",
            "clips": stable_clips,
        },
    )
    requested_call_ids = _install_recovered_function_call(
        monkeypatch,
        recovered_function_call,
    )

    def fake_create_download_url(
        *,
        s3_client: object,
        bucket_name: str,
        generated_clip_object_key: str,
    ) -> str:
        assert s3_client is not None
        assert bucket_name == "test-video-bucket"
        assert generated_clip_object_key == (
            "jobs/job-1/generated-clips/attempt_000.mp4"
        )

        return "https://downloads.example.com/attempt_000.mp4"

    monkeypatch.setattr(
        routes,
        "create_presigned_generated_clip_download_url",
        fake_create_download_url,
    )

    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-1",
        "status": "completed",
        "clips": [
            {
                "clip_id": "attempt_000",
                "start_time_seconds": 12.5,
                "end_time_seconds": 20.0,
                "download_url": (
                    "https://downloads.example.com/"
                    "attempt_000.mp4"
                ),
            },
        ],
        "error": None,
    }

    assert requested_call_ids == [
        "fc-completed",
    ]
    assert recovered_function_call.get_timeouts == [
        0,
    ]

    assert job_store.records["job-1"] == {
        "status": "completed",
        "modal_call_id": "fc-completed",
        "clips": stable_clips,
        "error": None,
    }
    assert "download_url" not in stable_clips[0]


def test_read_completed_job_generates_fresh_download_urls(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    job_store: FakeJobStore,
) -> None:
    stable_clips = [
        {
            "clip_id": "attempt_000",
            "start_time_seconds": 1.0,
            "end_time_seconds": 8.0,
            "object_key": (
                "jobs/job-1/generated-clips/attempt_000.mp4"
            ),
        },
    ]

    job_store.records["job-1"] = {
        "status": "completed",
        "modal_call_id": "fc-completed",
        "clips": stable_clips,
        "error": None,
    }

    generated_url_count = 0

    def fake_create_download_url(
        **_: object,
    ) -> str:
        nonlocal generated_url_count

        generated_url_count += 1

        return (
            "https://downloads.example.com/"
            f"attempt_000.mp4?version={generated_url_count}"
        )

    monkeypatch.setattr(
        routes,
        "create_presigned_generated_clip_download_url",
        fake_create_download_url,
    )

    first_response = client.get("/jobs/job-1")
    second_response = client.get("/jobs/job-1")

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert first_response.json()["clips"][0]["download_url"] == (
        "https://downloads.example.com/"
        "attempt_000.mp4?version=1"
    )
    assert second_response.json()["clips"][0]["download_url"] == (
        "https://downloads.example.com/"
        "attempt_000.mp4?version=2"
    )

    assert job_store.records["job-1"]["clips"] == stable_clips
    assert "download_url" not in stable_clips[0]


def test_read_job_status_treats_no_attempt_result_as_completed(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    job_store: FakeJobStore,
) -> None:
    job_store.records["job-1"] = {
        "status": "processing",
        "modal_call_id": "fc-no-attempts",
        "clips": None,
        "error": None,
    }

    recovered_function_call = FakeRecoveredFunctionCall(
        result={
            "job_id": "job-1",
            "clips": [],
        },
    )
    _install_recovered_function_call(
        monkeypatch,
        recovered_function_call,
    )

    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-1",
        "status": "completed",
        "clips": [],
        "error": None,
    }

    assert job_store.records["job-1"] == {
        "status": "completed",
        "modal_call_id": "fc-no-attempts",
        "clips": [],
        "error": None,
    }


def test_read_job_status_caches_safe_worker_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    job_store: FakeJobStore,
) -> None:
    job_store.records["job-1"] = {
        "status": "processing",
        "modal_call_id": "fc-failed",
        "clips": None,
        "error": None,
    }

    recovered_function_call = FakeRecoveredFunctionCall(
        exception=RuntimeError(
            "Internal information that must not reach the client"
        ),
    )
    requested_call_ids = _install_recovered_function_call(
        monkeypatch,
        recovered_function_call,
    )

    first_response = client.get("/jobs/job-1")
    second_response = client.get("/jobs/job-1")

    expected_response = {
        "job_id": "job-1",
        "status": "failed",
        "clips": [],
        "error": "Video processing failed",
    }

    assert first_response.status_code == 200
    assert first_response.json() == expected_response

    assert second_response.status_code == 200
    assert second_response.json() == expected_response

    assert requested_call_ids == [
        "fc-failed",
    ]
    assert recovered_function_call.get_timeouts == [
        0,
    ]

    assert job_store.records["job-1"] == {
        "status": "failed",
        "modal_call_id": "fc-failed",
        "clips": None,
        "error": "Video processing failed",
    }
