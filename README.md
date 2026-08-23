# AI Judo Coach System

AI Judo Coach is a Python video-processing system that identifies sections of judo footage containing potential throw attempts and extracts them as separate MP4 clips.

The system processes an input video through pose estimation, player tracking, pose-sequence construction, and an LSTM clip-classification model.

## Pipeline Overview

```text
Input video
→ remove audio and normalise frame rate
→ divide video into overlapping 7-second windows
→ YOLO11x-pose estimation and ByteTrack tracking
→ global two-player pose assignment
→ interpolation and pose-sequence quality assessment
→ construct [210, 68] classifier input
→ classify each window
→ merge overlapping positive windows
→ cap and rank final intervals
→ extract final MP4 clips
```

The clip-classification model predicts:

- `0` — no throw attempt
- `1` — throw attempt

Each classifier input contains 210 frames and 68 pose-coordinate features per frame.

## Project Structure

```text
ai-judo-coach-system/
├── .github/
│   └── workflows/
│       └── tests.yml
├── src/
│   └── ai_judo_coach/
│       ├── attempt_clip_generation/
│       ├── exceptions/
│       ├── inference/
│       ├── pipeline/
│       ├── schemas/
│       ├── video/
│       ├── __init__.py
│       ├── config.py
│       └── main.py
├── tests/
├── weights/
│   ├── judo_clipper_classification_model_v1/
│   │   ├── model_metadata.yaml
│   │   └── model_weights.pt
│   └── ultralytics_v11x_yolo/
│       └── yolo11x-pose.pt
├── wheels/
│   └── v1_clip_classification_model-0.1.0-py3-none-any.whl
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requirements

### Python

The project requires Python 3.12:

```text
Python >=3.12,<3.13
```

Verify the installed version:

```bash
python3 --version
```

### FFmpeg and FFprobe

The application uses `ffmpeg-python`, which requires the FFmpeg and FFprobe executables to be installed separately and available on the system `PATH`.

#### Ubuntu or Debian

```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS

```bash
brew install ffmpeg
```

Verify the installation:

```bash
ffmpeg -version
ffprobe -version
```

Installing the `ffmpeg-python` package does not install the FFmpeg system executable.

### Model Artefacts

The following model artefacts must be present:

```text
weights/judo_clipper_classification_model_v1/model_metadata.yaml
weights/judo_clipper_classification_model_v1/model_weights.pt
weights/ultralytics_v11x_yolo/yolo11x-pose.pt
```

The private clip-classification package wheel must also be present:

```text
wheels/v1_clip_classification_model-0.1.0-py3-none-any.whl
```

## Development Installation

Create and activate a Python 3.12 virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python3 -m pip install --upgrade pip
```

Install the private clip-classification package:

```bash
python3 -m pip install \
  ./wheels/v1_clip_classification_model-0.1.0-py3-none-any.whl
```

Install the application and its API and development dependencies in editable mode:

```bash
python3 -m pip install -e ".[api,dev]"
```

Editable mode allows changes under `src/` to take effect without reinstalling the application.

## Production Installation

Create and activate the production virtual environment, then run:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The production `requirements.txt` installs:

1. The private clip-classification wheel.
2. The application and its API dependencies.

Do not combine the production requirements installation with a separate editable installation of the same application.

## Dependency Management

`pyproject.toml` is the authoritative source for application dependencies.

Core runtime dependencies include:

- PyTorch
- Ultralytics
- NumPy
- Decord
- OpenCV
- `ffmpeg-python`
- the private v1 clip-classification package

Development dependencies are declared under the `dev` optional dependency group.

API dependencies are declared under the `api` optional dependency group.

Avoid generating the main dependency configuration directly from `pip freeze`, as this includes transitive dependencies and machine-specific CUDA packages.

## Model Device Configuration

Model devices are configured in:

```text
src/ai_judo_coach/config.py
```

Example:

```python
CLASSIFIER_DEVICE: str = "auto"
YOLO_DEVICE: str = "auto"
```

When `auto` is selected:

- CUDA is used when it is available.
- CPU is used otherwise.

The YOLO and clip-classification models are loaded once and reused while processing the windows from one input video.

## Clip-Classification Model

The released v1 classifier is a two-layer bidirectional LSTM with a feed-forward classification head.

Its production release is stored in:

```text
weights/judo_clipper_classification_model_v1/
├── model_metadata.yaml
└── model_weights.pt
```

The metadata defines:

- model architecture;
- expected input shape;
- input dtype;
- label mapping;
- non-finite-value policy;
- classification threshold.

The production classification threshold is:

```text
0.55
```

The model returns a raw logit. The installed inference package applies sigmoid and the frozen classification threshold.

The classifier can be loaded directly with:

```python
from v1_clip_classification_model.inference import (
    JudoClipClassifier,
)


classifier = JudoClipClassifier.from_release(
    release_directory=(
        "weights/judo_clipper_classification_model_v1"
    ),
    device="auto",
)
```

One complete classifier input can then be processed with:

```python
result = classifier.predict(model_input)
```

The result includes:

```python
result.logit
result.probability
result.prediction
result.class_name
result.threshold
```

## Attempt-Interval Selection

Initial classifier windows are seven seconds long and overlap because the pipeline advances by a shorter stride.

Positive windows are post-processed using the following policy:

1. Sort positive windows by start time.
2. Merge overlapping or touching windows.
3. Retain the highest classifier score contributing to each merged region.
4. Cap long regions around their highest-scoring source window.
5. Retain at most the configured maximum number of intervals.
6. Restore chronological order.
7. Extract one MP4 for each final interval.

The relevant limits are configured in `config.py`, including:

```python
MAX_GENERATED_ATTEMPT_CLIP_DURATION_SEC: float = 14.0
MAX_GENERATED_ATTEMPT_CLIPS: int = 10
```

These limits are product-level safeguards rather than learned model parameters.

## Generated Files

Each processing job should receive its own temporary output directory.

A job directory is expected to contain files similar to:

```text
<job-directory>/
├── input_cleanse/
│   └── cleansed_input.mp4
└── generated_clips/
    ├── attempt_000.mp4
    ├── attempt_001.mp4
    └── attempt_002.mp4
```

Generated clips use zero-padded identifiers so filenames sort correctly.

The pipeline returns internal `GeneratedAttemptClip` objects containing:

- clip ID;
- start time;
- end time;
- backend file path.

Backend file paths must not be exposed directly to frontend clients. A future API layer should serve or upload the MP4s and return frontend-facing URLs.

If no throw attempts are detected, the pipeline returns an empty list.

## Temporary File Ownership

The API or higher-level job handler should create and own the temporary job directory.

The directory must remain available until generated MP4s have been served or uploaded. It should then be deleted according to an explicit cleanup policy.

Do not create a `TemporaryDirectory` inside the pipeline and return paths from it, because those files would be deleted when the pipeline function returns.

## Running Tests

Install the development dependencies first:

```bash
python3 -m pip install \
  ./wheels/v1_clip_classification_model-0.1.0-py3-none-any.whl

python3 -m pip install -e ".[api,dev]"
```

Run the tests:

```bash
pytest
```

GitHub Actions installs FFmpeg, the private classifier wheel, and the application before running the test suite.

## Packaging

Build the application package with:

```bash
python3 -m build
```

Generated distributions are written to:

```text
dist/
```

The `build/`, `dist/`, and `*.egg-info/` directories are generated artefacts and should not be committed.

## Deployment Notes

A deployment requires:

1. The installed `ai-judo-coach-system` package.
2. The installed private clip-classification package.
3. The classifier release bundle.
4. The YOLO pose-model weights.
5. FFmpeg and FFprobe.
6. A location for temporary processing files.
7. Persistent or object storage if generated clips must remain available.

For a single backend instance, generated MP4s may initially be served from local persistent storage.

For a multi-instance deployment, generated clips should be uploaded to shared object storage and returned to the frontend using public or signed URLs.

The processing workload includes YOLO, PyTorch, tracking, and video encoding. A deployment environment must therefore provide sufficient memory, execution time, and CPU or GPU resources.

## Known Limitations

- The clip classifier makes one prediction for each complete seven-second window.
- It does not identify exact frame-level throw boundaries.
- Merged output intervals represent connected positive classifier windows rather than precise attempt durations.
- Closely spaced throw attempts may be merged into one output interval.
- Performance depends on the quality of pose estimation, tracking, player assignment, interpolation, and sequence construction.
- Generated output is currently limited by configured clip-count and clip-duration safeguards.
- Model weights are separate binary artefacts and must be included in the deployment process.

