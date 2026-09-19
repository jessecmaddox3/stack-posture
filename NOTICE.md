# Notices and third-party components

> **TL;DR:** The code in this repository is MIT licensed. Dependencies, the
> downloaded pose model, and any optional Gemini service use have their own
> licenses or terms.

## Runtime dependencies

Stack depends on MediaPipe, OpenCV, NumPy, rumps/PyObjC, and the Google Gen AI
SDK. Their licenses and notices are distributed with their respective packages.
The dependency versions are pinned in `pyproject.toml` and `uv.lock`.

## Pose model

`scripts/fetch_model.sh` downloads the [MediaPipe Pose Landmarker Full
model](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
from Google's public MediaPipe model storage. The model is not included in this
repository or in release archives. The [BlazePose GHUM model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf)
identifies Apache License 2.0 (page 2). Setup downloads the unmodified version-1
Full float16 task, generation `1682642785209422`, verifies its size and SHA-256,
then replaces the old file atomically. A failed download preserves the old file.

- Size: `9398198` bytes.
- SHA-256: `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`.
- Exact URL and downloader: [model_download.py](src/posture/model_download.py).
- Model license: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The promotional header was generated with ChatGPT, without personal reference
images. Its prompt and provenance are in [docs/artwork.md](docs/artwork.md).
The dashboard demo and screenshot contain invented measurements only.

## Dashboard asset

`src/posture/dashboard/vendor/chart.umd.min.js` is [Chart.js 4.4.0](https://www.chartjs.org/),
distributed under the MIT License. Its upstream copyright notice is retained in
the file, including the bundled `@kurkle/color` helper's credit. Full upstream
MIT permission notices are in `src/posture/dashboard/vendor/LICENSES.txt` and
are embedded in generated dashboards, including the standalone demo.

## Optional Gemini use

Gemini is disabled by default. If enabled, cropped camera frames are sent to
Google's Gemini API and are governed by Google's applicable service terms and
privacy terms. The local monitor works without a Google account or API key.
