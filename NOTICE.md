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
repository or in release archives. Review the model's upstream terms before
redistributing it. The script checks the expected byte size after download.

## Dashboard asset

`src/posture/dashboard/vendor/chart.umd.min.js` is [Chart.js 4.4.0](https://www.chartjs.org/),
distributed under the MIT License. Its upstream copyright notice is retained in
the file.

## Optional Gemini use

Gemini is disabled by default. If enabled, cropped camera frames are sent to
Google's Gemini API and are governed by Google's applicable service terms and
privacy terms. The local monitor works without a Google account or API key.
