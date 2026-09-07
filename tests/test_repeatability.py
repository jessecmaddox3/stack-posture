"""Tests for scripts/repeatability.py's pure summarising logic.

The script itself needs a live Gemini API key and the network, so it is not
run end to end here; running it against a real key is left to the maintainer
to do manually. What matters for a
committed harness is that its reporting logic, especially the all-failed
paths, works before it is ever pointed at a real key. Those paths are the
single most likely first-run outcome: the configured model ID
(gemini-3.5-flash-lite) is unverified, and a 404 there fails every call.

scripts/ is not on the test path, so summarise() is loaded directly from the
script file rather than restructuring the project layout for one script.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "repeatability.py"


def _load_summarise():
    spec = importlib.util.spec_from_file_location("repeatability_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarise


summarise = _load_summarise()


def _success(run: int, rating: str) -> dict:
    return {"run": run, "rating": rating, "issue": "none",
            "details": "looks fine", "tip": "none"}


def _failure(run: int, error: str) -> dict:
    return {"run": run, "error": error}


def test_all_runs_failed_model_unavailable_gives_no_repeatability_claim():
    results = [_failure(i, "MODEL_UNAVAILABLE") for i in range(1, 4)]

    exit_code, lines = summarise(results, "gemini-3.5-flash-lite")

    assert exit_code == 1
    text = "\n".join(lines)
    assert "gemini-3.5-flash-lite" in text
    assert "gemini_model" in text
    assert "No repeatability conclusion is possible from zero successful runs." in text
    assert "VERDICT" not in text


def test_all_runs_failed_api_error_does_not_crash_and_makes_no_claim():
    # Before the extraction this raised IndexError on most_common(1)[0]
    # against an empty Counter, and would also have divided by zero.
    results = [_failure(i, "API_ERROR") for i in range(1, 4)]

    exit_code, lines = summarise(results, "gemini-3.5-flash-lite")

    assert exit_code == 1
    text = "\n".join(lines)
    assert "No repeatability conclusion is possible from zero successful runs." in text
    assert "VERDICT" not in text
    # This is an API failure, not a bad model id: no model-update instruction.
    assert "gemini_model" not in text


def test_all_runs_identical_does_not_overclaim_from_one_frame():
    results = [_success(i, "GOOD") for i in range(1, 11)]

    exit_code, lines = summarise(results, "gemini-3.5-flash-lite")

    assert exit_code == 0
    text = "\n".join(lines)
    assert "identical on all 10 runs of this ONE frame" in text
    assert "not sufficient" in text
    assert "run this again on several borderline frames" in text


def test_split_result_reports_modal_share_and_flags_not_repeatable():
    results = [_success(i, "GOOD") for i in range(1, 8)] + \
        [_success(i, "DECENT") for i in range(8, 11)]

    exit_code, lines = summarise(results, "gemini-3.5-flash-lite")

    assert exit_code == 0
    text = "\n".join(lines)
    assert "NOT repeatable" in text
    assert "70%" in text


def test_mixed_success_and_failure_uses_successful_runs_as_denominator():
    # 3 GOOD + 1 DECENT among 4 successful runs, plus 6 failed attempts.
    # The modal share must be 3/4 = 75%. A bug that divides by the number of
    # attempts (10) instead of successful runs (4) would report 30%.
    results = (
        [_success(1, "GOOD"), _success(2, "GOOD"), _success(3, "GOOD"),
         _success(4, "DECENT")]
        + [_failure(i, "API_ERROR") for i in range(5, 11)]
    )

    exit_code, lines = summarise(results, "gemini-3.5-flash-lite")

    assert exit_code == 0
    text = "\n".join(lines)
    assert "Ratings across 4 successful runs" in text
    assert "75%" in text
    assert "30%" not in text
