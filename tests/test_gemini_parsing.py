import json
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from posture.config import Config
from posture.gemini import RESPONSE_SCHEMA, CircuitBreaker, GeminiClient, build_prompt, parse_verdict
from posture.types import Outcome, Rating


# --- parsing ---

def good_payload(**overrides):
    payload = {"person_present": True, "rating": "POOR", "issue": "forward head",
               "details": "Head is well ahead of the shoulders.",
               "tip": "Tuck your chin back."}
    payload.update(overrides)
    return json.dumps(payload)


def test_parses_a_clean_json_verdict():
    verdict = parse_verdict(good_payload())
    assert verdict.rating is Rating.POOR
    assert verdict.issue == "forward head"
    assert verdict.person_present is True


def test_parses_json_wrapped_in_markdown_fences():
    # The exact failure mode that made v2 silently record UNKNOWN rows.
    text = f"```json\n{good_payload()}\n```"
    assert parse_verdict(text).rating is Rating.POOR


def test_parses_json_with_leading_prose():
    text = f"Here is my assessment:\n{good_payload()}"
    assert parse_verdict(text).rating is Rating.POOR


def test_absent_person_yields_a_verdict_with_no_rating():
    verdict = parse_verdict(json.dumps({
        "person_present": False, "rating": "GOOD", "issue": "", "details": "", "tip": ""}))
    assert verdict.person_present is False
    assert verdict.rating is None


def test_unknown_rating_string_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        parse_verdict(good_payload(rating="TERRIBLE"))


def test_malformed_json_raises():
    with pytest.raises(ValueError):
        parse_verdict("not json at all")


def test_truncated_json_raises():
    with pytest.raises(ValueError):
        parse_verdict('{"person_present": true, "rating": "PO')


def test_empty_response_raises():
    with pytest.raises(ValueError):
        parse_verdict("")


def test_missing_required_field_raises():
    with pytest.raises(ValueError):
        parse_verdict(json.dumps({"person_present": True, "rating": "GOOD"}))


def test_lowercase_rating_is_accepted():
    assert parse_verdict(good_payload(rating="poor")).rating is Rating.POOR


@pytest.mark.parametrize("text", ["null", "5", "true", "[1, 2, 3]"])
def test_scalar_or_array_json_raises_rather_than_crashing(text):
    # A bare JSON scalar or array parses fine (valid JSON) but is not a dict,
    # so "field not in payload" would raise an uncaught TypeError instead of
    # the ValueError every other malformed shape produces here.
    with pytest.raises(ValueError):
        parse_verdict(text)


# --- prompt ---

def test_prompt_labels_each_camera_role():
    prompt = build_prompt(["front", "side"])
    assert "front" in prompt.lower() and "side" in prompt.lower()


def test_prompt_does_not_bias_toward_finding_problems():
    # v2 said "Be reasonably strict. Most people have at least slightly poor posture."
    prompt = build_prompt(["front"]).lower()
    assert "strict" not in prompt
    assert "most people" not in prompt


def test_prompt_contains_no_local_measurements():
    # Blind mode: anchoring the model would make the agreement statistic meaningless.
    prompt = build_prompt(["front", "side"]).lower()
    for leak in ("craniovertebral", "cva", "measured", "degrees", "baseline"):
        assert leak not in prompt


# --- circuit breaker ---

def test_circuit_starts_closed():
    assert CircuitBreaker(threshold=3, cooldown_s=900).is_open(now=0.0) is False


def test_circuit_opens_after_the_threshold():
    cb = CircuitBreaker(threshold=3, cooldown_s=900)
    for _ in range(3):
        cb.record_failure(now=0.0)
    assert cb.is_open(now=0.0) is True


def test_circuit_stays_closed_below_the_threshold():
    cb = CircuitBreaker(threshold=3, cooldown_s=900)
    cb.record_failure(now=0.0)
    cb.record_failure(now=0.0)
    assert cb.is_open(now=0.0) is False


def test_success_resets_the_failure_count():
    cb = CircuitBreaker(threshold=3, cooldown_s=900)
    cb.record_failure(now=0.0)
    cb.record_failure(now=0.0)
    cb.record_success()
    cb.record_failure(now=0.0)
    assert cb.is_open(now=0.0) is False


def test_circuit_closes_after_the_cooldown():
    cb = CircuitBreaker(threshold=1, cooldown_s=900)
    cb.record_failure(now=0.0)
    assert cb.is_open(now=500.0) is True
    assert cb.is_open(now=901.0) is False


def test_trip_permanently_ignores_the_cooldown():
    # A retired model will never come back; do not keep retrying it hourly.
    cb = CircuitBreaker(threshold=3, cooldown_s=900)
    cb.trip_permanently()
    assert cb.is_open(now=0.0) is True
    assert cb.is_open(now=100_000.0) is True


# --- client error mapping ---

def make_client(monkeypatch):
    monkeypatch.setattr("posture.gemini.get_api_key", lambda: "test-key")
    with patch("posture.gemini.genai.Client"):
        return GeminiClient(Config(gemini_enabled=True, gemini_model="gemini-3.5-flash-lite"))


def test_missing_api_key_yields_api_error(monkeypatch):
    monkeypatch.setattr("posture.gemini.get_api_key", lambda: None)
    client = GeminiClient(Config(gemini_enabled=True))
    verdict, outcome = client.assess({})
    assert verdict is None
    assert outcome is Outcome.API_ERROR


@pytest.mark.parametrize("enabled", [False, "false", "true", 1])
def test_client_never_reads_keys_without_explicit_boolean_opt_in(monkeypatch, enabled):
    key_reader = MagicMock(side_effect=AssertionError("Keychain must stay untouched"))
    monkeypatch.setattr("posture.gemini.get_api_key", key_reader)
    client = GeminiClient(Config(gemini_enabled=enabled))
    assert client.available is False
    key_reader.assert_not_called()


def test_404_maps_to_model_unavailable_and_trips_permanently(monkeypatch):
    from google.genai import errors
    client = make_client(monkeypatch)
    error = errors.ClientError.__new__(errors.ClientError)
    error.code = 404
    error.message = "model not found"
    client._client.models.generate_content = MagicMock(side_effect=error)
    verdict, outcome = client.assess({"front": b"\xff\xd8fake"})
    assert outcome is Outcome.MODEL_UNAVAILABLE
    assert client.breaker.is_open(now=1e9) is True


def test_server_error_maps_to_api_error_without_permanent_trip(monkeypatch):
    from google.genai import errors
    client = make_client(monkeypatch)
    error = errors.ServerError.__new__(errors.ServerError)
    error.code = 503
    error.message = "unavailable"
    client._client.models.generate_content = MagicMock(side_effect=error)
    verdict, outcome = client.assess({"front": b"\xff\xd8fake"})
    assert outcome is Outcome.API_ERROR
    assert client.breaker.is_open(now=1e9) is False


def test_unparseable_response_maps_to_api_error(monkeypatch):
    client = make_client(monkeypatch)
    response = MagicMock()
    response.text = "I cannot help with that request."
    client._client.models.generate_content = MagicMock(return_value=response)
    verdict, outcome = client.assess({"front": b"\xff\xd8fake"})
    assert verdict is None
    assert outcome is Outcome.API_ERROR


def test_successful_assess_returns_a_verdict(monkeypatch):
    client = make_client(monkeypatch)
    response = MagicMock()
    response.text = good_payload()
    client._client.models.generate_content = MagicMock(return_value=response)
    verdict, outcome = client.assess({"front": b"\xff\xd8fake"})
    assert outcome is None
    assert verdict.rating is Rating.POOR


@pytest.mark.parametrize("text", ["null", "5", "true", "[1, 2, 3]"])
def test_assess_returns_api_error_for_scalar_or_array_json_response(monkeypatch, text):
    # The exact failure the reviewer demonstrated: a scalar/array JSON response
    # reached parse_verdict's "field not in payload" check, raised an uncaught
    # TypeError, and propagated straight out of assess() instead of being
    # reported as (None, API_ERROR) like every other malformed response.
    client = make_client(monkeypatch)
    response = MagicMock()
    response.text = text
    client._client.models.generate_content = MagicMock(return_value=response)
    verdict, outcome = client.assess({"front": b"\xff\xd8fake"})
    assert verdict is None
    assert outcome is Outcome.API_ERROR


def test_empty_frames_maps_to_crop_unavailable_not_api_error(monkeypatch):
    # Task 13's crop step fails closed: an empty jpegs_by_role means every role
    # was refused locally, before the API was ever contacted. Recording that as
    # API_ERROR would be a lie in the stored data and would feed the breaker
    # failures the API never caused.
    client = make_client(monkeypatch)
    verdict, outcome = client.assess({})
    assert verdict is None
    assert outcome is Outcome.CROP_UNAVAILABLE
    assert client.breaker.is_open(now=1e9) is False


# --- preflight ---

def test_preflight_ok_returns_none(monkeypatch):
    client = make_client(monkeypatch)
    client._client.models.generate_content = MagicMock(return_value=MagicMock())
    assert client.preflight() is None


def test_preflight_404_maps_to_model_unavailable_and_trips_permanently(monkeypatch):
    from google.genai import errors
    client = make_client(monkeypatch)
    error = errors.ClientError.__new__(errors.ClientError)
    error.code = 404
    error.message = "model not found"
    client._client.models.generate_content = MagicMock(side_effect=error)
    outcome = client.preflight()
    assert outcome is Outcome.MODEL_UNAVAILABLE
    assert client.breaker.is_open(now=1e9) is True


def test_preflight_server_error_maps_to_api_error_without_permanent_trip(monkeypatch):
    from google.genai import errors
    client = make_client(monkeypatch)
    error = errors.ServerError.__new__(errors.ServerError)
    error.code = 503
    error.message = "unavailable"
    client._client.models.generate_content = MagicMock(side_effect=error)
    outcome = client.preflight()
    assert outcome is Outcome.API_ERROR
    assert client.breaker.is_open(now=1e9) is False


# --- request construction ---

def test_assess_sends_one_image_part_per_role_and_the_full_config(monkeypatch):
    client = make_client(monkeypatch)
    response = MagicMock()
    response.text = good_payload()
    mock_generate_content = MagicMock(return_value=response)
    client._client.models.generate_content = mock_generate_content

    client.assess({"front": b"\xff\xd8front", "side": b"\xff\xd8side"})

    assert mock_generate_content.call_count == 1
    _, kwargs = mock_generate_content.call_args

    contents = kwargs["contents"]
    image_parts = [part for part in contents if isinstance(part, types.Part)]
    assert len(image_parts) == 2  # one Part per role supplied

    joined_text = "\n".join(c for c in contents if isinstance(c, str))
    assert "FRONT" in joined_text and "SIDE" in joined_text

    config = kwargs["config"]
    assert config is not None
    assert config.response_mime_type == "application/json"
    assert config.response_schema == RESPONSE_SCHEMA
    assert config.http_options is not None
    assert config.http_options.timeout is not None
    assert config.http_options.timeout > 0
