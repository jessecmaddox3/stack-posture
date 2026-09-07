"""Gemini client. Structured output, explicit error mapping, circuit breaker.

An unavailable model maps to MODEL_UNAVAILABLE, trips the breaker permanently,
and surfaces in the menu bar instead of being mistaken for an absent person.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors, types

from posture.config import Config
from posture.secrets import get_api_key
from posture.types import Outcome, Rating

logger = logging.getLogger("posture")

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "person_present": {"type": "BOOLEAN"},
        "rating": {"type": "STRING", "enum": ["GOOD", "DECENT", "POOR"]},
        "issue": {"type": "STRING"},
        "details": {"type": "STRING"},
        "tip": {"type": "STRING"},
    },
    "required": ["person_present", "rating", "issue", "details", "tip"],
}

_VIEW_NOTES = {
    "front": "Image labelled FRONT is taken head-on, facing the person.",
    "side": "Image labelled SIDE is taken from the person's side, in profile.",
}

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class GeminiVerdict:
    person_present: bool
    rating: Rating | None
    issue: str
    details: str
    tip: str


def build_prompt(roles: list[str]) -> str:
    """Blind assessment prompt. Contains no local measurements by design.

    Deliberately omits an instruction to be strict, which would invite
    over-detection.
    """
    views = "\n".join(_VIEW_NOTES[r] for r in roles if r in _VIEW_NOTES)
    return f"""You are assessing the seated posture of a person at a desk.

{views}

First decide whether a person is visible and seated. If not, set person_present
to false and leave the other fields empty.

If a person is visible, assess their seated posture, considering head and neck
position, shoulder alignment, and the curve of the back.

Judge ONLY the person's body. Ignore the camera entirely: its height, angle,
distance, framing, lens distortion, lighting and image quality are properties of
the desk setup, not of the posture, and none of them are things the person can
correct by sitting differently. A low or high camera angle is not a posture
problem. Never name the camera, the angle, the framing or the image in any
field. If the only thing you would otherwise remark on is the camera, treat the
posture as unremarkable and rate it on the body alone.

Rate as one of:
  GOOD   - upright and well aligned
  DECENT - minor issues, broadly acceptable
  POOR   - clear slouching, hunching, or misalignment

Report what you actually observe. If the posture looks fine, say GOOD.

Fields:
  issue   - a short phrase naming the main problem with their BODY, or "none"
  details - one or two sentences on what their body is doing
  tip     - one specific change to their body they can make right now, or "none"
"""


def parse_verdict(text: str) -> GeminiVerdict:
    """Parse a structured verdict, tolerating fences and prose around the JSON.

    Raises ValueError on anything unparseable. The caller maps that to
    API_ERROR rather than inventing a rating, because v2's habit of silently
    recording UNKNOWN rows polluted the whole trend.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.MULTILINE)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(candidate)
        if not match:
            raise ValueError(f"no JSON object in response: {text[:120]!r}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON: {exc}") from exc

    if not isinstance(payload, dict):
        # Valid JSON, but a bare scalar or array ("null", "5", "true", "[1]").
        # "field not in payload" would raise a TypeError on these instead of
        # the ValueError every other malformed shape produces, and that
        # TypeError would escape assess() uncaught.
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")

    for field in ("person_present", "rating", "issue", "details", "tip"):
        if field not in payload:
            raise ValueError(f"missing required field: {field}")

    present = bool(payload["person_present"])
    rating = None
    if present:
        raw = str(payload["rating"]).strip().upper()
        try:
            rating = Rating(raw)
        except ValueError as exc:
            raise ValueError(f"unrecognised rating: {raw!r}") from exc

    return GeminiVerdict(
        person_present=present, rating=rating,
        issue=str(payload["issue"]), details=str(payload["details"]),
        tip=str(payload["tip"]),
    )


class CircuitBreaker:
    """Stop hammering a failing endpoint, and never retry a retired model."""

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None
        self._permanent = False

    def record_failure(self, now: float) -> None:
        self._failures += 1
        if self._failures >= self._threshold and self._opened_at is None:
            self._opened_at = now
            logger.warning("circuit breaker opened after %d failures", self._failures)

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def trip_permanently(self) -> None:
        self._permanent = True
        logger.error("circuit breaker tripped permanently: the model is unavailable")

    def is_open(self, now: float) -> bool:
        if self._permanent:
            return True
        if self._opened_at is None:
            return False
        if now - self._opened_at >= self._cooldown_s:
            self._failures = 0
            self._opened_at = None
            return False
        return True


class GeminiClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.breaker = CircuitBreaker(config.circuit_breaker_threshold,
                                      config.circuit_breaker_cooldown_s)
        key = get_api_key()
        self._client = genai.Client(api_key=key) if key else None

    @property
    def available(self) -> bool:
        return self._client is not None

    def preflight(self) -> Outcome | None:
        """Verify the configured model responds. Run once at launch.

        This is the check whose absence let v2 die silently for two months.
        """
        if self._client is None:
            return Outcome.API_ERROR
        try:
            self._client.models.generate_content(
                model=self.config.gemini_model,
                contents=["Reply with the single word: ok"],
                config=types.GenerateContentConfig(
                    max_output_tokens=10,
                    http_options=types.HttpOptions(timeout=self.config.api_timeout_s * 1000),
                ),
            )
            return None
        except errors.ClientError as exc:
            if getattr(exc, "code", None) == 404:
                self.breaker.trip_permanently()
                logger.error("model %s is unavailable (404)", self.config.gemini_model)
                return Outcome.MODEL_UNAVAILABLE
            logger.error("preflight client error: %s", exc)
            return Outcome.API_ERROR
        except Exception as exc:
            logger.error("preflight failed: %s", exc)
            return Outcome.API_ERROR

    def assess(self, jpegs_by_role: dict[str, bytes]
               ) -> tuple[GeminiVerdict | None, Outcome | None]:
        """Assess cropped frames. Returns (verdict, outcome); outcome None on success."""
        if self._client is None:
            logger.warning("no API key configured")
            return None, Outcome.API_ERROR
        if not jpegs_by_role:
            # Every role was refused by Task 13's fail-closed crop before the
            # API was ever contacted. This is not an API failure: it must not
            # count toward the breaker, and recording it as API_ERROR would be
            # a lie in the stored data.
            return None, Outcome.CROP_UNAVAILABLE

        roles = sorted(jpegs_by_role)
        contents: list = [build_prompt(roles)]
        for role in roles:
            contents.append(f"Image labelled {role.upper()}:")
            contents.append(types.Part.from_bytes(data=jpegs_by_role[role],
                                                  mime_type="image/jpeg"))

        try:
            response = self._client.models.generate_content(
                model=self.config.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.0,
                    max_output_tokens=500,
                    http_options=types.HttpOptions(timeout=self.config.api_timeout_s * 1000),
                ),
            )
        except errors.ClientError as exc:
            if getattr(exc, "code", None) == 404:
                self.breaker.trip_permanently()
                return None, Outcome.MODEL_UNAVAILABLE
            self.breaker.record_failure(time.monotonic())
            logger.error("gemini client error: %s", exc)
            return None, Outcome.API_ERROR
        except Exception as exc:
            self.breaker.record_failure(time.monotonic())
            logger.error("gemini call failed: %s", exc)
            return None, Outcome.API_ERROR

        try:
            verdict = parse_verdict(response.text or "")
        except ValueError as exc:
            self.breaker.record_failure(time.monotonic())
            logger.error("could not parse gemini response: %s", exc)
            return None, Outcome.API_ERROR

        self.breaker.record_success()
        return verdict, None
