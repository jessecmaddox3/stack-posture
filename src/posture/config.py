"""Configuration. Defaults in code, overridden by TOML, never holding secrets."""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

DATA_DIR = Path.home() / ".posture_monitor"
CONFIG_PATH = DATA_DIR / "config.toml"

VALID_CAMERA_ROLES = ("front", "side")


@dataclass(frozen=True)
class Config:
    # Cadence. Starting values, expected to be tuned after a week of real use.
    sample_interval_s: int = 30
    record_interval_s: int = 120
    sustained_samples: int = 3          # consecutive bad samples before a nudge
    nudge_cooldown_s: int = 600

    # Gemini. THREE independent switches, because there are two independent
    # reasons to call the API and one master off switch:
    #
    #   gemini_enabled = false        stops everything. No client is created and
    #                                 no frame is ever uploaded. This is the
    #                                 setting for a work laptop.
    #   comparison_sample_rate = 0.0  stops the COMPARISON path only. Nothing is
    #                                 rolled into the agreement statistics.
    #                                 Coaching calls are unaffected: they are a
    #                                 separate path with a separate knob below.
    #   max_coaching_calls_per_day = 0  stops the COACHING path only.
    #
    # Calls per day, at these defaults and an 8-hour desk day (240 record
    # windows of 120s each):
    #
    #   comparison path: 240 * 0.15                  = ~36/day, whatever the posture
    #   coaching path:   at most max_coaching_calls_per_day = 12/day
    #   worst case total:                            = ~48/day, ~1050/month
    #
    # The comparison figure is an expectation and is independent of how the day
    # goes; the coaching figure is a hard ceiling. Before the budget existed the
    # coaching path was unbounded and a bad day cost ~199 calls, ~4400/month.
    gemini_model: str = "gemini-3.5-flash-lite"
    # OFF by default, deliberately. This app photographs someone at their desk,
    # and turning that into an upload is a decision only they can make. Default
    # on would also make the published claim "nothing leaves your Mac" false for
    # anyone who never opened the config, which is most people. Opting in is one
    # line; opting out of something you did not know was happening is not.
    gemini_enabled: bool = False
    gemini_blind: bool = True           # do NOT show it local metrics during comparison
    # Fraction of record windows uploaded purely to measure local-vs-Gemini
    # agreement. Rolled BEFORE the local rating is consulted, so the comparison
    # set stays unbiased; see posture/gating.py.
    comparison_sample_rate: float = 0.15
    # Hard ceiling on gated coaching calls per calendar day, counted across both
    # successes and API failures (both upload a photo and both are billable
    # attempts). 0 disables coaching calls entirely. This is the only bound on
    # the gated path: without it every DECENT or POOR window calls, so the cost
    # scales with how bad the back is, which is exactly backwards.
    max_coaching_calls_per_day: int = 12
    min_seconds_between_api_calls: int = 60
    api_timeout_s: int = 20
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_s: int = 900

    # Capture
    camera_indices: dict[str, int] = field(default_factory=lambda: {"front": 0})
    capture_warmup_frames: int = 3

    # Calibration image store (Task 12). Unbounded by choice; wiped manually.
    store_calibration_frames: bool = False

    model_path: Path = DATA_DIR / "models" / "pose_landmarker_full.task"
    db_path: Path = DATA_DIR / "history.db"
    log_path: Path = DATA_DIR / "posture_monitor.log"

    def validate(self) -> None:
        """Reject ambiguous consent and invalid limits before opening any device/service."""
        for key in ("gemini_enabled", "gemini_blind", "store_calibration_frames"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be true or false, without quotes")
        for key in ("sample_interval_s", "record_interval_s", "nudge_cooldown_s",
                    "min_seconds_between_api_calls", "api_timeout_s",
                    "circuit_breaker_cooldown_s", "comparison_sample_rate"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            if key in ("sample_interval_s", "api_timeout_s") and value == 0:
                raise ValueError(f"{key} must be greater than zero")
        if self.comparison_sample_rate > 1:
            raise ValueError("comparison_sample_rate must be between 0 and 1")
        for key in ("sustained_samples", "max_coaching_calls_per_day",
                    "capture_warmup_frames", "circuit_breaker_threshold"):
            minimum = 1 if key in ("sustained_samples", "circuit_breaker_threshold") else 0
            value = getattr(self, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be a whole number at least {minimum}")
        if not isinstance(self.gemini_model, str) or not self.gemini_model.strip():
            raise ValueError("gemini_model must be a nonempty model name")
        roles = self.camera_indices
        if not isinstance(roles, dict) or not roles:
            raise ValueError("cameras must contain at least one front or side camera")
        unknown = set(roles) - set(VALID_CAMERA_ROLES)
        if unknown:
            raise ValueError(f"unknown camera role(s): {sorted(unknown)}")
        if any(type(index) is not int or index < 0 for index in roles.values()):
            raise ValueError("camera indices must be nonnegative whole numbers")
        if len(set(roles.values())) != len(roles):
            raise ValueError("two camera roles share the same index")

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or CONFIG_PATH
        cfg = cls()
        if not path.exists():
            return cfg
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        cameras = raw.pop("cameras", None)
        known = {f for f in cls.__dataclass_fields__}
        updates = {k: v for k, v in raw.items() if k in known}
        for key in ("model_path", "db_path", "log_path"):
            if key in updates:
                if not isinstance(updates[key], str) or not updates[key].strip():
                    raise ValueError(f"{key} must be a nonempty path string")
                updates[key] = Path(updates[key]).expanduser()
        if cameras is not None:
            updates["camera_indices"] = cameras
        cfg = replace(cfg, **updates)
        cfg.validate()
        return cfg
