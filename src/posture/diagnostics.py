"""A copyable troubleshooting bundle.

Answers "what was the state when it broke", not just "what did the log say".
Everything here is plain text so it can be pasted into a message unchanged.

Redaction is enforced here rather than trusted upstream: the bundle is written
to be shared, and one careless log line should not turn that into a credential
leak.
"""
from __future__ import annotations

import platform
import re
import sqlite3
import sys
from dataclasses import asdict

from posture.config import Config
from posture.store.migrations import SCHEMA_VERSION
from posture.store.queries import active_baseline, coaching_calls_today

LOG_TAIL_LINES = 200

_PATTERNS = (
    # Google API keys are a stable, recognisable shape.
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    re.compile(r"ya29\.[0-9A-Za-z_\-]{5,}"),
    re.compile(r"(?i)bearer\s+[0-9A-Za-z._\-]{8,}"),
    # key-ish name followed by a value, in assignment, TOML, JSON, or log form
    re.compile(r"(?i)([\"\']?[A-Za-z_]*(?:api[_-]?key|token|secret|password)"
               r"[A-Za-z_]*[\"\']?\s*[:=]\s*)[\"\']?[^\s\"\',]+"),
)


def redact(text: str) -> str:
    """Blank anything that looks like a credential, keeping the surrounding line.

    Deliberately errs toward redacting: a bundle missing one value is a small
    problem, a bundle carrying a live key is a large one. The last pattern keeps
    its captured prefix so the reader can still see WHICH setting was blanked.
    """
    out = text
    for pattern in _PATTERNS[:-1]:
        out = pattern.sub("[REDACTED]", out)
    out = _PATTERNS[-1].sub(r"\1[REDACTED]", out)
    return out


# Verified before dispatch against: a Google AIza key, a ya29 OAuth token, a
# Bearer header, and api_key/GEMINI_API_KEY/"apiKey" assignment forms. All five
# redact. Benign lines (timestamps, "model: gemini-3.5-flash-lite",
# "outcome OK: 412", breaker messages) pass through unchanged. Note the JSON form
# leaves a trailing quote, as in '"apiKey": [REDACTED]"'. That is cosmetic and
# expected; do not "fix" it by widening the pattern into surrounding text.


def _baseline_section(conn: sqlite3.Connection) -> list[str]:
    baseline = active_baseline(conn)
    if baseline is None:
        return ["  no baseline: calibration has not been run"]
    lines = [f"  created: {baseline.created_at}", f"  label: {baseline.label}",
             f"  facing_sign: {baseline.facing_sign}"]
    for name, value in asdict(baseline.metrics).items():
        if value is not None:
            lines.append(f"  {name}: {value}")
    return lines


def _history_section(conn: sqlite3.Connection) -> list[str]:
    row = conn.execute(
        "SELECT COUNT(*) AS n, MIN(date) AS first, MAX(date) AS last FROM checks"
    ).fetchone()
    outcomes = conn.execute(
        "SELECT outcome, COUNT(*) AS n FROM checks GROUP BY outcome ORDER BY n DESC"
    ).fetchall()
    errors = conn.execute(
        "SELECT api_error, COUNT(*) AS n FROM checks "
        "WHERE api_error IS NOT NULL GROUP BY api_error ORDER BY n DESC"
    ).fetchall()
    lines = [f"  checks: {row['n']}", f"  range: {row['first']} to {row['last']}"]
    lines += [f"  outcome {r['outcome']}: {r['n']}" for r in outcomes]
    lines += [f"  api_error {r['api_error']}: {r['n']}" for r in errors]
    return lines


def build_diagnostics(conn: sqlite3.Connection, config: Config, log_text: str,
                      camera_indices: tuple[int, ...]) -> str:
    """Assemble the bundle. Pure: callers supply the log text and camera list.

    Taking those as arguments rather than reading them keeps this testable and
    keeps a slow camera probe off whatever thread is building the report.
    """
    sections: list[str] = []
    sections.append("=== stack diagnostics ===")
    sections.append(f"python: {sys.version.split()[0]}")
    sections.append(f"macos: {platform.mac_ver()[0]}")
    sections.append(f"schema: {SCHEMA_VERSION}")

    sections.append("")
    sections.append("cameras:")
    sections.append(f"  discovered: {list(camera_indices) or 'none'}")
    sections.append(f"  configured: {config.camera_indices}")

    sections.append("")
    sections.append("gemini:")
    sections.append(f"  enabled: {config.gemini_enabled}")
    sections.append(f"  model: {config.gemini_model}")
    sections.append(f"  comparison sample rate: {config.comparison_sample_rate}")
    # Both halves of the call volume, so "why is it calling so much" is
    # answerable from the bundle rather than from reading the source. The sample
    # rate alone never explained the gated calls, which is how they went
    # unnoticed in the first place.
    sections.append(f"  coaching budget/day: {config.max_coaching_calls_per_day}")
    sections.append(f"  coaching calls today: {coaching_calls_today(conn)}")

    sections.append("")
    sections.append("baseline:")
    sections += _baseline_section(conn)

    sections.append("")
    sections.append("history:")
    sections += _history_section(conn)

    sections.append("")
    sections.append(f"log (last {LOG_TAIL_LINES} lines):")
    tail = log_text.splitlines()[-LOG_TAIL_LINES:]
    sections += [f"  {line}" for line in tail] or ["  (empty)"]

    return redact("\n".join(sections))


def read_log_tail(path, lines: int = LOG_TAIL_LINES) -> str:
    """Best effort. A missing log is itself worth reporting, not worth raising."""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace")
                          .splitlines()[-lines:])
    except OSError as exc:
        return f"(could not read {path}: {exc})"
