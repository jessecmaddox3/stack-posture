import json
import re
from datetime import date, timedelta

import pytest

from posture.dashboard.generate import TEMPLATE_PATH, VENDOR_PATH, build_dashboard_data, generate
from posture.store.db import connect, migrate
from posture.store.queries import CheckRecord, insert_check, save_baseline
from posture.types import Outcome, PostureMetrics, Rating


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "d.db")
    migrate(c)
    yield c
    c.close()


def make_baseline(conn, label="test") -> int:
    """Create a real baselines row and return its id.

    checks.baseline_id is a genuine FOREIGN KEY and connect() sets
    PRAGMA foreign_keys=ON, so a made up integer raises IntegrityError.
    """
    return save_baseline(conn, PostureMetrics(cva_deg=60.0), label).id


def _metrics_for(roles: list[str], cva: float | None,
                 side_angle: bool) -> PostureMetrics:
    """Metrics a check on these camera roles could actually have produced.

    capability names the camera roles a row carries a MEASUREMENT from, not the
    cameras that returned a frame, so a fixture claiming ["front", "side"] while
    holding no side-derived number describes a row the monitor cannot write.
    Deriving the metrics from the roles keeps every capability assertion below
    about something the real pipeline could have stored.

    side_angle=False is the exception, and it is a real one: a side camera that
    returned a frame and detected the user but produced no readable angle. That row
    HAS a side camera in cameras and no side-derived metric, which is precisely
    the case the capability column now distinguishes.
    """
    if "side" in roles and side_angle and cva is None:
        cva = 58.0
    return PostureMetrics(
        shoulder_tilt_deg=2.0 if "front" in roles else None,
        cva_deg=cva if "side" in roles and side_angle else None,
    )


def add(conn, *, day, rating, cva=None, gemini=None, sample=False, issue="rounded",
        cameras=None, baseline_id=None, outcome=Outcome.OK, ts=None,
        api_error=None, side_angle=True):
    # A cva reading implies a side camera contributed, unless the caller is
    # explicitly describing a different setup.
    roles = cameras if cameras is not None else (
        ["front", "side"] if cva is not None else ["front"])
    insert_check(conn, CheckRecord(
        ts=ts or f"{day}T10:00:00", date=day, outcome=outcome, baseline_id=baseline_id,
        cameras=roles, metrics=_metrics_for(roles, cva, side_angle),
        local_rating=rating, local_driver=None, gemini_rating=gemini,
        gemini_issue=issue, gemini_details="d", gemini_tip="t",
        gemini_model="gemini-3.5-flash-lite" if gemini else None,
        is_comparison_sample=sample, api_error=api_error))


def test_today_score_is_none_when_there_is_no_data_for_today(conn):
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    add(conn, day=yesterday, rating=Rating.GOOD)
    data = build_dashboard_data(conn)
    # v2 showed yesterday's number labelled "Today's Score" (spec M6).
    assert data["today_score"] is None


def test_today_score_reflects_today(conn):
    add(conn, day=date.today().isoformat(), rating=Rating.GOOD)
    assert build_dashboard_data(conn)["today_score"] == 100.0


def test_trend_compares_the_last_seven_days_to_the_seven_before(conn):
    # 14 days: the older 7 are POOR, the newer 7 are GOOD, so the trend is +80.
    for offset in range(14):
        day = (date.today() - timedelta(days=offset)).isoformat()
        add(conn, day=day, rating=Rating.GOOD if offset < 7 else Rating.POOR)
    assert build_dashboard_data(conn)["trend"] == pytest.approx(80.0)


def test_trend_is_none_without_enough_history(conn):
    add(conn, day=date.today().isoformat(), rating=Rating.GOOD)
    data = build_dashboard_data(conn)
    assert data["trend"] is None
    assert "at least" in data["trend_note"]


def test_the_trend_window_is_calendar_days_not_rows(conn):
    # Two desk setups per day means two rows per day. Slicing the newest 14 ROWS
    # covers only 7 calendar days, so a genuine improvement across the fortnight
    # collapses to roughly zero. Verified against the real query: 10 days of two
    # setups returns 20 rows whose first 7 span 4 calendar days.
    for offset in range(14):
        day = (date.today() - timedelta(days=offset)).isoformat()
        rating = Rating.GOOD if offset < 7 else Rating.POOR
        add(conn, day=day, rating=rating, cameras=["front", "side"])
        add(conn, day=day, rating=rating, cameras=["front"])
    assert build_dashboard_data(conn)["trend"] == pytest.approx(80.0)


def test_headline_numbers_use_one_capability_and_report_the_rest(conn):
    # A front-only GOOD and a two-camera GOOD are not the same measurement, so
    # averaging them lets a change of desk read as a change of back.
    today = date.today().isoformat()
    for _ in range(3):
        add(conn, day=today, rating=Rating.GOOD, cameras=["front", "side"])
    add(conn, day=today, rating=Rating.POOR, cameras=["front"])
    data = build_dashboard_data(conn)
    assert data["capability"] == "front+side"
    assert data["today_score"] == 100.0   # not 80.0, which is the blended value
    assert data["excluded_checks"] == 1


def test_a_recalibration_suppresses_the_trend_with_a_reason(conn):
    # Scores either side of a recalibration are measured against different
    # references. Printing a number here would be a lie with a decimal point.
    older, newer = make_baseline(conn, "before"), make_baseline(conn, "after")
    for offset in range(14):
        day = (date.today() - timedelta(days=offset)).isoformat()
        add(conn, day=day, rating=Rating.GOOD if offset < 7 else Rating.POOR,
            baseline_id=newer if offset < 7 else older)
    data = build_dashboard_data(conn)
    assert data["trend"] is None
    assert "recalibrated" in data["trend_note"]


def test_one_baseline_across_the_window_still_reports_a_trend(conn):
    # The guard must not swallow every trend. With a single baseline it stays on.
    only = make_baseline(conn)
    for offset in range(14):
        day = (date.today() - timedelta(days=offset)).isoformat()
        add(conn, day=day, rating=Rating.GOOD if offset < 7 else Rating.POOR,
            baseline_id=only)
    data = build_dashboard_data(conn)
    assert data["trend"] == pytest.approx(80.0)
    assert data["trend_note"] is None


def test_cva_series_has_gaps_where_the_side_camera_was_absent(conn):
    # Relative dates, because cva_series is bounded by date now: fixed dates
    # here would quietly fall out of the 60-day window and stop testing
    # anything a couple of months from the day they were written.
    day = lambda back: (date.today() - timedelta(days=back)).isoformat()  # noqa: E731
    add(conn, day=day(3), rating=Rating.GOOD, cva=60.0)
    add(conn, day=day(2), rating=Rating.GOOD, cva=None)
    add(conn, day=day(1), rating=Rating.GOOD, cva=55.0)
    series = build_dashboard_data(conn)["cva"]
    assert len(series) == 2  # the day with no reading is simply absent


def test_cva_series_carries_negative_readings_through_to_the_page(conn, tmp_path):
    """cva_deg is signed now, so a collapsed day is a negative daily average.

    The series is a SQL AVG and the chart's y axis is auto-scaled, neither of
    which assumes a positive range. This pins that: a day below zero must reach
    the payload as a negative number rather than being dropped or clamped.
    """
    add(conn, day=(date.today() - timedelta(days=3)).isoformat(),
        rating=Rating.GOOD, cva=60.0)
    add(conn, day=(date.today() - timedelta(days=2)).isoformat(),
        rating=Rating.POOR, cva=-40.0)
    series = build_dashboard_data(conn)["cva"]
    assert [row["cva"] for row in series] == [60.0, -40.0]
    assert "-40.0" in generate(conn, tmp_path / "out.html").read_text()


def test_agreement_uses_only_comparison_samples(conn):
    add(conn, day="2026-07-29", rating=Rating.GOOD, gemini=Rating.POOR, sample=True)
    add(conn, day="2026-07-29", rating=Rating.POOR, gemini=Rating.POOR, sample=False)
    agreement = build_dashboard_data(conn)["agreement"]
    assert agreement["sampled"] == 1
    assert agreement["compared"] == 1
    assert agreement["matched"] == 0
    # The gated call is out of the statistic entirely, not sitting in the
    # no-verdict bucket: it was never in the comparison set to begin with.
    assert agreement["no_verdict"] == 0


def test_agreement_rate_is_computed(conn):
    add(conn, day="2026-07-29", rating=Rating.GOOD, gemini=Rating.GOOD, sample=True)
    add(conn, day="2026-07-29", rating=Rating.POOR, gemini=Rating.POOR, sample=True)
    add(conn, day="2026-07-29", rating=Rating.GOOD, gemini=Rating.POOR, sample=True)
    agreement = build_dashboard_data(conn)["agreement"]
    assert agreement["sampled"] == 3
    assert agreement["compared"] == 3
    assert agreement["matched"] == 2
    assert agreement["rate"] == pytest.approx(2 / 3)
    # With nothing failing, the two denominators coincide. That is the case in
    # which F2 was invisible, so pin it rather than leaving it implied.
    assert agreement["no_verdict"] == 0
    assert agreement["no_verdict_reasons"] == []


def test_cva_empty_state_is_present_when_there_is_no_side_camera_data(conn, tmp_path):
    # Front-camera-only users must get an explanation, not a blank chart.
    add(conn, day=date.today().isoformat(), rating=Rating.GOOD, cva=None)
    data = build_dashboard_data(conn)
    assert data["cva"] == []
    html = generate(conn, tmp_path / "out.html").read_text()
    assert "needs a second camera" in html


def test_generated_html_embeds_no_cdn_reference(conn, tmp_path):
    add(conn, day=date.today().isoformat(), rating=Rating.GOOD)
    html = generate(conn, tmp_path / "out.html").read_text()
    assert "cdn.jsdelivr.net" not in html
    assert "new Chart(" in html


def test_generated_html_escapes_model_text(conn, tmp_path):
    add(conn, day=date.today().isoformat(), rating=Rating.POOR,
        gemini=Rating.POOR, sample=True,
        issue='<img src=x onerror="alert(1)">')
    html = generate(conn, tmp_path / "out.html").read_text()
    # Data is embedded as JSON and rendered via textContent, so no live tag appears.
    assert "<img src=x onerror=" not in html


def test_data_payload_is_valid_json(conn, tmp_path):
    add(conn, day=date.today().isoformat(), rating=Rating.GOOD)
    html = generate(conn, tmp_path / "out.html").read_text()
    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";\n", start)
    json.loads(html[start:end])


# --- F1/F2: the amendment computed trend_note, capability, excluded_checks,
# and a distinct all-setups total, but never taught the template to read any
# of them. A suppressed trend rendered as a bare "no data", indistinguishable
# from "not enough history" versus "you just recalibrated", and the page
# stated a capability-filtered count under a label ("checks recorded") that
# claimed to cover everything. These tests pin that the template actually
# references the fields, not just that generate.py computes them (that part
# was already covered and was never the gap).


def test_trend_note_capability_and_excluded_checks_are_wired_into_the_page(conn, tmp_path):
    # This is the reviewer's own diagnostic, run the other way: before the fix,
    # grep -n "trend_note|excluded_checks|DATA.capability" on the generated
    # HTML returned no match at all. These three names must now be read by
    # the rendering script, not merely present in the JSON payload.
    add(conn, day=date.today().isoformat(), rating=Rating.GOOD)
    html = generate(conn, tmp_path / "out.html").read_text()
    assert "DATA.trend_note" in html
    assert "DATA.capability" in html
    assert "DATA.excluded_checks" in html


def test_trend_card_assigns_the_note_through_textcontent(conn, tmp_path):
    # Wiring proof beyond "the name appears somewhere": the note must flow
    # into the DOM through the same trendCard() path that decides whether to
    # show a number or an explanation.
    add(conn, day=date.today().isoformat(), rating=Rating.GOOD)
    html = generate(conn, tmp_path / "out.html").read_text()
    assert "note_el.textContent = note" in html
    assert "trendCard(DATA.trend, DATA.trend_note)" in html


def test_capability_line_states_the_setup_the_window_and_the_excluded_count(conn, tmp_path):
    add(conn, day=date.today().isoformat(), rating=Rating.GOOD)
    html = generate(conn, tmp_path / "out.html").read_text()
    assert "capabilityLine.textContent = capText" in html
    assert "last 60 days" in html
    assert "DATA.excluded_checks > 0" in html
    assert "are not included in the per-setup numbers below" in html


def test_total_checks_labels_do_not_overstate_each_others_scope(conn, tmp_path):
    # 3 checks on the dominant "front" setup, 1 on a different setup, same
    # day. total_checks must count only the dominant setup (3);
    # total_checks_scored must count every setup's SCORED checks (4). The
    # page must reference the all-setups total under its own name rather
    # than reusing total_checks for both claims.
    today = date.today().isoformat()
    for _ in range(3):
        add(conn, day=today, rating=Rating.GOOD, cameras=["front"])
    add(conn, day=today, rating=Rating.POOR, cameras=["front", "side"])
    data = build_dashboard_data(conn)
    assert data["total_checks"] == 3
    assert data["total_checks_scored"] == 4
    html = generate(conn, tmp_path / "out.html").read_text()
    assert "DATA.total_checks_scored" in html
    assert "Checks on this setup" in html


# --- F6: the excluded-checks sentence said those checks "are not included
# above", but the capability line sits ABOVE the cards, and the number just
# above it (total_checks_recorded) DOES include them. Only the per-setup
# cards below exclude them. The wording must be anchored to what it means
# (the per-setup cards), not to page position, which breaks the moment
# anything is reordered.


def test_excluded_checks_wording_is_anchored_to_the_cards_not_page_position(conn, tmp_path):
    today = date.today().isoformat()
    for _ in range(3):
        add(conn, day=today, rating=Rating.GOOD, cameras=["front", "side"])
    add(conn, day=today, rating=Rating.POOR, cameras=["front"])
    html = generate(conn, tmp_path / "out.html").read_text()
    assert "are not included in the per-setup numbers below" in html
    assert "are not included above" not in html


# --- F7: total_checks_scored (formerly total_checks_all) sums daily_scores,
# which filters to outcome = 'OK' AND local_rating IS NOT NULL. A
# PERSON_ABSENT or API_ERROR check is a real event, still listed in the
# Recent checks table (recent_checks is unfiltered), but invisible to that
# sum. Labelling it "checks recorded" was true of no number the page had.
# total_checks_recorded is the genuine count, any outcome; total_checks_
# unscored is the gap between it and total_checks_scored.


def test_total_checks_recorded_includes_outcomes_daily_scores_cannot_see(conn, tmp_path):
    today = date.today().isoformat()
    add(conn, day=today, rating=Rating.GOOD)
    add(conn, day=today, rating=None, outcome=Outcome.PERSON_ABSENT)
    add(conn, day=today, rating=None, outcome=Outcome.API_ERROR)
    data = build_dashboard_data(conn)
    assert data["total_checks_scored"] == 1
    assert data["total_checks_recorded"] == 3
    assert data["total_checks_unscored"] == 2
    html = generate(conn, tmp_path / "out.html").read_text()
    assert "DATA.total_checks_recorded" in html
    assert "DATA.total_checks_unscored" in html


# --- F3: MIN_TREND_DAYS was an unpinned constant. Lowering it to 1 kept the
# whole suite green, because no test constructed a window with exactly 2 days
# on one side and checked that the trend was STILL suppressed there.


def test_trend_requires_at_least_three_days_of_recent_data(conn):
    # Exactly 2 days of recent data, with a full and otherwise-valid previous
    # week, must still suppress the trend. MIN_TREND_DAYS pins this at 3:
    # weakening it to 2 (2 >= 2) or 1 (2 >= 1) would let this print a number.
    for offset in range(2):
        day = (date.today() - timedelta(days=offset)).isoformat()
        add(conn, day=day, rating=Rating.GOOD)
    for offset in range(7, 14):
        day = (date.today() - timedelta(days=offset)).isoformat()
        add(conn, day=day, rating=Rating.POOR)
    data = build_dashboard_data(conn)
    assert data["trend"] is None
    assert "at least" in data["trend_note"]


# --- F4: test_headline_numbers_use_one_capability_and_report_the_rest passed
# under the drop-the-capability-filter mutant for today_score, protected only
# by "front" < "front+side" string sorting lining up with which capability
# was actually dominant. This flips that: the dominant setup ("front", by
# count) sorts FIRST, so a capability-blind mutant's last-write-wins value
# (whatever the alphabetically LAST same-date row happens to be) disagrees
# with the correct one instead of coinciding with it by accident.


def test_today_score_uses_the_dominant_capability_even_when_it_sorts_first(conn):
    today = date.today().isoformat()
    for _ in range(3):
        add(conn, day=today, rating=Rating.POOR, cameras=["front"])
    add(conn, day=today, rating=Rating.GOOD, cameras=["front", "side"])
    data = build_dashboard_data(conn)
    assert data["capability"] == "front"
    assert data["today_score"] == 20.0
    assert data["excluded_checks"] == 1


# --- F5: the "<" escaping in generate() only prevents an HTML PARSER from
# seeing a tag; once the browser's JSON.parse decodes the payload, the
# runtime string value is the raw, unescaped model text again. textContent is
# the only thing standing between that string and the DOM. A real assertion
# that nothing executes would need a browser; this is a static proxy that
# says so rather than pretending otherwise.


def test_template_never_uses_innerhtml_and_helpers_assign_via_textcontent(conn):
    src = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert "innerHTML" not in src
    assert "td.textContent" in src
    assert "span.textContent" in src
    assert "note_el.textContent" in src


# Widened per review: the check above forbids exactly one spelling, so
# outerHTML, insertAdjacentHTML, document.write, setHTMLUnsafe,
# createContextualFragment, srcdoc, eval, and Function would all have passed
# it silently, and it only ever read TEMPLATE_PATH (the pre-inline template),
# never generate()'s actual assembled output, so a sink introduced during
# assembly would have been invisible. Still a static proxy, not a proof: a
# spelling built at runtime (e.g. el['inner'+'HTML']) would still pass. A
# genuine guarantee needs a browser, which this task forbids running.
_HTML_INJECTION_SINKS = (
    "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write",
    "setHTMLUnsafe", "createContextualFragment", "srcdoc", "eval(", "Function(",
)


def test_no_known_html_injection_sink_appears_in_the_template(conn):
    src = TEMPLATE_PATH.read_text(encoding="utf-8")
    for spelling in _HTML_INJECTION_SINKS:
        assert spelling not in src, f"{spelling!r} found in template.html"


def test_no_known_html_injection_sink_appears_in_generate_output(conn, tmp_path):
    # Confirmed separately that the vendored Chart.js contains zero of these
    # spellings except one legitimate internal "Function(" call that has
    # nothing to do with how generate() handles model-derived text, so the
    # inlined vendor bundle is excluded here rather than producing a false
    # alarm about third-party code this project does not control.
    add(conn, day=date.today().isoformat(), rating=Rating.POOR,
        gemini=Rating.POOR, sample=True, issue="<script>alert(1)</script>")
    html = generate(conn, tmp_path / "out.html").read_text()
    vendor_src = VENDOR_PATH.read_text(encoding="utf-8")
    own_output = html.replace(vendor_src, "")
    for spelling in _HTML_INJECTION_SINKS:
        assert spelling not in own_output, f"{spelling!r} found in generated output"


# --- Minor from fix round 2: today_score can be None for two different
# reasons (no checks today at all, versus today's checks all landing on a
# non-dominant setup), and only the second one has any explanation anywhere
# on the page. That is the same "blank with no reason" problem F1 fixed for
# the trend.


def test_today_checked_on_other_setup_is_true_when_todays_data_is_excluded(conn, tmp_path):
    # History makes "front" dominant, but today's only check used "front+side".
    for offset in range(1, 8):
        day = (date.today() - timedelta(days=offset)).isoformat()
        add(conn, day=day, rating=Rating.GOOD, cameras=["front"])
    add(conn, day=date.today().isoformat(), rating=Rating.POOR, cameras=["front", "side"])
    data = build_dashboard_data(conn)
    assert data["capability"] == "front"
    assert data["today_score"] is None
    assert data["today_checked_on_other_setup"] is True
    html = generate(conn, tmp_path / "out.html").read_text()
    assert "DATA.today_checked_on_other_setup" in html
    assert "different camera setup" in html


def test_today_checked_on_other_setup_is_false_when_there_is_simply_no_data(conn):
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    add(conn, day=yesterday, rating=Rating.GOOD)
    data = build_dashboard_data(conn)
    assert data["today_score"] is None
    assert data["today_checked_on_other_setup"] is False


# --- Critical: the page contradicted itself about baselines. The trend card
# blanked itself because a recalibration makes the two halves of the window
# incomparable, and eight pixels below that explanation the line chart drew a
# continuous stroke straight across the same recalibration, while "Average
# score (last 60 days)" blended both rulers into one number. The chart is the
# thing a person actually reads, so drawing the comparison the page has just
# refused to make is worse than either behaviour on its own.


def _fortnight_across_a_recalibration(conn):
    """14 days: the older week on one baseline, the newer week on the next."""
    older, newer = make_baseline(conn, "before"), make_baseline(conn, "after")
    for offset in range(14):
        day = (date.today() - timedelta(days=offset)).isoformat()
        add(conn, day=day, rating=Rating.GOOD if offset < 7 else Rating.POOR,
            baseline_id=newer if offset < 7 else older)
    return older, newer


def test_the_score_chart_splits_into_one_series_per_baseline(conn):
    older, newer = _fortnight_across_a_recalibration(conn)
    data = build_dashboard_data(conn)

    segments = data["daily_segments"]
    assert [s["baseline_id"] for s in segments] == [older, newer]
    # Every series must be padded to the full x axis so the two runs stay on
    # their real dates, and no single series may carry a value on both sides
    # of the boundary: that is what makes Chart.js leave a visible gap there
    # instead of joining two different rulers with one stroke.
    labels = data["daily_labels"]
    assert len(labels) == 14
    for segment in segments:
        assert len(segment["data"]) == len(labels)
        present = [i for i, v in enumerate(segment["data"]) if v is not None]
        assert present == list(range(present[0], present[-1] + 1))
    assert sum(1 for s in segments for v in s["data"] if v is not None) == 14
    # And the boundary itself is named, so the note beside the chart can say
    # where the ruler changed rather than leaving the gap unexplained.
    assert data["baseline_breaks"] == [labels[7]]


def test_the_score_chart_stays_one_series_without_a_recalibration(conn):
    # The split must not fragment ordinary history into meaningless pieces.
    only = make_baseline(conn)
    for offset in range(14):
        day = (date.today() - timedelta(days=offset)).isoformat()
        add(conn, day=day, rating=Rating.GOOD, baseline_id=only)
    data = build_dashboard_data(conn)
    assert len(data["daily_segments"]) == 1
    assert data["baseline_breaks"] == []
    assert all(v is not None for v in data["daily_segments"][0]["data"])


def test_the_older_baselines_history_is_kept_not_dropped(conn):
    # Blanking the chart the way the trend card blanks itself would throw away
    # history that is still worth seeing. The older run must still be plotted.
    older, _ = _fortnight_across_a_recalibration(conn)
    data = build_dashboard_data(conn)
    old_segment = next(s for s in data["daily_segments"] if s["baseline_id"] == older)
    assert [v for v in old_segment["data"] if v is not None] == [20.0] * 7


def test_average_score_is_scoped_to_the_current_baseline_and_says_so(conn):
    # The older week scores 20 and the newer week scores 100. Blending them
    # gives 60.0, a number measured against no ruler that exists.
    _fortnight_across_a_recalibration(conn)
    data = build_dashboard_data(conn)
    assert data["average_score"] == 100.0
    assert data["average_score"] != 60.0
    assert "last 60 days" not in data["average_score_label"]
    assert "current baseline" in data["average_score_label"]
    assert "recalibrat" in data["average_score_note"]


def test_average_score_still_covers_the_window_without_a_recalibration(conn):
    only = make_baseline(conn)
    for offset in range(4):
        day = (date.today() - timedelta(days=offset)).isoformat()
        add(conn, day=day, rating=Rating.GOOD if offset < 2 else Rating.POOR,
            baseline_id=only)
    data = build_dashboard_data(conn)
    assert data["average_score"] == 60.0
    assert data["average_score_label"] == "Average score (last 60 days)"
    assert data["average_score_note"] is None


def test_todays_score_is_not_blended_across_a_same_day_recalibration(conn):
    # Recalibrating at lunchtime puts two rulers on one date. Today's score
    # must be the one measured against the ruler in force now.
    older, newer = make_baseline(conn, "before"), make_baseline(conn, "after")
    today = date.today().isoformat()
    add(conn, day=today, rating=Rating.POOR, baseline_id=older, ts=f"{today}T09:00:00")
    add(conn, day=today, rating=Rating.GOOD, baseline_id=newer, ts=f"{today}T14:00:00")
    data = build_dashboard_data(conn)
    assert data["today_score"] == 100.0  # not 60.0, the blend of both rulers


def _block(html: str, marker: str, closer: str) -> str:
    """The source of one statement in the generated page, marker to closer.

    Whole-file substring checks are too coarse here: the CVA chart already sets
    spanGaps and the break note's wording is a string literal that survives
    being made unreachable, so "the spelling appears somewhere in the HTML"
    passes under mutants that disable exactly the behaviour under test. Slicing
    the one statement first is what makes these assertions discriminating.
    """
    start = html.index(marker)
    end = html.index(closer, start) + len(closer)
    return html[start:end]


def test_the_chart_break_and_scoped_average_are_wired_into_the_page(conn, tmp_path):
    # Payload fields the template never reads are not a fix. This is the same
    # gap F1 closed for trend_note.
    _fortnight_across_a_recalibration(conn)
    html = generate(conn, tmp_path / "out.html").read_text()
    assert "DATA.daily_labels" in html
    assert "DATA.average_score_label" in html
    assert "DATA.average_score_note" in html

    # The score chart must build one dataset per baseline segment and must keep
    # spanGaps off: with it on, Chart.js bridges the nulls between segments and
    # redraws the very continuous line this fix exists to remove.
    chart = _block(html, "new Chart(document.getElementById('scoreChart')", "\n});")
    assert "DATA.daily_segments.map" in chart
    assert "spanGaps: false" in chart
    assert "spanGaps: true" not in chart

    # The gap must be explained where it appears, and only when it is real.
    note = _block(html, "if (DATA.baseline_breaks.length > 0) {", "\n}")
    assert "el.textContent" in note
    assert "DATA.baseline_breaks.join" in note
    assert "not directly comparable" in note
    assert "el.style.display = 'block'" in note


# --- F4: the note under the craniovertebral chart told the reader that a gap
# in the line means the side camera was not connected. That is not what a gap
# means. cva_series selects on json_extract(metrics, '$.cva_deg') IS NOT NULL,
# so a day is absent whenever no side-derived ANGLE was stored, which includes
# a side camera that was plugged in all day and could not see an ear.

def _cva_chart_note(html: str) -> str:
    """The explanatory paragraph under the CVA chart, sliced out and nothing else.

    Asserting against the whole file is not an assertion about this sentence:
    "side camera" appears in the empty-state paragraph directly above it, in the
    JSON payload, and in the vendored bundle, so a whole-file check can pass
    while this paragraph goes on saying something false.
    """
    notes = re.findall(r'<p class="note">(.*?)</p>', html, re.S)
    matching = [n for n in notes if "Measured from the side camera" in n]
    assert len(matching) == 1, f"expected exactly one CVA note, found {len(matching)}"
    return " ".join(matching[0].split())


def test_the_cva_gap_note_does_not_claim_a_gap_means_a_disconnected_camera(conn, tmp_path):
    add(conn, day=date.today().isoformat(), rating=Rating.GOOD, cva=55.0)
    note = _cva_chart_note(generate(conn, tmp_path / "out.html").read_text())

    assert "Days with no line are days the side camera was not connected." not in note
    assert "Days with no line are days with no side-derived reading." in note
    assert "a connected one that produced no readable angle" in note


def test_a_connected_side_camera_with_no_angle_really_does_leave_a_gap(conn, tmp_path):
    """The note's new claim, checked against the query rather than assumed.

    Without this the sentence is just different prose. This builds the exact
    case it describes: a day whose check recorded the side camera as having
    contributed nothing measurable, sitting between two days that did.
    """
    day = lambda back: (date.today() - timedelta(days=back)).isoformat()  # noqa: E731
    add(conn, day=day(3), rating=Rating.GOOD, cva=60.0)
    # Side camera connected and detected, but no angle came out of it.
    add(conn, day=day(2), rating=Rating.GOOD, cameras=["front", "side"],
        side_angle=False)
    add(conn, day=day(1), rating=Rating.GOOD, cva=55.0)

    series = build_dashboard_data(conn)["cva"]
    assert [row["date"] for row in series] == [day(3), day(1)]


# --- F2: the agreement denominator silently dropped every sampled check that
# never produced a verdict, so the rate described only the calls that worked.
# Reproduced by a reviewer: 20 checks sampled for comparison, the page saying
# "4 of 5". api_error reaches the diagnostics bundle but nothing on the page
# hinted at the gap, and the dropped rows are not a random subset: a crop
# refusal tracks poor detection, which tracks the extreme postures the
# comparison exists to check.

def _agreement_block(html: str) -> str:
    """Just the JS that renders the agreement card, not the whole page.

    The agreement text is assembled in the browser, so the only thing a file
    can be asserted against is the source that builds it. Slicing the block out
    is what makes that an assertion about this card: "sampled checks" also
    appears in the payload and in the paragraph under the card, so a whole-file
    check would pass while the card itself went on printing the old sentence.
    """
    start = html.index("const agreementEl")
    return html[start:html.index("const tbody", start)]


def _seed_twenty_samples(conn):
    """Twenty checks in the comparison set. Five produced a verdict, four matched."""
    day = date.today().isoformat()
    for _ in range(4):
        add(conn, day=day, rating=Rating.GOOD, gemini=Rating.GOOD, sample=True)
    add(conn, day=day, rating=Rating.GOOD, gemini=Rating.POOR, sample=True)
    for _ in range(12):
        add(conn, day=day, rating=Rating.GOOD, sample=True, api_error="API_ERROR")
    for _ in range(3):
        add(conn, day=day, rating=Rating.GOOD, sample=True,
            api_error="CROP_UNAVAILABLE")


def test_the_agreement_card_reports_the_full_sampled_denominator(conn):
    _seed_twenty_samples(conn)
    a = build_dashboard_data(conn)["agreement"]

    assert a["sampled"] == 20
    assert a["compared"] == 5
    assert a["matched"] == 4
    assert a["no_verdict"] == 15
    assert a["rate"] == pytest.approx(4 / 5)


def test_the_agreement_card_says_why_the_missing_samples_are_missing(conn):
    _seed_twenty_samples(conn)
    a = build_dashboard_data(conn)["agreement"]
    assert a["no_verdict_reasons"] == [
        {"reason": "API_ERROR", "count": 12},
        {"reason": "CROP_UNAVAILABLE", "count": 3},
    ]


def test_a_sampled_check_with_no_local_rating_is_named_rather_than_dropped(conn):
    """The other way a sampled check leaves the matrix, and it must not vanish either.

    agreement_matrix requires BOTH ratings. A sampled check whose local rating
    is NULL (no baseline metric in common with what was measured) is excluded
    for a reason that has nothing to do with Gemini, so it gets its own bucket
    instead of being folded into an API failure or disappearing.
    """
    day = date.today().isoformat()
    add(conn, day=day, rating=Rating.GOOD, gemini=Rating.GOOD, sample=True)
    add(conn, day=day, rating=None, gemini=Rating.POOR, sample=True)

    a = build_dashboard_data(conn)["agreement"]
    assert a["sampled"] == 2
    assert a["compared"] == 1
    assert a["no_verdict_reasons"] == [{"reason": "NO_LOCAL_RATING", "count": 1}]


def test_the_page_labels_the_agreement_denominator_and_shows_the_shortfall(conn, tmp_path):
    _seed_twenty_samples(conn)
    block = _agreement_block(generate(conn, tmp_path / "out.html").read_text())

    # The old wording claimed the rate was over the sampled checks outright.
    assert "' sampled checks)'" not in block
    assert "sampled checks that produced a verdict" in block
    assert "produced no verdict" in block
    assert "no_verdict_reasons" in block


def test_an_all_failed_comparison_set_shows_no_agreement_rate_at_all(conn):
    """Every sampled check failed, so there is no rate. 0 of 0 is not 100%.

    Dividing here at all would either crash or print a number invented from an
    empty denominator, which is the same overstatement F2 is about, in its most
    extreme form.
    """
    day = date.today().isoformat()
    for _ in range(6):
        add(conn, day=day, rating=Rating.GOOD, sample=True,
            api_error="MODEL_UNAVAILABLE")

    a = build_dashboard_data(conn)["agreement"]
    assert a["sampled"] == 6
    assert a["compared"] == 0
    assert a["rate"] is None
    assert a["no_verdict_reasons"] == [{"reason": "MODEL_UNAVAILABLE", "count": 6}]


def test_the_recent_table_shows_the_instruction_not_the_verdict(conn, tmp_path):
    """The popup says "Reset"; this table must not say "POOR" beside it.

    The stored value is still POOR, because it is persisted and compared. Only
    the wording a reader sees changes, and it has to match the popup: this table
    is where he checks whether a nudge he remembers was fair.
    """
    add(conn, day=date.today().isoformat(), rating=Rating.POOR)
    html = generate(conn, tmp_path / "d.html").read_text()
    mapping = html.split("const RATING_WORDS =")[1].split(";")[0]
    assert "GOOD: 'Stacked'" in mapping
    assert "DECENT: 'Drifting'" in mapping
    assert "POOR: 'Reset'" in mapping
    # And the badge is rendered through it, not from the raw rating.
    body = html.split("function badgeCell")[1].split("function")[0]
    assert "RATING_WORDS[rating]" in body
    assert "span.textContent = rating;" not in body
