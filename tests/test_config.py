import textwrap

import pytest

from posture.config import Config


def test_defaults_when_no_file(tmp_path):
    cfg = Config.load(tmp_path / "missing.toml")
    assert cfg.sample_interval_s == 30
    assert cfg.sustained_samples == 3
    assert cfg.comparison_sample_rate == 0.15
    assert cfg.gemini_model == "gemini-3.5-flash-lite"


def test_file_overrides_defaults(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent("""
        sample_interval_s = 10
        gemini_model = "gemini-3.5-flash"
    """))
    cfg = Config.load(p)
    assert cfg.sample_interval_s == 10
    assert cfg.gemini_model == "gemini-3.5-flash"
    assert cfg.sustained_samples == 3  # untouched default survives


def test_camera_roles_parse(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[cameras]\nfront = 0\nside = 2\n")
    cfg = Config.load(p)
    assert cfg.camera_indices == {"front": 0, "side": 2}


def test_unknown_keys_are_ignored(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent("""
        sample_interval_s = 15
        this_key_does_not_exist = "surprise"
    """))
    cfg = Config.load(p)
    assert cfg.sample_interval_s == 15
    assert not hasattr(cfg, "this_key_does_not_exist")


def test_unknown_camera_roles_are_rejected(tmp_path):
    """A typo must not silently become a side camera.

    Everything not named "front" is treated as a side view, so `frnot = 0` would
    manufacture craniovertebral and trunk angles out of a front-facing image.
    """
    p = tmp_path / "config.toml"
    p.write_text("[cameras]\nfrnot = 0\n")
    with pytest.raises(ValueError, match="unknown camera role"):
        Config.load(p)


def test_duplicate_camera_indices_are_rejected(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[cameras]\nfront = 0\nside = 0\n")
    with pytest.raises(ValueError, match="same index"):
        Config.load(p)


def test_valid_roles_are_accepted(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[cameras]\nfront = 0\nside = 2\n")
    assert Config.load(p).camera_indices == {"front": 0, "side": 2}


def test_the_daily_coaching_budget_has_a_default_and_is_overridable(tmp_path):
    # The one knob that bounds what the gated path can cost in a day. Without a
    # default in code it would only bound anyone who happened to write it into
    # their TOML, which is nobody.
    assert Config.load(tmp_path / "missing.toml").max_coaching_calls_per_day == 12
    p = tmp_path / "config.toml"
    p.write_text("max_coaching_calls_per_day = 0\n")
    assert Config.load(p).max_coaching_calls_per_day == 0


def test_gemini_is_off_until_someone_turns_it_on():
    """The privacy-relevant default, pinned.

    This app photographs someone at their desk. Whether that becomes an upload
    is their decision, not the default's. It is also the claim the public page
    makes ("nothing leaves your Mac"), and a claim nothing enforces is one edit
    from becoming false. Flipping this back to True must break a named test, not
    pass quietly.
    """
    assert Config().gemini_enabled is False


def test_turning_gemini_on_is_a_single_explicit_line(tmp_path):
    # Opting in must stay easy, or the default becomes a wall rather than a
    # choice. Note gemini_enabled has to sit ABOVE any [table] header or TOML
    # nests it inside that table and it silently does nothing.
    path = tmp_path / "config.toml"
    path.write_text("gemini_enabled = true\n\n[cameras]\nfront = 0\n")
    assert Config.load(path).gemini_enabled is True
