from posture.demo import dashboard_data, main


def test_visual_demo_is_synthetic_and_self_contained(tmp_path, monkeypatch):
    output = tmp_path / "preview.html"
    monkeypatch.setattr("sys.argv", ["demo", "--dashboard", str(output)])
    assert main() == 0
    html = output.read_text()
    assert "Synthetic demo." in html
    assert "__DATA_JSON__" not in html
    assert '<script src=' not in html
    assert '2014-2022 Chart.js Contributors' in html
    assert '2018-2021 Jukka Kurkela' in html
    data = dashboard_data()
    assert data["total_checks_recorded"] == len(data["recent"]) == 14
    assert data["agreement"]["sampled"] == 0
