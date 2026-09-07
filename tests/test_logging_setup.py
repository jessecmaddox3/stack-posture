from posture.logging_setup import setup_logging


def test_creates_the_log_file_and_writes(tmp_path):
    log_path = tmp_path / "nested" / "posture.log"
    logger = setup_logging(log_path)
    logger.info("hello")
    for handler in logger.handlers:
        handler.flush()
    assert log_path.exists()
    assert "hello" in log_path.read_text()


def test_handler_is_rotating_and_bounded(tmp_path):
    logger = setup_logging(tmp_path / "p.log")
    rotating = [h for h in logger.handlers if hasattr(h, "maxBytes")]
    assert rotating and rotating[0].maxBytes == 2_000_000
    assert rotating[0].backupCount == 3


def test_repeated_setup_does_not_duplicate_handlers(tmp_path):
    setup_logging(tmp_path / "p.log")
    logger = setup_logging(tmp_path / "p.log")
    assert len(logger.handlers) == 2
