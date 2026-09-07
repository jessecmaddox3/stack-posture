"""Rotating file log plus stderr."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(path: Path, level: int = logging.INFO) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("posture")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    rotating = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    rotating.setFormatter(fmt)
    logger.addHandler(rotating)

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    logger.propagate = False
    return logger
