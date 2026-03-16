from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO", debug: bool = False) -> None:
    """loguru ロガーを設定する。"""
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stdout, format=fmt, level=level, colorize=True)

    if debug:
        logger.add(
            "logs/debug_{time}.log",
            format=fmt,
            level="DEBUG",
            rotation="1 day",
            retention="7 days",
        )
