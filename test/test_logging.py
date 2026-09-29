"""Logging control tests"""

import logging
import subprocess
import sys

import pytest

import cmdstanpy


@pytest.mark.parametrize(
    "level", [logging.NOTSET, logging.DEBUG, logging.WARNING, logging.ERROR]
)
def test_logger_initialization_preserves_level(level: int) -> None:
    # A fresh process avoids pytest's logging handlers and the cached logger.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import logging
import cmdstanpy

logger = logging.getLogger('cmdstanpy')
logger.setLevel({level})
logger = cmdstanpy.utils.get_logger()
print(logger.level)
logger.warning('warning message')
logger.error('error message')
""",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    expected_level = logging.DEBUG if level == logging.NOTSET else level
    assert int(result.stdout) == expected_level
    assert ('warning message' in result.stderr) == (level <= logging.WARNING)
    assert 'error message' in result.stderr


def test_disable_logging(caplog: pytest.LogCaptureFixture) -> None:
    logger = cmdstanpy.utils.logging.get_logger()

    with caplog.at_level(logging.INFO, logger="cmdstanpy"):
        logger.info("before")
    assert any("before" in m for m in caplog.messages)

    caplog.clear()
    cmdstanpy.disable_logging()

    with caplog.at_level(logging.INFO, logger="cmdstanpy"):
        logger.info("after")

    assert not caplog.messages
    logger.disabled = False


def test_disable_logging_context_manager(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = cmdstanpy.utils.logging.get_logger()

    with caplog.at_level(logging.INFO, logger="cmdstanpy"):
        logger.info("before")
    assert any("before" in m for m in caplog.messages)

    caplog.clear()
    with cmdstanpy.disable_logging():
        with caplog.at_level(logging.INFO, logger="cmdstanpy"):
            logger.info("inside context manager")

    assert not caplog.messages

    with caplog.at_level(logging.INFO, logger="cmdstanpy"):
        logger.info("after")

    assert any("after" in m for m in caplog.messages)
    logger.disabled = False


def test_disable_logging_context_manager_nested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = cmdstanpy.utils.logging.get_logger()

    with caplog.at_level(logging.INFO, logger="cmdstanpy"):
        logger.info("before")
    assert any("before" in m for m in caplog.messages)

    caplog.clear()
    with cmdstanpy.disable_logging():
        with cmdstanpy.enable_logging():
            with caplog.at_level(logging.INFO, logger="cmdstanpy"):
                logger.info("inside context manager")

    assert any("inside context manager" in m for m in caplog.messages)

    caplog.clear()
    with cmdstanpy.enable_logging():
        with cmdstanpy.disable_logging():
            with caplog.at_level(logging.INFO, logger="cmdstanpy"):
                logger.info("inside context manager")

    assert not caplog.messages
    logger.disabled = False
