"""Logging control tests"""

import logging

import cmdstanpy


def test_disable_logging(caplog):
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


def test_disable_logging_context_manager(caplog):
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


def test_disable_logging_context_manager_nested(caplog):
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
