"""
CmdStanPy logging
"""
import functools
import logging


@functools.lru_cache(maxsize=None)
def get_logger() -> logging.Logger:
    """cmdstanpy logger"""
    logger = logging.getLogger('cmdstanpy')
    if len(logger.handlers) == 0:
        # send all messages to handlers
        logger.setLevel(logging.DEBUG)
        # add a default handler to the logger to INFO and higher
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                "%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger
