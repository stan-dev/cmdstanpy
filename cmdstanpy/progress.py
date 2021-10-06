"""
Record tqdm progress bar fail during session
"""

import logging

SHOW_PROGRESS: bool = True


def allow_show_progress() -> bool:
    return SHOW_PROGRESS


def disable_progress(e: Exception) -> None:
    print("DISABLE")
    # pylint: disable=global-statement
    global SHOW_PROGRESS
    if SHOW_PROGRESS:
        logging.getLogger('cmdstanpy').error(
            'Error in progress bar initialization:\n'
            '\t%s\n'
            'Disabling progress bars for this session',
            str(e),
        )
    SHOW_PROGRESS = False
