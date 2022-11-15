"""Testing utilities for CmdStanPy."""

import contextlib
import logging
import platform
import re
from typing import List, Type
from unittest import mock
from importlib import reload
import pytest


mark_windows_only = pytest.mark.skipif(
    platform.system() != 'Windows', reason='only runs on windows'
)
mark_not_windows = pytest.mark.skipif(
    platform.system() == 'Windows', reason='does not run on windows'
)


# pylint: disable=invalid-name
@contextlib.contextmanager
def raises_nested(expected_exception: Type[Exception], match: str) -> None:
    """A version of assertRaisesRegex that checks the full traceback.

    Useful for when an exception is raised from another and you wish to
    inspect the inner exception.
    """
    with pytest.raises(expected_exception) as ctx:
        yield
    exception: Exception = ctx.value
    lines = []
    while exception:
        lines.append(str(exception))
        exception = exception.__cause__
    text = "\n".join(lines)
    assert re.search(match, text), f"pattern `{match}` does not match `{text}`"


@contextlib.contextmanager
def without_import(library, module):
    with mock.patch.dict('sys.modules', {library: None}):
        reload(module)
        yield
    reload(module)


def check_present(
    caplog: pytest.LogCaptureFixture,
    *conditions: List[tuple],
    clear: bool = True,
) -> None:
    """
    Check that all desired records exist.
    """
    for condition in conditions:
        logger, level, message = condition
        if isinstance(level, str):
            level = getattr(logging, level)
        found = any(
            logger == logger_ and level == level_ and message.match(message_)
            if isinstance(message, re.Pattern)
            else message == message_
            for logger_, level_, message_ in caplog.record_tuples
        )
        if not found:
            raise ValueError(f"logs did not contain the record {condition}")
    if clear:
        caplog.clear()
