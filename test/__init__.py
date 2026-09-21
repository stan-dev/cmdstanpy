"""Testing utilities for CmdStanPy."""

import contextlib
import logging
import os
import platform
import re
import time
from importlib import reload
from types import ModuleType
from typing import Generator, Optional, Type
from unittest import mock

import pytest

mark_windows_only = pytest.mark.skipif(
    platform.system() != 'Windows', reason='only runs on windows'
)
mark_not_windows = pytest.mark.skipif(
    platform.system() == 'Windows', reason='does not run on windows'
)


def delete_file(path: str, timeout: float = 5.0) -> None:
    """
    Delete a file, retrying briefly on Windows.

    Antivirus software scans a binary the first time it is executed and
    holds it open while doing so, which makes Windows deny the delete with
    ``PermissionError`` until the scan finishes. Defender is disabled on the
    x86_64 CI images but cannot be disabled on the ARM64 ones.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if platform.system() != 'Windows' or time.monotonic() > deadline:
                raise
            time.sleep(0.1)


# pylint: disable=invalid-name
@contextlib.contextmanager
def raises_nested(
    expected_exception: Type[Exception], match: str
) -> Generator[None, None, None]:
    """A version of assertRaisesRegex that checks the full traceback.

    Useful for when an exception is raised from another and you wish to
    inspect the inner exception.
    """
    with pytest.raises(expected_exception) as ctx:
        yield
    exception: Optional[BaseException] = ctx.value
    lines = []
    while exception:
        lines.append(str(exception))
        exception = exception.__cause__
    text = "\n".join(lines)
    assert re.search(match, text), f"pattern `{match}` does not match `{text}`"


@contextlib.contextmanager
def without_import(
    library: str, module: ModuleType
) -> Generator[None, None, None]:
    with mock.patch.dict('sys.modules', {library: None}):
        reload(module)
        yield
    reload(module)


def check_present(
    caplog: pytest.LogCaptureFixture,
    *conditions: tuple,
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
            (
                logger == logger_
                and level == level_
                and message.match(message_)
                if isinstance(message, re.Pattern)
                else message == message_
            )
            for logger_, level_, message_ in caplog.record_tuples
        )
        if not found:
            raise ValueError(f"logs did not contain the record {condition}")
    if clear:
        caplog.clear()
