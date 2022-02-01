"""The global configuration for the test suite"""
import atexit
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


# after we have run all tests, use git to delete the built files in data/


@pytest.fixture(scope='session', autouse=True)
def cleanup_test_files():

    import cmdstanpy

    # see https://github.com/pytest-dev/pytest/issues/5502
    atexit.unregister(cmdstanpy._cleanup_tmpdir)

    yield

    shutil.rmtree(cmdstanpy._TMPDIR, ignore_errors=True)

    subprocess.Popen(
        ['git', 'clean', '-fX', DATAFILES_PATH],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
