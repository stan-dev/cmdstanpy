"""The global configuration for the test suite"""
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


# after we have run all tests, use git to delete the built files in data/


@pytest.fixture(scope='session', autouse=True)
def cleanup_test_files():
    yield
    subprocess.Popen(
        ['git', 'clean', '-fX', DATAFILES_PATH],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
