"""install_cmdstan test"""
from unittest import mock
import pytest
from cmdstanpy.install_cmdstan import (
    CmdStanInstallError,
    CmdStanRetrieveError,
    is_version_available,
    latest_version,
    rebuild_cmdstan,
    retrieve_version,
)


def test_is_version_available() -> None:
    # check http error for bad version
    assert not is_version_available('2.222.222-rc222')


def test_latest_version() -> None:
    # examples of known previous version:  2.24-rc1, 2.23.0
    version = latest_version()
    nums = version.split('.')
    assert len(nums) >= 2
    assert nums[0][0].isdigit()
    assert nums[1][0].isdigit()


def test_retrieve_version() -> None:
    # check http error for bad version
    with pytest.raises(
        CmdStanRetrieveError, match='not available from github.com'
    ):
        retrieve_version('no_such_version')
    with pytest.raises(ValueError):
        retrieve_version(None)
    with pytest.raises(ValueError):
        retrieve_version('')


def test_rebuild_bad_path() -> None:
    with mock.patch.dict("os.environ", CMDSTAN="~/some/fake/path"):
        with pytest.raises(
            CmdStanInstallError, match="you sure it is installed"
        ):
            rebuild_cmdstan(latest_version())
