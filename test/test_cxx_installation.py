"""install_cxx_toolchain tests"""

from test import mark_windows_only, mark_not_windows
import pytest

from cmdstanpy import install_cxx_toolchain


@mark_windows_only
def test_config() -> None:
    """Test config output."""

    config = install_cxx_toolchain.get_config('C:\\RTools', True)

    config_reference = [
        '/SP-',
        '/VERYSILENT',
        '/SUPPRESSMSGBOXES',
        '/CURRENTUSER',
        'LANG="English"',
        '/DIR="RTools"',
        '/NOICONS',
        '/NORESTART',
    ]

    assert config == config_reference


@mark_not_windows
def test_install_not_windows() -> None:
    """Try to install on unsupported platform."""

    with pytest.raises(
        NotImplementedError,
        match=r'Download for the C\+\+ toolchain on the current platform '
        r'has not been implemented:\s*\S+',
    ):
        install_cxx_toolchain.run_rtools_install({})


@mark_windows_only
def test_normalize_version() -> None:
    """Test supported versions."""

    for ver in ['4.0', '4', '40']:
        assert install_cxx_toolchain.normalize_version(ver) == '4.0'

    for ver in ['3.5', '35']:
        assert install_cxx_toolchain.normalize_version(ver) == '3.5'


@mark_windows_only
def test_toolchain_name() -> None:
    """Check toolchain name."""
    assert install_cxx_toolchain.get_toolchain_name() == 'RTools'
