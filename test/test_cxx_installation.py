"""install_cxx_toolchain tests"""

import platform
import unittest

import pytest

from cmdstanpy import install_cxx_toolchain


class InstallCxxScriptTest(unittest.TestCase):
    @pytest.mark.skipif(
        platform.system() != 'Windows', reason='Windows only tests'
    )
    def test_config(self):
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

        self.assertEqual(config, config_reference)

    @pytest.mark.skipif(
        platform.system() == 'Windows', reason='Windows only tests'
    )
    def test_install_not_windows(self):
        """Try to install on unsupported platform."""

        with self.assertRaisesRegex(
            NotImplementedError,
            r'Download for the C\+\+ toolchain on the current platform has not '
            r'been implemented:\s*\S+',
        ):
            install_cxx_toolchain.run_rtools_install({})

    @pytest.mark.skipif(
        platform.system() != 'Windows', reason='Windows only tests'
    )
    def test_normalize_version(self):
        """Test supported versions."""

        for ver in ['4.0', '4', '40']:
            self.assertEqual(
                install_cxx_toolchain.normalize_version(ver), '4.0'
            )

        for ver in ['3.5', '35']:
            self.assertEqual(
                install_cxx_toolchain.normalize_version(ver), '3.5'
            )

    @pytest.mark.skipif(
        platform.system() != 'Windows', reason='Windows only tests'
    )
    def test_toolchain_name(self):
        """Check toolchain name."""

        self.assertEqual(install_cxx_toolchain.get_toolchain_name(), 'RTools')


if __name__ == '__main__':
    unittest.main()
