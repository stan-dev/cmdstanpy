"""install_cxx_toolchain tests"""

import unittest
import platform

from cmdstanpy import install_cxx_toolchain


class InstallCxxScriptTest(unittest.TestCase):
    def test_config(self):
        """Test config output."""
        if platform.system() != 'Windows':
            return
        else:
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

    def test_install_not_windows(self):
        """Try to install on unsupported platform."""
        if platform.system() == 'Windows':
            return

        with self.assertRaisesRegex(
            NotImplementedError,
            r'Download for the C\+\+ toolchain on the current platform has not '
            r'been implemented:\s*\S+',
        ):
            install_cxx_toolchain.main()

    def test_normalize_version(self):
        """Test supported versions."""
        if platform.system() != 'Windows':
            return
        else:
            for ver in ['4.0', '4', '40']:
                self.assertEqual(
                    install_cxx_toolchain.normalize_version(ver), '4.0'
                )

            for ver in ['3.5', '35']:
                self.assertEqual(
                    install_cxx_toolchain.normalize_version(ver), '3.5'
                )

    def test_toolchain_name(self):
        """Check toolchain name."""
        if platform.system() != 'Windows':
            return
        else:
            self.assertEqual(
                install_cxx_toolchain.get_toolchain_name(), 'RTools'
            )


if __name__ == '__main__':
    unittest.main()
