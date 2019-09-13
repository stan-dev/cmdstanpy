import argparse
import os
import unittest
from unittest import mock
import platform
import shutil

from cmdstanpy import TMPDIR
from cmdstanpy.install_cxx_toolchain import main as install_cxx


class install_cxx_script(unittest.TestCase):
    def test_not_windows(self):
        if platform.system() == "Windows":
            return

        with self.assertRaisesRegex(
            NotImplementedError,
            'Download for the C++ toolchain on the current platform has not been implemented: \s+',
        ):
            install_cxx()

    @mock.patch(
        'argparse.ArgumentParser.parse_args',
        return_value=argparse.Namespace(version="3.5", dir=TMPDIR),
    )
    def test_windows(self, mock_args):
        if platform.system() != "Windows":
            return

        install_cxx()
        self.assertTrue(os.path.exists(os.path.join(TMPDIR, "RTools")))
        shutil.rmtree(os.path.join(TMPDIR, "RTools"), ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
