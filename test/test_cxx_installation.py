import argparse
import os
import unittest
import platform
import tempfile

from cmdstanpy import TMPDIR
from cmdstanpy.install_cxx_toolchain import main as install_cxx

here = os.path.dirname(os.path.abspath(__file__))
datafiles_path = os.path.join(here, 'data')


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
        assertTrue(os.path.exists(os.path.join(TMPDIR, "RTools")))


if __name__ == '__main__':
    unittest.main()
