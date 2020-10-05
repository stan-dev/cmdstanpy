"""install_cmdstan test"""

import os
import unittest

from cmdstanpy import _TMPDIR
from cmdstanpy.install_cmdstan import (
    is_version_available,
    latest_version,
    retrieve_version,
    CmdStanRetrieveError,
)
from cmdstanpy.utils import install_cmdstan

class InstallCmdStanTest(unittest.TestCase):
    def test_is_version_available(self):
        # check http error for bad version
        self.assertTrue(is_version_available('2.24.1'))
        self.assertFalse(is_version_available('2.222.222-rc222'))

    def test_latest_version(self):
        # examples of known previous version:  2.24-rc1, 2.23.0
        version = latest_version()
        nums = version.split('.')
        self.assertTrue(len(nums) >= 2)
        self.assertTrue(nums[0][0].isdigit())
        self.assertTrue(nums[1][0].isdigit())

    def test_retrieve_version(self):
        # check http error for bad version
        with self.assertRaisesRegex(
            CmdStanRetrieveError, 'not available from github.com'
        ):
            retrieve_version('no_such_version')

    def test_install_cmdstan_specify_dir(self):
        cur_path = ''
        if 'CMDSTAN' in os.environ:
            cur_path = os.environ['CMDSTAN']
            del os.environ['CMDSTAN']
        self.assertFalse('CMDSTAN' in os.environ)
        version = '2.24.1'
        print('l 179')
        retcode = install_cmdstan(version=version, dir=_TMPDIR, overwrite=True)
        print('l 181: {}'.format(retcode))
        expect_cmdstan_path = os.path.join(_TMPDIR, '-'.join(['cmdstan', version]))
        self.assertTrue('CMDSTAN' in os.environ)
        self.assertEqual(expect_cmdstan_path, os.environ['CMDSTAN'])
        os.environ['CMDSTAN'] = cur_path




if __name__ == '__main__':
    unittest.main()
