"""install_cmdstan test"""

import os
import platform
import shutil
from time import time
import unittest

from cmdstanpy.install_cmdstan import (
    is_version_available,
    latest_version,
    retrieve_version,
    CmdStanRetrieveError,
)
from cmdstanpy.utils import install_cmdstan

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


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
        cur_cmdstan_path = ''
        if 'CMDSTAN' in os.environ:
            cur_cmdstan_path = os.environ['CMDSTAN']
            del os.environ['CMDSTAN']
        self.assertFalse('CMDSTAN' in os.environ)

        if platform.system() == 'Windows':
            return

        version = '2.24.1'
        tmpdir = os.path.join(DATAFILES_PATH, 'tmp-' + str(time()))
        os.mkdir(tmpdir)
        print('before install')
        retcode = install_cmdstan(version=version, dir=tmpdir)
        print('after install, retcode: {}'.format(retcode))
        expect_cmdstan_path = os.path.join(
            tmpdir, '-'.join(['cmdstan', version])
        )
        self.assertTrue('CMDSTAN' in os.environ)
        self.assertEqual(expect_cmdstan_path, os.environ['CMDSTAN'])

        # cleanup
        os.environ['CMDSTAN'] = cur_cmdstan_path
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
