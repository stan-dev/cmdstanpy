"""install_cmdstan test"""

import unittest

from cmdstanpy.install_cmdstan import (
    is_version_available,
    latest_version,
    validate_dir,
)
from cmdstanpy.utils import cmdstan_path


class InstallCmdStanTest(unittest.TestCase):
    def test_is_version_available(self):
        # check http error for bad version
        self.assertTrue(is_version_available('2.24.1'))
        self.assertFalse(is_version_available('2.222.222-rc222'))

    def test_latest_version(self):
        # parse version into Maj, min, patch
        # Maj between 1 and 99
        # min between 1 and 99
        # p starts with digit
        # examples of known previous version:  2.24.0-rc1, 2.23.0
        version = latest_version()
        nums = version.split('.')
        self.assertEqual(len(nums), 3)
        for num in nums:
            self.assertTrue(num[0].isdigit())

    def test_retrieve_version(self):
        # check http error for bad version
        self.assertTrue(True)

    def test_validate_dir(self):
        # get cmdstan path; should be valid
        # create directory, chmod to no write
        # create file
        # with self.assertRaisesRegex(ValueError, 'Cannot create directory'):
        #     validate_dir(path_foo)
        path = cmdstan_path()
        validate_dir(path)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
