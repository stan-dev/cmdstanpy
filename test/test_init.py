import os
import os.path
import unittest
import cmdstanpy
from importlib import reload


class CmdStanPathTest(unittest.TestCase):
    def test_default_path(self):
        abs_rel_path = os.path.abspath(os.path.join('releases', 'cmdstan'))
        self.assertEqual(abs_rel_path, cmdstanpy.CMDSTAN_PATH)

    def test_env_path(self):
        old_version = os.path.abspath(
            os.path.join('releases', 'cmdstan-2.18.1'))
        os.environ['CMDSTAN'] = old_version
        reload(cmdstanpy)
        self.assertEqual(old_version, cmdstanpy.CMDSTAN_PATH)


if __name__ == '__main__':
    unittest.main()
