"""RunSet tests"""

import os
import unittest

from cmdstanpy.cmdstan_args import SamplerArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION
from cmdstanpy.stanfit import RunSet

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class RunSetTest(unittest.TestCase):
    def test_check_retcodes(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=4)
        self.assertIn('RunSet: chains=4', runset.__repr__())
        self.assertIn('method=sample', runset.__repr__())

        retcodes = runset._retcodes
        self.assertEqual(4, len(retcodes))
        for i in range(len(retcodes)):
            self.assertEqual(-1, runset._retcode(i))
        runset._set_retcode(0, 0)
        self.assertEqual(0, runset._retcode(0))
        for i in range(1, len(retcodes)):
            self.assertEqual(-1, runset._retcode(i))
        self.assertFalse(runset._check_retcodes())
        for i in range(1, len(retcodes)):
            runset._set_retcode(i, 0)
        self.assertTrue(runset._check_retcodes())

    def test_get_err_msgs(self):
        exe = os.path.join(DATAFILES_PATH, 'logistic' + EXTENSION)
        rdata = os.path.join(DATAFILES_PATH, 'logistic.data.R')
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='logistic',
            model_exe=exe,
            chain_ids=[1, 2, 3],
            data=rdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=3)
        for i in range(3):
            runset._set_retcode(i, 70)
            stdout_file = 'chain-' + str(i + 1) + '-missing-data-stdout.txt'
            path = os.path.join(DATAFILES_PATH, stdout_file)
            runset._stdout_files[i] = path
        errs = '\n\t'.join(runset._get_err_msgs())
        self.assertIn('Exception', errs)

    def test_output_filenames(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=4)
        self.assertIn('bernoulli-', runset._csv_files[0])
        self.assertIn('-1-', runset._csv_files[0])
        self.assertIn('-4-', runset._csv_files[3])

    def test_commands(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=4)
        self.assertIn('id=1', runset._cmds[0])
        self.assertIn('id=4', runset._cmds[3])


if __name__ == '__main__':
    unittest.main()
