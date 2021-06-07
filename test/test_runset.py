"""RunSet tests"""

import os
import unittest

from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import CmdStanArgs, SamplerArgs
from cmdstanpy.stanfit import RunSet
from cmdstanpy.utils import EXTENSION, cmdstan_version_at

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class RunSetTest(unittest.TestCase):
    def test_check_repr(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        chain_ids = [1, 2, 3, 4]  # default
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=chain_ids,
            data=jdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args)

        self.assertIn('RunSet: chains=4', runset.__repr__())
        self.assertIn('method=sample', runset.__repr__())
        self.assertIn('retcodes=[-1, -1, -1, -1]', runset.__repr__())
        self.assertIn('csv_files:', runset.__repr__())
        self.assertNotIn('diagnostics_files:', runset.__repr__())

    def test_check_retcodes(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        chain_ids = [1, 2, 3, 4]  # default
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=chain_ids,
            data=jdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args)

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
        rdata = os.path.join(DATAFILES_PATH, 'logistic.missing_data.R')
        sampler_args = SamplerArgs()
        chain_ids = [1, 2, 3]
        cmdstan_args = CmdStanArgs(
            model_name='logistic',
            model_exe=exe,
            chain_ids=chain_ids,
            data=rdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=3, chain_ids=chain_ids)
        if not cmdstan_version_at(2, 27):
            for i in range(3):
                runset._set_retcode(i, 70)
                stdout_file = 'chain-' + str(i + 1) + '-missing-data-stdout.txt'
                path = os.path.join(DATAFILES_PATH, stdout_file)
                runset._stdout_files[i] = path
            errs = runset.get_err_msgs()
            self.assertIn('Exception: variable does not exist', errs)
        else:
            for i in range(3):
                runset._set_retcode(i, 1)
                stderr_file = 'chain-' + str(i + 1) + '-missing-data-stderr.txt'
                path = os.path.join(DATAFILES_PATH, stderr_file)
                runset._stderr_files[i] = path
            errs = runset.get_err_msgs()
            self.assertIn('Exception: variable does not exist', errs)

    def test_output_filenames(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        chain_ids = [1, 2, 3, 4]
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=chain_ids,
            data=jdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args)
        self.assertIn('bernoulli-', runset._csv_files[0])
        self.assertIn('-1-', runset._csv_files[0])
        self.assertIn('-4-', runset._csv_files[3])

    def test_commands(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        chain_ids = [1, 2, 3, 4]
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=chain_ids,
            data=jdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args)
        self.assertIn('id=1', runset._cmds[0])
        self.assertIn('id=4', runset._cmds[3])

    def test_save_diagnostics(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        chain_ids = [1, 2, 3, 4]
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=chain_ids,
            data=jdata,
            method_args=sampler_args,
            save_diagnostics=True,
        )
        runset = RunSet(args=cmdstan_args)
        self.assertIn(_TMPDIR, runset.diagnostic_files[0])

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=chain_ids,
            data=jdata,
            method_args=sampler_args,
            save_diagnostics=True,
            output_dir=os.path.abspath('.'),
        )
        runset = RunSet(args=cmdstan_args)
        self.assertIn(os.path.abspath('.'), runset.diagnostic_files[0])

    def test_chain_ids(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        chain_ids = [11, 12, 13, 14]
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=chain_ids,
            data=jdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=4, chain_ids=chain_ids)
        self.assertIn('id=11', runset._cmds[0])
        self.assertIn('-11-', runset._csv_files[0])
        self.assertIn('id=14', runset._cmds[3])
        self.assertIn('-14-', runset._csv_files[3])

    def test_ctor_checks(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        chain_ids = [11, 12, 13, 14]
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=chain_ids,
            data=jdata,
            method_args=sampler_args,
        )
        with self.assertRaises(ValueError):
            RunSet(args=cmdstan_args, chains=0)
        with self.assertRaises(ValueError):
            RunSet(args=cmdstan_args, chains=4, chain_ids=[1, 2, 3])


if __name__ == '__main__':
    unittest.main()
