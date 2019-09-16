import os
import unittest

from cmdstanpy.cmdstan_args import SamplerArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION
from cmdstanpy.stanfit import RunSet

here = os.path.dirname(os.path.abspath(__file__))
datafiles_path = os.path.join(here, 'data')


class RunSetTest(unittest.TestCase):
    def test_check_retcodes(self):
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=4)
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


if __name__ == '__main__':
    unittest.main()
