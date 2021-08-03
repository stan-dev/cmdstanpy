"""Metadata tests"""

import os
import unittest

from cmdstanpy.cmdstan_args import CmdStanArgs, SamplerArgs
from cmdstanpy.stanfit import InferenceMetadata, RunSet
from cmdstanpy.utils import EXTENSION, check_sampler_csv

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')

DATAFILES_PATH = os.path.join(HERE, 'data')
GOODFILES_PATH = os.path.join(DATAFILES_PATH, 'runset-good')
BADFILES_PATH = os.path.join(DATAFILES_PATH, 'runset-bad')


class InferenceMetadataTest(unittest.TestCase):
    def test_good(self):
        # construct fit using existing sampler output
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs(
            iter_sampling=100, max_treedepth=11, adapt_delta=0.95
        )
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_dir=DATAFILES_PATH,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args)
        runset._csv_files = [
            os.path.join(DATAFILES_PATH, 'runset-good', 'bern-1.csv'),
            os.path.join(DATAFILES_PATH, 'runset-good', 'bern-2.csv'),
            os.path.join(DATAFILES_PATH, 'runset-good', 'bern-3.csv'),
            os.path.join(DATAFILES_PATH, 'runset-good', 'bern-4.csv'),
        ]
        retcodes = runset._retcodes
        for i in range(len(retcodes)):
            runset._set_retcode(i, 0)
        config = check_sampler_csv(
            path=runset.csv_files[i],
            is_fixed_param=False,
            iter_sampling=100,
            iter_warmup=1000,
            save_warmup=False,
            thin=1,
        )
        expected = 'Metadata:\n{}\n'.format(config)
        metadata = InferenceMetadata(config)
        actual = '{}'.format(metadata)
        self.assertEqual(expected, actual)
        self.assertEqual(config, metadata.cmdstan_config)

        hmc_vars = {
            'lp__',
            'accept_stat__',
            'stepsize__',
            'treedepth__',
            'n_leapfrog__',
            'divergent__',
            'energy__',
        }

        method_vars_cols = metadata.method_vars_cols
        self.assertEqual(hmc_vars, method_vars_cols.keys())
        bern_model_vars = {'theta'}
        self.assertEqual(bern_model_vars, metadata.stan_vars_dims.keys())
        self.assertEqual((), metadata.stan_vars_dims['theta'])
        self.assertEqual(bern_model_vars, metadata.stan_vars_cols.keys())
        self.assertEqual((7,), metadata.stan_vars_cols['theta'])


if __name__ == '__main__':
    unittest.main()
