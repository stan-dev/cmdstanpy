import os
import unittest
import json
from math import fabs
from cmdstanpy.cmdstan_args import Method, VariationalArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION
from cmdstanpy.model import Model
from cmdstanpy.stanfit import RunSet, StanVariational
from contextlib import contextmanager
import logging
from multiprocessing import cpu_count
import numpy as np
import sys
from testfixtures import LogCapture

here = os.path.dirname(os.path.abspath(__file__))
datafiles_path = os.path.join(here, 'data')


class StanVariationalTest(unittest.TestCase):
    def test_set_variational_attrs(self):
        stan = os.path.join(datafiles_path, 'variational', 'eta_should_be_big.stan')
        model = Model(stan_file=stan)
        model.compile()
        no_data = {}
        args = VariationalArgs(algorithm='meanfield')
        cmdstan_args = CmdStanArgs(
            model_name=model.name,
            model_exe=model.exe_file,
            chain_ids=None,
            data=no_data,
            method_args=args
        )
        runset = RunSet(args=cmdstan_args, chains=1)
        vi = StanVariational(runset)
        self.assertIn('StanVariational: model=eta_should_be_big', vi.__repr__())
        self.assertIn('method=variational', vi.__repr__())

        # check StanVariational.__init__ state
        self.assertEqual(vi._column_names,())
        self.assertEqual(vi._variational_mean,{})
        self.assertEqual(vi._variational_sample,None)

        # process csv file, check attrs
        output = os.path.join(datafiles_path, 'variational', 'eta_big_output.csv')
        vi._set_variational_attrs(output)
        self.assertEqual(vi.column_names,('lp__', 'log_p__', 'log_g__', 'mu.1', 'mu.2'))
        self.assertAlmostEqual(vi.variational_params_dict['mu.1'], 31.0299, places=2)
        self.assertAlmostEqual(vi.variational_params_dict['mu.2'], 28.8141, places=2)
        self.assertEqual(vi.variational_sample.shape, (1000, 5))


class VariationalTest(unittest.TestCase):
    def test_variational_good(self):
        stan = os.path.join(datafiles_path, 'variational', 'eta_should_be_big.stan')
        model = Model(stan_file=stan)
        model.compile()
        vi = model.variational(algorithm='meanfield', seed=12345)
        self.assertEqual(vi.column_names,('lp__', 'log_p__', 'log_g__', 'mu.1', 'mu.2'))

        self.assertAlmostEqual(vi.variational_params_np[3], 31.0418, places=2)
        self.assertAlmostEqual(vi.variational_params_np[4], 27.4463, places=2)

        self.assertAlmostEqual(vi.variational_params_dict['mu.1'], 31.0418, places=2)
        self.assertAlmostEqual(vi.variational_params_dict['mu.2'], 27.4463, places=2)

        self.assertEqual(
            vi.variational_params_np[0], vi.variational_params_pd['lp__'][0]
        )
        self.assertEqual(
            vi.variational_params_np[3], vi.variational_params_pd['mu.1'][0]
        )
        self.assertEqual(
            vi.variational_params_np[4], vi.variational_params_pd['mu.2'][0]
        )

        self.assertEqual(vi.variational_sample.shape, (1000, 5))

    def test_variational_missing_args(self):
        self.assertTrue(True)


    def test_variational_eta_small(self):
        stan = os.path.join(datafiles_path, 'variational', 'eta_should_be_small.stan')
        model = Model(stan_file=stan)
        model.compile()
        vi = model.variational(algorithm='meanfield', seed=12345)
        self.assertEqual(vi.column_names,('lp__', 'log_p__', 'log_g__', 'mu.1', 'mu.2'))
        self.assertAlmostEqual(fabs(vi.variational_params_dict['mu.1']), 0.08, places=1)
        self.assertAlmostEqual(fabs(vi.variational_params_dict['mu.2']), 0.09, places=1)
        self.assertTrue(True)


    def test_variational_eta_fail(self):
        stan = os.path.join(datafiles_path, 'variational', 'eta_should_fail.stan')
        model = Model(stan_file=stan)
        model.compile()
        with self.assertRaisesRegex(RuntimeError, 'algorithm may not have converged'):
            vi = model.variational(algorithm='meanfield', seed=12345)


if __name__ == '__main__':
    unittest.main()
