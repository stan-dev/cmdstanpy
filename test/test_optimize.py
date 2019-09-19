import os
import unittest
import json

from cmdstanpy.cmdstan_args import Method, OptimizeArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION
from cmdstanpy.model import Model
from cmdstanpy.stanfit import RunSet, StanMLE
from contextlib import contextmanager
import logging
from multiprocessing import cpu_count
import numpy as np
import sys
from testfixtures import LogCapture

here = os.path.dirname(os.path.abspath(__file__))
datafiles_path = os.path.join(here, 'data')


class StanMLETest(unittest.TestCase):
    def test_set_mle_attrs(self):
        stan = os.path.join(datafiles_path, 'optimize', 'rosenbrock.stan')
        exe = os.path.join(datafiles_path, 'optimize', 'rosenbrock')
        no_data = {}
        args = OptimizeArgs(algorithm='Newton')
        cmdstan_args = CmdStanArgs(
            model_name='rosenbrock',
            model_exe=exe,
            chain_ids=None,
            data=no_data,
            method_args=args,
        )
        runset = RunSet(args=cmdstan_args, chains=1)
        mle = StanMLE(runset)
        self.assertEqual(mle._column_names,None)
        self.assertEqual(mle._mle,None)

        output = os.path.join(datafiles_path, 'optimize', 'rosenbrock_mle.csv')
        mle._set_mle_attrs(output)
        self.assertEqual(mle.column_names,('lp__', 'x', 'y'))
        self.assertAlmostEqual(mle.optimized_params_dict['x'], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_dict['y'], 1, places=3)
        


class OptimizeTest(unittest.TestCase):
    def test_optimize_good(self):
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        model = Model(stan_file=stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        jinit = os.path.join(datafiles_path, 'bernoulli.init.json')
        mle = model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='BFGS',
            init_alpha=0.001,
            iter=100,
        )

        # test numpy output
        self.assertAlmostEqual(mle.optimized_params_np[0], -5, places=2)
        self.assertAlmostEqual(mle.optimized_params_np[1], 0.2, places=3)

        # test pandas output
        self.assertEqual(
            mle.optimized_params_np[0], mle.optimized_params_pd['lp__'][0]
        )
        self.assertEqual(
            mle.optimized_params_np[1], mle.optimized_params_pd['theta'][0]
        )

        # test dict output
        self.assertEqual(
            mle.optimized_params_np[0], mle.optimized_params_dict['lp__']
        )
        self.assertEqual(
            mle.optimized_params_np[1], mle.optimized_params_dict['theta']
        )

    def test_optimize_good_dict(self):
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        model = Model(stan_file=stan, exe_file=exe)
        with open(os.path.join(datafiles_path, 'bernoulli.data.json')) as d:
            data = json.load(d)
        with open(os.path.join(datafiles_path, 'bernoulli.init.json')) as d:
            init = json.load(d)
        mle = model.optimize(
            data=data,
            seed=1239812093,
            inits=init,
            algorithm='BFGS',
            init_alpha=0.001,
            iter=100,
        )
        # test numpy output
        self.assertAlmostEqual(mle.optimized_params_np[0], -5, places=2)
        self.assertAlmostEqual(mle.optimized_params_np[1], 0.2, places=3)


    def test_optimize_rosenbrock(self):
        stan = os.path.join(datafiles_path, 'optimize', 'rosenbrock.stan')
        rose_model = Model(stan_file=stan)
        rose_model.compile()
        no_data = {}
        mle = rose_model.optimize(
            data=no_data,
            seed=1239812093,
            inits=None,
            algorithm='BFGS'
        )
        self.assertEqual(mle.column_names,('lp__', 'x', 'y'))
        self.assertAlmostEqual(mle.optimized_params_dict['x'], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_dict['y'], 1, places=3)


    def test_optimize_bad(self):
        stan = os.path.join(datafiles_path, 'optimize', 'exponential_boundary.stan')
        exp_bound_model = Model(stan_file=stan)
        exp_bound_model.compile()
        no_data = {}
        with self.assertRaisesRegex(Exception, 'Error during optimizing, error code 70'):
            exp_bound_model.optimize(
                data=no_data,
                seed=1239812093,
                inits=None,
                algorithm='BFGS'
            )


if __name__ == '__main__':
    unittest.main()