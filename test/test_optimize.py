"""CmdStan method optimize tests"""

import os
import json
import unittest
import pytest
import numpy as np

from cmdstanpy.cmdstan_args import OptimizeArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION
from cmdstanpy.model import CmdStanModel
from cmdstanpy.stanfit import RunSet, CmdStanMLE

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class CmdStanMLETest(unittest.TestCase):

    # pylint: disable=no-self-use
    @pytest.fixture(scope='class', autouse=True)
    def do_clean_up(self):
        for root, _, files in os.walk(DATAFILES_PATH):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext.lower() in ('.o', '.hpp', '.exe', ''):
                    filepath = os.path.join(root, filename)
                    os.remove(filepath)

    def test_instantiate(self):
        stan = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
        model = CmdStanModel(stan_file=stan)
        no_data = {}
        args = OptimizeArgs(algorithm='Newton')
        cmdstan_args = CmdStanArgs(
            model_name=model.name,
            model_exe=model.exe_file,
            chain_ids=None,
            data=no_data,
            method_args=args,
        )
        runset = RunSet(args=cmdstan_args, chains=1)
        runset._csv_files = [
            os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv')
        ]
        mle = CmdStanMLE(runset)
        self.assertIn('CmdStanMLE: model=rosenbrock', mle.__repr__())
        self.assertIn('method=optimize', mle.__repr__())
        self.assertEqual(mle.column_names, ('lp__', 'x', 'y'))
        self.assertAlmostEqual(mle.optimized_params_dict['x'], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_dict['y'], 1, places=3)


class OptimizeTest(unittest.TestCase):
    def test_optimize_good(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        jinit = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')
        mle = model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='BFGS',
            init_alpha=0.001,
            iter=100,
        )

        # test numpy output
        self.assertTrue(isinstance(mle.optimized_params_np, np.ndarray))
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
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        model = CmdStanModel(stan_file=stan, exe_file=exe)
        with open(os.path.join(DATAFILES_PATH, 'bernoulli.data.json')) as fd:
            data = json.load(fd)
        with open(os.path.join(DATAFILES_PATH, 'bernoulli.init.json')) as fd:
            init = json.load(fd)
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
        stan = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
        rose_model = CmdStanModel(stan_file=stan)
        no_data = {}
        mle = rose_model.optimize(
            data=no_data, seed=1239812093, inits=None, algorithm='BFGS'
        )
        self.assertEqual(mle.column_names, ('lp__', 'x', 'y'))
        self.assertAlmostEqual(mle.optimized_params_dict['x'], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_dict['y'], 1, places=3)

    def test_optimize_bad(self):
        stan = os.path.join(
            DATAFILES_PATH, 'optimize', 'exponential_boundary.stan'
        )
        exp_bound_model = CmdStanModel(stan_file=stan)
        no_data = {}
        with self.assertRaisesRegex(
            Exception, 'Error during optimizing, error code 70'
        ):
            exp_bound_model.optimize(
                data=no_data, seed=1239812093, inits=None, algorithm='BFGS'
            )


if __name__ == '__main__':
    unittest.main()
