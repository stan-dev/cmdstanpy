"""CmdStan method optimize tests"""

import json
import os
import unittest

import numpy as np
import pytest

from cmdstanpy.cmdstan_args import CmdStanArgs, OptimizeArgs
from cmdstanpy.model import CmdStanModel
from cmdstanpy.stanfit import from_csv, CmdStanMLE, RunSet
from cmdstanpy.utils import EXTENSION

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

    def test_instantiate_from_csvfiles(self):
        csvfiles_path = os.path.join(DATAFILES_PATH, 'optimize')
        mle = from_csv(dir=csvfiles_path)
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
            algorithm='LBFGS',
            init_alpha=0.001,
            iter=100,
            tol_obj=1e-12,
            tol_rel_obj=1e4,
            tol_grad=1e-8,
            tol_rel_grad=1e7,
            tol_param=1e-8,
            history_size=5,
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

    def test_negative_parameter_values(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        jinit = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')

        with self.assertRaisesRegex(ValueError, 'must be greater than'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_obj=-1,
            )

        with self.assertRaisesRegex(ValueError, 'must be greater than'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_rel_obj=-1,
            )

        with self.assertRaisesRegex(ValueError, 'must be greater than'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_grad=-1,
            )

        with self.assertRaisesRegex(ValueError, 'must be greater than'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_rel_grad=-1,
            )

        with self.assertRaisesRegex(ValueError, 'must be greater than'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_param=-1,
            )

        with self.assertRaisesRegex(ValueError, 'must be greater than'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                history_size=-1,
            )

    def test_parameters_are_floats(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        jinit = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')

        with self.assertRaisesRegex(ValueError, 'must be type of float'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_obj="rabbit",
            )

        with self.assertRaisesRegex(ValueError, 'must be type of float'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_rel_obj="rabbit",
            )

        with self.assertRaisesRegex(ValueError, 'must be type of float'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_grad="rabbit",
            )

        with self.assertRaisesRegex(ValueError, 'must be type of float'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_rel_grad="rabbit",
            )

        with self.assertRaisesRegex(ValueError, 'must be type of float'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_param="rabbit",
            )

        with self.assertRaisesRegex(ValueError, 'must be type of int'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                history_size="rabbit",
            )

    def test_parameters_and_optimizer_compatible(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        jinit = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')

        with self.assertRaisesRegex(
            ValueError, 'must not be set when algorithm is Newton'
        ):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='Newton',
                tol_obj=1,
            )

        with self.assertRaisesRegex(
            ValueError, 'must not be set when algorithm is Newton'
        ):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='Newton',
                tol_rel_obj=1,
            )

        with self.assertRaisesRegex(
            ValueError, 'must not be set when algorithm is Newton'
        ):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='Newton',
                tol_grad=1,
            )

        with self.assertRaisesRegex(
            ValueError, 'must not be set when algorithm is Newton'
        ):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='Newton',
                tol_rel_grad=1,
            )

        with self.assertRaisesRegex(
            ValueError, 'must not be set when algorithm is Newton'
        ):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='Newton',
                tol_param=1,
            )

        with self.assertRaisesRegex(
            ValueError,
            'history_size must not be set when algorithm is Newton or BFGS',
        ):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='Newton',
                history_size=1,
            )

        with self.assertRaisesRegex(
            ValueError,
            'history_size must not be set when algorithm is Newton or BFGS',
        ):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='BFGS',
                history_size=1,
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
        mle = rose_model.optimize(seed=1239812093, inits=None, algorithm='BFGS')
        self.assertEqual(mle.column_names, ('lp__', 'x', 'y'))
        self.assertAlmostEqual(mle.optimized_params_dict['x'], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_dict['y'], 1, places=3)

    def test_optimize_no_data(self):
        stan = os.path.join(DATAFILES_PATH, 'optimize', 'no_data.stan')
        rose_model = CmdStanModel(stan_file=stan)
        mle = rose_model.optimize(seed=1239812093)
        self.assertEqual(mle.column_names, ('lp__', 'a'))
        self.assertAlmostEqual(mle.optimized_params_dict['a'], 0, places=3)

    def test_optimize_bad(self):
        stan = os.path.join(
            DATAFILES_PATH, 'optimize', 'exponential_boundary.stan'
        )
        exp_bound_model = CmdStanModel(stan_file=stan)
        no_data = {}
        with self.assertRaisesRegex(RuntimeError, 'Error during optimization'):
            exp_bound_model.optimize(
                data=no_data, seed=1239812093, inits=None, algorithm='BFGS'
            )


if __name__ == '__main__':
    unittest.main()
