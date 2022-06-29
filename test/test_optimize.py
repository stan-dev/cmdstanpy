"""CmdStan method optimize tests"""

import contextlib
import io
import json
import os
import shutil
import unittest

import numpy as np
from testfixtures import LogCapture

from cmdstanpy.cmdstan_args import CmdStanArgs, OptimizeArgs
from cmdstanpy.model import CmdStanModel
from cmdstanpy.stanfit import CmdStanMLE, RunSet, from_csv
from cmdstanpy.utils import EXTENSION

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class CmdStanMLETest(unittest.TestCase):
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
        self.assertIn('CmdStanMLE: model=rosenbrock', repr(mle))
        self.assertIn('method=optimize', repr(mle))
        self.assertEqual(mle.column_names, ('lp__', 'x', 'y'))
        self.assertAlmostEqual(mle.optimized_params_dict['x'], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_dict['y'], 1, places=3)

    def test_instantiate_from_csvfiles(self):
        csvfiles_path = os.path.join(
            DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv'
        )
        mle = from_csv(path=csvfiles_path)
        self.assertIn('CmdStanMLE: model=rosenbrock', repr(mle))
        self.assertIn('method=optimize', repr(mle))
        self.assertEqual(mle.column_names, ('lp__', 'x', 'y'))
        self.assertAlmostEqual(mle.optimized_params_dict['x'], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_dict['y'], 1, places=3)

    def test_instantiate_from_csvfiles_save_iterations(self):
        csvfiles_path = os.path.join(
            DATAFILES_PATH, 'optimize', 'eight_schools_mle_iters.csv'
        )
        mle = from_csv(path=csvfiles_path)
        self.assertIn('CmdStanMLE: model=eight_schools', repr(mle))
        self.assertIn('method=optimize', repr(mle))
        self.assertEqual(
            mle.column_names,
            (
                'lp__',
                'mu',
                'theta[1]',
                'theta[2]',
                'theta[3]',
                'theta[4]',
                'theta[5]',
                'theta[6]',
                'theta[7]',
                'theta[8]',
                'tau',
            ),
        )
        self.assertAlmostEqual(
            mle.optimized_params_dict['mu'], 1.06401, places=3
        )
        self.assertAlmostEqual(
            mle.optimized_params_dict['theta[1]'], 1.06401, places=3
        )
        self.assertEqual(mle.optimized_iterations_np.shape, (173, 11))

    def test_rosenbrock(self):
        stan = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
        model = CmdStanModel(stan_file=stan)
        mle = model.optimize(algorithm='LBFGS')
        self.assertIn('CmdStanMLE: model=rosenbrock', repr(mle))
        self.assertIn('method=optimize', repr(mle))
        self.assertTrue(mle.converged)
        self.assertEqual(mle.column_names, ('lp__', 'x', 'y'))
        self.assertAlmostEqual(mle.stan_variable('x'), 1, places=3)
        self.assertAlmostEqual(mle.stan_variable('y'), 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_pd['x'][0], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_np[1], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_dict['x'], 1, places=3)
        with LogCapture() as log:
            self.assertEqual(mle.optimized_iterations_np, None)
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'Intermediate iterations not saved to CSV output file. '
                'Rerun the optimize method with "save_iterations=True".',
            )
        )
        with LogCapture() as log:
            self.assertEqual(mle.optimized_iterations_pd, None)
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'Intermediate iterations not saved to CSV output file. '
                'Rerun the optimize method with "save_iterations=True".',
            )
        )

        mle = model.optimize(
            algorithm='LBFGS', save_iterations=True, seed=12345
        )
        self.assertTrue(mle.converged)
        self.assertAlmostEqual(mle.stan_variable('x'), 1, places=3)
        self.assertAlmostEqual(mle.stan_variable('y'), 1, places=3)

        self.assertEqual(mle.optimized_params_np.shape, (3,))
        self.assertAlmostEqual(mle.optimized_params_np[1], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_pd['x'][0], 1, places=3)
        self.assertAlmostEqual(mle.optimized_params_dict['x'], 1, places=3)

        last_iter = mle.optimized_iterations_np.shape[0] - 1
        self.assertNotEqual(
            mle.optimized_iterations_np[0, 1],
            mle.optimized_iterations_np[last_iter, 1],
        )
        for i in range(3):
            self.assertEqual(
                mle.optimized_params_np[i],
                mle.optimized_iterations_np[last_iter, i],
            )

    def test_eight_schools(self):
        stan = os.path.join(DATAFILES_PATH, 'eight_schools.stan')
        rdata = os.path.join(DATAFILES_PATH, 'eight_schools.data.R')
        model = CmdStanModel(stan_file=stan)
        with self.assertRaises(RuntimeError):
            model.optimize(data=rdata, algorithm='LBFGS')

        mle = model.optimize(
            data=rdata, algorithm='LBFGS', require_converged=False
        )
        self.assertIn('CmdStanMLE: model=eight_schools', repr(mle))
        self.assertIn('method=optimize', repr(mle))
        self.assertFalse(mle.converged)
        with LogCapture() as log:
            self.assertEqual(mle.optimized_params_pd.shape, (1, 11))
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'Invalid estimate, optimization failed to converge.',
            )
        )
        with LogCapture() as log:
            mle.stan_variable('tau')
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'Invalid estimate, optimization failed to converge.',
            )
        )

    def test_variable_bern(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_model = CmdStanModel(stan_file=stan)
        bern_mle = bern_model.optimize(
            data=jdata,
            seed=1239812093,
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
        self.assertEqual(1, len(bern_mle.metadata.stan_vars_dims))
        self.assertIn('theta', bern_mle.metadata.stan_vars_dims)
        self.assertEqual(bern_mle.metadata.stan_vars_dims['theta'], ())
        theta = bern_mle.stan_variable(var='theta')
        self.assertTrue(isinstance(theta, float))
        with self.assertRaises(ValueError):
            bern_mle.stan_variable(var='eta')
        with self.assertRaises(ValueError):
            bern_mle.stan_variable(var='lp__')

    def test_variables_3d(self):
        stan = os.path.join(DATAFILES_PATH, 'multidim_vars.stan')
        jdata = os.path.join(DATAFILES_PATH, 'logistic.data.R')
        multidim_model = CmdStanModel(stan_file=stan)
        multidim_mle = multidim_model.optimize(
            data=jdata,
            seed=1239812093,
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
        self.assertEqual(3, len(multidim_mle.metadata.stan_vars_dims))
        self.assertIn('y_rep', multidim_mle.metadata.stan_vars_dims)
        self.assertEqual(
            multidim_mle.metadata.stan_vars_dims['y_rep'], (5, 4, 3)
        )
        var_y_rep = multidim_mle.stan_variable(var='y_rep')
        self.assertEqual(var_y_rep.shape, (5, 4, 3))
        var_beta = multidim_mle.stan_variable(var='beta')
        self.assertEqual(var_beta.shape, (2,))  # 1-element tuple
        var_frac_60 = multidim_mle.stan_variable(var='frac_60')
        self.assertTrue(isinstance(var_frac_60, float))
        vars = multidim_mle.stan_variables()
        self.assertEqual(len(vars), len(multidim_mle.metadata.stan_vars_dims))
        self.assertIn('y_rep', vars)
        self.assertEqual(vars['y_rep'].shape, (5, 4, 3))
        self.assertIn('beta', vars)
        self.assertEqual(vars['beta'].shape, (2,))
        self.assertIn('frac_60', vars)
        self.assertTrue(isinstance(vars['frac_60'], float))

        multidim_mle_iters = multidim_model.optimize(
            data=jdata,
            seed=1239812093,
            algorithm='LBFGS',
            init_alpha=0.001,
            iter=100,
            tol_obj=1e-12,
            tol_rel_obj=1e4,
            tol_grad=1e-8,
            tol_rel_grad=1e7,
            tol_param=1e-8,
            history_size=5,
            save_iterations=True,
        )
        vars_iters = multidim_mle_iters.stan_variables(inc_iterations=True)
        self.assertEqual(
            len(vars_iters), len(multidim_mle_iters.metadata.stan_vars_dims)
        )
        self.assertIn('y_rep', vars_iters)
        self.assertEqual(vars_iters['y_rep'].shape, (8, 5, 4, 3))
        self.assertIn('beta', vars_iters)
        self.assertEqual(vars_iters['beta'].shape, (8, 2))
        self.assertIn('frac_60', vars_iters)
        self.assertEqual(vars_iters['frac_60'].shape, (8,))


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
                tol_obj=-1.0,
            )

        with self.assertRaisesRegex(ValueError, 'must be greater than'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_rel_obj=-1.0,
            )

        with self.assertRaisesRegex(ValueError, 'must be greater than'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_grad=-1.0,
            )

        with self.assertRaisesRegex(ValueError, 'must be greater than'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_rel_grad=-1.0,
            )

        with self.assertRaisesRegex(ValueError, 'must be greater than'):
            model.optimize(
                data=jdata,
                seed=1239812093,
                inits=jinit,
                algorithm='LBFGS',
                tol_param=-1.0,
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

    def test_single_row_csv(self):
        stan = os.path.join(DATAFILES_PATH, 'matrix_var.stan')
        model = CmdStanModel(stan_file=stan)
        mle = model.optimize()
        self.assertTrue(isinstance(mle.stan_variable('theta'), float))
        z_as_ndarray = mle.stan_variable(var="z")
        self.assertEqual(z_as_ndarray.shape, (4, 3))
        for i in range(4):
            for j in range(3):
                self.assertEqual(int(z_as_ndarray[i, j]), i + 1)

    def test_show_console(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

        sys_stdout = io.StringIO()
        with contextlib.redirect_stdout(sys_stdout):
            bern_model.optimize(
                data=jdata,
                show_console=True,
            )
        console = sys_stdout.getvalue()
        self.assertIn('Chain [1] method = optimize', console)

    def test_exe_only(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        exe_only = os.path.join(DATAFILES_PATH, 'exe_only')
        shutil.copyfile(bern_model.exe_file, exe_only)
        os.chmod(exe_only, 0o755)

        bern2_model = CmdStanModel(exe_file=exe_only)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        mle = bern2_model.optimize(data=jdata)
        self.assertEqual(
            mle.optimized_params_np[0], mle.optimized_params_dict['lp__']
        )
        self.assertEqual(
            mle.optimized_params_np[1], mle.optimized_params_dict['theta']
        )

    def test_complex_output(self):
        stan = os.path.join(DATAFILES_PATH, 'complex_var.stan')
        model = CmdStanModel(stan_file=stan)
        fit = model.optimize()

        self.assertEqual(fit.stan_variable('zs').shape, (2, 3))
        self.assertEqual(fit.stan_variable('z'), 3 + 4j)
        # make sure the name 'imag' isn't magic
        self.assertEqual(fit.stan_variable('imag').shape, (2,))

    def test_attrs(self):
        stan = os.path.join(DATAFILES_PATH, 'named_output.stan')
        model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        fit = model.optimize(data=jdata)

        self.assertEqual(fit.a, 4.5)
        self.assertEqual(fit.b.shape, (3,))
        self.assertIsInstance(fit.theta, float)

        self.assertEqual(fit.stan_variable('thin'), 3.5)

        self.assertIsInstance(fit.optimized_params_np, np.ndarray)
        self.assertEqual(fit.stan_variable('optimized_params_np'), 0)

        with self.assertRaisesRegex(AttributeError, 'Unknown variable name:'):
            dummy = fit.c


if __name__ == '__main__':
    unittest.main()
