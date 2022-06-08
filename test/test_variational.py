"""CmdStan method variational tests"""

import contextlib
import io
import os
import shutil
import unittest
from math import fabs

import numpy as np
from testfixtures import LogCapture

from cmdstanpy.cmdstan_args import CmdStanArgs, VariationalArgs
from cmdstanpy.model import CmdStanModel
from cmdstanpy.stanfit import CmdStanVB, RunSet, from_csv

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class CmdStanVBTest(unittest.TestCase):
    def test_instantiate(self):
        stan = os.path.join(
            DATAFILES_PATH, 'variational', 'eta_should_be_big.stan'
        )
        model = CmdStanModel(stan_file=stan)
        no_data = {}
        args = VariationalArgs(algorithm='meanfield')
        cmdstan_args = CmdStanArgs(
            model_name=model.name,
            model_exe=model.exe_file,
            chain_ids=None,
            data=no_data,
            method_args=args,
        )
        runset = RunSet(args=cmdstan_args, chains=1)
        runset._csv_files = [
            os.path.join(DATAFILES_PATH, 'variational', 'eta_big_output.csv')
        ]
        variational = CmdStanVB(runset)
        self.assertIn('CmdStanVB: model=eta_should_be_big', repr(variational))
        self.assertIn('method=variational', repr(variational))
        self.assertEqual(
            variational.column_names,
            ('lp__', 'log_p__', 'log_g__', 'mu[1]', 'mu[2]'),
        )
        self.assertEqual(variational.eta, 100)

        self.assertAlmostEqual(
            variational.variational_params_dict['mu[1]'], 311.545, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu[2]'], 532.801, places=2
        )
        self.assertEqual(variational.variational_sample.shape, (1000, 5))

    def test_instantiate_from_csvfiles(self):
        csvfiles_path = os.path.join(DATAFILES_PATH, 'variational')
        variational = from_csv(path=csvfiles_path)
        self.assertIn('CmdStanVB: model=eta_should_be_big', repr(variational))
        self.assertIn('method=variational', repr(variational))
        self.assertEqual(
            variational.column_names,
            ('lp__', 'log_p__', 'log_g__', 'mu[1]', 'mu[2]'),
        )
        self.assertEqual(variational.eta, 100)

        self.assertAlmostEqual(
            variational.variational_params_dict['mu[1]'], 311.545, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu[2]'], 532.801, places=2
        )
        self.assertEqual(variational.variational_sample.shape, (1000, 5))

    def test_variables(self):
        # pylint: disable=C0103
        stan = os.path.join(
            DATAFILES_PATH, 'variational', 'eta_should_be_big.stan'
        )
        model = CmdStanModel(stan_file=stan)
        variational = model.variational(algorithm='meanfield', seed=999999)
        self.assertEqual(
            variational.column_names,
            ('lp__', 'log_p__', 'log_g__', 'mu[1]', 'mu[2]'),
        )
        self.assertEqual(1, len(variational.metadata.stan_vars_dims))
        self.assertIn('mu', variational.metadata.stan_vars_dims)
        self.assertEqual(variational.metadata.stan_vars_dims['mu'], (2,))
        mu = variational.stan_variable(var='mu')
        self.assertEqual(mu.shape, (2,))
        with self.assertRaises(ValueError):
            variational.stan_variable(var='eta')
        with self.assertRaises(ValueError):
            variational.stan_variable(var='lp__')

    def test_variables_3d(self):
        # construct fit using existing sampler output
        stan = os.path.join(DATAFILES_PATH, 'multidim_vars.stan')
        jdata = os.path.join(DATAFILES_PATH, 'logistic.data.R')
        multidim_model = CmdStanModel(stan_file=stan)
        multidim_variational = multidim_model.variational(
            data=jdata,
            seed=1239812093,
            algorithm='meanfield',
        )
        self.assertEqual(3, len(multidim_variational.metadata.stan_vars_dims))
        self.assertIn('y_rep', multidim_variational.metadata.stan_vars_dims)
        self.assertEqual(
            multidim_variational.metadata.stan_vars_dims['y_rep'], (5, 4, 3)
        )
        var_y_rep = multidim_variational.stan_variable(var='y_rep')
        self.assertEqual(var_y_rep.shape, (5, 4, 3))
        var_beta = multidim_variational.stan_variable(var='beta')
        self.assertEqual(var_beta.shape, (2,))  # 1-element tuple
        var_frac_60 = multidim_variational.stan_variable(var='frac_60')
        self.assertTrue(isinstance(var_frac_60, float))
        vars = multidim_variational.stan_variables()
        self.assertEqual(
            len(vars), len(multidim_variational.metadata.stan_vars_dims)
        )
        self.assertIn('y_rep', vars)
        self.assertEqual(vars['y_rep'].shape, (5, 4, 3))
        self.assertIn('beta', vars)
        self.assertEqual(vars['beta'].shape, (2,))
        self.assertIn('frac_60', vars)
        self.assertTrue(isinstance(vars['frac_60'], float))


class VariationalTest(unittest.TestCase):
    def test_variational_good(self):
        stan = os.path.join(
            DATAFILES_PATH, 'variational', 'eta_should_be_big.stan'
        )
        model = CmdStanModel(stan_file=stan)
        variational = model.variational(algorithm='meanfield', seed=999999)
        self.assertEqual(
            variational.column_names,
            ('lp__', 'log_p__', 'log_g__', 'mu[1]', 'mu[2]'),
        )
        # fixed seed, id=1 by default will give known output values
        self.assertEqual(variational.eta, 100)
        self.assertAlmostEqual(
            variational.variational_params_dict['mu[1]'], 311.545, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu[2]'], 532.801, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_np[0],
            variational.variational_params_pd['lp__'][0],
        )
        self.assertEqual(
            variational.variational_params_np[3],
            variational.variational_params_dict['mu[1]'],
        )
        self.assertAlmostEqual(
            variational.variational_params_np[4],
            variational.variational_params_dict['mu[2]'],
        )
        self.assertEqual(variational.variational_sample.shape, (1000, 5))

    def test_variational_eta_small(self):
        stan = os.path.join(
            DATAFILES_PATH, 'variational', 'eta_should_be_small.stan'
        )
        model = CmdStanModel(stan_file=stan)
        variational = model.variational(algorithm='meanfield', seed=12345)
        self.assertEqual(
            variational.column_names,
            ('lp__', 'log_p__', 'log_g__', 'mu[1]', 'mu[2]'),
        )
        self.assertAlmostEqual(
            fabs(variational.variational_params_dict['mu[1]']), 0.08, places=1
        )
        self.assertAlmostEqual(
            fabs(variational.variational_params_dict['mu[2]']), 0.09, places=1
        )

    def test_variational_eta_fail(self):
        stan = os.path.join(
            DATAFILES_PATH, 'variational', 'eta_should_fail.stan'
        )
        model = CmdStanModel(stan_file=stan)
        with self.assertRaisesRegex(
            RuntimeError,
            r'algorithm may not have converged\.\n.*require_converged',
        ):
            model.variational(algorithm='meanfield', seed=12345)

        with LogCapture() as log:
            model.variational(
                algorithm='meanfield', seed=12345, require_converged=False
            )
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'The algorithm may not have converged.\n'
                'Proceeding because require_converged is set to False',
            )
        )

    def test_single_row_csv(self):
        stan = os.path.join(DATAFILES_PATH, 'matrix_var.stan')
        model = CmdStanModel(stan_file=stan)
        # testing data parsing, allow non-convergence
        vb_fit = model.variational(require_converged=False, seed=12345)
        self.assertTrue(isinstance(vb_fit.stan_variable('theta'), float))
        z_as_ndarray = vb_fit.stan_variable(var="z")
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
            # testing data parsing, allow non-convergence
            bern_model.variational(
                data=jdata,
                show_console=True,
                require_converged=False,
                seed=12345,
            )
        console = sys_stdout.getvalue()
        self.assertIn('Chain [1] method = variational', console)

    def test_exe_only(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        exe_only = os.path.join(DATAFILES_PATH, 'exe_only')
        shutil.copyfile(bern_model.exe_file, exe_only)
        os.chmod(exe_only, 0o755)

        bern2_model = CmdStanModel(exe_file=exe_only)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        variational = bern2_model.variational(
            data=jdata,
            require_converged=False,
            seed=12345,
            algorithm='meanfield',
        )
        self.assertEqual(variational.variational_sample.shape, (1000, 4))

    def test_complex_output(self):
        stan = os.path.join(DATAFILES_PATH, 'complex_var.stan')
        model = CmdStanModel(stan_file=stan)
        fit = model.variational(
            require_converged=False,
            seed=12345,
            algorithm='meanfield',
        )

        self.assertEqual(fit.stan_variable('zs').shape, (2, 3))
        self.assertEqual(fit.stan_variable('z'), 3 + 4j)
        # make sure the name 'imag' isn't magic
        self.assertEqual(fit.stan_variable('imag').shape, (2,))

    def test_attrs(self):
        stan = os.path.join(DATAFILES_PATH, 'named_output.stan')
        model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        fit = model.variational(
            data=jdata,
            require_converged=False,
            seed=12345,
            algorithm='meanfield',
        )

        self.assertEqual(fit.a, 4.5)
        self.assertEqual(fit.b.shape, (3,))
        self.assertIsInstance(fit.theta, float)

        self.assertEqual(fit.stan_variable('thin'), 3.5)

        self.assertIsInstance(fit.variational_params_np, np.ndarray)
        self.assertEqual(fit.stan_variable('variational_params_np'), 0)

        with self.assertRaisesRegex(AttributeError, 'Unknown variable name:'):
            dummy = fit.c


if __name__ == '__main__':
    unittest.main()
