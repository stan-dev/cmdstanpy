"""CmdStan method variational tests"""

import os
import unittest
from math import fabs

import pytest
from testfixtures import LogCapture

from cmdstanpy.cmdstan_args import CmdStanArgs, VariationalArgs
from cmdstanpy.model import CmdStanModel
from cmdstanpy.stanfit import CmdStanVB, RunSet, from_csv

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class CmdStanVBTest(unittest.TestCase):

    # pylint: disable=no-self-use
    @pytest.fixture(scope='class', autouse=True)
    def do_clean_up(self):
        for root, _, files in os.walk(
            os.path.join(DATAFILES_PATH, 'variational')
        ):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext.lower() in ('.o', '.d', '.hpp', '.exe', '') and (
                    filename != ".gitignore"
                ):
                    filepath = os.path.join(root, filename)
                    os.remove(filepath)

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
        self.assertIn(
            'CmdStanVB: model=eta_should_be_big', variational.__repr__()
        )
        self.assertIn('method=variational', variational.__repr__())
        self.assertEqual(
            variational.column_names,
            ('lp__', 'log_p__', 'log_g__', 'mu[1]', 'mu[2]'),
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu[1]'], 31.0299, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu[2]'], 28.8141, places=2
        )
        self.assertEqual(variational.variational_sample.shape, (1000, 5))

    def test_instantiate_from_csvfiles(self):
        csvfiles_path = os.path.join(DATAFILES_PATH, 'variational')
        variational = from_csv(path=csvfiles_path)
        self.assertIn(
            'CmdStanVB: model=eta_should_be_big', variational.__repr__()
        )
        self.assertIn('method=variational', variational.__repr__())
        self.assertEqual(
            variational.column_names,
            ('lp__', 'log_p__', 'log_g__', 'mu[1]', 'mu[2]'),
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu[1]'], 31.0299, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu[2]'], 28.8141, places=2
        )
        self.assertEqual(variational.variational_sample.shape, (1000, 5))

    def test_variables(self):
        # pylint: disable=C0103
        stan = os.path.join(
            DATAFILES_PATH, 'variational', 'eta_should_be_big.stan'
        )
        model = CmdStanModel(stan_file=stan)
        variational = model.variational(algorithm='meanfield', seed=12345)
        self.assertEqual(
            variational.column_names,
            ('lp__', 'log_p__', 'log_g__', 'mu[1]', 'mu[2]'),
        )
        self.assertEqual(1, len(variational.metadata.stan_vars_dims))
        self.assertTrue('mu' in variational.metadata.stan_vars_dims)
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
        self.assertTrue('y_rep' in multidim_variational.metadata.stan_vars_dims)
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
        self.assertTrue('y_rep' in vars)
        self.assertEqual(vars['y_rep'].shape, (5, 4, 3))
        self.assertTrue('beta' in vars)
        self.assertEqual(vars['beta'].shape, (2,))
        self.assertTrue('frac_60' in vars)
        self.assertTrue(isinstance(vars['frac_60'], float))


class VariationalTest(unittest.TestCase):

    # pylint: disable=no-self-use
    @pytest.fixture(scope='class', autouse=True)
    def do_clean_up(self):
        for root, _, files in os.walk(
            os.path.join(DATAFILES_PATH, 'variational')
        ):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext.lower() in ('.o', '.d', '.hpp', '.exe', '') and (
                    filename != ".gitignore"
                ):
                    filepath = os.path.join(root, filename)
                    os.remove(filepath)

    def test_variational_good(self):
        stan = os.path.join(
            DATAFILES_PATH, 'variational', 'eta_should_be_big.stan'
        )
        model = CmdStanModel(stan_file=stan)
        variational = model.variational(algorithm='meanfield', seed=12345)
        self.assertEqual(
            variational.column_names,
            ('lp__', 'log_p__', 'log_g__', 'mu[1]', 'mu[2]'),
        )

        self.assertAlmostEqual(
            variational.variational_params_np[3], 31.0418, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_np[4], 27.4463, places=2
        )

        self.assertAlmostEqual(
            variational.variational_params_dict['mu[1]'], 31.0418, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu[2]'], 27.4463, places=2
        )

        self.assertEqual(
            variational.variational_params_np[0],
            variational.variational_params_pd['lp__'][0],
        )
        self.assertEqual(
            variational.variational_params_np[3],
            variational.variational_params_pd['mu[1]'][0],
        )
        self.assertEqual(
            variational.variational_params_np[4],
            variational.variational_params_pd['mu[2]'][0],
        )

        self.assertEqual(variational.variational_sample.shape, (1000, 5))

    def test_variational_missing_args(self):
        self.assertTrue(True)

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
        self.assertTrue(True)

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
        vb_fit = model.variational()
        self.assertTrue(isinstance(vb_fit.stan_variable('theta'), float))
        z_as_ndarray = vb_fit.stan_variable(var="z")
        self.assertEqual(z_as_ndarray.shape, (4, 3))
        for i in range(4):
            for j in range(3):
                self.assertEqual(int(z_as_ndarray[i, j]), i + 1)


if __name__ == '__main__':
    unittest.main()
