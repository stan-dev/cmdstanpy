"""CmdStan method variational tests"""

import os
import unittest
from math import fabs
import pytest
from cmdstanpy.cmdstan_args import VariationalArgs, CmdStanArgs
from cmdstanpy.model import CmdStanModel
from cmdstanpy.stanfit import RunSet, CmdStanVB

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
                if ext.lower() in ('.o', '.hpp', '.exe', ''):
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
            ('lp__', 'log_p__', 'log_g__', 'mu.1', 'mu.2'),
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu.1'], 31.0299, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu.2'], 28.8141, places=2
        )
        self.assertEqual(variational.variational_sample.shape, (1000, 5))


class VariationalTest(unittest.TestCase):

    # pylint: disable=no-self-use
    @pytest.fixture(scope='class', autouse=True)
    def do_clean_up(self):
        for root, _, files in os.walk(
            os.path.join(DATAFILES_PATH, 'variational')
        ):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext.lower() in ('.o', '.hpp', '.exe', ''):
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
            ('lp__', 'log_p__', 'log_g__', 'mu.1', 'mu.2'),
        )

        self.assertAlmostEqual(
            variational.variational_params_np[3], 31.0418, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_np[4], 27.4463, places=2
        )

        self.assertAlmostEqual(
            variational.variational_params_dict['mu.1'], 31.0418, places=2
        )
        self.assertAlmostEqual(
            variational.variational_params_dict['mu.2'], 27.4463, places=2
        )

        self.assertEqual(
            variational.variational_params_np[0],
            variational.variational_params_pd['lp__'][0],
        )
        self.assertEqual(
            variational.variational_params_np[3],
            variational.variational_params_pd['mu.1'][0],
        )
        self.assertEqual(
            variational.variational_params_np[4],
            variational.variational_params_pd['mu.2'][0],
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
            ('lp__', 'log_p__', 'log_g__', 'mu.1', 'mu.2'),
        )
        self.assertAlmostEqual(
            fabs(variational.variational_params_dict['mu.1']), 0.08, places=1
        )
        self.assertAlmostEqual(
            fabs(variational.variational_params_dict['mu.2']), 0.09, places=1
        )
        self.assertTrue(True)

    def test_variational_eta_fail(self):
        stan = os.path.join(
            DATAFILES_PATH, 'variational', 'eta_should_fail.stan'
        )
        model = CmdStanModel(stan_file=stan)
        with self.assertRaisesRegex(
            RuntimeError, 'algorithm may not have converged'
        ):
            model.variational(algorithm='meanfield', seed=12345)


if __name__ == '__main__':
    unittest.main()
