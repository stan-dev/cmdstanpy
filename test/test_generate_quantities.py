"""CmdStan method generate_quantities tests"""

import contextlib
import io
import json
import logging
import os
import unittest
from test import CustomTestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_raises
from testfixtures import LogCapture

import cmdstanpy.stanfit
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.model import CmdStanModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class GenerateQuantitiesTest(CustomTestCase):
    def test_from_csv_files(self):
        # fitted_params sample - list of filenames
        goodfiles_path = os.path.join(DATAFILES_PATH, 'runset-good', 'bern')
        csv_files = []
        for i in range(4):
            csv_files.append('{}-{}.csv'.format(goodfiles_path, i + 1))

        # gq_model
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

        bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=csv_files)

        self.assertEqual(
            bern_gqs.runset._args.method, Method.GENERATE_QUANTITIES
        )
        self.assertIn('CmdStanGQ: model=bernoulli_ppc', repr(bern_gqs))
        self.assertIn('method=generate_quantities', repr(bern_gqs))

        self.assertEqual(bern_gqs.runset.chains, 4)
        for i in range(bern_gqs.runset.chains):
            self.assertEqual(bern_gqs.runset._retcode(i), 0)
            csv_file = bern_gqs.runset.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))

        column_names = [
            'y_rep[1]',
            'y_rep[2]',
            'y_rep[3]',
            'y_rep[4]',
            'y_rep[5]',
            'y_rep[6]',
            'y_rep[7]',
            'y_rep[8]',
            'y_rep[9]',
            'y_rep[10]',
        ]
        self.assertEqual(bern_gqs.column_names, tuple(column_names))

        # draws()
        self.assertEqual(bern_gqs.draws().shape, (100, 4, 10))
        with LogCapture() as log:
            logging.getLogger()
            bern_gqs.draws(inc_warmup=True)
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                "Sample doesn't contain draws from warmup iterations, "
                'rerun sampler with "save_warmup=True".',
            )
        )

        # draws_pd()
        self.assertEqual(bern_gqs.draws_pd().shape, (400, 10))
        self.assertEqual(
            bern_gqs.draws_pd(inc_sample=True).shape[1],
            bern_gqs.mcmc_sample.draws_pd().shape[1]
            + bern_gqs.draws_pd().shape[1],
        )

        self.assertEqual(
            list(bern_gqs.draws_pd(vars=['y_rep']).columns),
            column_names,
        )

    def test_from_csv_files_bad(self):
        # gq model
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

        # no filename
        with self.assertRaises(ValueError):
            model.generate_quantities(data=jdata, mcmc_sample=[])

        # Stan CSV flles corrupted
        goodfiles_path = os.path.join(
            DATAFILES_PATH, 'runset-bad', 'bad-draws-bern'
        )
        csv_files = []
        for i in range(4):
            csv_files.append('{}-{}.csv'.format(goodfiles_path, i + 1))

        with self.assertRaisesRegex(
            Exception, 'Invalid sample from Stan CSV files'
        ):
            model.generate_quantities(data=jdata, mcmc_sample=csv_files)

    def test_from_mcmc_sample(self):
        # fitted_params sample
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_fit = bern_model.sample(
            data=jdata,
            chains=4,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
        )
        # gq_model
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)

        bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)

        self.assertEqual(
            bern_gqs.runset._args.method, Method.GENERATE_QUANTITIES
        )
        self.assertIn('CmdStanGQ: model=bernoulli_ppc', repr(bern_gqs))
        self.assertIn('method=generate_quantities', repr(bern_gqs))
        self.assertEqual(bern_gqs.runset.chains, 4)
        for i in range(bern_gqs.runset.chains):
            self.assertEqual(bern_gqs.runset._retcode(i), 0)
            csv_file = bern_gqs.runset.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))

    def test_from_mcmc_sample_draws(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_fit = bern_model.sample(
            data=jdata,
            chains=4,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
        )
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)

        bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)

        self.assertEqual(bern_gqs.draws_pd().shape, (400, 10))
        self.assertEqual(
            bern_gqs.draws_pd(inc_sample=True).shape[1],
            bern_gqs.mcmc_sample.draws_pd().shape[1]
            + bern_gqs.draws_pd().shape[1],
        )
        row1_sample_pd = bern_fit.draws_pd().iloc[0]
        row1_gqs_pd = bern_gqs.draws_pd().iloc[0]
        self.assertTrue(
            np.array_equal(
                pd.concat((row1_sample_pd, row1_gqs_pd), axis=0).values,
                bern_gqs.draws_pd(inc_sample=True).iloc[0].values,
            )
        )
        # draws_xr
        xr_data = bern_gqs.draws_xr()
        self.assertEqual(xr_data.y_rep.dims, ('chain', 'draw', 'y_rep_dim_0'))
        self.assertEqual(xr_data.y_rep.values.shape, (4, 100, 10))

        xr_var = bern_gqs.draws_xr(vars='y_rep')
        self.assertEqual(xr_var.y_rep.dims, ('chain', 'draw', 'y_rep_dim_0'))
        self.assertEqual(xr_var.y_rep.values.shape, (4, 100, 10))

        xr_var = bern_gqs.draws_xr(vars=['y_rep'])
        self.assertEqual(xr_var.y_rep.dims, ('chain', 'draw', 'y_rep_dim_0'))
        self.assertEqual(xr_var.y_rep.values.shape, (4, 100, 10))

        xr_data_plus = bern_gqs.draws_xr(inc_sample=True)
        self.assertEqual(
            xr_data_plus.y_rep.dims, ('chain', 'draw', 'y_rep_dim_0')
        )
        self.assertEqual(xr_data_plus.y_rep.values.shape, (4, 100, 10))
        self.assertEqual(xr_data_plus.theta.dims, ('chain', 'draw'))
        self.assertEqual(xr_data_plus.theta.values.shape, (4, 100))

    def test_from_mcmc_sample_variables(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_fit = bern_model.sample(
            data=jdata,
            chains=4,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
        )
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)

        bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)

        theta = bern_gqs.stan_variable(var='theta')
        self.assertEqual(theta.shape, (400,))
        y_rep = bern_gqs.stan_variable(var='y_rep')
        self.assertEqual(y_rep.shape, (400, 10))
        with self.assertRaises(ValueError):
            bern_gqs.stan_variable(var='eta')
        with self.assertRaises(ValueError):
            bern_gqs.stan_variable(var='lp__')

        vars_dict = bern_gqs.stan_variables()
        var_names = list(
            bern_gqs.mcmc_sample.metadata.stan_vars_cols.keys()
        ) + list(bern_gqs.metadata.stan_vars_cols.keys())
        self.assertEqual(set(var_names), set(list(vars_dict.keys())))

    def test_save_warmup(self):
        # fitted_params sample
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_fit = bern_model.sample(
            data=jdata,
            chains=4,
            parallel_chains=2,
            seed=12345,
            iter_warmup=100,
            iter_sampling=100,
            save_warmup=True,
        )
        # gq_model
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)

        with LogCapture() as log:
            logging.getLogger()
            bern_gqs = model.generate_quantities(
                data=jdata, mcmc_sample=bern_fit
            )
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'Sample contains saved warmup draws which will be used to '
                'generate additional quantities of interest.',
            )
        )
        self.assertEqual(bern_gqs.draws().shape, (100, 4, 10))
        self.assertEqual(
            bern_gqs.draws(concat_chains=False, inc_warmup=False).shape,
            (100, 4, 10),
        )
        self.assertEqual(
            bern_gqs.draws(concat_chains=False, inc_warmup=True).shape,
            (200, 4, 10),
        )
        self.assertEqual(
            bern_gqs.draws(concat_chains=True, inc_warmup=False).shape,
            (400, 10),
        )
        self.assertEqual(
            bern_gqs.draws(concat_chains=True, inc_warmup=True).shape,
            (800, 10),
        )

        self.assertEqual(bern_gqs.draws_pd().shape, (400, 10))
        self.assertEqual(bern_gqs.draws_pd(inc_warmup=False).shape, (400, 10))
        self.assertEqual(bern_gqs.draws_pd(inc_warmup=True).shape, (800, 10))
        self.assertEqual(
            bern_gqs.draws_pd(vars=['y_rep'], inc_warmup=False).shape,
            (400, 10),
        )
        self.assertEqual(
            bern_gqs.draws_pd(vars='y_rep', inc_warmup=False).shape,
            (400, 10),
        )

        theta = bern_gqs.stan_variable(var='theta')
        self.assertEqual(theta.shape, (400,))
        y_rep = bern_gqs.stan_variable(var='y_rep')
        self.assertEqual(y_rep.shape, (400, 10))
        with self.assertRaises(ValueError):
            bern_gqs.stan_variable(var='eta')
        with self.assertRaises(ValueError):
            bern_gqs.stan_variable(var='lp__')

        vars_dict = bern_gqs.stan_variables()
        var_names = list(
            bern_gqs.mcmc_sample.metadata.stan_vars_cols.keys()
        ) + list(bern_gqs.metadata.stan_vars_cols.keys())
        self.assertEqual(set(var_names), set(list(vars_dict.keys())))

        xr_data = bern_gqs.draws_xr()
        self.assertEqual(xr_data.y_rep.dims, ('chain', 'draw', 'y_rep_dim_0'))
        self.assertEqual(xr_data.y_rep.values.shape, (4, 100, 10))

        xr_data_plus = bern_gqs.draws_xr(inc_sample=True)
        self.assertEqual(
            xr_data_plus.y_rep.dims, ('chain', 'draw', 'y_rep_dim_0')
        )
        self.assertEqual(xr_data_plus.y_rep.values.shape, (4, 100, 10))
        self.assertEqual(xr_data_plus.theta.dims, ('chain', 'draw'))
        self.assertEqual(xr_data_plus.theta.values.shape, (4, 100))

        xr_data_plus = bern_gqs.draws_xr(vars='theta', inc_sample=True)
        self.assertEqual(xr_data_plus.theta.dims, ('chain', 'draw'))
        self.assertEqual(xr_data_plus.theta.values.shape, (4, 100))

        xr_data_plus = bern_gqs.draws_xr(inc_sample=True, inc_warmup=True)
        self.assertEqual(
            xr_data_plus.y_rep.dims, ('chain', 'draw', 'y_rep_dim_0')
        )
        self.assertEqual(xr_data_plus.y_rep.values.shape, (4, 200, 10))
        self.assertEqual(xr_data_plus.theta.dims, ('chain', 'draw'))
        self.assertEqual(xr_data_plus.theta.values.shape, (4, 200))

    def test_sample_plus_quantities_dedup(self):
        # fitted_params - model GQ block: y_rep is PPC of theta
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_fit = model.sample(
            data=jdata,
            chains=4,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
        )
        # gq_model - y_rep[n] == y[n]
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc_dup.stan')
        model = CmdStanModel(stan_file=stan)
        bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)
        # check that models have different y_rep values
        assert_raises(
            AssertionError,
            assert_array_equal,
            bern_fit.stan_variable(var='y_rep'),
            bern_gqs.stan_variable(var='y_rep'),
        )
        # check that stan_variable returns values from gq model
        with open(jdata) as fd:
            bern_data = json.load(fd)
        y_rep = bern_gqs.stan_variable(var='y_rep')
        for i in range(10):
            self.assertEqual(y_rep[0, i], bern_data['y'][i])

    def test_no_xarray(self):
        with self.without_import('xarray', cmdstanpy.stanfit.gq):
            with self.assertRaises(ImportError):
                # if this fails the testing framework is the problem
                import xarray as _  # noqa

            stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
            bern_model = CmdStanModel(stan_file=stan)
            jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
            bern_fit = bern_model.sample(
                data=jdata,
                chains=4,
                parallel_chains=2,
                seed=12345,
                iter_sampling=100,
            )
            stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
            model = CmdStanModel(stan_file=stan)

            bern_gqs = model.generate_quantities(
                data=jdata, mcmc_sample=bern_fit
            )

            with self.assertRaises(RuntimeError):
                bern_gqs.draws_xr()

    def test_single_row_csv(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_fit = bern_model.sample(
            data=jdata,
            chains=1,
            seed=12345,
            iter_sampling=1,
        )
        stan = os.path.join(DATAFILES_PATH, 'matrix_var.stan')
        model = CmdStanModel(stan_file=stan)
        gqs = model.generate_quantities(mcmc_sample=bern_fit)
        z_as_ndarray = gqs.stan_variable(var="z")
        self.assertEqual(z_as_ndarray.shape, (1, 4, 3))  # flattens chains
        z_as_xr = gqs.draws_xr(vars="z")
        self.assertEqual(z_as_xr.z.data.shape, (1, 1, 4, 3))  # keeps chains
        for i in range(4):
            for j in range(3):
                self.assertEqual(int(z_as_ndarray[0, i, j]), i + 1)
                self.assertEqual(int(z_as_xr.z.data[0, 0, i, j]), i + 1)

    def test_show_console(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_fit = bern_model.sample(
            data=jdata,
            chains=4,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
        )
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)

        sys_stdout = io.StringIO()
        with contextlib.redirect_stdout(sys_stdout):
            model.generate_quantities(
                data=jdata,
                mcmc_sample=bern_fit,
                show_console=True,
            )
        console = sys_stdout.getvalue()
        self.assertIn('Chain [1] method = generate', console)
        self.assertIn('Chain [2] method = generate', console)
        self.assertIn('Chain [3] method = generate', console)
        self.assertIn('Chain [4] method = generate', console)

    def test_complex_output(self):
        stan_bern = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        model_bern = CmdStanModel(stan_file=stan_bern)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        fit_sampling = model_bern.sample(chains=1, iter_sampling=10, data=jdata)

        stan = os.path.join(DATAFILES_PATH, 'complex_var.stan')
        model = CmdStanModel(stan_file=stan)
        fit = model.generate_quantities(mcmc_sample=fit_sampling)

        self.assertEqual(fit.stan_variable('zs').shape, (10, 2, 3))
        self.assertEqual(fit.stan_variable('z')[0], 3 + 4j)
        # make sure the name 'imag' isn't magic
        self.assertEqual(fit.stan_variable('imag').shape, (10, 2))

        self.assertNotIn("zs_dim_2", fit.draws_xr())
        # getting a raw scalar out of xarray is heavy
        self.assertEqual(
            fit.draws_xr().z.isel(chain=0, draw=1).data[()], 3 + 4j
        )

    def test_attrs(self):
        stan_bern = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        model_bern = CmdStanModel(stan_file=stan_bern)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        fit_sampling = model_bern.sample(chains=1, iter_sampling=10, data=jdata)

        stan = os.path.join(DATAFILES_PATH, 'named_output.stan')
        model = CmdStanModel(stan_file=stan)
        fit = model.generate_quantities(data=jdata, mcmc_sample=fit_sampling)

        self.assertEqual(fit.a[0], 4.5)
        self.assertEqual(fit.b.shape, (10, 3))
        self.assertEqual(fit.theta.shape, (10,))

        fit.draws()
        self.assertEqual(fit.stan_variable('draws')[0], 0)

        with self.assertRaisesRegex(AttributeError, 'Unknown variable name:'):
            dummy = fit.c


if __name__ == '__main__':
    unittest.main()
