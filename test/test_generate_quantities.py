"""CmdStan method generate_quantities tests"""

import logging
import os
import unittest

from numpy.testing import assert_array_equal, assert_raises
from testfixtures import LogCapture

from cmdstanpy.cmdstan_args import Method
from cmdstanpy.model import CmdStanModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class GenerateQuantitiesTest(unittest.TestCase):
    def test_gen_quantities_csv_files(self):
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
        self.assertIn('CmdStanGQ: model=bernoulli_ppc', bern_gqs.__repr__())
        self.assertIn('method=generate_quantities', bern_gqs.__repr__())

        self.assertEqual(bern_gqs.runset.chains, 4)
        for i in range(bern_gqs.runset.chains):
            self.assertEqual(bern_gqs.runset._retcode(i), 0)
            csv_file = bern_gqs.runset.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))

        self.assertEqual(bern_gqs.generated_quantities.shape, (400, 10))
        self.assertEqual(bern_gqs.generated_quantities_pd.shape, (400, 10))

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
        self.assertEqual(
            bern_gqs.sample_plus_quantities_pd().shape[1],
            bern_gqs.mcmc_sample.draws_pd().shape[1]
            + bern_gqs.generated_quantities_pd.shape[1],
        )

    def test_gen_quantities_csv_files_bad(self):
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

    def test_gen_quanties_mcmc_sample(self):
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
        self.assertIn('CmdStanGQ: model=bernoulli_ppc', bern_gqs.__repr__())
        self.assertIn('method=generate_quantities', bern_gqs.__repr__())

        self.assertEqual(bern_gqs.runset.chains, 4)
        for i in range(bern_gqs.runset.chains):
            self.assertEqual(bern_gqs.runset._retcode(i), 0)
            csv_file = bern_gqs.runset.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))

        self.assertEqual(bern_gqs.generated_quantities.shape, (400, 10))
        self.assertEqual(bern_gqs.generated_quantities_pd.shape, (400, 10))
        self.assertEqual(
            bern_gqs.sample_plus_quantities_pd().shape[1],
            bern_gqs.mcmc_sample.draws_pd().shape[1]
            + bern_gqs.generated_quantities_pd.shape[1],
        )

    def test_gen_quanties_mcmc_sample_save_warmup(self):
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
            self.assertEqual(bern_gqs.generated_quantities.shape, (800, 10))
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'Sample contains saved wormup draws which will be used to '
                'generate additional quantities of interest.',
            )
        )

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
        assert_raises(
            AssertionError,
            assert_array_equal,
            bern_fit.stan_variable(name='y_rep'),
            bern_gqs.stan_variable(name='y_rep'),
        )


if __name__ == '__main__':
    unittest.main()
