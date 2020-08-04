"""CmdStan method generate_quantities tests"""

import os
import unittest

from cmdstanpy.cmdstan_args import Method
from cmdstanpy.model import CmdStanModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class GenerateQuantitiesTest(unittest.TestCase):
    def test_gen_quantities_csv_files(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)

        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

        # synthesize list of filenames
        goodfiles_path = os.path.join(DATAFILES_PATH, 'runset-good', 'bern')
        csv_files = []
        for i in range(4):
            csv_files.append('{}-{}.csv'.format(goodfiles_path, i + 1))

        bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=csv_files)
        self.assertEqual(
            bern_gqs.runset._args.method, Method.GENERATE_QUANTITIES
        )
        self.assertIn('CmdStanGQ: model=bernoulli_ppc', bern_gqs.__repr__())
        self.assertIn('method=generate_quantities', bern_gqs.__repr__())

        # check results - ouput files, quantities of interest, draws
        self.assertEqual(bern_gqs.runset.chains, 4)
        for i in range(bern_gqs.runset.chains):
            self.assertEqual(bern_gqs.runset._retcode(i), 0)
            csv_file = bern_gqs.runset.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))
        column_names = [
            'y_rep.1',
            'y_rep.2',
            'y_rep.3',
            'y_rep.4',
            'y_rep.5',
            'y_rep.6',
            'y_rep.7',
            'y_rep.8',
            'y_rep.9',
            'y_rep.10',
        ]
        self.assertEqual(bern_gqs.column_names, tuple(column_names))
        self.assertEqual(
            bern_gqs.sample_plus_quantities.shape[1],
            bern_gqs.mcmc_sample.shape[1]
            + bern_gqs.generated_quantities_pd.shape[1],
        )

    def test_gen_quantities_csv_files_bad(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

        # synthesize list of filenames
        goodfiles_path = os.path.join(
            DATAFILES_PATH, 'runset-bad', 'bad-draws-bern'
        )
        csv_files = []
        for i in range(4):
            csv_files.append('{}-{}.csv'.format(goodfiles_path, i + 1))

        with self.assertRaisesRegex(Exception, 'Invalid mcmc_sample'):
            model.generate_quantities(data=jdata, mcmc_sample=csv_files)

    def test_gen_quanties_mcmc_sample(self):
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
        self.assertEqual(
            bern_gqs.runset._args.method, Method.GENERATE_QUANTITIES
        )
        self.assertIn('CmdStanGQ: model=bernoulli_ppc', bern_gqs.__repr__())
        self.assertIn('method=generate_quantities', bern_gqs.__repr__())

        # check results - ouput files, quantities of interest, draws
        self.assertEqual(bern_gqs.runset.chains, 4)
        for i in range(bern_gqs.runset.chains):
            self.assertEqual(bern_gqs.runset._retcode(i), 0)
            csv_file = bern_gqs.runset.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))
        column_names = [
            'y_rep.1',
            'y_rep.2',
            'y_rep.3',
            'y_rep.4',
            'y_rep.5',
            'y_rep.6',
            'y_rep.7',
            'y_rep.8',
            'y_rep.9',
            'y_rep.10',
        ]
        self.assertEqual(bern_gqs.column_names, tuple(column_names))
        self.assertEqual(
            bern_fit.get_drawset().shape, bern_gqs.mcmc_sample.shape
        )
        self.assertEqual(
            bern_gqs.sample_plus_quantities.shape[1],
            bern_gqs.mcmc_sample.shape[1]
            + bern_gqs.generated_quantities_pd.shape[1],
        )

    def test_sample_plus_quantities_dedup(self):
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
        bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)
        self.assertEqual(
            bern_gqs.sample_plus_quantities.shape[1],
            bern_gqs.mcmc_sample.shape[1],
        )


if __name__ == '__main__':
    unittest.main()
