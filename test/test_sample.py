"""CmdStan method sample tests"""

import os
import platform
import logging
import shutil
from multiprocessing import cpu_count
import tempfile
import stat
import unittest
from time import time
from testfixtures import LogCapture
import pytest

try:
    import ujson as json
except ImportError:
    import json

from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import Method, SamplerArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION, cmdstan_version_at
from cmdstanpy.stanfit import RunSet, CmdStanMCMC
from cmdstanpy.model import CmdStanModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')
GOODFILES_PATH = os.path.join(DATAFILES_PATH, 'runset-good')
BADFILES_PATH = os.path.join(DATAFILES_PATH, 'runset-bad')
SAMPLER_STATE = [
    'lp__',
    'accept_stat__',
    'stepsize__',
    'treedepth__',
    'n_leapfrog__',
    'divergent__',
    'energy__',
]
BERNOULLI_COLS = SAMPLER_STATE + ['theta']


class SampleTest(unittest.TestCase):

    # pylint: disable=no-self-use
    @pytest.fixture(scope='class', autouse=True)
    def do_clean_up(self):
        for root, _, files in os.walk(DATAFILES_PATH):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext.lower() in ('.o', '.hpp', '.exe', ''):
                    filepath = os.path.join(root, filename)
                    os.remove(filepath)

    def test_bernoulli_good(self, stanfile='bernoulli.stan'):
        stan = os.path.join(DATAFILES_PATH, stanfile)
        bern_model = CmdStanModel(stan_file=stan)

        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
        )
        self.assertIn('CmdStanMCMC: model=bernoulli', bern_fit.__repr__())
        self.assertIn('method=sample', bern_fit.__repr__())

        self.assertEqual(bern_fit.runset._args.method, Method.SAMPLE)

        for i in range(bern_fit.runset.chains):
            csv_file = bern_fit.runset.csv_files[i]
            stdout_file = bern_fit.runset.stdout_files[i]
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(stdout_file))

        self.assertEqual(bern_fit.runset.chains, 2)
        self.assertEqual(bern_fit.num_draws, 100)
        self.assertEqual(bern_fit.column_names, tuple(BERNOULLI_COLS))

        bern_draws = bern_fit.draws()
        self.assertEqual(bern_draws.shape, (100, 2, len(BERNOULLI_COLS)))
        self.assertEqual(bern_fit.metric_type, 'diag_e')
        self.assertEqual(bern_fit.stepsize.shape, (2,))
        self.assertEqual(bern_fit.metric.shape, (2, 1))

        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_warmup=1000,
            iter_sampling=100,
            metric='dense_e',
        )
        self.assertIn('CmdStanMCMC: model=bernoulli', bern_fit.__repr__())
        self.assertIn('method=sample', bern_fit.__repr__())

        self.assertEqual(bern_fit.runset._args.method, Method.SAMPLE)

        for i in range(bern_fit.runset.chains):
            csv_file = bern_fit.runset.csv_files[i]
            stdout_file = bern_fit.runset.stdout_files[i]
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(stdout_file))

        self.assertEqual(bern_fit.runset.chains, 2)
        self.assertEqual(bern_fit.num_draws, 100)
        self.assertEqual(bern_fit.column_names, tuple(BERNOULLI_COLS))

        bern_sample = bern_fit.sample
        self.assertEqual(bern_sample.shape, (100, 2, len(BERNOULLI_COLS)))
        self.assertEqual(bern_fit.metric_type, 'dense_e')
        self.assertEqual(bern_fit.stepsize.shape, (2,))
        self.assertEqual(bern_fit.metric.shape, (2, 1, 1))

        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
            output_dir=DATAFILES_PATH,
        )
        for i in range(bern_fit.runset.chains):
            csv_file = bern_fit.runset.csv_files[i]
            stdout_file = bern_fit.runset.stdout_files[i]
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(stdout_file))
        bern_draws = bern_fit.draws()
        self.assertEqual(bern_draws.shape, (100, 2, len(BERNOULLI_COLS)))
        for i in range(bern_fit.runset.chains):  # cleanup datafile_path dir
            os.remove(bern_fit.runset.csv_files[i])
            if os.path.exists(bern_fit.runset.stdout_files[i]):
                os.remove(bern_fit.runset.stdout_files[i])
            if os.path.exists(bern_fit.runset.stderr_files[i]):
                os.remove(bern_fit.runset.stderr_files[i])
        rdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.R')
        bern_fit = bern_model.sample(
            data=rdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
        )
        bern_draws = bern_fit.draws()
        self.assertEqual(bern_draws.shape, (100, 2, len(BERNOULLI_COLS)))

        data_dict = {'N': 10, 'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
        bern_fit = bern_model.sample(
            data=data_dict,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
        )
        bern_draws = bern_fit.draws()
        self.assertEqual(bern_draws.shape, (100, 2, len(BERNOULLI_COLS)))

    def test_init_types(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
            inits=1.1,
        )
        self.assertIn('init=1.1', bern_fit.runset.__repr__())

        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
            inits=1,
        )
        self.assertIn('init=1', bern_fit.runset.__repr__())

        # Save init to json
        inits_path1 = os.path.join(_TMPDIR, 'inits_test_1.json')
        with open(inits_path1, 'w') as fd:
            json.dump({'theta': 0.1}, fd)
        inits_path2 = os.path.join(_TMPDIR, 'inits_test_2.json')
        with open(inits_path2, 'w') as fd:
            json.dump({'theta': 0.9}, fd)

        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
            inits=inits_path1,
        )
        self.assertIn(
            'init={}'.format(inits_path1.replace('\\', '\\\\')),
            bern_fit.runset.__repr__(),
        )

        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
            inits=[inits_path1, inits_path2],
        )
        self.assertIn(
            'init={}'.format(inits_path1.replace('\\', '\\\\')),
            bern_fit.runset.__repr__(),
        )

        with self.assertRaises(ValueError):
            bern_model.sample(
                data=jdata,
                chains=2,
                parallel_chains=2,
                seed=12345,
                iter_sampling=100,
                inits=(1, 2),
            )

        with self.assertRaises(ValueError):
            bern_model.sample(
                data=jdata,
                chains=2,
                parallel_chains=2,
                seed=12345,
                iter_sampling=100,
                inits=-1,
            )

    def test_bernoulli_bad(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)

        with self.assertRaisesRegex(RuntimeError, 'variable does not exist'):
            bern_model.sample(
                chains=2, parallel_chains=2, seed=12345, iter_sampling=100
            )

        with self.assertRaisesRegex(RuntimeError, 'variable does not exist'):
            bern_model.sample(
                data={'foo': 1},
                chains=2,
                parallel_chains=2,
                seed=12345,
                iter_sampling=100,
            )
        if platform.system() != 'Windows':
            jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
            dirname1 = 'tmp1' + str(time())
            os.mkdir(dirname1, mode=644)
            dirname2 = 'tmp2' + str(time())
            path = os.path.join(dirname1, dirname2)
            with self.assertRaisesRegex(
                ValueError, 'invalid path for output files'
            ):
                bern_model.sample(data=jdata, chains=1, output_dir=path)
            os.rmdir(dirname1)

    def test_multi_proc(self):
        logistic_stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
        logistic_model = CmdStanModel(stan_file=logistic_stan)
        logistic_data = os.path.join(DATAFILES_PATH, 'logistic.data.R')

        with LogCapture() as log:
            logging.getLogger()
            logistic_model.sample(
                data=logistic_data, chains=4, parallel_chains=1
            )
        log.check_present(
            ('cmdstanpy', 'INFO', 'finish chain 1'),
            ('cmdstanpy', 'INFO', 'start chain 2'),
        )
        with LogCapture() as log:
            logging.getLogger()
            logistic_model.sample(
                data=logistic_data, chains=4, parallel_chains=2
            )
        if cpu_count() >= 4:
            # finish chains 1, 2 before starting chains 3, 4
            log.check_present(
                ('cmdstanpy', 'INFO', 'finish chain 1'),
                ('cmdstanpy', 'INFO', 'start chain 4'),
            )
        if cpu_count() >= 4:
            with LogCapture() as log:
                logging.getLogger()
                logistic_model.sample(
                    data=logistic_data, chains=4, parallel_chains=4
                )
                log.check_present(
                    ('cmdstanpy', 'INFO', 'start chain 4'),
                    ('cmdstanpy', 'INFO', 'finish chain 1'),
                )

        with LogCapture() as log:
            logging.getLogger()
            logistic_model.sample(
                data=logistic_data,
                chains=1,
                parallel_chains=1,
                threads_per_chain=7,
            )
        log.check_present(('cmdstanpy', 'DEBUG', 'total threads: 7'))
        with LogCapture() as log:
            logging.getLogger()
            logistic_model.sample(
                data=logistic_data,
                chains=7,
                parallel_chains=1,
                threads_per_chain=5,
            )
        log.check_present(('cmdstanpy', 'DEBUG', 'total threads: 5'))
        with LogCapture() as log:
            logging.getLogger()
            logistic_model.sample(
                data=logistic_data,
                chains=1,
                parallel_chains=7,
                threads_per_chain=5,
            )
        log.check_present(
            (
                'cmdstanpy',
                'INFO',
                'Requesting 7 parallel_chains for 1 chains, '
                'running all chains in parallel.',
            )
        )
        with LogCapture() as log:
            logging.getLogger()
            logistic_model.sample(
                data=logistic_data, chains=7, threads_per_chain=5
            )
            cores = max(min(cpu_count(), 7), 1)
            expect = 'total threads: {}'.format(cores * 5)
        log.check_present(('cmdstanpy', 'DEBUG', expect))
        with self.assertRaisesRegex(
            ValueError, 'parallel_chains must be a positive integer'
        ):
            logistic_model.sample(
                data=logistic_data, chains=4, parallel_chains=-4
            )
        with self.assertRaisesRegex(
            ValueError, 'threads_per_chain must be a positive integer'
        ):
            logistic_model.sample(
                data=logistic_data, chains=4, threads_per_chain=-4
            )

    def test_fixed_param_good(self):
        stan = os.path.join(DATAFILES_PATH, 'datagen_poisson_glm.stan')
        datagen_model = CmdStanModel(stan_file=stan)
        no_data = {}
        datagen_fit = datagen_model.sample(
            data=no_data, seed=12345, iter_sampling=100, fixed_param=True
        )
        self.assertEqual(datagen_fit.runset._args.method, Method.SAMPLE)
        self.assertEqual(datagen_fit.metric_type, None)
        self.assertEqual(datagen_fit.metric, None)
        self.assertEqual(datagen_fit.stepsize, None)

        for i in range(datagen_fit.runset.chains):
            csv_file = datagen_fit.runset.csv_files[i]
            stdout_file = datagen_fit.runset.stdout_files[i]
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(stdout_file))

        self.assertEqual(datagen_fit.runset.chains, 1)

        column_names = [
            'lp__',
            'accept_stat__',
            'N',
            'y_sim[1]',
            'y_sim[2]',
            'y_sim[3]',
            'y_sim[4]',
            'y_sim[5]',
            'y_sim[6]',
            'y_sim[7]',
            'y_sim[8]',
            'y_sim[9]',
            'y_sim[10]',
            'y_sim[11]',
            'y_sim[12]',
            'y_sim[13]',
            'y_sim[14]',
            'y_sim[15]',
            'y_sim[16]',
            'y_sim[17]',
            'y_sim[18]',
            'y_sim[19]',
            'y_sim[20]',
            'x_sim[1]',
            'x_sim[2]',
            'x_sim[3]',
            'x_sim[4]',
            'x_sim[5]',
            'x_sim[6]',
            'x_sim[7]',
            'x_sim[8]',
            'x_sim[9]',
            'x_sim[10]',
            'x_sim[11]',
            'x_sim[12]',
            'x_sim[13]',
            'x_sim[14]',
            'x_sim[15]',
            'x_sim[16]',
            'x_sim[17]',
            'x_sim[18]',
            'x_sim[19]',
            'x_sim[20]',
            'pop_sim[1]',
            'pop_sim[2]',
            'pop_sim[3]',
            'pop_sim[4]',
            'pop_sim[5]',
            'pop_sim[6]',
            'pop_sim[7]',
            'pop_sim[8]',
            'pop_sim[9]',
            'pop_sim[10]',
            'pop_sim[11]',
            'pop_sim[12]',
            'pop_sim[13]',
            'pop_sim[14]',
            'pop_sim[15]',
            'pop_sim[16]',
            'pop_sim[17]',
            'pop_sim[18]',
            'pop_sim[19]',
            'pop_sim[20]',
            'alpha_sim',
            'beta_sim',
            'eta[1]',
            'eta[2]',
            'eta[3]',
            'eta[4]',
            'eta[5]',
            'eta[6]',
            'eta[7]',
            'eta[8]',
            'eta[9]',
            'eta[10]',
            'eta[11]',
            'eta[12]',
            'eta[13]',
            'eta[14]',
            'eta[15]',
            'eta[16]',
            'eta[17]',
            'eta[18]',
            'eta[19]',
            'eta[20]',
        ]
        self.assertEqual(datagen_fit.column_names, tuple(column_names))
        self.assertEqual(datagen_fit.num_draws, 100)
        self.assertEqual(datagen_fit.draws().shape, (100, 1, len(column_names)))
        self.assertEqual(datagen_fit.metric, None)
        self.assertEqual(datagen_fit.metric_type, None)
        self.assertEqual(datagen_fit.stepsize, None)

    def test_bernoulli_file_with_space(self):
        self.test_bernoulli_good('bernoulli with space in name.stan')

    def test_bernoulli_path_with_space(self):
        self.test_bernoulli_good(
            'path with space/' 'bernoulli_path_with_space.stan'
        )


class CmdStanMCMCTest(unittest.TestCase):
    def test_validate_good_run(self):
        # construct fit using existing sampler output
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs(
            iter_sampling=100, max_treedepth=11, adapt_delta=0.95
        )
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_dir=DATAFILES_PATH,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args)
        runset._csv_files = [
            os.path.join(DATAFILES_PATH, 'runset-good', 'bern-1.csv'),
            os.path.join(DATAFILES_PATH, 'runset-good', 'bern-2.csv'),
            os.path.join(DATAFILES_PATH, 'runset-good', 'bern-3.csv'),
            os.path.join(DATAFILES_PATH, 'runset-good', 'bern-4.csv'),
        ]
        self.assertEqual(4, runset.chains)
        retcodes = runset._retcodes
        for i in range(len(retcodes)):
            runset._set_retcode(i, 0)
        self.assertTrue(runset._check_retcodes())

        fit = CmdStanMCMC(runset)
        self.assertEqual(100, fit.num_draws)
        self.assertEqual(len(BERNOULLI_COLS), len(fit.column_names))
        self.assertEqual('lp__', fit.column_names[0])

        draws_pd = fit.draws_pd()
        self.assertEqual(
            draws_pd.shape,
            (fit.runset.chains * fit.num_draws, len(fit.column_names)),
        )

        summary = fit.summary()
        self.assertIn('5%', list(summary.columns))
        self.assertIn('50%', list(summary.columns))
        self.assertIn('95%', list(summary.columns))
        self.assertNotIn('1%', list(summary.columns))
        self.assertNotIn('99%', list(summary.columns))

        summary = fit.summary(percentiles=[1, 45, 99])
        self.assertIn('1%', list(summary.columns))
        self.assertIn('45%', list(summary.columns))
        self.assertIn('99%', list(summary.columns))
        self.assertNotIn('5%', list(summary.columns))
        self.assertNotIn('50%', list(summary.columns))
        self.assertNotIn('95%', list(summary.columns))

        with self.assertRaises(ValueError):
            fit.summary(percentiles=[])

        with self.assertRaises(ValueError):
            fit.summary(percentiles=[-1])

        diagnostics = fit.diagnose()
        self.assertIn(
            'Treedepth satisfactory for all transitions.', diagnostics
        )
        self.assertIn('No divergent transitions found.', diagnostics)
        self.assertIn('E-BFMI satisfactory for all transitions.', diagnostics)
        self.assertIn('Effective sample size satisfactory.', diagnostics)

    def test_validate_big_run(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        sampler_args = SamplerArgs(iter_warmup=1500, iter_sampling=1000)
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2],
            seed=12345,
            output_dir=DATAFILES_PATH,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=2)
        runset._csv_files = [
            os.path.join(DATAFILES_PATH, 'runset-big', 'output_icar_nyc-1.csv'),
            os.path.join(DATAFILES_PATH, 'runset-big', 'output_icar_nyc-1.csv'),
        ]
        fit = CmdStanMCMC(runset)
        phis = ['phi[{}]'.format(str(x + 1)) for x in range(2095)]
        column_names = SAMPLER_STATE + phis
        self.assertEqual(fit.num_draws, 1000)
        self.assertEqual(fit.column_names, tuple(column_names))
        self.assertEqual(fit.metric_type, 'diag_e')
        self.assertEqual(fit.stepsize.shape, (2,))
        self.assertEqual(fit.metric.shape, (2, 2095))
        self.assertEqual((1000, 2, 2102), fit.draws().shape)
        phis = fit.draws_pd(params=['phi'])
        self.assertEqual((2000, 2095), phis.shape)
        phi1 = fit.draws_pd(params=['phi[1]'])
        self.assertEqual((2000, 1), phi1.shape)
        mo_phis = fit.draws_pd(params=['phi[1]', 'phi[10]', 'phi[100]'])
        self.assertEqual((2000, 3), mo_phis.shape)
        phi2095 = fit.draws_pd(params=['phi[2095]'])
        self.assertEqual((2000, 1), phi2095.shape)
        with self.assertRaisesRegex(
            ValueError, r'unknown parameter: phi\[2096\]'
        ):
            fit.draws_pd(params=['phi[2096]'])
        with self.assertRaisesRegex(ValueError, 'unknown parameter: ph'):
            fit.draws_pd(params=['ph'])

    # pylint: disable=no-self-use
    def test_custom_metric(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_model = CmdStanModel(stan_file=stan)
        jmetric = os.path.join(DATAFILES_PATH, 'bernoulli.metric.json')
        # just test that it runs without error
        bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=200,
            metric=jmetric,
        )

        jmetric2 = os.path.join(DATAFILES_PATH, 'bernoulli.metric-2.json')
        bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=200,
            metric=[jmetric, jmetric2],
        )

    def test_custom_stepsize(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_model = CmdStanModel(stan_file=stan)
        # just test that it runs without error
        bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=200,
            step_size=1,
        )

        bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=200,
            step_size=[1, 2],
        )

    def test_custom_seed(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_model = CmdStanModel(stan_file=stan)
        # just test that it runs without error
        bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=[44444, 55555],
            iter_sampling=200,
        )

    def test_adapt_schedule(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_model = CmdStanModel(stan_file=stan)
        bern_fit = bern_model.sample(
            data=jdata,
            chains=1,
            seed=12345,
            iter_sampling=200,
            iter_warmup=200,
            adapt_init_phase=11,
            adapt_metric_window=12,
            adapt_step_size=13,
        )
        txt_file = bern_fit.runset.stdout_files[0]
        with open(txt_file, 'r') as fd:
            lines = fd.readlines()
            stripped = [line.strip() for line in lines]
            self.assertIn('init_buffer = 11', stripped)
            self.assertIn('window = 12', stripped)
            self.assertIn('term_buffer = 13', stripped)

    def test_save_csv(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_model = CmdStanModel(stan_file=stan)
        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=200,
        )
        for i in range(bern_fit.runset.chains):
            csv_file = bern_fit.runset.csv_files[i]
            stdout_file = bern_fit.runset.stdout_files[i]
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(stdout_file))

        # save files to good dir
        bern_fit.save_csvfiles(dir=DATAFILES_PATH)
        for i in range(bern_fit.runset.chains):
            csv_file = bern_fit.runset.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))
        with self.assertRaisesRegex(
            ValueError, 'file exists, not overwriting: '
        ):
            bern_fit.save_csvfiles(dir=DATAFILES_PATH)

        tmp2_dir = os.path.join(HERE, 'tmp2')
        os.mkdir(tmp2_dir)
        bern_fit.save_csvfiles(dir=tmp2_dir)
        for i in range(bern_fit.runset.chains):
            csv_file = bern_fit.runset.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))
        for i in range(bern_fit.runset.chains):  # cleanup datafile_path dir
            os.remove(bern_fit.runset.csv_files[i])
            if os.path.exists(bern_fit.runset.stdout_files[i]):
                os.remove(bern_fit.runset.stdout_files[i])
            if os.path.exists(bern_fit.runset.stderr_files[i]):
                os.remove(bern_fit.runset.stderr_files[i])
        shutil.rmtree(tmp2_dir, ignore_errors=True)

        # regenerate to tmpdir, save to good dir
        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_sampling=200,
        )
        bern_fit.save_csvfiles()  # default dir
        for i in range(bern_fit.runset.chains):
            csv_file = bern_fit.runset.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))
        for i in range(bern_fit.runset.chains):  # cleanup default dir
            os.remove(bern_fit.runset.csv_files[i])
            if os.path.exists(bern_fit.runset.stdout_files[i]):
                os.remove(bern_fit.runset.stdout_files[i])
            if os.path.exists(bern_fit.runset.stderr_files[i]):
                os.remove(bern_fit.runset.stderr_files[i])

        with self.assertRaisesRegex(ValueError, 'cannot access csv file'):
            bern_fit.save_csvfiles(dir=DATAFILES_PATH)

        if platform.system() != 'Windows':
            with self.assertRaisesRegex(Exception, 'cannot save to path: '):
                dir = tempfile.mkdtemp(dir=_TMPDIR)
                os.chmod(dir, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                bern_fit.save_csvfiles(dir=dir)

    def test_diagnose_divergences(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1],
            output_dir=DATAFILES_PATH,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=1)
        runset._csv_files = [
            os.path.join(
                DATAFILES_PATH, 'diagnose-good', 'corr_gauss_depth8-1.csv'
            )
        ]
        fit = CmdStanMCMC(runset)
        # TODO - use cmdstan test files instead
        expected = '\n'.join(
            [
                'Checking sampler transitions treedepth.',
                '424 of 1000 (42%) transitions hit the maximum '
                'treedepth limit of 8, or 2^8 leapfrog steps.',
                'Trajectories that are prematurely terminated '
                'due to this limit will result in slow exploration.',
                'For optimal performance, increase this limit.',
            ]
        )
        self.assertIn(expected, fit.diagnose().replace('\r\n', '\n'))

    def test_validate_bad_run(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs(max_treedepth=11, adapt_delta=0.95)

        # some chains had errors
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_dir=DATAFILES_PATH,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args)
        for i in range(4):
            runset._set_retcode(i, 0)
        self.assertTrue(runset._check_retcodes())

        # errors reported
        runset._stderr_files = [
            os.path.join(
                DATAFILES_PATH, 'runset-bad', 'bad-transcript-bern-1.txt'
            ),
            os.path.join(
                DATAFILES_PATH, 'runset-bad', 'bad-transcript-bern-2.txt'
            ),
            os.path.join(
                DATAFILES_PATH, 'runset-bad', 'bad-transcript-bern-3.txt'
            ),
            os.path.join(
                DATAFILES_PATH, 'runset-bad', 'bad-transcript-bern-4.txt'
            ),
        ]
        self.assertIn('Exception', runset.get_err_msgs())

        # csv file headers inconsistent
        runset._csv_files = [
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-hdr-bern-1.csv'),
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-hdr-bern-2.csv'),
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-hdr-bern-3.csv'),
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-hdr-bern-4.csv'),
        ]
        with self.assertRaisesRegex(ValueError, 'header mismatch'):
            CmdStanMCMC(runset)

        # bad draws
        runset._csv_files = [
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-draws-bern-1.csv'),
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-draws-bern-2.csv'),
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-draws-bern-3.csv'),
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-draws-bern-4.csv'),
        ]
        with self.assertRaisesRegex(ValueError, 'draws'):
            CmdStanMCMC(runset)

        # mismatch - column headers, draws
        runset._csv_files = [
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-cols-bern-1.csv'),
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-cols-bern-2.csv'),
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-cols-bern-3.csv'),
            os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-cols-bern-4.csv'),
        ]
        with self.assertRaisesRegex(
            ValueError, 'bad draw, expecting 9 items, found 8'
        ):
            CmdStanMCMC(runset)

    def test_save_warmup(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

        bern_model = CmdStanModel(stan_file=stan)
        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            iter_warmup=200,
            iter_sampling=100,
            save_warmup=True,
        )
        self.assertEqual(bern_fit.column_names, tuple(BERNOULLI_COLS))
        self.assertEqual(bern_fit.num_draws, 300)
        self.assertEqual(bern_fit.num_draws_warmup, 200)
        self.assertEqual(bern_fit.num_draws_sampling, 100)
        self.assertEqual(bern_fit.draws().shape, (100, 2, len(BERNOULLI_COLS)))
        self.assertEqual(
            bern_fit.draws(inc_warmup=False).shape,
            (100, 2, len(BERNOULLI_COLS)),
        )
        self.assertEqual(
            bern_fit.draws(inc_warmup=True).shape, (300, 2, len(BERNOULLI_COLS))
        )
        self.assertEqual(bern_fit.draws_pd().shape, (200, len(BERNOULLI_COLS)))
        self.assertEqual(
            bern_fit.draws_pd(inc_warmup=False).shape,
            (200, len(BERNOULLI_COLS)),
        )
        self.assertEqual(
            bern_fit.draws_pd(inc_warmup=True).shape, (600, len(BERNOULLI_COLS))
        )

    def test_save_warmup_thin(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

        bern_model = CmdStanModel(stan_file=stan)
        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            iter_warmup=200,
            iter_sampling=100,
            thin=5,
            save_warmup=True,
        )
        self.assertEqual(bern_fit.column_names, tuple(BERNOULLI_COLS))
        self.assertEqual(bern_fit.num_draws, 60)
        self.assertEqual(bern_fit.draws().shape, (20, 2, len(BERNOULLI_COLS)))
        self.assertEqual(
            bern_fit.draws(inc_warmup=True).shape, (60, 2, len(BERNOULLI_COLS))
        )

    def test_dont_save_warmup(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

        bern_model = CmdStanModel(stan_file=stan)
        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            iter_warmup=200,
            iter_sampling=100,
            save_warmup=False,
        )
        self.assertEqual(bern_fit.column_names, tuple(BERNOULLI_COLS))
        self.assertEqual(bern_fit.num_draws, 100)
        self.assertEqual(bern_fit.draws().shape, (100, 2, len(BERNOULLI_COLS)))
        with LogCapture() as log:
            self.assertEqual(
                bern_fit.draws(inc_warmup=True).shape,
                (100, 2, len(BERNOULLI_COLS)),
            )
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'draws from warmup iterations not available,'
                ' must run sampler with "save_warmup=True".',
            )
        )
        with LogCapture() as log:
            self.assertEqual(
                bern_fit.draws_pd(inc_warmup=True).shape,
                (200, len(BERNOULLI_COLS)),
            )
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'draws from warmup iterations not available,'
                ' must run sampler with "save_warmup=True".',
            )
        )

    def test_deprecated(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

        bern_model = CmdStanModel(stan_file=stan)
        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            iter_warmup=200,
            iter_sampling=100,
            save_warmup=True,
        )
        with LogCapture() as log:
            self.assertEqual(
                bern_fit.sample.shape, (100, 2, len(BERNOULLI_COLS))
            )
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'method "sample" will be deprecated,'
                ' use method "draws" instead.',
            )
        )
        with LogCapture() as log:
            self.assertEqual(
                bern_fit.warmup.shape, (300, 2, len(BERNOULLI_COLS))
            )
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                'method "warmup" has been deprecated, instead use method'
                ' "draws(inc_warmup=True)", returning draws from both'
                ' warmup and sampling iterations.',
            )
        )

    def test_sampler_diags(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_model = CmdStanModel(stan_file=stan)
        bern_fit = bern_model.sample(
            data=jdata, chains=2, seed=12345, iter_warmup=100, iter_sampling=100
        )
        diags = bern_fit.sampler_diagnostics()
        self.assertEqual(SAMPLER_STATE, list(diags))
        for key in diags:
            self.assertEqual(diags[key].shape, (100, 2))

    def test_variable_bern(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_model = CmdStanModel(stan_file=stan)
        bern_fit = bern_model.sample(
            data=jdata, chains=2, seed=12345, iter_warmup=100, iter_sampling=100
        )
        self.assertEqual(1, len(bern_fit._stan_variable_dims))
        self.assertTrue('theta' in bern_fit._stan_variable_dims)
        self.assertEqual(bern_fit._stan_variable_dims['theta'], 1)
        theta = bern_fit.stan_variable(name='theta')
        self.assertEqual(theta.shape, (200, 1))
        with self.assertRaises(ValueError):
            bern_fit.stan_variable(name='eta')
        with self.assertRaises(ValueError):
            bern_fit.stan_variable(name='lp__')

    def test_variable_lv(self):
        # pylint: disable=C0103
        # construct fit using existing sampler output
        exe = os.path.join(DATAFILES_PATH, 'lotka-volterra' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'lotka-volterra.data.json')
        sampler_args = SamplerArgs(iter_sampling=20)
        cmdstan_args = CmdStanArgs(
            model_name='lotka-volterra',
            model_exe=exe,
            chain_ids=[1],
            seed=12345,
            data=jdata,
            output_dir=DATAFILES_PATH,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=1)
        runset._csv_files = [os.path.join(DATAFILES_PATH, 'lotka-volterra.csv')]
        runset._set_retcode(0, 0)
        fit = CmdStanMCMC(runset)
        self.assertEqual(20, fit.num_draws)
        self.assertEqual(8, len(fit._stan_variable_dims))
        self.assertTrue('z' in fit._stan_variable_dims)
        self.assertEqual(fit._stan_variable_dims['z'], (20, 2))
        z = fit.stan_variable(name='z')
        self.assertEqual(z.shape, (20, 40))
        theta = fit.stan_variable(name='theta')
        self.assertEqual(theta.shape, (20, 4))

    def test_variables(self):
        # construct fit using existing sampler output
        exe = os.path.join(DATAFILES_PATH, 'lotka-volterra' + EXTENSION)
        jdata = os.path.join(DATAFILES_PATH, 'lotka-volterra.data.json')
        sampler_args = SamplerArgs(iter_sampling=20)
        cmdstan_args = CmdStanArgs(
            model_name='lotka-volterra',
            model_exe=exe,
            chain_ids=[1],
            seed=12345,
            data=jdata,
            output_dir=DATAFILES_PATH,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=1)
        runset._csv_files = [os.path.join(DATAFILES_PATH, 'lotka-volterra.csv')]
        runset._set_retcode(0, 0)
        fit = CmdStanMCMC(runset)
        self.assertEqual(20, fit.num_draws)
        self.assertEqual(8, len(fit._stan_variable_dims))
        self.assertTrue('z' in fit._stan_variable_dims)
        self.assertEqual(fit._stan_variable_dims['z'], (20, 2))
        vars = fit.stan_variables()
        self.assertEqual(len(vars), len(fit._stan_variable_dims))
        self.assertTrue('z' in vars)
        self.assertEqual(vars['z'].shape, (20, 40))
        self.assertTrue('theta' in vars)
        self.assertEqual(vars['theta'].shape, (20, 4))

    def test_validate(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_model = CmdStanModel(stan_file=stan)
        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            iter_warmup=200,
            iter_sampling=100,
            thin=2,
            save_warmup=True,
            validate_csv=False,
        )
        # check error messages
        with LogCapture() as log:
            logging.getLogger()
            self.assertIsNone(bern_fit.column_names)
        expect = 'csv files not yet validated'
        msg = log.actual()[-1][-1]
        self.assertTrue(msg.startswith(expect))

        with LogCapture() as log:
            logging.getLogger()
            self.assertIsNone(bern_fit.stan_variable_dims)
        expect = 'csv files not yet validated'
        msg = log.actual()[-1][-1]
        self.assertTrue(msg.startswith(expect))

        with LogCapture() as log:
            logging.getLogger()
            self.assertIsNone(bern_fit.metric_type)
        expect = 'csv files not yet validated'
        msg = log.actual()[-1][-1]
        self.assertTrue(msg.startswith(expect))

        with LogCapture() as log:
            logging.getLogger()
            self.assertIsNone(bern_fit.metric)
        expect = 'csv files not yet validated'
        msg = log.actual()[-1][-1]
        self.assertTrue(msg.startswith(expect))

        with LogCapture() as log:
            logging.getLogger()
            self.assertIsNone(bern_fit.stepsize)
        expect = 'csv files not yet validated'
        msg = log.actual()[-1][-1]
        self.assertTrue(msg.startswith(expect))

        # check computations match
        self.assertEqual(bern_fit.num_draws, 150)
        bern_fit.validate_csv_files()
        self.assertEqual(bern_fit.num_draws, 150)
        self.assertEqual(len(bern_fit.column_names), 8)
        self.assertEqual(len(bern_fit.stan_variable_dims), 1)
        self.assertEqual(bern_fit.metric_type, 'diag_e')

    def test_validate_sample_sig_figs(self, stanfile='bernoulli.stan'):
        if cmdstan_version_at(2, 25):
            stan = os.path.join(DATAFILES_PATH, stanfile)
            bern_model = CmdStanModel(stan_file=stan)

            jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
            bern_fit = bern_model.sample(
                data=jdata,
                chains=1,
                seed=12345,
                iter_sampling=100,
            )
            bern_draws = bern_fit.draws()
            theta = format(bern_draws[99, 0, 7], '.18g')
            self.assertFalse(theta.startswith('0.21238045821757600'))

            bern_fit_17 = bern_model.sample(
                data=jdata,
                chains=1,
                seed=12345,
                iter_sampling=100,
                sig_figs=17,
            )
            bern_draws_17 = bern_fit_17.draws()
            theta_17 = format(bern_draws_17[99, 0, 7], '.18g')
            print("theta_17", theta_17)
            if platform.system() == "Windows":
                self.assertTrue(theta_17.startswith('0.212357704884271165'))
            else:
                self.assertTrue(theta_17.startswith('0.212348884540145805'))
            self.assertFalse(theta_17.startswith('0.21239'))

            bern_fit = bern_model.sample(
                data=jdata,
                chains=1,
                seed=12345,
                iter_sampling=100,
                sig_figs=9,
            )
            bern_draws = bern_fit.draws()
            theta_9 = format(bern_draws[99, 0, 7], '.18g')
            print("theta_9", theta_9)
            if platform.system() == "Windows":
                self.assertTrue(theta_9.startswith('0.212357705000000008'))
            else:
                self.assertTrue(theta_9.startswith('0.21234888499999998'))
            self.assertFalse(theta_9.startswith('0.21239'))

            with self.assertRaises(ValueError):
                bern_model.sample(
                    data=jdata,
                    chains=1,
                    seed=12345,
                    iter_sampling=100,
                    sig_figs=27,
                )
                with self.assertRaises(ValueError):
                    bern_model.sample(
                        data=jdata,
                        chains=1,
                        seed=12345,
                        iter_sampling=100,
                        sig_figs=-1,
                    )

    def test_validate_summary_sig_figs(self):
        # construct fit using existing sampler output
        exe = os.path.join(DATAFILES_PATH, 'logistic' + EXTENSION)
        rdata = os.path.join(DATAFILES_PATH, 'logistic.data.R')
        sampler_args = SamplerArgs(iter_sampling=100)
        cmdstan_args = CmdStanArgs(
            model_name='logistic',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=rdata,
            output_dir=DATAFILES_PATH,
            sig_figs=17,
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args)
        runset._csv_files = [
            os.path.join(DATAFILES_PATH, 'logistic_output_1.csv'),
            os.path.join(DATAFILES_PATH, 'logistic_output_2.csv'),
            os.path.join(DATAFILES_PATH, 'logistic_output_3.csv'),
            os.path.join(DATAFILES_PATH, 'logistic_output_4.csv'),
        ]
        retcodes = runset._retcodes
        for i in range(len(retcodes)):
            runset._set_retcode(i, 0)
        self.assertTrue(runset._check_retcodes())
        fit = CmdStanMCMC(runset)

        sum_default = fit.summary()
        beta1_default = format(sum_default.iloc[1, 0], '.18g')
        self.assertTrue(beta1_default.startswith('1.3'))

        if cmdstan_version_at(2, 25):
            sum_17 = fit.summary(sig_figs=17)
            beta1_17 = format(sum_17.iloc[1, 0], '.18g')
            self.assertTrue(beta1_17.startswith('1.345767078273258'))

            sum_10 = fit.summary(sig_figs=10)
            beta1_10 = format(sum_10.iloc[1, 0], '.18g')
            self.assertTrue(beta1_10.startswith('1.34576707'))

        with self.assertRaises(ValueError):
            fit.summary(sig_figs=20)
        with self.assertRaises(ValueError):
            fit.summary(sig_figs=-1)


if __name__ == '__main__':
    unittest.main()
