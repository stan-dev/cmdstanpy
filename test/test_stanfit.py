import os
import unittest

from cmdstanpy.cmdstan_args import SamplerArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION
from cmdstanpy.stanfit import StanFit
from cmdstanpy.model import Model

datafiles_path = os.path.join('test', 'data')
goodfiles_path = os.path.join(datafiles_path, 'runset-good')
badfiles_path = os.path.join(datafiles_path, 'runset-bad')


class StanFitTest(unittest.TestCase):
    def test_check_retcodes(self):
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            method_args=sampler_args,
        )
        fit = StanFit(args=cmdstan_args, chains=4)
        retcodes = fit._retcodes
        self.assertEqual(4, len(retcodes))
        for i in range(len(retcodes)):
            self.assertEqual(-1, fit._retcode(i))
        fit._set_retcode(0, 0)
        self.assertEqual(0, fit._retcode(0))
        for i in range(1, len(retcodes)):
            self.assertEqual(-1, fit._retcode(i))
        self.assertFalse(fit._check_retcodes())
        for i in range(1, len(retcodes)):
            fit._set_retcode(i, 0)
        self.assertTrue(fit._check_retcodes())

    def test_validate_good_run(self):
        # construct fit using existing sampler output
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        output = os.path.join(goodfiles_path, 'bern')
        sampler_args = SamplerArgs(
            sampling_iters=100, max_treedepth=11, adapt_delta=0.95
        )
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_basename=output,
            method_args=sampler_args,
        )
        fit = StanFit(args=cmdstan_args, chains=4)
        retcodes = fit._retcodes
        for i in range(len(retcodes)):
            fit._set_retcode(i, 0)
        self.assertTrue(fit._check_retcodes())
        fit._check_console_msgs()
        fit._validate_csv_files()
        self.assertEqual(4, fit.chains)
        self.assertEqual(100, fit.draws)
        self.assertEqual(8, len(fit.column_names))
        self.assertEqual('lp__', fit.column_names[0])

        df = fit.get_drawset()
        self.assertEqual(
            df.shape, (fit.chains * fit.draws, len(fit.column_names))
        )
        _ = fit.summary()

        # TODO - use cmdstan test files instead
        expected = '\n'.join(
            [
                'Checking sampler transitions treedepth.',
                'Treedepth satisfactory for all transitions.',
                '\nChecking sampler transitions for divergences.',
                'No divergent transitions found.',
                '\nChecking E-BFMI - sampler transitions HMC potential energy.',
                'E-BFMI satisfactory for all transitions.',
                '\nEffective sample size satisfactory.',
            ]
        )
        self.assertIn(expected, fit.diagnose().replace("\r\n", "\n"))

    def test_validate_big_run(self):
        exe = os.path.join(
            datafiles_path, 'bernoulli' + EXTENSION
        )  # fake out validation
        output = os.path.join(datafiles_path, 'runset-big', 'output_icar_nyc')
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2],
            seed=12345,
            output_basename=output,
            method_args=sampler_args,
        )
        fit = StanFit(args=cmdstan_args, chains=2)
        fit._validate_csv_files()
        sampler_state = [
            'lp__',
            'accept_stat__',
            'stepsize__',
            'treedepth__',
            'n_leapfrog__',
            'divergent__',
            'energy__',
        ]
        phis = ['phi.{}'.format(str(x + 1)) for x in range(2095)]
        column_names = sampler_state + phis
        self.assertEqual(fit.columns, len(column_names))
        self.assertEqual(fit.column_names, tuple(column_names))
        self.assertEqual(fit.metric_type, 'diag_e')
        self.assertEqual(fit.stepsize.shape, (2,))
        self.assertEqual(fit.metric.shape, (2, 2095))
        self.assertEqual((1000, 2, 2102), fit.sample.shape)
        phis = fit.get_drawset(params=['phi'])
        self.assertEqual((2000, 2095), phis.shape)
        phi1 = fit.get_drawset(params=['phi.1'])
        self.assertEqual((2000, 1), phi1.shape)
        mo_phis = fit.get_drawset(params=['phi.1', 'phi.10', 'phi.100'])
        self.assertEqual((2000, 3), mo_phis.shape)
        phi2095 = fit.get_drawset(params=['phi.2095'])
        self.assertEqual((2000, 1), phi2095.shape)
        with self.assertRaises(Exception):
            fit.get_drawset(params=['phi.2096'])
        with self.assertRaises(Exception):
            fit.get_drawset(params=['ph'])

    def test_save_csv(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        bern_model = Model(stan_file=stan, exe_file=exe)
        bern_model.compile()
        bern_fit = bern_model.sample(
            data=jdata, chains=4, cores=2, seed=12345, sampling_iters=200
        )

        for i in range(bern_fit.chains):
            csv_file = bern_fit.csv_files[i]
            txt_file = ''.join([os.path.splitext(csv_file)[0], '.txt'])
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(txt_file))

        # save files to good dir
        basename = 'bern_save_csvfiles_test'
        bern_fit.save_csvfiles(dir=datafiles_path, basename=basename)
        for i in range(bern_fit.chains):
            csv_file = bern_fit.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))
        with self.assertRaisesRegex(Exception, 'file exists'):
            bern_fit.save_csvfiles(dir=datafiles_path, basename=basename)
        for i in range(bern_fit.chains):  # cleanup datafile_path dir
            os.remove(bern_fit.csv_files[i])
            os.remove(bern_fit.console_files[i])

        # regenerate to tmpdir, save to good dir
        bern_fit = bern_model.sample(
            data=jdata, chains=4, cores=2, seed=12345, sampling_iters=200
        )
        bern_fit.save_csvfiles(basename=basename)  # default dir
        for i in range(bern_fit.chains):
            csv_file = bern_fit.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))
        for i in range(bern_fit.chains):  # cleanup default dir
            os.remove(bern_fit.csv_files[i])
            os.remove(bern_fit.console_files[i])

    def test_diagnose_divergences(self):
        exe = os.path.join(
            datafiles_path, 'bernoulli' + EXTENSION
        )  # fake out validation
        output = os.path.join(
            datafiles_path, 'diagnose-good', 'corr_gauss_depth8'
        )
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1],
            output_basename=output,
            method_args=sampler_args,
        )
        fit = StanFit(args=cmdstan_args, chains=1)
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
        self.assertIn(expected, fit.diagnose().replace("\r\n", "\n"))

    def test_validate_bad_run(self):
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        sampler_args = SamplerArgs(
            sampling_iters=100, max_treedepth=11, adapt_delta=0.95
        )

        # some chains had errors
        output = os.path.join(badfiles_path, 'bad-transcript-bern')
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_basename=output,
            method_args=sampler_args,
        )
        fit = StanFit(args=cmdstan_args, chains=4)
        with self.assertRaisesRegex(Exception, 'Exception'):
            fit._check_console_msgs()

        # csv file headers inconsistent
        output = os.path.join(badfiles_path, 'bad-hdr-bern')
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_basename=output,
            method_args=sampler_args,
        )
        fit = StanFit(args=cmdstan_args, chains=4)
        retcodes = fit._retcodes
        for i in range(len(retcodes)):
            fit._set_retcode(i, 0)
        self.assertTrue(fit._check_retcodes())
        with self.assertRaisesRegex(ValueError, 'header mismatch'):
            fit._validate_csv_files()

        # bad draws
        output = os.path.join(badfiles_path, 'bad-draws-bern')
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_basename=output,
            method_args=sampler_args,
        )
        fit = StanFit(args=cmdstan_args, chains=4)
        retcodes = fit._retcodes
        for i in range(len(retcodes)):
            fit._set_retcode(i, 0)
        self.assertTrue(fit._check_retcodes())
        with self.assertRaisesRegex(ValueError, 'draws'):
            fit._validate_csv_files()

        # mismatch - column headers, draws
        output = os.path.join(badfiles_path, 'bad-cols-bern')
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_basename=output,
            method_args=sampler_args,
        )
        fit = StanFit(args=cmdstan_args, chains=4)
        retcodes = fit._retcodes
        for i in range(len(retcodes)):
            fit._set_retcode(i, 0)
        self.assertTrue(fit._check_retcodes())
        with self.assertRaisesRegex(ValueError, 'bad draw'):
            fit._validate_csv_files()


if __name__ == '__main__':
    unittest.main()
