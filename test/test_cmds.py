import io
import os
import os.path
import sys
import unittest

from cmdstanpy import TMPDIR
from cmdstanpy.lib import Model, SamplerArgs, RunSet
from cmdstanpy.cmds import compile_model, sample, summary, diagnose
from cmdstanpy.cmds import get_drawset, save_csvfiles

datafiles_path = os.path.join('test', 'data')


class CompileTest(unittest.TestCase):
    def test_good(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        if os.path.exists(exe):
            os.remove(exe)
        model = compile_model(stan)
        self.assertEqual(stan, model.stan_file)
        self.assertTrue(model.exe_file.endswith(exe))

    def test_bad(self):
        stan = os.path.join(TMPDIR, 'bbad.stan')
        with self.assertRaises(Exception):
            model = compile_model(stan)

    # TODO: test compile with existing exe - timestamp on exe unchanged
    # TODO: test overwrite with existing exe - timestamp on exe updated


class SampleTest(unittest.TestCase):
    def test_bernoulli_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        output = os.path.join(TMPDIR, 'test1-bernoulli-output')
        post_sample = sample(model,
                                 chains=4,
                                 cores=2,
                                 seed=12345,
                                 post_warmup_draws_per_chain=100,
                                 data_file=jdata,
                                 csv_output_file=output,
                                 nuts_max_depth=11,
                                 adapt_delta=0.95)
        for i in range(post_sample.chains):
            csv_file = post_sample.csv_files[i]
            txt_file = ''.join([os.path.splitext(csv_file)[0], '.txt'])
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(txt_file))

        self.assertEqual(post_sample.chains,4)
        self.assertEqual(post_sample.draws,100)
        column_names = ['lp__','accept_stat__','stepsize__','treedepth__',
                            'n_leapfrog__','divergent__','energy__', 'theta']
        self.assertEqual(post_sample.column_names, tuple(column_names))

        post_sample.assemble_sample()
        self.assertEqual(post_sample.sample.shape, (100, 4, len(column_names)))
        self.assertEqual(post_sample.metric_type, 'diag_e')
        self.assertEqual(post_sample.stepsize.shape,(4,))
        self.assertEqual(post_sample.metric.shape,(4, 1))

    def test_bernoulli_2(self):
        # tempfile for outputs
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        post_sample = sample(model,
                                 chains=4,
                                 cores=2,
                                 seed=12345,
                                 post_warmup_draws_per_chain=100,
                                 data_file=jdata,
                                 nuts_max_depth=11,
                                 adapt_delta=0.95)
        for i in range(post_sample.chains):
            csv_file = post_sample.csv_files[i]
            txt_file = ''.join([os.path.splitext(csv_file)[0], '.txt'])
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(txt_file))

    def test_bernoulli_rdata(self):
        rdata = os.path.join(datafiles_path, 'bernoulli.data.R')
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        output = os.path.join(TMPDIR, 'test3-bernoulli-output')
        model = compile_model(stan)
        post_sample = sample(model, data_file=rdata, csv_output_file=output)
        for i in range(post_sample.chains):
            csv_file = post_sample.csv_files[i]
            txt_file = ''.join([os.path.splitext(csv_file)[0], '.txt'])
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(txt_file))

    def test_bernoulli_data(self):
        data_dict = { 'N' : 10, 'y' : [0,1,0,0,0,0,0,0,0,1] }
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        output = os.path.join(TMPDIR, 'test3-bernoulli-output')
        model = compile_model(stan)
        post_sample = sample(model, data=data_dict, csv_output_file=output)
        for i in range(post_sample.chains):
            csv_file = post_sample.csv_files[i]
            txt_file = ''.join([os.path.splitext(csv_file)[0], '.txt'])
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(txt_file))

    def test_missing_input(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        output = os.path.join(TMPDIR, 'test4-bernoulli-output')
        model = compile_model(stan)
        with self.assertRaisesRegex(Exception, 'Error during sampling'):
            post_sample = sample(model, csv_output_file=output)


class DrawsetTest(unittest.TestCase):
    def test_bernoulli(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        post_sample = sample(model,
                                 chains=4,
                                 cores=2,
                                 seed=12345,
                                 post_warmup_draws_per_chain=200,
                                 data_file=jdata)
        post_sample.assemble_sample()
        df = get_drawset(post_sample)
        self.assertEqual(df.shape,
                             (post_sample.chains * post_sample.draws,
                                        len(post_sample.column_names)))

    def test_sample_big(self):
        # construct runset using existing sampler output
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(datafiles_path, 'runset-big', 'output_icar_nyc')
        args = SamplerArgs(model,
                           output_file=output)
        runset = RunSet(chains=2, args=args)
        runset.validate_csv_files()
        runset.assemble_sample()
        sampler_state = ['lp__','accept_stat__','stepsize__','treedepth__',
                             'n_leapfrog__','divergent__','energy__']
        phis = ['phi.{}'.format(str(x+1)) for x in range(2095)]
        column_names = sampler_state + phis
        self.assertEqual(runset.columns, len(column_names))
        self.assertEqual(runset.column_names, tuple(column_names))
        self.assertEqual(runset.metric_type, 'diag_e')
        self.assertEqual(runset.stepsize.shape,(2,))
        self.assertEqual(runset.metric.shape,(2, 2095))
        self.assertEqual((1000,2,2102), runset.sample.shape)
        phis = get_drawset(runset, params=['phi'])
        self.assertEqual((2000,2095), phis.shape)
        phi1 = get_drawset(runset, params=['phi.1'])
        self.assertEqual((2000,1), phi1.shape)
        mo_phis = get_drawset(runset, params=['phi.1', 'phi.10', 'phi.100'])
        self.assertEqual((2000,3), mo_phis.shape)
        phi2095 = get_drawset(runset, params=['phi.2095'])
        self.assertEqual((2000,1), phi2095.shape)
        with self.assertRaises(Exception):
            get_drawset(runset, params=['phi.2096'])
        with self.assertRaises(Exception):
            get_drawset(runset, params=['ph'])


class SummaryTest(unittest.TestCase):
    def test_bernoulli(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        post_sample = sample(model,
                                 chains=4,
                                 cores=2,
                                 seed=12345,
                                 post_warmup_draws_per_chain=200,
                                 data_file=jdata)
        df = summary(post_sample)
        self.assertTrue(df.shape == (2, 9))


class DiagnoseTest(unittest.TestCase):
    def diagnose_no_problems(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        post_sample = sample(model,
                                 chains=4,
                                 cores=2,
                                 seed=12345,
                                 post_warmup_draws_per_chain=200,
                                 data_file=jdata)
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        diagnose(post_sample)
        sys.stdout = sys.__stdout__
        self.assertEqual(capturedOutput.getvalue(), 'No problems detected.\n')

    def test_diagnose_divergences(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(datafiles_path, 'diagnose-good',
                                  'corr_gauss_depth8')
        args = SamplerArgs(model, output_file=output)
        runset = RunSet(args=args, chains=1)

        # TODO - use cmdstan test files instead
        expected = ''.join([
            '424 of 1000 (42%) transitions hit the maximum ',
            'treedepth limit of 8, or 2^8 leapfrog steps. ',
            'Trajectories that are prematurely terminated ',
            'due to this limit will result in slow ',
            'exploration and you should increase the ',
            'limit to ensure optimal performance.\n'])

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        diagnose(runset)
        sys.stdout = sys.__stdout__
        self.assertEqual(capturedOutput.getvalue(), expected)

class SaveCsvfilesTest(unittest.TestCase):
    def test_bernoulli(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        post_sample = sample(model,
                                 chains=4,
                                 cores=2,
                                 seed=12345,
                                 post_warmup_draws_per_chain=200,
                                 data_file=jdata)
        for i in range(post_sample.chains):
            csv_file = post_sample.csv_files[i]
            txt_file = ''.join([os.path.splitext(csv_file)[0], '.txt'])
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(txt_file))

        basename = 'bern_save_csvfiles_test'
        save_csvfiles(post_sample, datafiles_path, basename) # good
        for i in range(post_sample.chains):
            csv_file = post_sample.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))

        with self.assertRaisesRegex(Exception, 'cannot save'):
            save_csvfiles(post_sample,
                              os.path.join('no', 'such', 'dir'), basename)

        with self.assertRaisesRegex(Exception, 'file exists'):
            save_csvfiles(post_sample, datafiles_path, basename)

        save_csvfiles(post_sample, basename=basename) # default dir
        for i in range(post_sample.chains):
            csv_file = post_sample.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))

        for i in range(post_sample.chains):    # cleanup
            os.remove(post_sample.csv_files[i])


if __name__ == '__main__':
    unittest.main()
