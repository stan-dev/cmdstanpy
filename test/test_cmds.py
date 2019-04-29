import os
import os.path
import unittest

from cmdstanpy.lib import Model, RunSet, SamplerArgs
from cmdstanpy.cmds import compile_model, diagnose, sample, summary

datafiles_path = os.path.expanduser(
    os.path.join("~", "github", "stan-dev", "cmdstanpy", "test", "files-data"))
tmpfiles_path = os.path.expanduser(
    os.path.join("~", "github", "stan-dev", "cmdstanpy", "test", "files-tmp"))

# TODO: need base test to cleanup tmp files


class CompileTest(unittest.TestCase):
    def test_compile_good(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if os.path.exists(exe):
            os.remove(exe)
        model = compile_model(stan)
        self.assertEqual("bernoulli", model.name)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(exe, model.exe_file)

    def test_compile_bad(self):
        stan = os.path.join(tmpfiles_path, "bbad.stan")
        with self.assertRaises(Exception):
            model = compile_model(stan)

    # TODO: test compile with existing exe - timestamp on exe unchanged
    # TODO: test overwrite with existing exe - timestamp on exe updated


class SampleTest(unittest.TestCase):
    def test_sample_bernoulli_1(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, name="bernoulli", exe_file=exe)
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(tmpfiles_path, "test1-bernoulli.output")
        transcript = os.path.join(tmpfiles_path, "test1-bernoulli.run")
        runset = sample(model,
                        chains=4,
                        cores=2,
                        seed=12345,
                        post_warmup_draws_per_chain=100,
                        data_file=jdata,
                        csv_output_file=output,
                        nuts_max_depth=11,
                        adapt_delta=0.95,
                        console_output_file=transcript)
        for i in range(runset.chains):
            self.assertEqual(0, runset.get_retcode(i))
        for i in range(runset.chains):
            csv = ''.join([output, "-", str(i + 1), ".csv"])
            txt = ''.join([transcript, "-", str(i + 1), ".txt"])
            self.assertTrue(os.path.exists(csv))
            self.assertTrue(os.path.exists(txt))

    def test_sample_bernoulli_2(self):
        # default console output file
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, name="bernoulli", exe_file=exe)
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(tmpfiles_path, "test2-bernoulli.output")
        runset = sample(model,
                        chains=4,
                        cores=2,
                        seed=12345,
                        post_warmup_draws_per_chain=100,
                        data_file=jdata,
                        csv_output_file=output,
                        nuts_max_depth=11,
                        adapt_delta=0.95)
        for i in range(runset.chains):
            self.assertEqual(0, runset.get_retcode(i))
        for i in range(runset.chains):
            csv = ''.join([output, "-", str(i + 1), ".csv"])
            txt = ''.join([output, "-", str(i + 1), ".txt"])
            self.assertTrue(os.path.exists(csv))
            self.assertTrue(os.path.exists(txt))

    def test_sample_bernoulli_rdata(self):
        rdata = os.path.join(datafiles_path, "bernoulli.data.R")
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        output = os.path.join(tmpfiles_path, "test3-bernoulli.output")
        model = compile_model(stan)
        runset = sample(model, data_file=rdata, csv_output_file=output)
        for i in range(runset.chains):
            self.assertEqual(0, runset.get_retcode(i))

    def test_sample_missing_input(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        output = os.path.join(tmpfiles_path, "test4-bernoulli.output")
        model = compile_model(stan)
        with self.assertRaisesRegexp(Exception, "Error during sampling"):
            runset = sample(model, csv_output_file=output)


class SummaryTest(unittest.TestCase):
    def test_summary_1_good(self):
        rdata = os.path.join(datafiles_path, "bernoulli.data.R")
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        output = os.path.join(tmpfiles_path, "summary-inputs")
        model = compile_model(stan)
        runset = sample(model, data_file=rdata, csv_output_file=output)
        transcript = os.path.join(tmpfiles_path, "summary-test1.txt")
        summary(runset, transcript)
        self.assertTrue(os.path.exists(transcript))

    def test_summary_2_good(self):
        rdata = os.path.join(datafiles_path, "bernoulli.data.R")
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        output = os.path.join(tmpfiles_path, "summary-inputs")
        model = compile_model(stan)
        runset = sample(model, data_file=rdata, csv_output_file=output)
        transcript = os.path.join(tmpfiles_path, "summary-test2.txt")
        summary(runset, transcript, sig_figs=10)
        self.assertTrue(os.path.exists(transcript))


class DiagnoseTest(unittest.TestCase):
    def test_diagnose_divergences(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(datafiles_path, "diagnose-good", "corr_gauss_depth8")
        args = SamplerArgs(model,
                           seed=12345,
                           data_file=jdata,
                           output_file=output)
        runset1 = RunSet(chains=1, args=args)
        runset1.set_retcode(0, 0)
        transcript = os.path.join(tmpfiles_path, "diagnose-divergences.txt")
        diagnose(runset1, transcript)
        self.assertTrue(os.path.exists(transcript))

    def test_diagnose_no_problems(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, name="bernoulli", exe_file=exe)
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(tmpfiles_path, "test1-bernoulli.output")
        runset2 = sample(model,
                        chains=4,
                        cores=2,
                        seed=12345,
                        post_warmup_draws_per_chain=100,
                        data_file=jdata,
                        csv_output_file=output,
                        nuts_max_depth=11,
                        adapt_delta=0.95)
        transcript = os.path.join(tmpfiles_path, "diagnose-ok.txt")
        diagnose(runset2, transcript)
        self.assertTrue(os.path.exists(transcript))
        with open(transcript, 'r') as myfile:
            contents = myfile.read()
        self.assertEqual(contents, 'Processing complete, no problems detected\n')

if __name__ == '__main__':
    unittest.main()
