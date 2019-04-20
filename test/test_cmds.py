import os
import os.path
import unittest

from cmdstanpy.lib import Model, RunSet
from cmdstanpy.cmds import *

datafiles_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                "cmdstanpy", "test", "files-data"))
tmpfiles_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                "cmdstanpy", "test", "files-tmp"))

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

    # test compile with existing exe - timestamp on exe unchanged
    # test overwrite with existing exe - timestamp on exe updated

class SampleTest(unittest.TestCase):

    def test_sample_bernoulli(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, name="bernoulli", exe_file=exe)
        print('model: {}'.format(model))
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        runset = sample(model, chains=4, cores=2, seed=12345,
                        post_warmup_draws_per_chain=100, data_file=jdata,
                        csv_output_file=output, nuts_max_depth=11, adapt_delta=0.95)
        for i in range(runset.chains):
            self.assertEqual(0, runset.get_retcode(i))
        transcript = os.path.join(tmpfiles_path, "bernoulli.output")
        for i in range(runset.chains):
            csv = ''.join([transcript,"-",str(i+1),".csv"])
            txt = ''.join([transcript,"-",str(i+1),".txt"])
            self.assertTrue(os.path.exists(csv))
            self.assertTrue(os.path.exists(txt))

    def test_sample_1_good(self):
        rdata = os.path.join(datafiles_path, "bernoulli.data.R")
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        model = compile_model(stan)
        runset = sample(model, data_file=rdata, csv_output_file=output)
        for i in range(runset.chains):
            self.assertEqual(0, runset.get_retcode(i))

    def test_sample_2_missing_input(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        model = compile_model(stan)
        runset = sample(model, csv_output_file=output)
        for i in range(runset.chains):
            self.assertEqual(70, runset.get_retcode(i))

class SummaryTest(unittest.TestCase):
    def test_summary_1_good(self):
        rdata = os.path.join(datafiles_path, "bernoulli.data.R")
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        model = compile_model(stan)
        runset = sample(model, data_file=rdata, csv_output_file=output)
        transcript = os.path.join(tmpfiles_path, "bernoulli.samples")
        ### stopped here
        pass


if __name__ == '__main__':
    unittest.main()
