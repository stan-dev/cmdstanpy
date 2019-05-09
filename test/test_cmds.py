import os
import os.path
import unittest
import sys

from cmdstanpy.lib import Model, RunSet, SamplerArgs, PosteriorSample
from cmdstanpy.cmds import compile_model, sample

datafiles_path = os.path.expanduser(
    os.path.join("~", "github", "stan-dev", "cmdstanpy", "test", "data"))
tmpfiles_path = os.path.expanduser(
    os.path.join("~", "github", "stan-dev", "cmdstanpy", "test", "files-tmp"))

# TODO: need base test to cleanup tmp files

class CompileTest(unittest.TestCase):
    def test_good(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if os.path.exists(exe):
            os.remove(exe)
        model = compile_model(stan)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(exe, model.exe_file)

    def test_bad(self):
        stan = os.path.join(tmpfiles_path, "bbad.stan")
        with self.assertRaises(Exception):
            model = compile_model(stan)

    # TODO: test compile with existing exe - timestamp on exe unchanged
    # TODO: test overwrite with existing exe - timestamp on exe updated


class SampleTest(unittest.TestCase):
    def test_bernoulli_1(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(tmpfiles_path, "test1-bernoulli-output")
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

    def test_bernoulli_2(self):
        # tempfile for outputs
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
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
        rdata = os.path.join(datafiles_path, "bernoulli.data.R")
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        output = os.path.join(tmpfiles_path, "test3-bernoulli-output")
        model = compile_model(stan)
        post_sample = sample(model, data_file=rdata, csv_output_file=output)
        for i in range(post_sample.chains):
            csv_file = post_sample.csv_files[i]
            txt_file = ''.join([os.path.splitext(csv_file)[0], '.txt'])
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(txt_file))

    def test_missing_input(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        output = os.path.join(tmpfiles_path, "test4-bernoulli-output")
        model = compile_model(stan)
        with self.assertRaisesRegex(Exception, "Error during sampling"):
            post_sample = sample(model, csv_output_file=output)
  
if __name__ == '__main__':
    unittest.main()

