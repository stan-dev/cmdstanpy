import io
import os
import os.path
import sys
import unittest

from cmdstanpy.lib import Model, PosteriorSample
from cmdstanpy.cmds import compile_model, sample

datafiles_path = os.path.expanduser(
    os.path.join("~", "github", "stan-dev", "cmdstanpy", "test", "files-data"))
goodfiles_path = os.path.expanduser(
    os.path.join("~", "github", "stan-dev", "cmdstanpy", "test", "files-data",
                 "runset-good"))
badfiles_path = os.path.expanduser(
    os.path.join("~", "github", "stan-dev", "cmdstanpy", "test", "files-data",
                 "runset-bad"))

class PostSampleTest(unittest.TestCase):
    def test_postsample_summary_good(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        post_sample = sample(model, data_file=jdata)
        df = post_sample.summary()
        self.assertTrue(df.shape == (8, 9))

    def test_postsample_diagnose_ok(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        post_sample = sample(model,
                                 post_warmup_draws_per_chain=100,
                                 data_file=jdata)
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput  
        post_sample.diagnose()
        sys.stdout = sys.__stdout__
        self.assertEqual(capturedOutput.getvalue(), '')

    def test_postsample_diagnose_divergences(self):
        output = os.path.join(datafiles_path, "diagnose-good", "corr_gauss_depth8-1.csv")
        expected = """424 of 1000 (42%) transitions hit the maximum treedepth limit of 8, or 2^8 leapfrog steps. Trajectories that are prematurely terminated due to this limit will result in slow exploration and you should increase the limit to ensure optimal performance.
"""
        run = { 'draws': 10, 'chains': 1, 'column_names': ['a', 'b', 'c']}
        csv_files = tuple([output])
        post_sample = PosteriorSample(run, csv_files)
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput  
        post_sample.diagnose()
        sys.stdout = sys.__stdout__
        self.assertEqual(capturedOutput.getvalue(), expected)


if __name__ == '__main__':
    unittest.main()
