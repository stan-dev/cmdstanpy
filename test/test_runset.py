import os
import os.path
import unittest
import json

from cmdstanpy import config
from cmdstanpy.lib import Model, RunSet, SamplerArgs, StanData
from cmdstanpy.utils import *

datafiles_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                "cmdstanpy", "test", "files-data"))
goodfiles_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                "cmdstanpy", "test", "files-data", "runset-good"))
badfiles_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                "cmdstanpy", "test", "files-data", "runset-bad"))


class RunSetTest(unittest.TestCase):

    def test_runset_1(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(goodfiles_path, "bernoulli.output")
        args = SamplerArgs(model, seed=12345, data_file=jdata, output_file=output,
                        post_warmup_draws=100,
                        nuts_max_depth=11, adapt_delta=0.95)
        runset = RunSet(chains=4, args=args)
        retcodes = runset.get_retcodes()
        self.assertEqual(4, len(retcodes))
        for i in range(len(retcodes)):
            self.assertEqual(-1, runset.get_retcode(i))
        runset.set_retcode(0,0)
        self.assertEqual(0, runset.get_retcode(0))
        for i in range(1,len(retcodes)):
            self.assertEqual(-1, runset.get_retcode(i))
        self.assertFalse(runset.is_success())
        for i in range(1,len(retcodes)):
            runset.set_retcode(i,0)
        self.assertTrue(runset.is_success())

    def test_validate_good(self):
        # construct runset using existing sampler output
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(goodfiles_path, "bern")
        args = SamplerArgs(model, seed=12345, data_file=jdata, output_file=output,
                        post_warmup_draws=100,
                        nuts_max_depth=11, adapt_delta=0.95)
        runset = RunSet(chains=4, args=args)
        retcodes = runset.get_retcodes()
        for i in range(len(retcodes)):
            runset.set_retcode(i,0)
        self.assertTrue(runset.is_success())
        dict = runset.validate()
        self.assertEqual(100, dict['draws'])
        self.assertEqual(8, len(dict['col_headers']))
        self.assertEqual("lp__", dict['col_headers'][0])

    def test_validate_bad_hdr(self):
        # construct runset using existing sampler output
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(badfiles_path, "bad-hdr-bern")
        args = SamplerArgs(model, seed=12345, data_file=jdata, output_file=output,
                        post_warmup_draws=100,
                        nuts_max_depth=11, adapt_delta=0.95)
        runset = RunSet(chains=4, args=args)
        retcodes = runset.get_retcodes()
        for i in range(len(retcodes)):
            runset.set_retcode(i,0)
        self.assertTrue(runset.is_success())
        with self.assertRaisesRegexp(ValueError, "header mismatch"):
            runset.validate()

    def test_validate_bad_draws(self):
        # construct runset using existing sampler output
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(badfiles_path, "bad-draws-bern")
        args = SamplerArgs(model, seed=12345, data_file=jdata, output_file=output,
                        post_warmup_draws=100,
                        nuts_max_depth=11, adapt_delta=0.95)
        runset = RunSet(chains=4, args=args)
        retcodes = runset.get_retcodes()
        for i in range(len(retcodes)):
            runset.set_retcode(i,0)
        self.assertTrue(runset.is_success())
        with self.assertRaisesRegexp(ValueError, "draws"):
            runset.validate()

    def test_validate_bad_cols(self):
        # construct runset using existing sampler output
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(badfiles_path, "bad-cols-bern")
        args = SamplerArgs(model, seed=12345, data_file=jdata, output_file=output,
                        post_warmup_draws=100,
                        nuts_max_depth=11, adapt_delta=0.95)
        runset = RunSet(chains=4, args=args)
        retcodes = runset.get_retcodes()
        for i in range(len(retcodes)):
            runset.set_retcode(i,0)
        self.assertTrue(runset.is_success())
        with self.assertRaisesRegexp(ValueError, "columns"):
            runset.validate()

if __name__ == '__main__':
    unittest.main()
