import os
import os.path
import unittest

from cmdstanpy.lib import Model, SamplerArgs

datafiles_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                "cmdstanpy", "test", "files-data"))
tmpfiles_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                "cmdstanpy", "test", "files-tmp"))

class SamplerArgsTest(unittest.TestCase):
    def test_samplerargs_min(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output)
        args.validate()
        cmd = args.compose_command('*')
        self.assertIn("id=*", cmd)
        self.assertIn("bernoulli.output-*.csv", cmd)

    def test_samplerargs_good(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        rdata = os.path.join(datafiles_path, "bernoulli.data.R")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        args = SamplerArgs(model, seed=12345, data_file=rdata, output_file=output,
                               nuts_max_depth=15, adapt_delta=0.99)
        cmd = args.compose_command('*')
        self.assertIn("random seed=12345", cmd)
        self.assertIn("data file=", cmd)
        self.assertIn("algorithm=hmc engine=nuts max_depth=15 adapt delta=0.99", cmd)

    def test_samplerargs_num_draws(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, post_warmup_draws=3, warmup_draws=7)
        cmd = args.compose_command('*')
        self.assertIn("num_samples=3", cmd)
        self.assertIn("num_warmup=7", cmd)

    def test_samplerargs_thin(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, thin=3)
        cmd = args.compose_command('*')
        self.assertIn("thin=3", cmd)


    def test_samplerargs_typical(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        jdata = os.path.join(datafiles_path, "bernoulli.data.json")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        args = SamplerArgs(model,
                           seed=12345, post_warmup_draws=100,
                           data_file=jdata, output_file=output,
                           nuts_max_depth=11, adapt_delta=0.90)
        cmd = args.compose_command('*')
        self.assertIn("bernoulli", cmd)
        self.assertIn("seed=12345", cmd)
        self.assertIn("num_samples=100", cmd)
        self.assertIn("bernoulli.data.json", cmd)
        self.assertIn("algorithm=hmc engine=nuts max_depth=11 adapt delta=0.9", cmd)

    def test_samplerargs_missing_args1(self):
        with self.assertRaises(Exception):
            args = SamplerArgs()

    def test_samplerargs_missing_args2(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        args = SamplerArgs(model)
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_model1(self):
        stan = os.path.join(datafiles_path, "bbad.stan")
        with self.assertRaises(Exception):
            model = Model(stan_file=stan, name="bbad")

    def test_samplerargs_bad_model2(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "no_such_exe")
        model = Model(stan_file=stan, name="bernoulli")
        args = SamplerArgs(model, )
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_output(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        args = SamplerArgs(model, output_file="/no/such/path/to.file")
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_seed1(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, seed="badseed")
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_seed2(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, seed=-10)
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_data(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, data_file="/no/such/path/to.file")
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_init_params(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, init_param_values="/no/such/path/to.file")
        with self.assertRaises(ValueError):
            args.validate()
            
    def test_samplerargs_bad_metric_file(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(tmpfiles_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, hmc_metric_file="/no/such/path/to.file")
        with self.assertRaises(ValueError):
            args.validate()


if __name__ == '__main__':
    unittest.main()
