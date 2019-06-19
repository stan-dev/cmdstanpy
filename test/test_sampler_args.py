import os
import os.path
import unittest

from cmdstanpy import TMPDIR
from cmdstanpy.lib import Model, SamplerArgs

datafiles_path = os.path.join("test", "data")


class SamplerArgsTest(unittest.TestCase):
    def test_samplerargs_min(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               output_file=output)
        args.validate()
        cmd = args.compose_command(0, ''.join([output,'-1.csv']))
        self.assertIn('id=1', cmd)

    def test_samplerargs_good(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        rdata = os.path.join(datafiles_path, 'bernoulli.data.R')
        output = os.path.join(TMPDIR, 'bernoulli.output')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               seed=12345,
                               data=rdata,
                               output_file=output,
                               max_treedepth=15,
                               adapt_delta=0.99)
        cmd = args.compose_command(0, ''.join([output,'-1.csv']))
        self.assertIn('random seed=12345', cmd)
        self.assertIn('data file=', cmd)
        self.assertIn(
            'algorithm=hmc engine=nuts max_depth=15 adapt delta=0.99', cmd)

    def test_samplerargs_num_iters(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               output_file=output,
                               sampling_iters=3,
                               warmup_iters=7)
        cmd = args.compose_command(0, ''.join([output,'-1.csv']))
        self.assertIn('num_samples=3', cmd)
        self.assertIn('num_warmup=7', cmd)

    def test_samplerargs_thin(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               output_file=output,
                               thin=3)
        cmd = args.compose_command(0, ''.join([output,'-1.csv']))
        self.assertIn('thin=3', cmd)

    def test_samplerargs_typical(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        output = os.path.join(TMPDIR, 'bernoulli.output')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               seed=12345,
                               sampling_iters=100,
                               data=jdata,
                               output_file=output,
                               max_treedepth=11,
                               adapt_delta=0.9)
        cmd = args.compose_command(0, ''.join([output,'-1.csv']))
        self.assertIn('bernoulli', cmd)
        self.assertIn('seed=12345', cmd)
        self.assertIn('num_samples=100', cmd)
        self.assertIn('bernoulli.data.json', cmd)
        self.assertIn('algorithm=hmc engine=nuts max_depth=11 adapt delta=0.9',
                      cmd)

    def test_samplerargs_chain_ids(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        output = os.path.join(TMPDIR, 'bernoulli.output')
        args = SamplerArgs(model,
                               chain_ids=[7,9],
                               data=jdata)
        cmd = args.compose_command(0, ''.join([output,'-7.csv']))
        self.assertIn('bernoulli', cmd)
        self.assertIn('bernoulli.data.json', cmd)
        self.assertIn('id=7', cmd)

    def test_samplerargs_missing_args(self):
        with self.assertRaises(Exception):
            args = SamplerArgs()

    def test_samplerargs_bad_output(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file='/no/such/path/to.file')

    def test_samplerargs_bad_seed1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file=output,
                                   seed='badseed')

    def test_samplerargs_bad_seed2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file=output,
                                   seed=-10)

    def test_samplerargs_bad_data(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file=output,
                                   data='/no/such/path/to.file')

    def test_samplerargs_bad_init_params(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file=output,
                                   inits='/no/such/path/to.file')

    def test_samplerargs_bad_metric_file(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file=output,
                                   metric='/no/such/path/to.file')


if __name__ == '__main__':
    unittest.main()
