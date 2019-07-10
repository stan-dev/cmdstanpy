import os
import unittest

from cmdstanpy import TMPDIR
from cmdstanpy.cmdstan_args import SamplerArgs, CmdStanArgs, FixedParamArgs, OptimizeArgs

datafiles_path = os.path.join("test", "data")


class OptimizeArgsTest(unittest.TestCase):
    def test_args_algorithm(self):
        args = OptimizeArgs(algorithm="xxx")
        self.assertRaises(ValueError, lambda: args.validate())
        args = OptimizeArgs(algorithm="Newton")
        args.validate()
        cmd = args.compose(None, 'output')
        self.assertIn("algorithm=newton", cmd)

    def test_args_algorithm_init_alpha(self):
        args = OptimizeArgs(init_alpha=2e-4)
        args.validate()
        cmd = args.compose(None, 'output')
        self.assertIn("init_alpha=0.0002", cmd)
        args = OptimizeArgs(init_alpha=-1.0)
        self.assertRaises(ValueError, lambda: args.validate())

    def test_args_algorithm_iter(self):
        args = OptimizeArgs(iter=400)
        args.validate()
        cmd = args.compose(None, 'output')
        self.assertIn("iter=400", cmd)
        args = OptimizeArgs(iter=-1)
        self.assertRaises(ValueError, lambda: args.validate())


class SamplerArgsTest(unittest.TestCase):
    def test_args_min(self):
        args = SamplerArgs()
        args.validate(chains=4)
        cmd = args.compose(idx=1, cmd='')
        self.assertIn('method=sample algorithm=hmc', cmd)

    def test_args_chains(self):
        args = SamplerArgs()
        with self.assertRaises(ValueError):
            args.validate(chains=None)

    def test_good(self):
        args = SamplerArgs(
            warmup_iters=10,
            sampling_iters=20,
            save_warmup=True,
            thin=7,
            max_treedepth=15,
            adapt_delta=0.99,
        )
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample', cmd)
        self.assertIn('num_warmup=10', cmd)
        self.assertIn('num_samples=20', cmd)
        self.assertIn('save_warmup=1', cmd)
        self.assertIn('thin=7', cmd)
        self.assertIn('algorithm=hmc engine=nuts', cmd)
        self.assertIn('max_depth=15 adapt delta=0.99', cmd)

        args = SamplerArgs(warmup_iters=10)
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample', cmd)
        self.assertIn('num_warmup=10', cmd)
        self.assertNotIn('num_samples=', cmd)
        self.assertNotIn('save_warmup=', cmd)
        self.assertNotIn('algorithm=hmc engine=nuts', cmd)

    def test_bad(self):
        args = SamplerArgs(warmup_iters=-10)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(warmup_iters=0, adapt_engaged=True)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(sampling_iters=-10)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(thin=-10)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(max_treedepth=-10)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(step_size=-10)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(step_size=[1.0, 1.1])
        with self.assertRaises(ValueError):
            args.validate(chains=1)

        args = SamplerArgs(step_size=[1.0, -1.1])
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(adapt_delta=1.1)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(adapt_delta=-0.1)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

    def test_adapt(self):
        args = SamplerArgs(adapt_engaged=False)
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample algorithm=hmc adapt engaged=0', cmd)

        args = SamplerArgs(adapt_engaged=True)
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample algorithm=hmc adapt engaged=1', cmd)

        args = SamplerArgs()
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertNotIn('engine=nuts', cmd)
        self.assertNotIn('engaged=1', cmd)

    def test_metric(self):
        args = SamplerArgs(metric='dense_e')
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample algorithm=hmc metric=dense_e', cmd)

        args = SamplerArgs(metric='dense')
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample algorithm=hmc metric=dense_e', cmd)

        args = SamplerArgs(metric='diag_e')
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample algorithm=hmc metric=diag_e', cmd)

        args = SamplerArgs(metric='diag')
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample algorithm=hmc metric=diag_e', cmd)

        args = SamplerArgs()
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertNotIn('metric=', cmd)

        jmetric = os.path.join(datafiles_path, 'bernoulli.metric.json')
        args = SamplerArgs(metric=jmetric)
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('metric=diag_e', cmd)
        self.assertIn('metric_file=', cmd)
        self.assertIn('bernoulli.metric.json', cmd)

        jmetric2 = os.path.join(datafiles_path, 'bernoulli.metric-2.json')
        args = SamplerArgs(metric=[jmetric, jmetric2])
        args.validate(chains=2)
        cmd = args.compose(0, '')
        self.assertIn('bernoulli.metric.json', cmd)
        cmd = args.compose(1, '')
        self.assertIn('bernoulli.metric-2.json', cmd)

        args = SamplerArgs(metric=[jmetric, jmetric])
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(metric=[jmetric, jmetric2])
        with self.assertRaises(ValueError):
            args.validate(chains=4)

        args = SamplerArgs(metric='/no/such/path/to.file')
        with self.assertRaises(ValueError):
            args.validate(chains=4)


class CmdStanArgsTest(unittest.TestCase):
    def test_compose(self):
        exe = os.path.join(datafiles_path, 'bernoulli')
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            method_args=sampler_args)
        with self.assertRaises(ValueError):
            cmdstan_args.compose_command(idx=4, csv_file='foo')
        with self.assertRaises(ValueError):
            cmdstan_args.compose_command(idx=-1, csv_file='foo')

    def test_no_chains(self):
        # we don't have chains for optimize
        exe = os.path.join(datafiles_path, 'bernoulli')
        sampler_args = FixedParamArgs()
        jinits = os.path.join(datafiles_path, 'bernoulli.init.json')
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=None,
            inits=jinits,
            method_args=sampler_args)
        self.assertIn("init=", cmdstan_args.compose_command(None, "out.csv"))

        with self.assertRaises(ValueError):
            CmdStanArgs(
                model_name='bernoulli',
                model_exe=exe,
                chain_ids=None,
                seed=[1, 2, 3],
                inits=jinits,
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            CmdStanArgs(
                model_name='bernoulli',
                model_exe=exe,
                chain_ids=None,
                inits=[jinits],
                method_args=sampler_args)

    def test_args_good(self):
        exe = os.path.join(datafiles_path, 'bernoulli')
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        sampler_args = SamplerArgs()

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            method_args=sampler_args)
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('id=1 random seed=', cmd)
        self.assertIn('data file=', cmd)
        self.assertIn('output file=', cmd)
        self.assertIn('method=sample algorithm=hmc', cmd)

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[7, 11, 18, 29],
            data=jdata,
            method_args=sampler_args)
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('id=7 random seed=', cmd)

    def test_args_inits(self):
        exe = os.path.join(datafiles_path, 'bernoulli')
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        sampler_args = SamplerArgs()

        jinits = os.path.join(datafiles_path, 'bernoulli.init.json')
        jinits1 = os.path.join(datafiles_path, 'bernoulli.init_1.json')
        jinits2 = os.path.join(datafiles_path, 'bernoulli.init_2.json')

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            inits=jinits,
            method_args=sampler_args)
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('init=', cmd)

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2],
            data=jdata,
            inits=[jinits1, jinits2],
            method_args=sampler_args)
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('bernoulli.init_1.json', cmd)
        cmd = cmdstan_args.compose_command(idx=1, csv_file='bern-output-1.csv')
        self.assertIn('bernoulli.init_2.json', cmd)

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            inits=0,
            method_args=sampler_args)
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('init=0', cmd)

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            inits=3.33,
            method_args=sampler_args)
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('init=3.33', cmd)

    def test_args_bad(self):
        sampler_args = SamplerArgs()

        with self.assertRaises(Exception):
            # missing
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe')

        with self.assertRaises(Exception):
            # missing
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe')

        with self.assertRaises(ValueError):
            # bad filepath
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                data='no/such/path/to.file',
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            # bad chain id
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, -4],
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            # bad seed
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                seed=4294967299,
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            # bad seed
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                seed=[1, 2, 3],
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            # bad seed
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                seed=-3,
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            # bad seed
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                seed='badseed',
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            # bad inits
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                inits=-5,
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            # bad inits
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                inits='no/such/path/to.file',
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            # bad inits
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                inits='no/such/path/to.file',
                method_args=sampler_args)

        jinits = os.path.join(datafiles_path, 'bernoulli.init.json')
        jinits1 = os.path.join(datafiles_path, 'bernoulli.init_1.json')
        jinits2 = os.path.join(datafiles_path, 'bernoulli.init_2.json')

        with self.assertRaises(ValueError):
            # bad inits - files must be unique
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                inits=[jinits, jinits],
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            # bad inits - files must be unique
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                inits=[jinits, jinits1, jinits2],
                method_args=sampler_args)

        with self.assertRaises(ValueError):
            # bad output basename
            cmdstan_args = CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                output_basename='no/such/path/to.file',
                method_args=sampler_args)


if __name__ == '__main__':
    unittest.main()
