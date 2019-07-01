import os
import unittest

from cmdstanpy import TMPDIR
from cmdstanpy.cmdstan_args import SamplerArgs, CmdStanArgs

datafiles_path = os.path.join("test", "data")


class SamplerArgsTest(unittest.TestCase):
    def test_args_min(self):
        args = SamplerArgs()
        args.validate(chains=4)
        cmd = args.compose(idx=1, cmd='')
        self.assertIn('method=sample', cmd)

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
        args = SamplerArgs(adapt_engaged = False)
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample', cmd)
        self.assertIn('algorithm=hmc adapt engaged=0', cmd)

        args = SamplerArgs(adapt_engaged = True)
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample', cmd)
        self.assertIn('algorithm=hmc adapt engaged=1', cmd)

        args = SamplerArgs()
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertNotIn('algorithm=hmc engine=nuts', cmd)
        self.assertNotIn('engaged=1', cmd)

    def test_metric(self):
        args = SamplerArgs(metric='dense_e')
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample', cmd)
        self.assertIn('algorithm=hmc metric=dense_e', cmd)

        args = SamplerArgs(metric='dense')
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample', cmd)
        self.assertIn('algorithm=hmc metric=dense_e', cmd)

        args = SamplerArgs(metric='diag_e')
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample', cmd)
        self.assertIn('algorithm=hmc metric=diag_e', cmd)

        args = SamplerArgs(metric='diag')
        args.validate(chains=4)
        cmd = args.compose(1, '')
        self.assertIn('method=sample', cmd)
        self.assertIn('algorithm=hmc metric=diag_e', cmd)

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
    def test_args_min(self):
        exe = os.path.join(datafiles_path, 'bernoulli')
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name = 'bernoulli',
            model_exe = exe,
            chain_ids = [1, 2, 3, 4],
            data = jdata,
            method_args = sampler_args)
        cmd = cmdstan_args.compose_command(idx=1, csv_file='bern-output')
        print(cmd)
        self.assertIn('method=sample', cmd)



if __name__ == '__main__':
    unittest.main()
