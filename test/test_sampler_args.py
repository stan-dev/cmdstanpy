import os
import os.path
import unittest

from cmdstanpy import TMPDIR
from cmdstanpy.lib import Model, SamplerArgs

datafiles_path = os.path.join("test", "data")


class SamplerArgsTest(unittest.TestCase):
    def test_args_min(self):
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

    def test_args_good(self):
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

    def test_args_typical(self):
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

    def test_args_many_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        jmetric = os.path.join(datafiles_path, 'bernoulli.metric.json')
        output = os.path.join(TMPDIR, 'bernoulli.output')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               seed=12345,
                               warmup_iters=100,
                               sampling_iters=100,
                               save_warmup=True,
                               thin=2,
                               metric=jmetric,
                               step_size=1.5,
                               data=jdata,
                               output_file=output,
                               max_treedepth=11,
                               adapt_delta=0.9)
        cmd = args.compose_command(0, ''.join([output,'-1.csv']))
        s1 = 'test/data/bernoulli id=1 random seed=12345 data file=test/data/bernoulli.data.json'
        s2 = 'method=sample num_samples=100 num_warmup=100 save_warmup=1 thin=2'
        s3 = 'algorithm=hmc engine=nuts max_depth=11 stepsize=1.5 metric=diag_e metric_file="test/data/bernoulli.metric.json" adapt delta=0.9'
        self.assertIn(s1, cmd)
        self.assertIn(s2, cmd)
        self.assertIn(s3, cmd)

    def test_args_chain_ids(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        args = SamplerArgs(model,
                               chain_ids=[7,9],
                               data=jdata)
        cmd = args.compose_command(0, 'output')
        self.assertIn('bernoulli', cmd)
        self.assertIn('bernoulli.data.json', cmd)
        self.assertIn('id=7', cmd)
        cmd = args.compose_command(1, 'output')
        self.assertIn('id=9', cmd)

    def test_args_chain_ids_bad(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaisesRegex(ValueError, 'invalid chain_id -99'):
            args = SamplerArgs(model,
                                   chain_ids=[7,-99])

    def test_args_missing_args_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(Exception):
            args = SamplerArgs()

    def test_args_missing_args_2(self):
        with self.assertRaises(Exception):
            args = SamplerArgs(model)

    def test_args_bad_seed_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file=output,
                                   seed='badseed')

    def test_args_bad_seed_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file=output,
                                   seed=-10)

    def test_args_bad_seed_3(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file=output,
                                   seed=[1, 2, 3])

    def test_args_bad_seed_4(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file=output,
                                   seed=4294967299)

    def test_args_bad_data(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        output = os.path.join(TMPDIR, 'bernoulli.output')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file=output,
                                   data='/no/such/path/to.file')

    def test_args_inits_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        jinits = os.path.join(datafiles_path, 'bernoulli.init.json')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               data=jdata,
                               inits=jinits)
        cmd = args.compose_command(0, 'output')
        s1 = 'data file=test/data/bernoulli.data.json init=test/data/bernoulli.init.json'
        self.assertIn(s1, cmd)

    def test_args_inits_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               data=jdata,
                               inits=0)
        cmd = args.compose_command(0, 'output')
        s1 = 'data file=test/data/bernoulli.data.json init=0'
        self.assertIn(s1, cmd)

    def test_args_inits_3(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               data=jdata,
                               inits=3.33)
        cmd = args.compose_command(0, 'output')
        s1 = 'data file=test/data/bernoulli.data.json init=3.33'
        self.assertIn(s1, cmd)

    def test_args_bad_inits_value(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   inits=-5)

    def test_args_bad_inits_file(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   inits='/no/such/path/to.file')

    def test_args_bad_inits_files_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jinits = os.path.join(datafiles_path, 'bernoulli.init.json')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   inits=[jinits, jinits, jinits])

    def test_args_bad_inits_files_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jinits = os.path.join(datafiles_path, 'bernoulli.init.json')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   inits=[jinits, 'no/such/file.json'])

    def test_args_iters_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               warmup_iters=123)
        cmd = args.compose_command(0, 'output')
        self.assertIn('num_warmup=123', cmd)

    def test_args_iters_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               sampling_iters=123)
        cmd = args.compose_command(0, 'output')
        self.assertIn('num_samples=123', cmd)

    def test_args_iters_3(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                warmup_iters=-123)

    def test_args_iters_4(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                sampling_iters=-123)

    def test_args_iters_5(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                warmup_iters=0,
                                adapt_engaged=True)

    def test_args_warmup_schedule_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               warmup_iters=200,
                               warmup_schedule=(0.1, 0.8, 0.1))
        cmd = args.compose_command(0, 'output')
        s1 = 'algorithm=hmc adapt init_buffer=20 term_buffer=20'

    def test_args_warmup_schedule_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                warmup_schedule=(-0.1, 0.8, 0.1))

    def test_args_warmup_schedule_3(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                warmup_schedule=(8.1, 0.8, 0.1))

    def test_args_iters_schedule_mismatch(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                warmup_iters=0,
                                warmup_schedule=(0.1, 0.8, 0.1))

    def test_args_iters_adapt_mismatch(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                warmup_iters=0,
                                adapt_engaged=True)

    def test_args_save_warmup_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               save_warmup=True)
        cmd = args.compose_command(0, 'output')
        self.assertIn('save_warmup=1', cmd)

    def test_args_save_warmup_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               save_warmup=False)
        cmd = args.compose_command(0, 'output')
        self.assertNotIn('save_warmup', cmd)

    def test_args_num_iters(self):
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

    def test_args_thin_good(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               thin=3)
        cmd = args.compose_command(0, 'output')
        self.assertIn('thin=3', cmd)

    def test_args_thin_bad(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                thin=-3)

    def test_args_max_treedepth_good(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               max_treedepth=15)
        cmd = args.compose_command(0, 'output')
        self.assertIn('max_depth=15', cmd)

    def test_args_max_treedepth_bad(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                max_treedepth=-3)

    def test_args_metric_file_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jmetric = os.path.join(datafiles_path, 'bernoulli.metric.json')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               metric=jmetric)
        cmd = args.compose_command(0, 'output')
        s1 = 'metric=diag_e metric_file="test/data/bernoulli.metric.json'
        self.assertIn(s1, cmd)
        
    def test_args_metric_file_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jmetric = os.path.join(datafiles_path, 'bernoulli.metric.json')
        jmetric2 = os.path.join(datafiles_path, 'bernoulli.metric-2.json')
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               metric=[jmetric, jmetric2])
        cmd = args.compose_command(1, 'output')
        s1 = 'metric=diag_e metric_file="test/data/bernoulli.metric-2.json'
        self.assertIn(s1, cmd)
                               
    def test_args_bad_metric_file(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   metric='/no/such/path/to.file')

                               
    def test_args_bad_metric_file_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jmetric = os.path.join(datafiles_path, 'bernoulli.metric.json')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   metric=[jmetric, '/no/such/path/to.file'])
                               
    def test_args_bad_metric_file_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        jmetric = os.path.join(datafiles_path, 'bernoulli.metric.json')
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   metric=[jmetric, jmetric, jmetric])

    def test_args_step_size_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               step_size=1.3)
        cmd = args.compose_command(0, 'output')
        self.assertIn('stepsize=1.3', cmd)

    def test_args_step_size_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               step_size=[1.31, 1.29])
        cmd = args.compose_command(1, 'output')
        self.assertIn('stepsize=1.29', cmd)

    def test_args_step_size_bad_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                step_size=-0.99)

    def test_args_step_size_bad_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                step_size=[1.31, -0.99])

    def test_args_step_size_bad_3(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                step_size=[2])

    def test_args_adapt_delta_1(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        args = SamplerArgs(model,
                               chain_ids=[1,2],
                               adapt_delta=.93)
        cmd = args.compose_command(0, 'output')
        self.assertIn('adapt delta=0.93', cmd)

    def test_args_adapt_delta_2(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                adapt_delta=-3)

    def test_args_adapt_delta_3(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                chain_ids=[1,2],
                                adapt_delta=1.3)

    def test_args_bad_output(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        with self.assertRaises(ValueError):
            args = SamplerArgs(model,
                                   chain_ids=[1,2],
                                   output_file='/no/such/path/to.file')


if __name__ == '__main__':
    unittest.main()
