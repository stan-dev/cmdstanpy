"""CmdStan argument tests"""

import os
import unittest

from cmdstanpy import TMPDIR
from cmdstanpy.cmdstan_args import (
    Method,
    SamplerArgs,
    CmdStanArgs,
    OptimizeArgs,
    GenerateQuantitiesArgs,
    VariationalArgs,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class OptimizeArgsTest(unittest.TestCase):
    def test_args_algorithm(self):
        args = OptimizeArgs(algorithm='non-valid_algorithm')
        with self.assertRaises(ValueError):
            args.validate()
        args = OptimizeArgs(algorithm='Newton')
        args.validate()
        cmd = args.compose(None, cmd=['output'])
        self.assertIn('algorithm=newton', ' '.join(cmd))

    def test_args_algorithm_init_alpha(self):
        args = OptimizeArgs(init_alpha=2e-4)
        args.validate()
        cmd = args.compose(None, cmd=['output'])

        self.assertIn('init_alpha=0.0002', ' '.join(cmd))
        args = OptimizeArgs(init_alpha=-1.0)
        with self.assertRaises(ValueError):
            args.validate()
        args = OptimizeArgs(init_alpha=1.0, algorithm='Newton')
        with self.assertRaises(ValueError):
            args.validate()

    def test_args_algorithm_iter(self):
        args = OptimizeArgs(iter=400)
        args.validate()
        cmd = args.compose(None, cmd=['output'])
        self.assertIn('iter=400', ' '.join(cmd))
        args = OptimizeArgs(iter=-1)
        with self.assertRaises(ValueError):
            args.validate()


class SamplerArgsTest(unittest.TestCase):
    def test_args_min(self):
        args = SamplerArgs()
        args.validate(chains=4)
        cmd = args.compose(idx=1, cmd=[])
        self.assertIn('method=sample algorithm=hmc', ' '.join(cmd))

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
        cmd = args.compose(1, cmd=[])
        self.assertIn('method=sample', ' '.join(cmd))
        self.assertIn('num_warmup=10', ' '.join(cmd))
        self.assertIn('num_samples=20', ' '.join(cmd))
        self.assertIn('save_warmup=1', ' '.join(cmd))
        self.assertIn('thin=7', ' '.join(cmd))
        self.assertIn('algorithm=hmc engine=nuts', ' '.join(cmd))
        self.assertIn('max_depth=15 adapt delta=0.99', ' '.join(cmd))

        args = SamplerArgs(warmup_iters=10)
        args.validate(chains=4)
        cmd = args.compose(1, cmd=[])
        self.assertIn('method=sample', ' '.join(cmd))
        self.assertIn('num_warmup=10', ' '.join(cmd))
        self.assertNotIn('num_samples=', ' '.join(cmd))
        self.assertNotIn('save_warmup=', ' '.join(cmd))
        self.assertNotIn('algorithm=hmc engine=nuts', ' '.join(cmd))

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

        args = SamplerArgs(warmup_iters=100, fixed_param=True)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(save_warmup=True, fixed_param=True)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(max_treedepth=12, fixed_param=True)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(metric='dense', fixed_param=True)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(step_size=0.5, fixed_param=True)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(adapt_delta=0.88, fixed_param=True)
        with self.assertRaises(ValueError):
            args.validate(chains=2)

    def test_adapt(self):
        args = SamplerArgs(adapt_engaged=False)
        args.validate(chains=4)
        cmd = args.compose(1, cmd=[])
        self.assertIn(
            'method=sample algorithm=hmc adapt engaged=0', ' '.join(cmd)
        )

        args = SamplerArgs(adapt_engaged=True)
        args.validate(chains=4)
        cmd = args.compose(1, cmd=[])
        self.assertIn(
            'method=sample algorithm=hmc adapt engaged=1', ' '.join(cmd)
        )

        args = SamplerArgs()
        args.validate(chains=4)
        cmd = args.compose(1, cmd=[])
        self.assertNotIn('engine=nuts', ' '.join(cmd))
        self.assertNotIn('engaged=1', ' '.join(cmd))

    def test_metric(self):
        args = SamplerArgs(metric='dense_e')
        args.validate(chains=4)
        cmd = args.compose(1, cmd=[])
        self.assertIn(
            'method=sample algorithm=hmc metric=dense_e', ' '.join(cmd)
        )

        args = SamplerArgs(metric='dense')
        args.validate(chains=4)
        cmd = args.compose(1, cmd=[])
        self.assertIn(
            'method=sample algorithm=hmc metric=dense_e', ' '.join(cmd)
        )

        args = SamplerArgs(metric='diag_e')
        args.validate(chains=4)
        cmd = args.compose(1, cmd=[])
        self.assertIn(
            'method=sample algorithm=hmc metric=diag_e', ' '.join(cmd)
        )

        args = SamplerArgs(metric='diag')
        args.validate(chains=4)
        cmd = args.compose(1, cmd=[])
        self.assertIn(
            'method=sample algorithm=hmc metric=diag_e', ' '.join(cmd)
        )

        args = SamplerArgs()
        args.validate(chains=4)
        cmd = args.compose(1, cmd=[])
        self.assertNotIn('metric=', ' '.join(cmd))

        jmetric = os.path.join(DATAFILES_PATH, 'bernoulli.metric.json')
        args = SamplerArgs(metric=jmetric)
        args.validate(chains=4)
        cmd = args.compose(1, cmd=[])
        self.assertIn('metric=diag_e', ' '.join(cmd))
        self.assertIn('metric_file=', ' '.join(cmd))
        self.assertIn('bernoulli.metric.json', ' '.join(cmd))

        jmetric2 = os.path.join(DATAFILES_PATH, 'bernoulli.metric-2.json')
        args = SamplerArgs(metric=[jmetric, jmetric2])
        args.validate(chains=2)
        cmd = args.compose(0, cmd=[])
        self.assertIn('bernoulli.metric.json', ' '.join(cmd))
        cmd = args.compose(1, cmd=[])
        self.assertIn('bernoulli.metric-2.json', ' '.join(cmd))

        args = SamplerArgs(metric=[jmetric, jmetric])
        with self.assertRaises(ValueError):
            args.validate(chains=2)

        args = SamplerArgs(metric=[jmetric, jmetric2])
        with self.assertRaises(ValueError):
            args.validate(chains=4)

        args = SamplerArgs(metric='/no/such/path/to.file')
        with self.assertRaises(ValueError):
            args.validate(chains=4)

    def test_fixed_param(self):
        args = SamplerArgs(fixed_param=True)
        args.validate(chains=1)
        cmd = args.compose(0, cmd=[])
        self.assertIn('method=sample algorithm=fixed_param', ' '.join(cmd))


class CmdStanArgsTest(unittest.TestCase):
    def test_compose(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli')
        sampler_args = SamplerArgs()
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            method_args=sampler_args,
        )
        with self.assertRaises(ValueError):
            cmdstan_args.compose_command(idx=4, csv_file='foo')
        with self.assertRaises(ValueError):
            cmdstan_args.compose_command(idx=-1, csv_file='foo')

    def test_no_chains(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        jinits = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')

        sampler_args = SamplerArgs()
        with self.assertRaises(ValueError):
            CmdStanArgs(
                model_name='bernoulli',
                model_exe=exe,
                chain_ids=None,
                seed=[1, 2, 3],
                data=jdata,
                inits=jinits,
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            CmdStanArgs(
                model_name='bernoulli',
                model_exe=exe,
                chain_ids=None,
                data=jdata,
                inits=[jinits],
                method_args=sampler_args,
            )

    def test_args_good(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            method_args=sampler_args,
        )
        self.assertEqual(cmdstan_args.method, Method.SAMPLE)
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('id=1 random seed=', ' '.join(cmd))
        self.assertIn('data file=', ' '.join(cmd))
        self.assertIn('output file=', ' '.join(cmd))
        self.assertIn('method=sample algorithm=hmc', ' '.join(cmd))

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[7, 11, 18, 29],
            data=jdata,
            method_args=sampler_args,
        )
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('id=7 random seed=', ' '.join(cmd))

    def test_args_inits(self):
        exe = os.path.join(DATAFILES_PATH, 'bernoulli')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        sampler_args = SamplerArgs()

        jinits = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')
        jinits1 = os.path.join(DATAFILES_PATH, 'bernoulli.init_1.json')
        jinits2 = os.path.join(DATAFILES_PATH, 'bernoulli.init_2.json')

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            inits=jinits,
            method_args=sampler_args,
        )
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('init=', ' '.join(cmd))

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2],
            data=jdata,
            inits=[jinits1, jinits2],
            method_args=sampler_args,
        )
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('bernoulli.init_1.json', ' '.join(cmd))
        cmd = cmdstan_args.compose_command(idx=1, csv_file='bern-output-1.csv')
        self.assertIn('bernoulli.init_2.json', ' '.join(cmd))

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            inits=0,
            method_args=sampler_args,
        )
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('init=0', ' '.join(cmd))

        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=[1, 2, 3, 4],
            data=jdata,
            inits=3.33,
            method_args=sampler_args,
        )
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        self.assertIn('init=3.33', ' '.join(cmd))

    # pylint: disable=no-value-for-parameter
    def test_args_bad(self):
        sampler_args = SamplerArgs()

        with self.assertRaises(Exception):
            # missing
            CmdStanArgs(model_name='bernoulli', model_exe='bernoulli.exe')

        with self.assertRaises(ValueError):
            # bad filepath
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                data='no/such/path/to.file',
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            # bad chain id
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, -4],
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            # bad seed
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                seed=4294967299,
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            # bad seed
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                seed=[1, 2, 3],
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            # bad seed
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                seed=-3,
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            # bad seed
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                seed='badseed',
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            # bad inits
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                inits=-5,
                method_args=sampler_args,
            )

        jinits = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')
        jinits1 = os.path.join(DATAFILES_PATH, 'bernoulli.init_1.json')
        jinits2 = os.path.join(DATAFILES_PATH, 'bernoulli.init_2.json')

        with self.assertRaises(ValueError):
            # bad inits - files must be unique
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                inits=[jinits, jinits],
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            # bad inits - files must be unique
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                inits=[jinits, jinits1, jinits2],
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            # bad inits - no such file
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                inits='no/such/path/to.file',
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            # bad output_dir name
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                output_dir='no/such/path',
                method_args=sampler_args,
            )

        with self.assertRaises(ValueError):
            # can't write to output_dir - valid for all platforms?
            read_only = os.path.join(
                os.path.dirname(os.path.dirname(TMPDIR)), 'read_only'
            )
            os.mkdir(read_only, mode=0o444)
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                output_dir=read_only,
                method_args=sampler_args,
            )


class GenerateQuantitesTest(unittest.TestCase):
    def test_args_fitted_params(self):
        args = GenerateQuantitiesArgs(csv_files=['no_such_file'])
        with self.assertRaises(ValueError):
            args.validate(chains=1)
        csv_files = [
            os.path.join(
                DATAFILES_PATH, 'runset-good', 'bern-{}.csv'.format(i + 1)
            )
            for i in range(4)
        ]
        args = GenerateQuantitiesArgs(csv_files=csv_files)
        args.validate(chains=4)
        cmd = args.compose(idx=1, cmd=[])
        self.assertIn('method=generate_quantities', ' '.join(cmd))
        self.assertIn('fitted_params={}'.format(csv_files[0]), ' '.join(cmd))


class VariationalTest(unittest.TestCase):
    def test_args_variational(self):
        args = VariationalArgs()
        self.assertTrue(True)

        args = VariationalArgs(output_samples=1)
        args.validate(chains=1)
        cmd = args.compose(idx=0, cmd=[])
        self.assertIn('method=variational', ' '.join(cmd))
        self.assertIn('output_samples=1', ' '.join(cmd))

        args = VariationalArgs(tol_rel_obj=1)
        args.validate(chains=1)
        cmd = args.compose(idx=0, cmd=[])
        self.assertIn('method=variational', ' '.join(cmd))
        self.assertIn('tol_rel_obj=1', ' '.join(cmd))

    def test_args_bad(self):
        args = VariationalArgs(algorithm='no_such_algo')
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(iter=0)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(iter=1.1)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(grad_samples=0)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(grad_samples=1.1)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(elbo_samples=0)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(elbo_samples=1.1)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(eta=-0.00003)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(adapt_iter=0)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(adapt_iter=1.1)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(tol_rel_obj=0)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(eval_elbo=0)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(eval_elbo=1.5)
        with self.assertRaises(ValueError):
            args.validate()

        args = VariationalArgs(output_samples=0)
        with self.assertRaises(ValueError):
            args.validate()


if __name__ == '__main__':
    unittest.main()
