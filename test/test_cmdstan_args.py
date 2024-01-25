"""CmdStan argument tests"""

import logging
import os
import platform
from test import check_present
from time import time

import numpy as np
import pytest

from cmdstanpy import _TMPDIR, cmdstan_path
from cmdstanpy.cmdstan_args import (
    CmdStanArgs,
    GenerateQuantitiesArgs,
    LaplaceArgs,
    Method,
    OptimizeArgs,
    PathfinderArgs,
    SamplerArgs,
    VariationalArgs,
)
from cmdstanpy.utils import cmdstan_version_before

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_args_algorithm() -> None:
    args = OptimizeArgs(algorithm='non-valid_algorithm')
    with pytest.raises(ValueError):
        args.validate()
    args = OptimizeArgs(algorithm='Newton')
    args.validate()
    cmd = args.compose(None, cmd=['output'])
    assert 'algorithm=newton' in ' '.join(cmd)


def test_args_algorithm_init_alpha() -> None:
    args = OptimizeArgs(algorithm='bfgs', init_alpha=2e-4)
    args.validate()
    cmd = args.compose(None, cmd=['output'])

    assert 'init_alpha=0.0002' in ' '.join(cmd)
    args = OptimizeArgs(init_alpha=2e-4)
    with pytest.raises(ValueError):
        args.validate()
    args = OptimizeArgs(init_alpha=-1.0)
    with pytest.raises(ValueError):
        args.validate()
    args = OptimizeArgs(init_alpha=1.0, algorithm='Newton')
    with pytest.raises(ValueError):
        args.validate()


def test_args_algorithm_iter() -> None:
    args = OptimizeArgs(iter=400)
    args.validate()
    cmd = args.compose(None, cmd=['output'])
    assert 'iter=400' in ' '.join(cmd)
    args = OptimizeArgs(iter=-1)
    with pytest.raises(ValueError):
        args.validate()


def test_args_min() -> None:
    args = SamplerArgs()
    args.validate(chains=4)
    cmd = args.compose(idx=1, cmd=[])
    assert 'method=sample algorithm=hmc' in ' '.join(cmd)


def test_args_chains() -> None:
    args = SamplerArgs()
    with pytest.raises(ValueError):
        args.validate(chains=None)


def test_good() -> None:
    args = SamplerArgs(
        iter_warmup=10,
        iter_sampling=20,
        save_warmup=True,
        thin=7,
        max_treedepth=15,
        adapt_delta=0.99,
    )
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'method=sample' in ' '.join(cmd)
    assert 'num_warmup=10' in ' '.join(cmd)
    assert 'num_samples=20' in ' '.join(cmd)
    assert 'save_warmup=1' in ' '.join(cmd)
    assert 'thin=7' in ' '.join(cmd)
    assert 'algorithm=hmc engine=nuts' in ' '.join(cmd)
    assert 'max_depth=15' in ' '.join(cmd)
    assert 'adapt engaged=1 delta=0.99' in ' '.join(cmd)

    args = SamplerArgs(iter_warmup=10)
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'method=sample' in ' '.join(cmd)
    assert 'num_warmup=10' in ' '.join(cmd)
    assert 'num_samples=' not in ' '.join(cmd)
    assert 'save_warmup=' not in ' '.join(cmd)
    assert 'algorithm=hmc engine=nuts' not in ' '.join(cmd)


def test_bad() -> None:
    args = SamplerArgs(iter_warmup=-10)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(iter_warmup=0, adapt_engaged=True)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(iter_sampling=-10)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(thin=-10)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(max_treedepth=-10)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(step_size=-10)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(step_size=[1.0, 1.1])
    with pytest.raises(ValueError):
        args.validate(chains=1)

    args = SamplerArgs(step_size=[1.0, -1.1])
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(adapt_delta=1.1)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(adapt_delta=-0.1)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(max_treedepth=12, fixed_param=True)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(metric='dense', fixed_param=True)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(step_size=0.5, fixed_param=True)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(adapt_delta=0.88, adapt_engaged=False)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(adapt_init_phase=0.88)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(adapt_metric_window=0.88)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(adapt_step_size=0.88)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(adapt_init_phase=-1)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(adapt_metric_window=-2)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(adapt_step_size=-3)
    with pytest.raises(ValueError):
        args.validate(chains=2)

    args = SamplerArgs(adapt_delta=0.88, fixed_param=True)
    with pytest.raises(ValueError):
        args.validate(chains=2)


def test_adapt() -> None:
    args = SamplerArgs(adapt_engaged=False)
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'method=sample algorithm=hmc adapt engaged=0' in ' '.join(cmd)

    args = SamplerArgs(adapt_engaged=True)
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'method=sample algorithm=hmc adapt engaged=1' in ' '.join(cmd)

    args = SamplerArgs(
        adapt_init_phase=26, adapt_metric_window=60, adapt_step_size=34
    )
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'method=sample algorithm=hmc adapt' in ' '.join(cmd)
    assert 'init_buffer=26' in ' '.join(cmd)
    assert 'window=60' in ' '.join(cmd)
    assert 'term_buffer=34' in ' '.join(cmd)

    args = SamplerArgs()
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'engine=nuts' not in ' '.join(cmd)
    assert 'adapt engaged=0' not in ' '.join(cmd)


def test_metric() -> None:
    args = SamplerArgs(metric='dense_e')
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'method=sample algorithm=hmc metric=dense_e' in ' '.join(cmd)

    args = SamplerArgs(metric='dense')
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'method=sample algorithm=hmc metric=dense_e' in ' '.join(cmd)

    args = SamplerArgs(metric='diag_e')
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'method=sample algorithm=hmc metric=diag_e' in ' '.join(cmd)

    args = SamplerArgs(metric='diag')
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'method=sample algorithm=hmc metric=diag_e' in ' '.join(cmd)

    args = SamplerArgs()
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'metric=' not in ' '.join(cmd)

    jmetric = os.path.join(DATAFILES_PATH, 'bernoulli.metric.json')
    args = SamplerArgs(metric=jmetric)
    args.validate(chains=4)
    cmd = args.compose(1, cmd=[])
    assert 'metric=diag_e' in ' '.join(cmd)
    assert 'metric_file=' in ' '.join(cmd)
    assert 'bernoulli.metric.json' in ' '.join(cmd)

    jmetric2 = os.path.join(DATAFILES_PATH, 'bernoulli.metric-2.json')
    args = SamplerArgs(metric=[jmetric, jmetric2])
    args.validate(chains=2)
    cmd = args.compose(0, cmd=[])
    assert 'bernoulli.metric.json' in ' '.join(cmd)
    cmd = args.compose(1, cmd=[])
    assert 'bernoulli.metric-2.json' in ' '.join(cmd)

    args = SamplerArgs(metric=[jmetric, jmetric2])
    with pytest.raises(ValueError):
        args.validate(chains=4)

    args = SamplerArgs(metric='/no/such/path/to.file')
    with pytest.raises(ValueError):
        args.validate(chains=4)


def test_fixed_param() -> None:
    args = SamplerArgs(fixed_param=True)
    args.validate(chains=1)
    cmd = args.compose(0, cmd=[])
    assert 'method=sample algorithm=fixed_param' in ' '.join(cmd)


def test_compose() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli')
    sampler_args = SamplerArgs()
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[1, 2, 3, 4],
        method_args=sampler_args,
    )
    with pytest.raises(ValueError):
        cmdstan_args.compose_command(idx=4, csv_file='foo')
    with pytest.raises(ValueError):
        cmdstan_args.compose_command(idx=-1, csv_file='foo')


def test_no_chains() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    jinits = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')

    sampler_args = SamplerArgs()
    with pytest.raises(ValueError):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=None,
            seed=[1, 2, 3],
            data=jdata,
            inits=jinits,
            method_args=sampler_args,
        )

    with pytest.raises(ValueError):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe=exe,
            chain_ids=None,
            data=jdata,
            inits=[jinits],
            method_args=sampler_args,
        )


def test_args_good() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs()

    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[1, 2, 3, 4],
        data=jdata,
        method_args=sampler_args,
        refresh=10,
    )
    assert cmdstan_args.method == Method.SAMPLE
    cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
    assert 'id=1 random seed=' in ' '.join(cmd)
    assert 'data file=' in ' '.join(cmd)
    assert 'output file=' in ' '.join(cmd)
    assert 'method=sample algorithm=hmc' in ' '.join(cmd)
    assert 'refresh=10' in ' '.join(cmd)

    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[7, 11, 18, 29],
        data=jdata,
        method_args=sampler_args,
    )
    cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
    assert 'id=7 random seed=' in ' '.join(cmd)

    # integer type
    rng = np.random.default_rng(42)
    seed = rng.integers(low=0, high=int(1e7))
    assert not isinstance(seed, int)
    assert isinstance(seed, np.integer)

    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[7, 11, 18, 29],
        data=jdata,
        seed=seed,
        method_args=sampler_args,
    )
    cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
    assert f'id=7 random seed={seed}' in ' '.join(cmd)

    dirname = 'tmp' + str(time())
    if os.path.exists(dirname):
        os.rmdir(dirname)
    CmdStanArgs(
        model_name='bernoulli',
        model_exe='bernoulli.exe',
        chain_ids=[1, 2, 3, 4],
        output_dir=dirname,
        method_args=sampler_args,
    )
    assert os.path.exists(dirname)
    os.rmdir(dirname)


def test_args_inits() -> None:
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
    assert 'init=' in ' '.join(cmd)

    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[1, 2],
        data=jdata,
        inits=[jinits1, jinits2],
        method_args=sampler_args,
    )
    cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
    assert 'bernoulli.init_1.json' in ' '.join(cmd)
    cmd = cmdstan_args.compose_command(idx=1, csv_file='bern-output-1.csv')
    assert 'bernoulli.init_2.json' in ' '.join(cmd)

    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[1, 2, 3, 4],
        data=jdata,
        inits=0,
        method_args=sampler_args,
    )
    cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
    assert 'init=0' in ' '.join(cmd)

    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[1, 2, 3, 4],
        data=jdata,
        inits=3.33,
        method_args=sampler_args,
    )
    cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
    assert 'init=3.33' in ' '.join(cmd)


# pylint: disable=no-value-for-parameter
def test_args_bad() -> None:
    sampler_args = SamplerArgs(iter_warmup=10, iter_sampling=20)

    with pytest.raises(
        Exception, match='missing 2 required positional arguments'
    ):
        CmdStanArgs(model_name='bernoulli', model_exe='bernoulli.exe')

    with pytest.raises(ValueError, match='no such file no/such/path/to.file'):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            data='no/such/path/to.file',
            method_args=sampler_args,
        )

    with pytest.raises(ValueError, match='invalid chain_id'):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, -4],
            method_args=sampler_args,
        )

    with pytest.raises(
        ValueError, match='Argument "seed" must be an integer between'
    ):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            seed=4294967299,
            method_args=sampler_args,
        )

    with pytest.raises(
        ValueError, match='Number of seeds must match number of chains'
    ):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            seed=[1, 2, 3],
            method_args=sampler_args,
        )

    with pytest.raises(
        ValueError, match='Argument "seed" must be an integer between'
    ):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            seed=-3,
            method_args=sampler_args,
        )

    with pytest.raises(
        ValueError, match='Argument "seed" must be an integer between'
    ):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            seed='badseed',
            method_args=sampler_args,
        )

    with pytest.raises(ValueError, match='Argument "inits" must be > 0'):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            inits=-5,
            method_args=sampler_args,
        )

    jinits = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')
    with pytest.raises(
        ValueError, match='Number of inits files must match number of chains'
    ):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            inits=[jinits, jinits],
            method_args=sampler_args,
        )

    with pytest.raises(ValueError, match='no such file'):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            inits='no/such/path/to.file',
            method_args=sampler_args,
        )

    fname = 'foo.txt'
    if os.path.exists(fname):
        os.remove(fname)
    with pytest.raises(
        ValueError, match='Specified output_dir is not a directory'
    ):
        open(fname, 'x').close()
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            output_dir=fname,
            method_args=sampler_args,
        )
    if os.path.exists(fname):
        os.remove(fname)

    # TODO: read-only dir test for Windows - set ACLs, not mode
    if platform.system() == 'Darwin' or platform.system() == 'Linux':
        with pytest.raises(ValueError):
            read_only = os.path.join(_TMPDIR, 'read_only')
            os.mkdir(read_only, mode=0o444)
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                output_dir=read_only,
                method_args=sampler_args,
            )

    with pytest.raises(
        ValueError, match='Argument "refresh" must be a positive integer value'
    ):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            method_args=sampler_args,
            refresh="a",
        )

    with pytest.raises(
        ValueError, match='Argument "refresh" must be a positive integer value'
    ):
        CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            method_args=sampler_args,
            refresh=0,
        )


def test_args_sig_figs(caplog: pytest.LogCaptureFixture) -> None:
    sampler_args = SamplerArgs()
    cmdstan_path()  # sets os.environ['CMDSTAN']
    if cmdstan_version_before(2, 25):
        with caplog.at_level(logging.WARNING):
            logging.getLogger()
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                sig_figs=12,
                method_args=sampler_args,
            )
        expect = (
            'Argument "sig_figs" invalid for CmdStan versions < 2.25, using '
            f'version {os.path.basename(cmdstan_path())} in directory '
            f'{os.path.dirname(cmdstan_path())}'
        )
        check_present(caplog, ('cmdstanpy', 'WARNING', expect))
    else:
        cmdstan_args = CmdStanArgs(
            model_name='bernoulli',
            model_exe='bernoulli.exe',
            chain_ids=[1, 2, 3, 4],
            sig_figs=12,
            method_args=sampler_args,
        )
        cmd = cmdstan_args.compose_command(idx=0, csv_file='bern-output-1.csv')
        assert 'sig_figs=' in ' '.join(cmd)
        with pytest.raises(ValueError):
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                sig_figs=-1,
                method_args=sampler_args,
            )
        with pytest.raises(ValueError):
            CmdStanArgs(
                model_name='bernoulli',
                model_exe='bernoulli.exe',
                chain_ids=[1, 2, 3, 4],
                sig_figs=20,
                method_args=sampler_args,
            )


def test_args_fitted_params() -> None:
    args = GenerateQuantitiesArgs(csv_files=['no_such_file'])
    with pytest.raises(ValueError):
        args.validate(chains=1)
    csv_files = [
        os.path.join(DATAFILES_PATH, 'runset-good', 'bern-{}.csv'.format(i + 1))
        for i in range(4)
    ]
    args = GenerateQuantitiesArgs(csv_files=csv_files)
    args.validate(chains=4)
    cmd = args.compose(idx=0, cmd=[])
    assert 'method=generate_quantities' in ' '.join(cmd)
    assert 'fitted_params={}'.format(csv_files[0]) in ' '.join(cmd)


def test_args_variational() -> None:
    args = VariationalArgs()

    args = VariationalArgs(output_samples=1)
    args.validate(chains=1)
    cmd = args.compose(idx=0, cmd=[])
    assert 'method=variational' in ' '.join(cmd)
    assert 'output_samples=1' in ' '.join(cmd)

    args = VariationalArgs(tol_rel_obj=0.01)
    args.validate(chains=1)
    cmd = args.compose(idx=0, cmd=[])
    assert 'method=variational' in ' '.join(cmd)
    assert 'tol_rel_obj=0.01' in ' '.join(cmd)

    args = VariationalArgs(adapt_engaged=True, adapt_iter=100)
    args.validate(chains=1)
    cmd = args.compose(idx=0, cmd=[])
    assert 'adapt engaged=1 iter=100' in ' '.join(cmd)

    args = VariationalArgs(adapt_engaged=False)
    args.validate(chains=1)
    cmd = args.compose(idx=0, cmd=[])
    assert 'adapt engaged=0' in ' '.join(cmd)

    args = VariationalArgs(eta=0.1)
    args.validate(chains=1)
    cmd = args.compose(idx=0, cmd=[])
    assert 'eta=0.1' in ' '.join(cmd)


def test_variational_args_bad() -> None:
    args = VariationalArgs(algorithm='no_such_algo')
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(iter=0)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(iter=1.1)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(grad_samples=0)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(grad_samples=1.1)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(elbo_samples=0)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(elbo_samples=1.1)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(eta=-0.00003)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(adapt_iter=0)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(adapt_iter=1.1)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(tol_rel_obj=0)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(eval_elbo=0)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(eval_elbo=1.5)
    with pytest.raises(ValueError):
        args.validate()

    args = VariationalArgs(output_samples=0)
    with pytest.raises(ValueError):
        args.validate()


def test_args_laplace():
    mode = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv')
    args = LaplaceArgs(mode=mode)
    args.validate()
    cmd = args.compose(0, cmd=[])
    full_cmd = ' '.join(cmd)
    assert 'method=laplace' in full_cmd
    assert 'rosenbrock_mle.csv' in full_cmd

    args = LaplaceArgs(mode=mode, jacobian=False)
    args.validate()
    cmd = args.compose(0, cmd=[])
    full_cmd = ' '.join(cmd)
    assert 'method=laplace' in full_cmd
    assert 'rosenbrock_mle.csv' in full_cmd
    assert 'jacobian=0' in full_cmd


def test_args_laplace_bad():
    fake_mode = os.path.join(DATAFILES_PATH, 'not_here.csv')
    args = LaplaceArgs(mode=fake_mode)
    with pytest.raises(ValueError):
        args.validate()

    mode = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv')
    args = LaplaceArgs(mode=mode, draws=0)
    with pytest.raises(ValueError):
        args.validate()

    args = LaplaceArgs(mode=mode, draws=1.1)
    with pytest.raises(ValueError):
        args.validate()


def test_args_pathfinder():
    args = PathfinderArgs()
    args.validate()
    cmd = args.compose(0, cmd=[])
    full_cmd = ' '.join(cmd)
    assert 'method=pathfinder' in full_cmd

    args = PathfinderArgs(init_alpha=0.1)
    args.validate()
    cmd = args.compose(0, cmd=[])
    full_cmd = ' '.join(cmd)
    assert 'method=pathfinder' in full_cmd
    assert 'init_alpha=0.1' in full_cmd

    args = PathfinderArgs(
        num_draws=93, num_psis_draws=42, num_paths=5, num_elbo_draws=10
    )
    args.validate()
    cmd = args.compose(0, cmd=[])
    full_cmd = ' '.join(cmd)
    assert 'method=pathfinder' in full_cmd
    assert 'num_draws=93' in full_cmd
    assert 'num_psis_draws=42' in full_cmd
    assert 'num_paths=5' in full_cmd
    assert 'num_elbo_draws=10' in full_cmd


@pytest.mark.parametrize(
    "arg,require_int",
    [
        ('init_alpha', False),
        ('tol_obj', False),
        ('tol_rel_obj', False),
        ('tol_grad', False),
        ('tol_rel_grad', False),
        ('tol_param', False),
        ('history_size', True),
        ('num_psis_draws', True),
        ('num_paths', True),
        ('max_lbfgs_iters', True),
        ('num_draws', True),
        ('num_elbo_draws', True),
    ],
)
def test_args_pathfinder_bad(arg, require_int):
    # pathfinder's only arg restrictions are on positiveness
    args = PathfinderArgs(**{arg: 0})
    with pytest.raises(ValueError):
        args.validate()
    args = PathfinderArgs(**{arg: -1})
    with pytest.raises(ValueError):
        args.validate()
    if require_int:
        args = PathfinderArgs(**{arg: 1.1})
        with pytest.raises(ValueError):
            args.validate()
