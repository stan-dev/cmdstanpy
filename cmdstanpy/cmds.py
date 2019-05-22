"""
First class functions
"""
import os
import platform
import subprocess
import tempfile

from pathlib import Path
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Dict

from cmdstanpy import TMPDIR
from cmdstanpy.lib import Model, StanData, RunSet, SamplerArgs, PosteriorSample
from cmdstanpy.utils import do_command, cmdstan_path


def compile_model(
    stan_file: str = None, opt_lvl: int = 1, overwrite: bool = False
) -> Model:
    """Compile the given Stan model file to an executable."""
    if stan_file is None:
        raise Exception('must specify argument "stan_file"')
    stan_file = Path(stan_file)
    if not stan_file.exists():
        raise Exception('no such stan_file {}'.format(stan_file))
    path = stan_file.parent.absolute()
    program_name = stan_file.name
    model_name = stan_file.stem

    hpp_name = model_name + '.hpp'
    hpp_file = path / hpp_name
    if overwrite or not hpp_file.exists():
        print('translating to {}'.format(hpp_file))
        stanc = 'stanc'
        if platform.system() == "Windows":
            stanc += '.exe'
        stanc_path = cmdstan_path() / 'bin' / stanc
        cmd = [stanc_path, '--o={}'.format(hpp_file.as_posix()), stan_file.as_posix()]
        print('stan to c++: make args {}'.format(cmd))
        do_command(cmd)
        if not hpp_file.exists():
            raise Exception('syntax error: {}'.format(stan_file))

    if platform.system() == "Windows":
        model_name += '.exe'
    exe_file = path / model_name
    if not overwrite and exe_file.exists():
        # print('model is up to date') # notify user or not?
        return Model(stan_file, exe_file)
    cmd = ['make', 'O={}'.format(opt_lvl), exe_file.as_posix()]
    print('compiling c++: make args {}'.format(cmd))
    try:
        do_command(cmd, cmdstan_path())
    except Exception:
        return Model(stan_file)
    return Model(stan_file, exe_file)


def sample(
    stan_model: Model = None,
    chains: int = 4,
    cores: int = 1,
    seed: int = None,
    data: Dict = None,
    data_file: str = None,
    init_param_values: Dict = None,
    init_param_values_file: str = None,
    csv_output_file: str = None,
    refresh: int = None,
    post_warmup_draws_per_chain: int = None,
    warmup_draws_per_chain: int = None,
    save_warmup: bool = False,
    thin: int = None,
    do_adaptation: bool = True,
    adapt_gamma: float = None,
    adapt_delta: float = None,
    adapt_kappa: float = None,
    adapt_t0: float = None,
    nuts_max_depth: int = None,
    hmc_metric: str = None,
    hmc_metric_file: str = None,
    hmc_stepsize: float = 1.0,
    hmc_stepsize_jitter: float = 0,
) -> RunSet:
    """Run or more chains of the NUTS/HMC sampler."""

    if data is not None and (
            data_file is not None and os.path.exists(data_file)):
        raise ValueError(
            'cannot specify both "data" and "data_file" arguments')
    if data is not None:
        if data_file is None:
            fd = tempfile.NamedTemporaryFile(
                mode='w+', suffix='.json', dir=TMPDIR, delete=False
            )
            data_file = fd.name
            print('input data tempfile: {}'.format(fd.name))
        sd = StanData(data_file)
        sd.write_json(data)

    if (
        init_param_values is not None
        and init_param_values_file is not None
        and os.path.exists(init_param_values_file)
    ):
        raise ValueError(
            'cannot specify both"init_param_values" '
            'and "init_param_values_file" arguments'
        )
    if init_param_values is not None:
        if init_param_values_file is None:
            fd = tempfile.NamedTemporaryFile(
                mode='w+', suffix='.json', dir=TMPDIR, delete=False
            )
            init_param_values_file = fd.name
            print('init params tempfile: {}'.format(fd.name))
        sd = StanData(init_param_values_file)
        sd.write_json(init_param_values)

    args = SamplerArgs(
        model=stan_model,
        seed=seed,
        data_file=data_file,
        init_param_values=init_param_values_file,
        output_file=csv_output_file,
        refresh=refresh,
        post_warmup_draws=post_warmup_draws_per_chain,
        warmup_draws=warmup_draws_per_chain,
        save_warmup=save_warmup,
        thin=thin,
        do_adaptation=do_adaptation,
        adapt_gamma=adapt_gamma,
        adapt_delta=adapt_delta,
        adapt_kappa=adapt_kappa,
        adapt_t0=adapt_t0,
        nuts_max_depth=nuts_max_depth,
        hmc_metric_file=hmc_metric_file,
        hmc_stepsize=hmc_stepsize,
        hmc_stepsize_jitter=hmc_stepsize_jitter,
    )
    args.validate()
    if chains < 1:
        raise ValueError(
            'chains must be a positive integer value, found {}'.format(chains)
        )
    if cores < 1:
        raise ValueError(
            'cores must be a positive integer value, found {}'.format(cores)
        )
    if cores > cpu_count():
        print('requested {} cores, only {} available'.format(
            cores, cpu_count()))
        cores = cpu_count()
    runset = RunSet(args=args, chains=chains)
    try:
        tp = ThreadPool(cores)
        for i in range(chains):
            tp.apply_async(do_sample, (runset, i))
    finally:
        tp.close()
        tp.join()
    if not runset.check_retcodes():
        msg = 'Error during sampling'
        for i in range(chains):
            if runset.retcode(i) != 0:
                msg = '{}, chain {} returned error code {}'.format(
                    msg, i, runset.retcode(i)
                )
        raise Exception(msg)
    run_dict = runset.validate_csv_files()
    post_sample = PosteriorSample(run_dict, runset.csv_files)
    return post_sample


def do_sample(runset: RunSet, idx: int) -> None:
    """
    Encapsulates call to sampler.
    Spawn process, capture console output to file, record returncode.
    """
    cmd = runset.cmds[idx]
    print('start chain {}.  '.format(idx + 1))
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    proc.wait()
    stdout, stderr = proc.communicate()
    transcript_file = runset.console_files[idx]
    print('finish chain {}.  '.format(idx + 1))
    with open(transcript_file, 'w+') as transcript:
        if stdout:
            transcript.write(stdout.decode('ascii'))
        if stderr:
            transcript.write('ERROR')
            transcript.write(stderr.decode('ascii'))
    runset.set_retcode(idx, proc.returncode)
