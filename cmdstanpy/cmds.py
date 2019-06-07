"""
First class functions.
"""
import os
import os.path
import platform
import shutil
import subprocess
import tempfile

import pandas as pd

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List

from cmdstanpy import TMPDIR
from cmdstanpy.lib import Model, StanData, RunSet, SamplerArgs
from cmdstanpy.utils import cmdstan_path


def compile_model(
    stan_file: str = None, opt_lvl: int = 1, overwrite: bool = False,
    include_paths: List[str] = None
) -> Model:
    """
    Compile the given Stan model file to an executable.

    :param stan_file: Path to Stan program

    :param opt_lvl: Optimization level for c++ compiler, one of {0, 1, 2, 3}
      where level 0 optimization results in the shortest compilation time
      with code that may run slowly and increasing optimization levels increase
      compile time and runtime performance.

    :param overwrite: When True, existing executible will be overwritten.
      Defaults to False.
    
    :param include_paths: list of paths to directories where Stan should look 
      for files to include.
    """
    if stan_file is None:
        raise Exception('must specify argument "stan_file"')
    if not os.path.exists(stan_file):
        raise Exception('no such stan_file {}'.format(stan_file))
    program_name = os.path.basename(stan_file)
    exe_file, _ = os.path.splitext(os.path.abspath(stan_file))
    hpp_file = '.'.join([exe_file, 'hpp'])
    if overwrite or not os.path.exists(hpp_file):
        print('translating to {}'.format(hpp_file))
        stanc_path = os.path.join(cmdstan_path(), 'bin', 'stanc')
        cmd = [stanc_path, '--o={}'.format(hpp_file), stan_file]
        if include_paths is not None:
            if any([not os.path.exists(d) for d in include_paths]):
                raise Exception('no such include path {}'.format(d))
            cmd.append('--include_paths=' + ','.join(include_paths))
        print('stan to c++: make args {}'.format(cmd))
        do_command(cmd)
        if not os.path.exists(hpp_file):
            raise Exception('syntax error'.format(stan_file))

    if platform.system().lower().startswith('win'):
        exe_file += '.exe'
    if not overwrite and os.path.exists(exe_file):
        # print('model is up to date') # notify user or not?
        return Model(stan_file, exe_file)
    exe_file_path = Path(exe_file).as_posix()
    cmd = ['make', 'O={}'.format(opt_lvl), exe_file_path]
    print('compiling c++: make args {}'.format(cmd))
    try:
        do_command(cmd, cmdstan_path())
    except Exception:
        return Model(stan_file)
    return Model(stan_file, exe_file)


def sample(
    stan_model: Model,
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
    """
    Run or more chains of the NUTS/HMC sampler.

    Optional parameters override CmdStan default arguments.

    :param stan_model: compiled Stan program
    :param chains: number of sampler chains, should be > 1
    :param cores: number of processes to run in parallel,
        shouldn't exceed number of available processors
    :param seed: seed for random number generator, if unspecified the seed
        seed is generated from the system time
    :param data: dictionary with entries for all data variables in the model
    :param data_file: path to input data file in JSON or Rdump format.
        *Note: cannot specify both `data` and `data_file` arguments*
    :param init_param_values: dictionary with entries for initial value of
        model parameters
    :param init_param_values_file: path to initial parameter values file in
        JSON or Rdump format.
        *Note: cannot specify both `init_param_values` and
        `init_param_values_file` arguments*
    :parm csv_output_file: basename
    :param refresh: console refresh rate, -1 for no updates
    :param post_warmup_draws_per_chain: number of draws after warmup
    :param warmup_draws_per_chain: number of draws during warmup
    :param save_warmup: when True, sampler saves warmup draws in output file
    :param thin: period between saved samples.
        *Note: default value 1 is strongly recommended*
    :param do_adaptation: when True, adapt stepsize, metric.
        *Note: True requires that number of warmup draws > 0;
        False requires that number of warmup draws == 0*
    :param adapt_gamma: adaptation regularization scale
    :param adapt_delta: adaptation target acceptance statistic
    :param adapt_kappa: adaptation relaxation exponent
    :param adapt_t0: adaptation interation offset
    :param nuts_max_depth: maximum allowed iterations of NUTS sampler
    :param hmc_metric: Euclidean metric, either `diag_e` or `dense_e`
    :param hmc_metric_file: path to file containing either vector specifying
        diagonal metric or matrix specifying dense metric
        in either JSON or Rdump format
    :param hmc_stepsize: initial stepsize for HMC sampler
    :param hmc_stepsize_jitter: amount of random jitter added to stepsize
        at each sampler iteration
    """
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
    runset.validate_csv_files()
    return runset


def summary(runset: RunSet) -> pd.DataFrame:
    """
    Run cmdstan/bin/stansummary over all output csv files.
    Echo stansummary stdout/stderr to console.
    Assemble csv tempfile contents into pandasDataFrame.

    :param runset: record of completed run of NUTS sampler
    """
    names = runset.column_names
    cmd_path = os.path.join(cmdstan_path(), 'bin', 'stansummary')
    tmp_csv_file = 'stansummary-{}-{}-chains-'.format(
        runset.model, runset.chains)
    fd, tmp_csv_path = tempfile.mkstemp(
        suffix='.csv', prefix=tmp_csv_file, dir=TMPDIR, text=True
        )
    cmd = '{} --csv_file={} {}'.format(
        cmd_path, tmp_csv_path, ' '.join(runset.csv_files)
        )
    do_command(cmd.split())  # breaks on all whitespace
    summary_data = pd.read_csv(
        tmp_csv_path, delimiter=',', header=0, index_col=0, comment='#'
        )
    mask = [
        x == 'lp__' or not x.endswith('__') for x in summary_data.index
        ]
    return summary_data[mask]


def diagnose(runset: RunSet) -> None:
    """
    Run cmdstan/bin/diagnose over all output csv files.
    Echo diagnose stdout/stderr to console.

    :param runset: record of completed run of NUTS sampler
    """
    cmd_path = os.path.join(cmdstan_path(), 'bin', 'diagnose')
    csv_files = ' '.join(runset.csv_files)
    cmd = '{} {} '.format(cmd_path, csv_files)
    result = do_command(cmd=cmd.split())
    if result is None:
        print('No problems detected.')
    else:
        print(result)


def get_drawset(runset: RunSet, params: List[str] = None) -> pd.DataFrame:
    """
    Returns the assembled sample as a pandas DataFrame consisting of
    one column per parameter and one row per draw.

    :param runset: record of completed run of NUTS sampler
    :param params: list of model parameter names.
    """
    pnames_base = [name.split('.')[0] for name in runset.column_names]
    if params is not None:
        for p in params:
            if not (p in runset._column_names or p in pnames_base):
                raise ValueError('unknown parameter: {}'.format(p))
    runset.assemble_sample()
    data = runset.sample.reshape(
        (runset.draws * runset.chains), len(runset.column_names), order='A'
        )
    df = pd.DataFrame(data=data, columns=runset.column_names)
    if params is None:
        return df
    mask = []
    for p in params:
        for name in runset.column_names:
            if p == name or p == name.split('.')[0]:
                mask.append(name)
    return df[mask]


def save_csvfiles(
        runset: RunSet, dir: str = None, basename: str = None) -> None:
    """
    Moves csvfiles to specified directory using specified basename,
    appending suffix '-<id>.csv' to each.

    :param runset: record of completed run of NUTS sampler
    :param dir: directory path
    :param basename:  base filename
    """
    if dir is None:
        dir = '.'
    test_path = os.path.join(dir, '.{}-test.tmp'.format(basename))
    try:
        with open(test_path, 'w') as fd:
            pass
        os.remove(test_path)  # cleanup
    except OSError:
        raise Exception('cannot save to path: {}'.format(dir))

    for i in range(runset.chains):
        if not os.path.exists(runset.csv_files[i]):
            raise ValueError(
                'cannot access csv file {}'.format(runset.csv_files[i]))
        to_path = os.path.join(dir, '{}-{}.csv'.format(basename, i+1))
        if os.path.exists(to_path):
            raise ValueError(
                'file exists, not overwriting: {}'.format(to_path))
        try:
            print('saving tmpfile: "{}" as: "{}"'.format(
                    runset.csv_files[i], to_path))
            shutil.move(runset.csv_files[i], to_path)
            runset.csv_files[i] = to_path
        except (IOError, OSError) as e:
            raise ValueError('cannot save to file: {}'.format(to_path)) from e


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


def do_command(cmd: str, cwd: str = None) -> str:
    """
    Spawn process, print stdout/stderr to console.
    Throws exception on non-zero returncode.
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    proc.wait()
    stdout, stderr = proc.communicate()
    if proc.returncode:
        if stderr:
            msg = 'ERROR\n {} '.format(stderr.decode('ascii').strip())
        raise Exception(msg)
    if stdout:
        return stdout.decode('ascii').strip()
    return None
