"""
First class functions.
"""
import os
import platform
import shutil
import subprocess
import tempfile

import pandas as pd

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Union, Tuple

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

    :param overwrite: When True, existing executable will be overwritten.
      Defaults to False.

    :param include_paths: List of paths to directories where Stan should look
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
            bad_paths = [d for d in include_paths if not os.path.exists(d)]
            if any(bad_paths):
                raise Exception(
                    'invalid include paths: {}'.format(', '.join(bad_paths))
                )
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
    data: Union[Dict, str] = None,
    chains: int = 4,
    cores: int = 1,
    seed: int = None,
    chain_id: Union[int, List[int]] = None,
    inits: Union[Dict, List[str], str] = None,
    warmup_iter: int = None,
    sampling_iter: int = None,
    wamup_schedule: Tuple[float, float, float] = None,
    save_warmup: bool = False,
    thin: int = None,
    max_treedepth: float = None,
    metric: Union[List[str], str] = None,
    step_size: float = 1.0,
    adapt_engaged: bool = True,
    target_accept_rate: float = None,
    output_file: str = None,
    show_progress: bool = False
) -> RunSet:
    """
    Run or more chains of the NUTS sampler to produce a set of draws
    from the posterior distribution of a model conditioned on some data.
    The caller must specify the model and data; all other arguments
    are optional.

    This function validates the specified configuration, composes a call to
    the CmdStan `sample` method and spawns one subprocess per chain to run
    the sampler and waits for all chains to run to completion.
    For each chain, it records the set return code, location of the sampler
    output files,  and the corresponding subprocess console outputs, if any.

    :param stan_model: Compiled Stan model.

    :param data: Values for all data variables in the model, specified either
        as a dictionary with entries matching the data variables,
        or as the path of a data file in JSON or Rdump format.

    :param chains: Number of sampler chains, should be > 1.

    :param cores: Number of processes to run in parallel. If this value
        exceeds the number of available processors, only max processors chains
        will be run in parallel.

    :param seed: The seed for random number generator or a list of per-chain
        seeds.  If unspecified, a seed is generated from the system time.  When
        the same seed is used across all chains, the chain-id is used to advance
        the RNG to avoid dependent samples.

    :param chain_id: The offset for the random number generator, either an integer or
        a list of unique per-chain offsets.  If unspecified, chain ids are numbered
        sequentially starting from 1.

    :param inits: Initial model parameter values.

        * By default, all parameters are randomly initialized between [-2, 2].
        * If the value is a number n > 0, the initialization range is [-n, n].
        * If the value is 0, all parameters are initialized to 0.
        * If the value is a dictionary, the entries are used for initialization. Missing parameter values are randomly initialized in range [-2, 2].
        * If the value is a string, it is the pathname to a data file in JSON or Rdump format of initial parameter values.
        * If the value is a list of strings, these are the per-chain data file paths.

    :param warmup_iter: Number of iterations during warmup for each chain.

    :param sampling_iter: Number of draws from the posterior for each chain.

    :param warmup_schedule: Triple specifying percentage of warmup iterations
        allocated to each phase of adaptation.  The default schedule is 
        ( 15%, 75%, 10%) where

        * Phase I is "fast" adaptation to find the typical set
        * Phase II is "slow" adaptation to find the metric
        * Phase III is "fast" adaptation to find the step_size.

        For further details, see `the Stan Reference Manual
        <https://mc-stan.org/docs/2_19/reference-manual/hmc-algorithm-parameters.html>`_.

    :param save_warmup: When True, sampler saves warmup draws as part of
        the Stan csv output file.

    :param thin: Period between saved samples. *Note: default value 1 is strongly recommended*

    :param max_treedepth: Maximum depth of trees evaluated by NUTS sampler
        per iteration.

    :param metric:  Specification of the mass matrix. One of:

        * If value is "diag", diagonal matrix is estimated.
        * If value is "dense", full matrix is estimated.
        * Otherwise, the value is a file path of list of filepaths where each file specifies the metric either as a vector of diagonal entries for a diagonal metric of a matrix for the dense metric. The data must be in JSON or Rdump format.

    :param step_size: Initial stepsize for HMC sampler.

    :param adapt_engaged: When True, adapt stepsize, metric.
        *Note: If True, `warmup_iter` must be > 0.*

    :param target_accept_rate: Adaptation target acceptance statistic.

    :parm csv_output_file: A path or file name which will be used as the
        base name for the sampler output files.  The csv output files produced by
        each chain are written to file `<basename>-<chain_id>.csv` and the console
        output and error messages are written to files `<basename>-<chain_id>.txt`.

    :param show_progress: When True, command sends progress messages to console.
        When False, command executes silently.

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

    The diagnose utility reads the outputs of all chains
    and checks for the following potential problems:

    + Transitions that hit the maximum treedepth
    + Divergent transitions
    + Low E-BFMI values (sampler transitions HMC potential energy)
    + Low effective sample sizes
    + High R-hat values

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
            transcript.write(stdout.decode('utf-8'))
        if stderr:
            transcript.write('ERROR')
            transcript.write(stderr.decode('utf-8'))
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
            msg = 'ERROR\n {} '.format(stderr.decode('utf-8').strip())
        raise Exception(msg)
    if stdout:
        return stdout.decode('utf-8').strip()
    return None
