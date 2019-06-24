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
    stan_file: str = None,
    opt_lvl: int = 2,
    overwrite: bool = False,
    include_paths: List[str] = None,
) -> Model:
    """
    Compile the given Stan model file to an executable.

    :param stan_file: Path to Stan program

    :param opt_lvl: Optimization level for C++ compiler, one of {0, 1, 2, 3}
      where level 0 optimization results in the shortest compilation time
      with code that may run slowly and increasing optimization levels increase
      compile time and runtime performance.

    :param overwrite: When True, existing executable will be overwritten.
      Defaults to False.

    :param include_paths: List of paths to directories where Stan should look
      for files to include in compilation of the C++ executable.
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
    seed: Union[int, List[int]] = None,
    chain_ids: Union[int, List[int]] = None,
    inits: Union[Dict, float, str, List[str]] = None,
    warmup_iters: int = None,
    sampling_iters: int = None,
    warmup_schedule: Tuple[float, float, float] = (0.15, 0.75, 0.10),
    save_warmup: bool = False,
    thin: int = None,
    max_treedepth: float = None,
    metric: Union[str, List[str]] = None,
    step_size: Union[float, List[float]] = None,
    adapt_engaged: bool = True,
    adapt_delta: float = None,
    csv_output_file: str = None,
    show_progress: bool = False,
) -> RunSet:
    """
    Run or more chains of the NUTS sampler to produce a set of draws
    from the posterior distribution of a model conditioned on some data.
    The caller must specify the model and data; all other arguments
    are optional.

    This function validates the specified configuration, composes a call to
    the CmdStan ``sample`` method and spawns one subprocess per chain to run
    the sampler and waits for all chains to run to completion.
    The composed call to CmdStan omits arguments left unspecified (i.e., value
    is ``None``) so that the default CmdStan configuration values will be used.

    For each chain, the ``RunSet`` object records the command, the return code,
    the paths to the sampler output files, and the corresponding subprocess
    console outputs, if any.

    :param stan_model: Compiled Stan model.

    :param data: Values for all data variables in the model, specified either
        as a dictionary with entries matching the data variables,
        or as the path of a data file in JSON or Rdump format.

    :param chains: Number of sampler chains, should be > 1.

    :param cores: Number of processes to run in parallel. Must be an integer
        between 1 and the number of CPUs in the system.

    :param seed: The seed for random number generator or a list of per-chain
        seeds. Must be an integer between 0 and 2^32 - 1. If unspecified,
        numpy.random.RandomState() is used to generate a seed which will be
        used for all chains. When the same seed is used across all chains,
        the chain-id is used to advance the RNG to avoid dependent samples.

    :param chain_ids: The offset for the random number generator, either
        an integer or a list of unique per-chain offsets.  If unspecified,
        chain ids are numbered sequentially starting from 1.

    :param inits: Specifies how the sampler initializes parameter values.
        Initializiation is either uniform random on a range centered on 0,
        exactly 0, or a dictionary or file of initial values for some or all
        parameters in the model.  The default initialization behavoir will
        initialize all parameter values on range [-2, 2].  If these values
        are too far from the expected parameter values, explicit initialization
        may improve adaptation. The following value types are allowed:

        * Single number ``n > 0`` - initialization range is [-n, n].
        * ``0`` - all parameters are initialized to 0.
        * dictionary - pairs parameter name : initial value.
        * string - pathname to a JSON or Rdump file of initial parameter values.
        * list of strings - per-chain pathname to data file.

    :param warmup_iters: Number of iterations during warmup for each chain.

    :param sampling_iters: Number of draws from the posterior for each chain.

    :param warmup_schedule: Triple specifying fraction of total warmup
        iterations allocated to each adaptation phase.  The default schedule
        is (.15, .75, .10) where:

        * Phase I is "fast" adaptation to find the typical set
        * Phase II is "slow" adaptation to find the metric
        * Phase III is "fast" adaptation to find the step_size.

        For further details, see the Stan Reference Manual, section
        HMC algorithm parameters.

    :param save_warmup: When True, sampler saves warmup draws as part of
        the Stan csv output file.

    :param thin: Period between saved samples.

    :param max_treedepth: Maximum depth of trees evaluated by NUTS sampler
        per iteration.

    :param metric: Specification of the mass matrix, either as a
        vector consisting of the diagonal elements of the covariance
        matrix (``diag`` or ``diag_e``) or the full covariance matrix
        (``dense`` or ``dense_e``).

        If the value of the metric argument is a string other than
        ``diag``, ``diag_e``, ``dense``, or ``dense_e``, it must be
        a valid filepath to a JSON or Rdump file which contains an entry
        ``inv_metric`` whose value is either the diagonal vector or
        the full covariance matrix. This can be used to restart sampling
        with no adaptation given the outputs of all chains from a previous run.

        If the value of the metric argument is a list of paths, its
        length must match the number of chains and all paths must be
        unique.

    :param step_size: Initial stepsize for HMC sampler.  The value is either
        a single number or a list of numbers which will be used as the global
        or per-chain initial step_size, respectively.

        The length of the list of step sizes must match the number of chains.
        This feature can be used to restart sampling with no adaptation
        given the outputs of all chains from a previous run.

    :param adapt_engaged: When True, adapt stepsize, metric.
        *Note: If True, ``warmup_iters`` must be > 0.*

    :param adapt_delta: Adaptation target Metropolis acceptance rate.
        The default value is 0.8.  Increasing this value, which must be
        strictly less than 1, causes adaptation to use smaller step sizes.
        It improves the effective sample size, but may increase the time
        per iteration.

    :param csv_output_file: A path or file name which will be used as the
        base name for the sampler output files.  The csv output files
        for each chain are written to file ``<basename>-<chain_id>.csv``
        and the console output and error messages are written to file
        ``<basename>-<chain_id>.txt``.

    :param show_progress: When True, command sends progress messages to
        console. When False, command executes silently.
    """
    if chains < 1:
        raise ValueError(
            'chains must be a positive integer value, found {}'.format(chains)
        )

    if chain_ids is None:
        chain_ids = [x + 1 for x in range(chains)]
    else:
        if type(chain_ids) is int:
            if chain_ids < 1:
                raise ValueError(
                    'chain_id must be a positive integer value,'
                    ' found {}'.format(chain_ids)
                )
            offset = chain_ids
            chain_ids = [x + offset + 1 for x in range(chains)]
        else:
            if not len(chain_ids) == chains:
                raise ValueError(
                    'chain_ids must correspond to number of chains'
                    ' specified {} chains, found {} chain_ids'.format(
                        chains, len(chain_ids)
                    )
                )
            for i in len(chain_ids):
                if chain_ids[i] < 1:
                    raise ValueError(
                        'chain_id must be a positive integer value,'
                        ' found {}'.format(chain_ids[i])
                    )

    if cores < 1:
        raise ValueError(
            'cores must be a positive integer value, found {}'.format(cores)
        )
    if cores > cpu_count():
        print(
            'requested {} cores, only {} available'.format(cores, cpu_count())
        )
        cores = cpu_count()

    if data is not None:
        if isinstance(data, dict):
            with tempfile.NamedTemporaryFile(
                mode='w+', suffix='.json', dir=TMPDIR, delete=False
            ) as fd:
                data_file = fd.name
                print('input data tempfile: {}'.format(fd.name))
            sd = StanData(data_file)
            sd.write_json(data)
            data_dict = data
            data = data_file

    if inits is not None:
        if isinstance(inits, dict):
            with tempfile.NamedTemporaryFile(
                mode='w+', suffix='.json', dir=TMPDIR, delete=False
            ) as fd:
                inits_file = fd.name
                print('inits tempfile: {}'.format(fd.name))
            sd = StanData(inits_file)
            sd.write_json(inits)
            inits_dict = inits
            inits = inits_file
        # TODO:  issue 49: inits can be initialization function

    args = SamplerArgs(
        model=stan_model,
        chain_ids=chain_ids,
        data=data,
        seed=seed,
        inits=inits,
        warmup_iters=warmup_iters,
        sampling_iters=sampling_iters,
        warmup_schedule=warmup_schedule,
        save_warmup=save_warmup,
        thin=thin,
        max_treedepth=max_treedepth,
        metric=metric,
        step_size=step_size,
        adapt_engaged=adapt_engaged,
        adapt_delta=adapt_delta,
        output_file=csv_output_file,
    )

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
        runset.model, runset.chains
    )
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
    mask = [x == 'lp__' or not x.endswith('__') for x in summary_data.index]
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
    runset: RunSet, dir: str = None, basename: str = None
) -> None:
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
                'cannot access csv file {}'.format(runset.csv_files[i])
            )
        to_path = os.path.join(dir, '{}-{}.csv'.format(basename, i + 1))
        if os.path.exists(to_path):
            raise ValueError('file exists, not overwriting: {}'.format(to_path))
        try:
            print(
                'saving tmpfile: "{}" as: "{}"'.format(
                    runset.csv_files[i], to_path
                )
            )
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
    proc = subprocess.Popen(
        cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
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
