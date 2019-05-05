import os
import os.path
import subprocess
import tempfile
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


from cmdstanpy import CMDSTAN_PATH, TMPDIR, STANSUMMARY_STATS
from cmdstanpy.lib import Model, RunSet, SamplerArgs, PosteriorSample

def compile_model(stan_file:str, opt_lvl:int=3, overwrite:bool=False) -> Model:
    """Compile the given Stan model file to an executable.
    """
    print("compile_model, CMDSTAN_PATH={}".format(CMDSTAN_PATH))
    if not os.path.exists(stan_file):
        raise Exception('no such stan_file {}'.format(stan_file))
    path = os.path.abspath(os.path.dirname(stan_file))
    program_name = os.path.basename(stan_file)
    model_name = os.path.splitext(program_name)[0]

    hpp_name = model_name + '.hpp'
    hpp_file = os.path.join(path, hpp_name)
    if overwrite or not os.path.exists(hpp_file):
        print('translating to {}'.format(hpp_file))
        stanc_path = os.path.join(CMDSTAN_PATH, 'bin', 'stanc')
        cmd = [stanc_path, '--o={}'.format(hpp_file), stan_file]
        print('stan to c++: make args {}'.format(cmd))
        do_command(cmd)
        if not os.path.exists(hpp_file):
            raise Exception('syntax error'.format(stan_file))

    exe_file = os.path.join(path, model_name)
    if not overwrite and os.path.exists(exe_file):
        # print('model is up to date') # notify user or not?
        return Model(stan_file, exe_file)
    cmd = ['make', 'O={}'.format(opt_lvl), exe_file]
    print('compiling c++: make args {}'.format(cmd))
    try:
        do_command(cmd, CMDSTAN_PATH)
    except Exception:
        return Model(stan_file)
    return Model(stan_file, exe_file)


def sample(stan_model:Model=None,
           chains:int=4,
           cores:int=1,
           seed:int=None,
           data_file:str=None,
           init_param_values:str=None,
           csv_output_file:str=None,
           console_output_file:str=None,
           refresh:int=None,
           post_warmup_draws_per_chain:int=None,
           warmup_draws_per_chain:int=None,
           save_warmup:bool=False,
           thin:int=None,
           do_adaptation:bool=True,
           adapt_gamma:float=None,
           adapt_delta:float=None,
           adapt_kappa:float=None,
           adapt_t0:float=None,
           nuts_max_depth:int=None,
           hmc_metric:str=None,
           hmc_metric_file:str=None,
           hmc_stepsize:float=1.0,
           hmc_stepsize_jitter:float=0) -> RunSet:
    """Run or more chains of the NUTS/HMC sampler."""
    args = SamplerArgs(model=stan_model,
                       seed=seed,
                       data_file=data_file,
                       init_param_values=init_param_values,
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
                       hmc_stepsize_jitter=hmc_stepsize_jitter)
    args.validate()
    if chains < 1:
        raise ValueError(
            'chains must be a positive integer value, found {}'.format(chains))
    if cores < 1:
        raise ValueError(
            'cores must be a positive integer value, found {}'.format(cores))
    if cores > cpu_count():
        print('requested {} cores, only {} available'.format(
            cores, cpu_count()))
        cores = cpu_count()
    runset = RunSet(args=args,
                    chains=chains,
                    console_file=console_output_file)
    tp = ThreadPool(cores)
    for i in range(chains):
        tp.apply_async(do_sample, (
            runset,
            i,
        ))
    tp.close()
    tp.join()
    if not runset.check_retcodes():
        msg = "Error during sampling"
        for i in range(chains):
            if runset.retcode(i) != 0:
                msg = '{}, chain {} returned error code {}'.format(
                    msg, i, runset.retcode(i))
        raise Exception(msg)
    column_names = runset.column_names # also checks runset is valid
    chains = runset.chains
    draws = runset.draws
    post_sample = PosteriorSample(chains, draws, column_names, runset.output_files)
    return post_sample

def summary(runset:RunSet=None) -> pd.DataFrame:
    """
    Run cmdstan/bin/stansummary over all sampler output csv files in runset.
    Echo stansummary stdout/stderr to console.
    Assemble csv tempfile contents into pandasDataFrame.
    """
    if runset is None:
        raise ValueError('summary command requires specified RunSet from sampler')
    names = runset.get_column_names()  # also checks runset is valid
    cmd_path = os.path.join(CMDSTAN_PATH, 'bin', 'stansummary')
    tmp_csv_file = 'stansummary-{}-{}-chains-'.format(
        runset.args.model.name, runset.chains)
    fd, tmp_csv_path = tempfile.mkstemp(suffix='.csv', prefix=tmp_csv_file, dir=TMPDIR, text=True)
    cmd = '{} --csv_file={} {}'.format(cmd_path, tmp_csv_path, ' '.join(runset.output_files))
    do_command(cmd.split())  # breaks on all whitespace
    summary_data = pd.read_csv(tmp_csv_path, delimiter=',', header=0, index_col=0, comment='#')
    return summary_data


def diagnose(runset:RunSet=None) -> None:
    """
    Run cmdstan/bin/diagnose on runset.
    Echo diagnose stdout/stderr to console.
    """
    if runset is None:
        raise ValueError('summary command requires specified RunSet from sampler')
    cmd_path = os.path.join(CMDSTAN_PATH, 'bin', 'diagnose')
    csv_files = ' '.join(runset.output_files)
    cmd = '{} {} '.format(cmd_path, csv_files)
    do_command(cmd=cmd.split())


def do_sample(runset:RunSet, idx:int) -> None:
    """
    Spawn process, capture console output to file, record returncode.
    """
    cmd = runset.cmds[idx]
    proc = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.wait()
    stdout, stderr = proc.communicate()
    transcript_file = runset.console_files[idx]
    with open(transcript_file, "w+") as transcript:
        if stdout:
            transcript.write(stdout.decode('ascii'))
        if stderr:
            transcript.write('ERROR')
            transcript.write(stderr.decode('ascii'))
    runset.set_retcode(idx, proc.returncode)


def do_command(cmd:str, cwd:str=None) -> None:
    """
    Spawn process, print stdout/stderr to console.
    Throws exception on non-zero returncode.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.wait()
    stdout, stderr = proc.communicate()
    if (proc.returncode):
        if stderr:
            print('ERROR\n {} '.format(stderr.decode('ascii').strip()))
        raise Exception('Command failed: {}'.format(cmd))
    if stdout:
        print(stdout.decode('ascii').strip())
