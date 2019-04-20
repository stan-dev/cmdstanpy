"""
User-facing commands
"""

import json
import logging
import os
import os.path
import subprocess
import sys
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from .config import *
from .lib import Model, RunSet, SamplerArgs
from .utils import is_int

def compile_model(stan_file, opt_lvl=3, overwrite=False):
    """Compile the given Stan model file to an executable.
    """
    if not os.path.exists(stan_file):
        raise Exception('no such stan_file {}'.format(stan_file))
        
    path = os.path.abspath(os.path.dirname(stan_file))
    program_name = os.path.basename(stan_file)
    model_name = program_name[:-5]

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
        print('model is up to date')
        return Model(stan_file, model_name, exe_file)

    cmd = ['make', 'O={}'.format(opt_lvl), exe_file]
    print('compiling c++: make args {}'.format(cmd))
    try:
        do_command(cmd, CMDSTAN_PATH)
    except Exception:
        return Model(stan_file, model_name)
    return Model(stan_file, model_name, exe_file)


def sample(stan_model = None,
               chains = 4,
               cores = 1,
               seed = None,
               data_file = None,
               init_param_values = None,
               csv_output_file = None,
               console_output_file = None,
               refresh = None,
               post_warmup_draws_per_chain = None,
               warmup_draws_per_chain = None,
               save_warmup = False,
               thin = None,
               do_adaptation = True,
               adapt_gamma = None,
               adapt_delta = None,
               adapt_kappa = None,
               adapt_t0 = None,
               nuts_max_depth = None,
               hmc_metric = None,
               hmc_metric_file = None,
               hmc_stepsize = 1):
    """Runs on or more chains of the NUTS/HMC sampler, writing set of draws from each chain to a file in stan-csv format 
    """
    # not enough to return runset - process chains
    # posterior_sample = process_runset(runset)
    # return posterior_sample
    args = SamplerArgs(model = stan_model,
                        seed = seed,
                        data_file = data_file,
                        init_param_values = init_param_values,
                        output_file = csv_output_file,
                        refresh = refresh,
                        post_warmup_draws =  post_warmup_draws_per_chain,
                        warmup_draws = warmup_draws_per_chain,
                        save_warmup = save_warmup,
                        thin = thin,
                        do_adaptation = do_adaptation,
                        adapt_gamma = adapt_gamma,
                        adapt_delta = adapt_delta,
                        adapt_kappa = adapt_kappa,
                        adapt_t0 = adapt_t0,
                        nuts_max_depth = nuts_max_depth,
                        hmc_metric_file = hmc_metric_file,
                        hmc_stepsize = hmc_stepsize)
    args.validate()
    if not is_int(chains) and chains > 0:
        raise ValueError('chains must be a positive integer value, found {}'.format(chains))
    if not is_int(cores) and cores > 0:
        raise ValueError('cores must be a positive integer value, found {}'.format(cores))

    if cores > cpu_count():
        logger.warning('requested {} cores but only {} cores available'.format(codes, cpu_count()))
        cores = cpu_count()
        
    if console_output_file is None:
        console_output_file = csv_output_file

    runset = RunSet(args = args,
                    chains = chains,
                    cores = cores,
                    transcript_file = console_output_file)
    tp = ThreadPool(cores)
    for i in range(chains):
        tp.apply_async(do_sample, (runset, i,))
    tp.close()
    tp.join()
    return runset

 
def summary(runset = None, outfile = None):
    if not runset.is_success():
        raise ValueError('invalid runset {}'.format(runset))
    if summary_outfile_file is None:
        summary_output_file = '{}-summary.csv'.format(self.output_file)
    if not os.path.exists(summary_output):
        try:
            open(summary_output_file,'w')
        except OSError:
            raise Exception('invalid summary output file name {}'.format(summary_output_file))

    cmd_path = os.path.join(CMDSTAN_PATH, 'bin', 'summary')
    csv_files = ' '.join(self.output_files)
    cmd = '{} --csv_file={} {}'.format(cmd_path, summary_output_file, csv_files)
    print(cmd)
    proc = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )
    proc.wait()
    stdout, stderr = proc.communicate()
    if stdout:
            print(stdout.decode('ascii'))
    if stderr:
            print('ERROR')
            print(stderr.decode('ascii'))
    if proc.returncode != 0:
            raise Exception('summary cmd failed, return code: {}'.format(proc.returncode))
    

def diagnose(runset, diagnose_output_file):
    cmd_path = os.path.join(CMDSTAN_PATH, 'bin', 'diagnose')
    pass


def do_sample(runset, idx):
    """Spawn process, capture stdout and std err to transcript file, return returncode.
    """
    cmd = runset.cmds[idx]
    proc = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )
    proc.wait()
    stdout, stderr = proc.communicate()

    transcript_file = runset.transcript_files[idx]
    with open(transcript_file, "w+") as transcript:
        if stdout:
            transcript.write(stdout.decode('ascii'))
        if stderr:
            transcript.write('ERROR')
            transcript.write(stderr.decode('ascii'))
    runset.set_retcode(idx, proc.returncode)

    

def do_command(cmd, cwd=None):
    """Spawn process, get output/err/returncode."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )
    proc.wait()
    stdout, stderr = proc.communicate()
    if stdout:
        print(stdout.decode('ascii').strip())
    if stderr:
        print('ERROR\n {} '.format(stderr.decode('ascii').strip()))
    if (proc.returncode):
        raise Exception('Command failed: {}'.format(cmd))

def check_command(chain_id, args, cmd):
    pass

def process_runset(runset):
    """Extract sample from runset."""

def read_stan_csv(filename):
    pass

