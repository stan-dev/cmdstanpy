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

from lib import Conf, Model, RunSet, SamplerArgs
from utils import do_command, do_sample, is_int

myconf = Conf()
cmdstan_path = myconf['cmdstan']

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
        stanc_path = os.path.join(cmdstan_path, 'bin', 'stanc')
        cmd = [stanc_path, '--o={}'.format(hpp_file), stan_file]
        do_command(cmd)
        if not os.path.exists(hpp_file):
            raise Exception('syntax error'.format(stan_file))

    exe_file = os.path.join(path, model_name)
    if not overwrite and os.path.exists(exe_file):
        print('model is up to date')
        return Model(stan_file, model_name, exe_file)

    cmd = ['make', 'O={}'.format(opt_lvl), exe_file]
    print(cmd)  # compiling is slow - need a spinner
    try:
        do_command(cmd, cmdstan_path)
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

    runset = RunSet(chains, cores, args, console_output_file)
    tp = ThreadPool(cores)
    for i in range(chains):
        tp.apply_async(do_sample, (runset, i,))
    tp.close()
    tp.join()
    return runset
 
