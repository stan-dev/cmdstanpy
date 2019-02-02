"""
User-facing commands
"""

import json
import logging
import os
import os.path
import subprocess
import sys

from lib import Conf, Model
from utils import SamplerArgs, _do_command

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
        _do_command(cmd)
        if not os.path.exists(hpp_file):
            raise Exception('syntax error'.format(stan_file))

    exe_file = os.path.join(path, model_name)
    if not overwrite and os.path.exists(exe_file):
        print('model is up to date')
        return Model(model_name, stan_file, exe_file)

    cmd = ['make', 'O={}'.format(opt_lvl), exe_file]
    print(cmd)  # compiling is slow - need a spinner
    try:
        _do_command(cmd, cmdstan_path)
    except Exception:
        return Model(model_name, stan_file)
    return Model(model_name, stan_file, exe_file)

def sample(stan_model = None,
               num_chains = None,
               num_cores = None,  #pass param  python subprocess 
               seed = None,
               data_file = None,
               init_param_values = None,
               output_file = None,
               diagnostic_file = None,
               refresh = None,
               num_samples = None,
               num_warmup = None,
               save_warmup = False,
               thin_samples = None,
               adapt_engaged = True,
               adapt_gamma = None,
               adapt_delta = None,
               adapt_kappa = None,
               adapt_t0 = None,
               fixed_param = False,
               NUTS_max_depth = None,
               HMC_diag_metric = None,
               HMC_metric_file = None,
               HMC_stepsize = None,
               HMC_stepsize_jitter = None):
    """Invoke HMC/NUTS sampler via compiled model.
    """
    args = SamplerArgs(stan_model, seed,
                data_file, init_param_values, output_file, diagnostic_file,
                refresh, num_samples, num_warmup, save_warmup, thin_samples,
                adapt_engaged,  adapt_gamma, adapt_delta, adapt_kappa, adapt_t0,
                fixed_param, NUTS_max_depth, HMC_diag_metric, HMC_metric_file,
                HMC_stepsize, HMC_stepsize_jitter)
    args.validate()
    if num_chains is None:
        num_chains = 1
    try:
        chains = int(num_chains)
    except Execption:
        raise ValueError('num_chains must be a positivie integer value, found {}'.format(num_chains))
    if (num_chains < 1):
        raise ValueError('num_chains must be a positivie integer value, found {}'.format(num_chains))
        
    for i in range(1,num_chains+1):
        cmd = args.compose_command(i)
        print(cmd)

        # for chains:
            # create RunSet
            # add "id=<chain_id>"
            # run cmd
            # wait
            # return RunSet
