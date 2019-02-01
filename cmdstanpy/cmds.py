import json
import logging
import os
import os.path
import subprocess
import sys

from lib import Conf, Model
myconf = Conf()
cmdstan_path = myconf['cmdstan']

def _do_command(cmd, cwd=None):
    """Spawn process, get output/err/returncode.
    """
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

def translate_model(stan_file, hpp_file):
    """Invoke stanc on the given Stan model file.
    """
    stanc_path = os.path.join(cmdstan_path, 'bin', 'stanc')
    cmd = [stanc_path, '--o={}'.format(hpp_file), stan_file]
    _do_command(cmd)

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
        translate_model(stan_file, hpp_file)
        if not os.path.exists(hpp_file):
            raise Exception('syntax error'.format(stan_file))

    target = os.path.join(path, model_name)
    if not overwrite and os.path.exists(target):
        print('model is up to date')
        return Model(model_name, stan_file, target)

    cmd = ['make', 'O={}'.format(opt_lvl), target]
    print(cmd)  # compiling is slow - need a spinner
    try:
        _do_command(cmd, cmdstan_path)
    except Exception:
        return Model(model_name, stan_file)
    return Model(model_name, stan_file, target)

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
    # check model has exe
    if stan_model is None:
        raise ValueError('no stan model specified')
    if stan_model.exe_file is None:
        raise ValueError('stan model must be compiled first, run command compile_model("{}")'.
                             format(stan_model.stan_file))
    if not os.path.exists(stan_model.exe_file):
        raise ValueError('cannot access model executible "{}"'.format(stan_model.exe_file))


    # method = sample
    cmd = stan_model.exe_file;
    if (num_samples is not None):
        cmd = '{} num_samples={}'.format(cmd, num_samples)
    if (num_warmup is not None):
        cmd = '{} num_warmup={}'.format(cmd, num_warmup)
    if (save_warmup):
        cmd = cmd + ' save_warmup=1'
    if (num_warmup is not None):
        cmd = '{} thin={}'.format(cmd, thin_samples)
    if (adapt_engaged and
        not (adapt_gamma is None and
             adapt_delta is None and
             adapt_kappa is None and
             adapt_t0 is None)):
        cmd = cmd + " adapt"
        if (adapt_gamma is not None):
            cmd = '{} gamma={}'.format(cmd, adapt_gamma)
        if (adapt_delta is not None):
            cmd = '{} delta={}'.format(cmd, adapt_delta)
        if (adapt_kappa is not None):
            cmd = '{} kappa={}'.format(cmd, adapt_kappa)
        if (adapt_t0 is not None):
            cmd = '{} t0={}'.format(cmd, adapt_t0)
    if (fixed_param and
        not (NUTS_max_depth is None and
             HMC_diag_metric is None and
             HMC_metric_file is None and
             HMC_stepsize is None and
             HMC_stepsize_jitter is None)):
        raise ValueError('conflicting specifications to sampler;'
                    + ' cannot specify "fixed_param" with HMC/NUTS controls'
                    + ' for treedepth, metric, and/or stepsize.')
    if (fixed_param):
        cmd = cmd + ' algorithm=fixed_param'
    else:
        if (NUTS_max_depth is not None):
            cmd = '{} max_depth={}'.format(cmd, max_depth)
        if (HMC_diag_metric is not None):
            cmd = '{} metric={}'.format(cmd, HMC_diag_metric)
        if (HMC_metric_file is not None):
            if not os.path.exists(HMC_metric_file):
                raise ValueError('no such file {}'.format(HMC_metric_file))
            cmd = '{} metric_file="{}"'.format(cmd, HMC_metric_file)
        if (HMC_stepsize is not None):
            cmd = '{} stepsize={}'.format(cmd, HMC_stepsize)
        if (HMC_stepsize_jitter is not None):
            cmd = '{} stepsize_jitter={}'.format(cmd, HMC_stepsize_jitter)
    # data
    if data_file is not None:
        if not os.path.exists(data_file):
            raise ValueError('no such file {}'.format(data_file))
        cmd = '{} data file="{}"'.format(cmd, data_file)
    # init
    if init_param_values is not None:
        if not os.path.exists(init_param_values):
            raise ValueError('no such file {}'.format(init_param_values))
        cmd = '{} init="{}"'.format(cmd, init_param_values)
    # output
    if not (output_file is None and diagnostic_file is None and refresh is None):
        cmd = cmd + ' output'
        if output_file is not None:
            if not os.path.exists(output_file):
                raise ValueError('no such file {}'.format(output_file))
            cmd = '{} file="{}"'.format(cmd, init_param_values)
        if diagnostic_file is not None:
            if not os.path.exists(diagnostic_file):
                raise ValueError('no such file {}'.format(diagnostic_file))
            cmd = '{} diagnostic_file="{}"'.format(cmd, diagnostic_file)
        if refresh is not None:
            cmd = '{} refresh={}'.format(cmd, diagnostic_file)
    # random seed
    if (seed is not None):
        cmd = '{} random seed={}'.format(cmd, seed)

    print(cmd)
    # for chains:
            # create RunSet
            # add "id=<chain_id>"
            # run cmd
            # wait
            # return RunSet
