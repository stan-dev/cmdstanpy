"""
Utility functions and classes
"""

import os
import os.path
import numpy as np

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

def _rdump_array(key, val):
    c = 'c(' + ', '.join(map(str, val.T.flat)) + ')'
    if (val.size, ) == val.shape:
        return '{key} <- {c}'.format(key=key, c=c)
    else:
        dim = '.Dim = c{0}'.format(val.shape)
        struct = '{key} <- structure({c}, {dim})'.format(key=key, c=c, dim=dim)
        return struct


def rdump(path, data):
    """Dump a dict of data to a R dump format file.
    """
    with open(path, 'w') as fd:
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.size > 1:
                line = _rdump_array(key, val)
            elif isinstance(val, list) and len(val) > 1:
                line = _rdump_array(key, np.asarray(val))
            else:

                try:
                    val = val.flat[0]
                except AttributeError:
                    pass
                line = '%s <- %s' % (key, val)
            fd.write(line)
            fd.write('\n')

class SamplerArgs(object):
    """Flattened arguments for NUTS/HMC sampler
    """
    def __init__(self,
                 stan_model = None,
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
        """Initialize object."""
        self.stan_model = stan_model
        self.seed = seed
        self.data_file = data_file
        self.init_param_values = init_param_values
        self.output_file = output_file
        self.diagnostic_file = diagnostic_file
        self.refresh = refresh
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.save_warmup = save_warmup
        self.thin_samples = thin_samples
        self.adapt_engaged = adapt_engaged
        self.adapt_gamma = adapt_gamma
        self.adapt_delta = adapt_delta
        self.adapt_kappa = adapt_kappa
        self.adapt_t0 = adapt_t0
        self.fixed_param = fixed_param
        self.NUTS_max_depth = NUTS_max_depth
        self.HMC_diag_metric = HMC_diag_metric
        self.HMC_metric_file = HMC_metric_file
        self.HMC_stepsize = HMC_stepsize
        self.HMC_stepsize_jitter = HMC_stepsize_jitter

    def validate(self):
        """Check arg consistancy, correctness.
        """
        if self.stan_model is None:
            raise ValueError('no stan model specified')
        if self.stan_model.exe_file is None:
            raise ValueError('stan model must be compiled first,' +
                                 ' run command compile_model("{}")'.
                                 format(self.stan_model.stan_file))
        if not os.path.exists(self.stan_model.exe_file):
            raise ValueError('cannot access model executible "{}"'.format(
                self.stan_model.exe_file))
        if (self.fixed_param and
            not (self.NUTS_max_depth is None and
                self.HMC_diag_metric is None and
                self.HMC_metric_file is None and
                self.HMC_stepsize is None and
                self.HMC_stepsize_jitter is None)):
            raise ValueError('conflicting specifications to sampler;'
                                 + ' cannot specify "fixed_param" with HMC/NUTS controls'
                                 + ' for treedepth, metric, and/or stepsize.')
        if self.data_file is not None:
            if not os.path.exists(self.data_file):
                raise ValueError('no such file {}'.format(self.data_file))
        if self.init_param_values is not None:
            if not os.path.exists(self.init_param_values):
                raise ValueError('no such file {}'.format(self.init_param_values))
        if (self.HMC_metric_file is not None):
            if not os.path.exists(self.HMC_metric_file):
                raise ValueError('no such file {}'.format(self.HMC_metric_file))
        if self.output_file is None:
            raise ValueError('please specify path to sampler output.csv file.')
        try:
            open(self.output_file,'w')
        except OSError:
            raise Exception('invalide output file path')
        if self.output_file is not None:
            if not os.path.exists(self.output_file):
                raise ValueError('no such file {}'.format(self.output_file))
        if self.diagnostic_file is not None:
            if not os.path.exists(self.diagnostic_file):
                raise ValueError('no such file {}'.format(self.diagnostic_file))
        pass

    def compose_command(self, chain_id):
        """compose command string for CmdStan for non-default arg values.
        """
        cmd = '{} id={}'.format(self.stan_model.exe_file, chain_id)
        if (self.num_samples is not None):
            cmd = '{} num_samples={}'.format(cmd, self.num_samples)
        if (self.num_warmup is not None):
            cmd = '{} num_warmup={}'.format(cmd, self.num_warmup)
        if (self.save_warmup):
            cmd = cmd + ' save_warmup=1'
        if (self.thin_samples is not None):
            cmd = '{} thin={}'.format(cmd, self.thin_samples)
        if (self.adapt_engaged and
                not (self.adapt_gamma is None and
                         self.adapt_delta is None and
                         self.adapt_kappa is None and
                         self.adapt_t0 is None)):
            cmd = cmd + " adapt"
        if (self.adapt_gamma is not None):
            cmd = '{} gamma={}'.format(cmd, self.adapt_gamma)
        if (self.adapt_delta is not None):
            cmd = '{} delta={}'.format(cmd, self.adapt_delta)
        if (self.adapt_kappa is not None):
            cmd = '{} kappa={}'.format(cmd, self.adapt_kappa)
        if (self.adapt_t0 is not None):
            cmd = '{} t0={}'.format(cmd, self.adapt_t0)
        if (self.fixed_param):
            cmd = cmd + ' algorithm=fixed_param'
        else:
            if (self.NUTS_max_depth is not None):
                cmd = '{} max_depth={}'.format(cmd, self.NUTS_max_depth)
            if (self.HMC_diag_metric is not None):
                cmd = '{} metric={}'.format(cmd, self.HMC_diag_metric)
            if (self.HMC_metric_file is not None):
                cmd = '{} metric_file="{}"'.format(cmd, self.HMC_metric_file)
            if (self.HMC_stepsize is not None):
                cmd = '{} stepsize={}'.format(cmd, self.HMC_stepsize)
            if (self.HMC_stepsize_jitter is not None):
                cmd = '{} stepsize_jitter={}'.format(cmd, self.HMC_stepsize_jitter)
        if self.data_file is not None:
            cmd = '{} data file="{}"'.format(cmd, self.data_file)
        if self.init_param_values is not None:
            cmd = '{} init="{}"'.format(cmd, self.init_param_values)
        cmd = cmd + ' output'
        output_file = '{}-{}.csv'.format(self.output_file, chain_id)
        cmd = '{} file="{}"'.format(cmd, output_file)
        if self.diagnostic_file is not None:
            diagnostic_file = '{}-{}.csv'.format(self.diagnostic_file, chain_id)
            cmd = '{} diagnostic_file="{}"'.format(cmd, diagnostic_file)
        if self.refresh is not None:
            cmd = '{} refresh={}'.format(cmd, self.refresh)
        if (self.seed is not None):
            cmd = '{} random seed={}'.format(cmd, self.seed)
        return cmd;
