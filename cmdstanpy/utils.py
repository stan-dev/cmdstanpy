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
                 model = None,
                 seed = None,
                 data_file = None,
                 init_param_values = None,
                 output_file = None,
                 refresh = None,
                 fixed_param = False,
                 num_samples = None,
                 num_warmup = None,
                 save_warmup = False,
                 thin_samples = None,
                 adapt_engaged = True,
                 adapt_gamma = None,
                 adapt_delta = None,
                 adapt_kappa = None,
                 adapt_t0 = None,
                 nuts_max_depth = None,
                 hmc_metric = None,
                 hmc_metric_file = None,
                 hmc_stepsize = None,
                 hmc_stepsize_jitter = None):
        """Initialize object."""
        self.model = model
        """Model object"""
        self.seed = seed
        self.data_file = data_file
        """full path to input data file name."""
        self.init_param_values = init_param_values
        """full path to input data file name."""
        self.output_file = output_file
        """full path to output file including basename, excluding suffix."""
        self.refresh = refresh
        """number of iterations between progress message updates."""
        self.num_samples = num_samples
        """number of sampling iterations."""
        self.num_warmup = num_warmup
        """number of warmup iterations."""
        self.save_warmup = save_warmup
        """include warmup iterations in output?"""
        self.thin_samples = thin_samples
        """period between saved samples."""
        self.fixed_param = fixed_param
        """Stan model has no parameters."""
        self.adapt_engaged = adapt_engaged
        """bool - do adaptation."""
        self.adapt_gamma = adapt_gamma
        """adaptation regularization scale."""
        self.adapt_delta = adapt_delta
        """adaptation target acceptance statistic."""
        self.adapt_kappa = adapt_kappa
        """adaptation relaxation exponent."""
        self.adapt_t0 = adapt_t0
        """adaptation iteration offset."""
        self.nuts_max_depth = nuts_max_depth
        """NUTS maximum tree depth."""
        self.hmc_metric = hmc_metric
        """HMC metric (mass matrix), one of unit, diag_e, dense_e."""
        self.hmc_metric_file = hmc_metric_file
        """initial value for mass matrix."""
        self.hmc_stepsize = hmc_stepsize
        """initial value for HMC stepsize."""
        self.hmc_stepsize_jitter = hmc_stepsize_jitter
        """initial value for HMC stepsize jitter."""

    def validate(self):
        """Check arg consistancy, correctness.
        """
        if self.model is None:
            raise ValueError('no stan model specified')
        if self.model.exe_file is None:
            raise ValueError('stan model must be compiled first,' +
                                 ' run command compile_model("{}")'.
                                 format(self.model.stan_file))
        if not os.path.exists(self.model.exe_file):
            raise ValueError('cannot access model executible "{}"'.format(
                self.model.exe_file))
        if self.seed is None:
            rng = np.random.RandomState()
            self.seed = rng.randint(1, 99999 + 1)
        if (self.fixed_param and
            not (self.nuts_max_depth is None and
                 self.adapt_gamma is None and
                 self.adapt_delta is None and
                 self.adapt_kappa is None and
                 self.adapt_t0 is None and
                 self.hmc_metric is None and
                 self.hmc_metric_file is None and
                 self.hmc_stepsize is None and
                     self.hmc_stepsize_jitter is None)):
            raise ValueError('conflicting specifications to sampler;'
                                 + ' cannot specify "fixed_param" with HMC/NUTS controls'
                                 + ' for adaptation, treedepth, metric, or stepsize.')
        if self.data_file is not None:
            if not os.path.exists(self.data_file):
                raise ValueError('no such file {}'.format(self.data_file))
        if self.init_param_values is not None:
            if not os.path.exists(self.init_param_values):
                raise ValueError('no such file {}'.format(self.init_param_values))
        if (self.hmc_metric_file is not None):
            if not os.path.exists(self.hmc_metric_file):
                raise ValueError('no such file {}'.format(self.hmc_metric_file))
        if self.output_file is None:
            raise ValueError('please specify path to sampler output.csv file.')
        try:
            open(self.output_file,'w')
        except OSError:
            raise Exception('invalide output file path')
        if self.output_file is not None:
            if not os.path.exists(self.output_file):
                raise ValueError('no such file {}'.format(self.output_file))
        # todo, check type/bounds on all other controls
        pass

    def compose_command(self, chain_id):
        """compose command string for CmdStan for non-default arg values.
        """
        cmd = '{} id={}'.format(self.model.exe_file, chain_id)
        if (self.seed is not None):
            cmd = '{} random seed={}'.format(cmd, self.seed)
        if self.data_file is not None:
            cmd = '{} data file="{}"'.format(cmd, self.data_file)
        if self.init_param_values is not None:
            cmd = '{} init="{}"'.format(cmd, self.init_param_values)
        cmd = cmd + ' output'
        output_file = '{}-{}.csv'.format(self.output_file, chain_id)
        cmd = '{} file="{}"'.format(cmd, output_file)
        if self.refresh is not None:
            cmd = '{} refresh={}'.format(cmd, self.refresh)
        cmd = cmd + ' method=sample'
        if (self.num_samples is not None):
            cmd = '{} num_samples={}'.format(cmd, self.num_samples)
        if (self.num_warmup is not None):
            cmd = '{} num_warmup={}'.format(cmd, self.num_warmup)
        if (self.save_warmup):
            cmd = cmd + ' save_warmup=1'
        if (self.thin_samples is not None):
            cmd = '{} thin={}'.format(cmd, self.thin_samples)
        if (self.fixed_param is True):
            cmd = cmd + ' algorithm=fixed_param'
        else:
            cmd = cmd + ' algorithm=hmc'
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
            if (self.nuts_max_depth is not None):
                cmd = '{} max_depth={}'.format(cmd, self.nuts_max_depth)
            if (self.hmc_metric is not None):
                cmd = '{} metric={}'.format(cmd, self.hmc_metric)
            if (self.hmc_metric_file is not None):
                cmd = '{} metric_file="{}"'.format(cmd, self.hmc_metric_file)
            if (self.hmc_stepsize is not None):
                cmd = '{} stepsize={}'.format(cmd, self.hmc_stepsize)
            if (self.hmc_stepsize_jitter is not None):
                cmd = '{} stepsize_jitter={}'.format(cmd, self.hmc_stepsize_jitter)
        return cmd;
