import os
import os.path
import re
import numpy as np
from .utils import is_int, rdump, scan_stan_csv


class Model(object):
    """Stan model."""

    def __init__(self, stan_file, name=None, exe_file=None):
        """Initialize object."""
        self.stan_file = stan_file
        """full path to Stan program src."""
        self.name = name
        """defaults to base name of Stan program file."""
        self.exe_file = exe_file
        """full path to compiled c++ executible."""
        if not os.path.exists(stan_file):
            raise ValueError('no such stan_file {}'.format(self.stan_file))

    def __repr__(self):
        return 'Model(name="{}", stan_file="{}", exe_file="{}")'.format(
            self.name, self.stan_file, self.exe_file)

    def code(self):
        """Return Stan program as a string."""
        code = None
        try:
            with open(self.stan_file, 'r') as fd:
                code = fd.read()
        except IOError:
            print('Cannot read file: {}'.format(self.stan_file))
        return code


class RunSet(object):
    """Record of running NUTS sampler on a model."""

    def __init__(self, args, chains=1, cores=1, console_file=None):
        """Initialize object."""
        self.chains = chains
        """number of chains."""
        self.cores = cores
        """max processes to run at once."""
        self.args = args
        """sampler args."""
        if console_file is None:
            self.console_file = self.args.output_file
        else:
            self.console_file = console_file
        """base filename for console output transcript files."""
        self.cmds = [args.compose_command(i + 1) for i in range(chains)]
        self.output_files = [
            '{}-{}.csv'.format(self.args.output_file, i + 1)
            for i in range(chains)
        ]
        """per-chain sample csv files."""
        self.console_files = [
            '{}-{}.txt'.format(self.console_file, i + 1)
            for i in range(chains)
        ]
        """per-chain console transcript files."""
        self.__retcodes = [-1 for _ in range(chains)]
        """per-chain return codes."""
        self.__sample_shape = None
        """sample shape: chains, draws, cols."""
        if chains < 1:
            raise ValueError(
                'chains must be positive integer value, found {i]}'.format(
                    self.chains))

    def __repr__(self):
        return 'RunSet(args={}, chains={}, cores={}, console={})'.format(
            self.args, self.chains, self.cores, self.console_file)

    def get_retcodes(self):
        return self.__retcodes

    def get_retcode(self, idx):
        return self.__retcodes[idx]

    def get_sample_shape(self):
        if self.__sample_shape is None and self.check_retcodes():
            self.validate_csv_files()
        return self.__sample_shape

    def set_retcode(self, idx, val):
        self.__retcodes[idx] = val

    def check_retcodes(self):
        """checks that all chains have retcode 0."""
        for i in range(self.chains):
            if self.__retcodes[i] != 0:
                return False
        return True

    def check_console_msgs(self):
        """checks console messages for each chain."""
        valid = True
        msg = ""
        for i in range(self.chains):
            with open(self.console_files[i], 'r') as fp:
                contents = fp.read()
                pat = re.compile(r'^Exception.*$', re.M)
                errors = re.findall(pat, contents)
                if (len(errors) > 0):
                    valid = False
                    msg = '{}chain {}: {}\n'.format(msg, i + 1, errors)
        if not valid:
            raise Exception(msg)

    def validate_csv_files(self):
        """checks drawset from each chain."""
        dzero = {}
        for i in range(self.chains):
            if i == 0:
                dzero = scan_stan_csv(self.output_files[i])
            else:
                d = scan_stan_csv(self.output_files[i])
                for key in dzero:
                    if key != 'id' and dzero[key] != d[key]:
                        raise ValueError(
                            'csv file header mismatch, '
                            'file {}, key {} is {}, expected {}'.format(
                                self.output_files[i], key, dzero[key], d[key]))
            self.__sample_shape = (self.chains, dzero['draws'],
                                   len(dzero['col_headers']))
        return dzero


class SamplerArgs(object):
    """Flattened arguments for NUTS/HMC sampler
    """

    def __init__(self,
                 model,
                 seed=None,
                 data_file=None,
                 init_param_values=None,
                 output_file=None,
                 refresh=None,
                 post_warmup_draws=None,
                 warmup_draws=None,
                 save_warmup=False,
                 thin=None,
                 do_adaptation=True,
                 adapt_gamma=None,
                 adapt_delta=None,
                 adapt_kappa=None,
                 adapt_t0=None,
                 nuts_max_depth=None,
                 hmc_metric_file=None,
                 hmc_stepsize=None):
        """Initialize object."""
        self.model = model
        """Model object"""
        self.seed = seed
        """seed for pseudo-random number generator."""
        self.data_file = data_file
        """full path to input data file name."""
        self.init_param_values = init_param_values
        """full path to initial parameter values file name."""
        self.output_file = output_file
        """full path to output file."""
        self.refresh = refresh
        """number of iterations between progress message updates."""
        self.post_warmup_draws = post_warmup_draws
        """number of post-warmup draws."""
        self.warmup_draws = warmup_draws
        """number of wramup draws."""
        self.save_warmup = save_warmup
        """boolean - include warmup iterations in output."""
        self.thin = thin
        """period between draws."""
        self.do_adaptation = do_adaptation
        """boolean - do adaptation during warmup."""
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
        self.hmc_metric_file = hmc_metric_file
        """initial value for HMC mass matrix."""
        self.hmc_stepsize = hmc_stepsize
        """initial value for HMC stepsize."""

    def validate(self):
        """Check arg consistency, correctness.
        """
        if self.model is None:
            raise ValueError('no stan model specified')
        if self.model.exe_file is None:
            raise ValueError('stan model must be compiled first,' +
                             ' run command compile_model("{}")'.format(
                                 self.model.stan_file))
        if not os.path.exists(self.model.exe_file):
            raise ValueError('cannot access model executible "{}"'.format(
                self.model.exe_file))
        if self.output_file is None:
            raise ValueError('no output file specified')
        if self.output_file.endswith('.csv'):
            self.output_file = self.output_file[:-4]
        try:
            open(self.output_file, 'w+')
            os.remove(self.output_file)  # cleanup after test
        except Exception:
            raise ValueError('invalid path for output csv files')
        if self.seed is None:
            rng = np.random.RandomState()
            self.seed = rng.randint(1, 99999 + 1)
        else:
            if not is_int(self.seed) or self.seed < 0 or self.seed > 2**32 - 1:
                raise ValueError(
                    'seed must be an integer value between 0 and 2**32-1, '
                    'found {}'.format(self.seed))
        if self.data_file is not None:
            if not os.path.exists(self.data_file):
                raise ValueError('no such file {}'.format(self.data_file))
        if self.init_param_values is not None:
            if not os.path.exists(self.init_param_values):
                raise ValueError('no such file {}'.format(
                    self.init_param_values))
        if (self.hmc_metric_file is not None):
            if not os.path.exists(self.hmc_metric_file):
                raise ValueError('no such file {}'.format(
                    self.hmc_metric_file))
        if self.post_warmup_draws is not None:
            if not is_int(
                    self.post_warmup_draws) or self.post_warmup_draws < 0:
                raise ValueError(
                    'post_warmup_draws must be a non-negative integer value'.
                    format(self.post_warmup_draws))
        if self.warmup_draws is not None:
            if not is_int(self.post_warmup_draws) or self.warmup_draws < 0:
                raise ValueError(
                    'warmup_draws must be a non-negative integer value'.format(
                        self.warmup_draws))
        # TODO: check type/bounds on all other controls

        pass

    def compose_command(self, chain_id):
        """compose command string for CmdStan for non-default arg values.
        """
        cmd = '{} id={}'.format(self.model.exe_file, chain_id)
        if (self.seed is not None):
            cmd = '{} random seed={}'.format(cmd, self.seed)
        if self.data_file is not None:
            cmd = '{} data file={}'.format(cmd, self.data_file)
        if self.init_param_values is not None:
            cmd = '{} init={}'.format(cmd, self.init_param_values)
        output_file = '{}-{}.csv'.format(self.output_file, chain_id)
        cmd = '{} output file={}'.format(cmd, output_file)
        if self.refresh is not None:
            cmd = '{} refresh={}'.format(cmd, self.refresh)
        cmd = cmd + ' method=sample'
        if (self.post_warmup_draws is not None):
            cmd = '{} num_samples={}'.format(cmd, self.post_warmup_draws)
        if (self.warmup_draws is not None):
            cmd = '{} num_warmup={}'.format(cmd, self.warmup_draws)
        if (self.save_warmup):
            cmd = cmd + ' save_warmup=1'
        if (self.thin is not None):
            cmd = '{} thin={}'.format(cmd, self.thin)
        cmd = cmd + ' algorithm=hmc'
        if (self.hmc_stepsize is not None):
            cmd = '{} stepsize={}'.format(cmd, self.hmc_stepsize)
        if (self.nuts_max_depth is not None):
            cmd = '{} engine=nuts max_depth={}'.format(cmd,
                                                       self.nuts_max_depth)
        if (self.do_adaptation and
                not (self.adapt_gamma is None and self.adapt_delta is None
                     and self.adapt_kappa is None and self.adapt_t0 is None)):
            cmd = cmd + " adapt"
        if (self.adapt_gamma is not None):
            cmd = '{} gamma={}'.format(cmd, self.adapt_gamma)
        if (self.adapt_delta is not None):
            cmd = '{} delta={}'.format(cmd, self.adapt_delta)
        if (self.adapt_kappa is not None):
            cmd = '{} kappa={}'.format(cmd, self.adapt_kappa)
        if (self.adapt_t0 is not None):
            cmd = '{} t0={}'.format(cmd, self.adapt_t0)
        if (self.hmc_metric_file is not None):
            cmd = '{} metric_file="{}"'.format(cmd, self.hmc_metric_file)
        return cmd


class StanData(object):
    """Stan model data or inits."""

    def __init__(self, rdump_file=None):
        """Initialize object."""
        self.rdump_file = rdump_file
        """path to rdump file."""
        if not os.path.exists(rdump_file):
            try:
                open(rdump_file, 'w')
            except OSError:
                raise Exception('invalid rdump_file name {}'.format(
                    self.rdump_file))

    def __repr__(self):
        return 'StanData(rdump_file="{}")'.format(self.rdump_file)

    def write_rdump(self, dict):
        rdump(self.rdump_file, dict)
