import os
import os.path
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import rdump, scan_stan_csv


class Model(object):
    """Stan model."""

    def __init__(self, stan_file:str, exe_file:str=None) -> None:
        """Initialize object."""
        self.stan_file = stan_file
        """full path to Stan program src."""
        self.exe_file = exe_file
        """full path to compiled c++ executible."""
        if not os.path.exists(stan_file):
            raise ValueError('no such file {}'.format(self.stan_file))
        if not exe_file is None:
            if not os.path.exists(exe_file):
                raise ValueError('no such file {}'.format(self.exe_file))
        filename = os.path.split(stan_file)[1]
        if len(filename) < 6 or not filename.endswith('.stan'):
            raise ValueError('invalid stan filename {}'.format(self.stan_file))
        self.name = os.path.splitext(filename)[0]

    def __repr__(self) -> str:
        return 'Model(stan_file="{}", exe_file="{}")'.format(
            self.stan_file, self.exe_file)

    def code(self) -> str:
        """Return Stan program as a string."""
        code = None
        try:
            with open(self.stan_file, 'r') as fd:
                code = fd.read()
        except IOError:
            print('Cannot read file: {}'.format(self.stan_file))
        return code

# rewrite - constructor takes Dict, optional filename
# see https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
# @clsmethod rdump, json (default)
class StanData(object):
    """Stan model data or inits."""

    def __init__(self, rdump_file:str=None) -> None:
        """Initialize object."""
        self.rdump_file = rdump_file
        """path to rdump file."""
        if not os.path.exists(rdump_file):
            try:
                with open(rdump_file, 'w') as fd:
                    pass
                os.remove(rdump_file)  # cleanup
            except OSError:
                raise Exception('invalid rdump_file name {}'.format(
                    self.rdump_file))

    def __repr__(self) -> str:
        return 'StanData(rdump_file="{}")'.format(self.rdump_file)

    def write_rdump(self, dict:Dict) -> None:
        rdump(self.rdump_file, dict)


class SamplerArgs(object):
    """Full set of arguments for NUTS/HMC sampler."""

    def __init__(self,
                 model:Model,
                 seed:int=None,
                 data_file:str=None,
                 init_param_values:str=None,
                 output_file:str=None,
                 refresh:int=None,
                 post_warmup_draws:int=None,
                 warmup_draws:int=None,
                 save_warmup:bool=False,
                 thin:int=None,
                 do_adaptation:bool=True,
                 adapt_gamma:float=None,
                 adapt_delta:float=None,
                 adapt_kappa:float=None,
                 adapt_t0:float=None,
                 nuts_max_depth:int=None,
                 hmc_metric_file:str=None,
                 hmc_stepsize:float=None,
                 hmc_stepsize_jitter:float=None) -> None:
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
        self.hmc_stepsize_jitter = hmc_stepsize_jitter
        """initial value for uniform random jitter of HMC stepsize."""

    def validate(self) -> None:
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
            with open(self.output_file, 'w+') as fd:
                pass
            os.remove(self.output_file)  # cleanup after test
        except Exception:
            raise ValueError('invalid path for output csv files')
        if self.seed is None:
            rng = np.random.RandomState()
            self.seed = rng.randint(1, 99999 + 1)
        else:
            if not isinstance(self.seed, int) or self.seed < 0 or self.seed > 2**32 - 1:
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
            if self.post_warmup_draws < 0:
                raise ValueError(
                    'post_warmup_draws must be a non-negative integer value'.
                    format(self.post_warmup_draws))
        if self.warmup_draws is not None:
            if self.warmup_draws < 0:
                raise ValueError(
                    'warmup_draws must be a non-negative integer value'.format(
                        self.warmup_draws))
        # TODO: check type/bounds on all other controls
        # positive int values
        pass

    def compose_command(self, chain_id:int) -> str:
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
        if (self.hmc_stepsize_jitter is not None):
            cmd = '{} stepsize_jitter={}'.format(cmd, self.hmc_stepsize_jitter)
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



# TODO: RunSet uses secure temp files - registers names of files, once created, not deleted
# add "save" operation - moves tempfiles to specified permanent dir
class RunSet(object):
    """Record of running NUTS sampler on a model."""

    def __init__(self, args:SamplerArgs, chains:int=1, console_file:str=None) -> None:
        """Initialize object."""
        self.__chains = chains
        """number of chains."""
        self.args = args
        """sampler args."""
        if console_file is None:
            self.__console_file = self.args.output_file
        else:
            self.__console_file = console_file
        """base filename for console output transcript files."""
        self.cmds = [args.compose_command(i + 1) for i in range(chains)]
        self.output_files = [
            '{}-{}.csv'.format(self.args.output_file, i + 1)
            for i in range(chains)
        ]
        """per-chain sample csv files."""
        self.console_files = [
            '{}-{}.txt'.format(self.__console_file, i + 1)
            for i in range(chains)
        ]
        """per-chain console transcript files."""
        self.__retcodes = [-1 for _ in range(chains)]
        """per-chain return codes."""
        self.__column_names = None
        if chains < 1:
            raise ValueError(
                'chains must be positive integer value, found {i]}'.format(chains))

    def __repr__(self) -> str:
        return 'RunSet(args={}, chains={}, console={})'.format(
            self.args, self.__chains, self.__console_file)

    @property
    def retcodes(self) -> List[int]:
        """Get list of retcodes for all chains."""
        return self.__retcodes

    def check_retcodes(self) -> bool:
        """Checks that all chains have retcode 0."""
        for i in range(self.chains):
            if self.__retcodes[i] != 0:
                return False
        return True

    def retcode(self, idx:int) -> int:
        """get retcode for chain[idx]."""
        return self.__retcodes[idx]

    def set_retcode(self, idx:int, val:int) -> None:
        """Set retcode for chain[idx] to val."""
        self.__retcodes[idx] = val

    @property
    def chains(self) -> int:
        return self.__chains

    @property
    def draws(self) -> int:
        """Get draws per chain."""
        if self.__draws is None and self.check_retcodes():
            sample_dict = self.validate_csv_files()
            self.__draws = sample_dict['draws']
            self.__column_names = sample_dict['column_names']  # call validate once
        return self.__draws

    @property
    def column_names(self) -> (str, ...):
        """Get csv file column names."""
        if self.__column_names is None and self.check_retcodes():
            sample_dict = self.validate_csv_files()
            self.__column_names = sample_dict['column_names']
            self.__draws = sample_dict['draws']  # call validate once
        return self.__column_names

    def check_console_msgs(self) -> bool:
        """Checks console messages for each chain."""
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

    def validate_csv_files(self) -> Dict:
        """
        Checks csv output files for all chains.
        Returns Dict with entries for sampler config and drawset .
        """
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
        return dzero



class PosteriorSample(object):
    """Assembled draws from all chains in a RunSet."""

    def __init__(self, chains:int=None, draws:int=None, column_names:(str, ...)=None,
                     csv_files:(str, ...)=None) -> None:
        """Initialize object."""
        self.__chains = chains
        """number of chains"""
        self.__draws = draws
        """number of chains"""
        self.__column_names = column_names
        """csv output header."""
        self.__csv_files = csv_files
        """sampler output csv files."""
        self.__sample = None
        """assembled draws across all chains, stored column major."""
        if chains is None:
            raise ValueError('must specify chains')
        if draws is None:
            raise ValueError('must specify draws')
        if column_names is None:
            raise ValueError('must specify columns')
        if len(column_names) == 0:
            raise ValueError('no column names specified')
        if csv_files is None:
            raise ValueError('must specify sampler output csv files')
        if len(csv_files) != chains:
            raise ValueError('expecting {} sampler output files, found {}'.format(
                chains, len(csv_files)))
        for i in range(chains):
            if not os.path.exists(csv_files[i]):
                raise ValueError('no such file {}'.format(csv_files[i]))

    def get_sample(self) -> np.ndarray:
        sample = np.empty((self.__draws, self.__chains, len(self.__column_names)),
                              dtype=float, order='F')
        for chain in range(self.__chains):
            print("chain: {}".format(chain))
            draw = 0
            with open(self.__csv_files[chain], 'r') as fd:
                for line in fd:
                    if line.startswith('#') or line.startswith('lp__,'):
                        continue
                    print(line.strip())
                    vs = [float(x) for x in line.split(',')]
                    sample[draw, chain, :] = vs
                    draw += 1
        return sample
    
    def extract(self) -> pd.DataFrame:
        if self.__sample is None:
            self.__sample = self.get_sample()
        data = self.__sample.reshape((self.__draws*self.__chains),len(self.__column_names),
                                         order='A')
        return pd.DataFrame(data=data, columns=self.__column_names)

#    def extract_sampler_state(self) -> pd.DataFrame:
#    def extract_sampler_params(self) -> pd.DataFrame:

    @property
    def draws(self) -> int:
        return self.__draws

    @property
    def chains(self) -> int:
        return self.__chains

    @property
    def columns(self) -> int:
        return len(self.__column_names)

    @property
    def column_names(self) -> (str, ...):
        return self.__column_names

    @property
    def sample(self) -> np.ndarray:
        if self.__sample is not None:
            return self.__sample
        self.__sample = self.get_sample()
        return self.__sample


