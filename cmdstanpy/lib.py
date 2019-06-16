import os
import re
import tempfile
from typing import Dict, List, Union, Tuple

import numpy as np

from cmdstanpy import TMPDIR
from cmdstanpy.utils import jsondump, rdump
from cmdstanpy.utils import check_csv


class Model(object):
    """Stan model."""

    def __init__(self, stan_file: str = None, exe_file: str = None) -> None:
        """Initialize object."""
        self.stan_file = stan_file
        """full path to Stan program src."""
        self.exe_file = exe_file
        """full path to compiled c++ executable."""
        if stan_file is None:
            raise ValueError('must specify Stan program file')
        if not os.path.exists(stan_file):
            raise ValueError('no such file {}'.format(self.stan_file))
        if exe_file is not None:
            if not os.path.exists(exe_file):
                raise ValueError('no such file {}'.format(self.exe_file))
        filename = os.path.split(stan_file)[1]
        if len(filename) < 6 or not filename.endswith('.stan'):
            raise ValueError('invalid stan filename {}'.format(self.stan_file))
        self._name = os.path.splitext(filename)[0]

    def __repr__(self) -> str:
        return 'Model(name={},  stan_file="{}", exe_file="{}")'.format(
            self._name, self.stan_file, self.exe_file
        )

    def code(self) -> str:
        """Return Stan program as a string."""
        code = None
        try:
            with open(self.stan_file, 'r') as fd:
                code = fd.read()
        except IOError:
            print('Cannot read file: {}'.format(self.stan_file))
        return code

    @property
    def name(self) -> str:
        return self._name


class StanData(object):
    """Stan model data or sampler inits."""

    def __init__(self, data_file: str = None) -> None:
        """Initialize object."""
        self._data_file = data_file
        """path to on-disk cmdstan input datafile."""
        if data_file is not None and not os.path.exists(data_file):
            try:
                with open(data_file, 'w') as fd:
                    pass
                os.remove(data_file)  # cleanup
            except OSError:
                raise Exception('invalid data_file name {}'.format(data_file))

    def __repr__(self) -> str:
        return 'StanData(data_file="{}")'.format(self._data_file)

    def write_rdump(self, dict: Dict) -> None:
        rdump(self._data_file, dict)

    def write_json(self, dict: Dict) -> None:
        jsondump(self._data_file, dict)

    @property
    def data_file(self) -> str:
        return self._data_file


class SamplerArgs(object):
    """Arguments for NUTS/HMC sampler."""

    def __init__(
        self,
        model: Model,
        data: str = None,
        seed: Union[int, List[int]] = None,
        inits: Union[float, str, List[str]] = None,
        warmup_iters: int = None,
        sampling_iters: int = None,
        warmup_schedule: Tuple[float, float, float] = None,
        save_warmup: bool = False,
        thin: int = None,
        max_treedepth: float = None,
        metric: Union[str, List[str]] = None,
        step_size: Union[float, List[float]] = None,
        adapt_engaged: bool = True,
        target_accept_rate: float = None,
        output_file: str = None,
    ) -> None:
        """Initialize object."""
        self.model = model
        self.seed = seed
        self.data = data
        self.inits = inits
        self.refresh = refresh
        self.sampling_iters = sampling_iters
        self.warmup_iters = warmup_iters
        self.save_warmup = save_warmup
        self.thin = thin
        self.adapt_engaged = adapt_engaged
        self.adapt_gamma = adapt_gamma
        self.nuts_max_depth = nuts_max_depth
        self.hmc_metric_file = hmc_metric_file
        self.step_size = step_size
        self.output_file = output_file


    def validate(self) -> None:
        """Check arg consistency, correctness.
        """
        if self.model is None:
            raise ValueError('no stan model specified')
        if self.model.exe_file is None:
            raise ValueError(
                'stan model must be compiled first,'
                + ' run command compile_model("{}")'.format(
                    self.model.stan_file)
            )
        if not os.path.exists(self.model.exe_file):
            raise ValueError(
                'cannot access model executable "{}"'.format(
                    self.model.exe_file)
            )
        if self.output_file is not None:
            try:
                with open(self.output_file, 'w+') as fd:
                    pass
                os.remove(self.output_file)  # cleanup
            except Exception:
                raise ValueError(
                    'invalid path for output files: {}'.format(
                        self.output_file)
                )
            if self.output_file.endswith('.csv'):
                self.output_file = self.output_file[:-4]
        if self.seed is None:
            rng = np.random.RandomState()
            self.seed = rng.randint(1, 99999 + 1)
        else:
            if (
                not isinstance(self.seed, int)
                or self.seed < 0
                or self.seed > 2 ** 32 - 1
            ):
                raise ValueError(
                    'seed must be an integer value between 0 and 2**32-1, '
                    'found {}'.format(self.seed)
                )
        if self.data is not None:
            if not os.path.exists(self.data):
                raise ValueError('no such file {}'.format(self.data))
# ************ stopped here **********
# validate inits
        if self.inits is not None:
            if not os.path.exists(self.inits):
                raise ValueError('no such file {}'.format(
                    self.inits))
# validate metric:  metric, path, or list of paths
        if self.hmc_metric_file is not None:
            if not os.path.exists(self.hmc_metric_file):
                raise ValueError('no such file {}'.format(
                    self.hmc_metric_file))
        if self.sampling_iters is not None:
            if self.sampling_iters < 0:
                raise ValueError(
                    'sampling_iters must be '
                    'a non-negative integer value'.format(
                        self.sampling_iters
                    )
                )
# validate warmup, warmup schedule
        if self.warmup_iters is not None:
            if self.warmup_iters < 0:
                raise ValueError(
                    'warmup_iters must be a '
                    'non-negative integer value'.format(
                        self.warmup_iters
                    )
                )
        # TODO: check type/bounds on all other controls
        # positive int values
        pass

    def compose_command(self, chain_id: int, csv_file: str) -> str:
        """compose command string for CmdStan for non-default arg values.
        """
        cmd = '{} id={}'.format(self.model.exe_file, chain_id)
        if self.seed is not None:
            cmd = '{} random seed={}'.format(cmd, self.seed)
        if self.data is not None:
            cmd = '{} data file={}'.format(cmd, self.data)
        if self.inits is not None:
            cmd = '{} init={}'.format(cmd, self.inits)
        cmd = '{} output file={}'.format(cmd, csv_file)
        if self.refresh is not None:
            cmd = '{} refresh={}'.format(cmd, self.refresh)
        cmd = cmd + ' method=sample'
        if self.sampling_iters is not None:
            cmd = '{} num_samples={}'.format(cmd, self.sampling_iters)
        if self.warmup_iters is not None:
            cmd = '{} num_warmup={}'.format(cmd, self.warmup_iters)
        if self.save_warmup:
            cmd = cmd + ' save_warmup=1'
        if self.thin is not None:
            cmd = '{} thin={}'.format(cmd, self.thin)
        cmd = cmd + ' algorithm=hmc'
        if self.step_size is not None:
            cmd = '{} stepsize={}'.format(cmd, self.step_size)
        if self.nuts_max_depth is not None:
            cmd = '{} engine=nuts max_depth={}'.format(
                cmd, self.nuts_max_depth)
        if self.adapt_engaged or not self.target_accept_rate is not None:
            cmd = cmd + " adapt engaged "
        if self.adapt_gamma is not None:
            cmd = '{} gamma={}'.format(cmd, self.adapt_gamma)
        if self.adapt_delta is not None:
            cmd = '{} delta={}'.format(cmd, self.adapt_delta)
        if self.adapt_kappa is not None:
            cmd = '{} kappa={}'.format(cmd, self.adapt_kappa)
        if self.adapt_t0 is not None:
            cmd = '{} t0={}'.format(cmd, self.adapt_t0)
        if self.hmc_metric_file is not None:
            cmd = '{} metric_file="{}"'.format(cmd, self.hmc_metric_file)
        return cmd


class RunSet(object):
    """Record of running NUTS sampler on a model."""

    def __init__(self, args: SamplerArgs, chains: int = 4, ) -> None:
        """Initialize object."""
        self._args = args
        self._chains = chains
        if chains < 1:
            raise ValueError(
                'chains must be positive integer value, '
                'found {i]}'.format(chains)
            )
        self.csv_files = []
        """per-chain sample csv files."""
        if args.output_file is None:
            csv_basename = 'stan-{}-draws'.format(args.model.name)
            for i in range(chains):
                fd = tempfile.NamedTemporaryFile(
                    mode='w+',
                    prefix='{}-{}-'.format(csv_basename, i + 1),
                    suffix='.csv',
                    dir=TMPDIR,
                    delete=False,
                )
                self.csv_files.append(fd.name)
        else:
            for i in range(chains):
                self.csv_files.append('{}-{}.csv'.format(
                    args.output_file, i + 1))
        self.console_files = []
        """per-chain sample console output files."""
        for i in range(chains):
            txt_file = ''.join(
                [os.path.splitext(self.csv_files[i])[0], '.txt']
                )
            self.console_files.append(txt_file)
        self.cmds = [
            args.compose_command(i + 1, self.csv_files[i])
            for i in range(chains)
        ]
        """per-chain sampler command."""
        self._retcodes = [-1 for _ in range(chains)]
        self._draws = None
        self._column_names = None
        self._num_params = None  # metric dim(s)
        self._metric_type = None
        self._metric = None
        self._stepsize = None
        self._sample = None

    def __repr__(self) -> str:
        repr = 'RunSet(args={}, chains={}'.format(self.args, self._chains)
        repr = '{}\n csv_files={}\nconsole_files={})'.format(
            repr, '\n\t'.join(self.csv_files), '\n\t'.join(self.console_files)
        )
        return repr

    @property
    def retcodes(self) -> List[int]:
        """per-chain return codes."""
        return self._retcodes

    def check_retcodes(self) -> bool:
        """True when all chains have retcode 0."""
        for i in range(self._chains):
            if self._retcodes[i] != 0:
                return False
        return True

    def retcode(self, idx: int) -> int:
        """get retcode for chain[idx]."""
        return self._retcodes[idx]

    def set_retcode(self, idx: int, val: int) -> None:
        """Set retcode for chain[idx] to val."""
        self._retcodes[idx] = val

    @property
    def model(self) -> str:
        """Stan model name"""
        return self._args.model.name

    @property
    def chains(self) -> int:
        """Number of sampler chains."""
        return self._chains

    @property
    def draws(self) -> int:
        """Number of draws per chain."""
        return self._draws

    @property
    def columns(self) -> int:
        """
        Total number of information items returned by sampler for each draw.
        Consists of sampler state, model parameters and computed quantities.
        """
        return len(self._column_names)

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of information items returned by sampler for each draw.
        Includes for sampler state labels and
        names of model parameters and computed quantities.
        """
        return self._column_names

    @property
    def metric_type(self) -> str:
        """Metric type, either 'diag_e' or 'dense_e'"""
        return self._metric_type

    @property
    def metric(self) -> np.ndarray:
        """Array of per-chain metric used by sampler."""
        if self._metric is None:
            self.assemble_sample()
        return self._metric

    @property
    def stepsize(self) -> np.ndarray:
        """Array or per-chain stepsize."""
        if self._stepsize is None:
            self.assemble_sample()
        return self._stepsize

    @property
    def sample(self) -> np.ndarray:
        """
        A 3-D numpy ndarray which contains all draws across all chain arranged
        as (draws, chains, columns) stored column major so that the values
        for each parameter are stored contiguously in memory, likewise
        all draws from a chain are contiguous.
        """
        if self._sample is None:
            self.assemble_sample()
        return self._sample

    def check_console_msgs(self) -> bool:
        """Checks console messages for each chain."""
        valid = True
        msg = ""
        for i in range(self._chains):
            with open(self.console_files[i], 'r') as fp:
                contents = fp.read()
                pat = re.compile(r'^Exception.*$', re.M)
                errors = re.findall(pat, contents)
                if len(errors) > 0:
                    valid = False
                    msg = '{}chain {}: {}\n'.format(msg, i + 1, errors)
        if not valid:
            raise Exception(msg)

    def validate_csv_files(self) -> None:
        """
        Checks that csv output files for all chains are consistent.
        Populates attributes for drawset, metric.
        Raises exception when inconsistencies detected.
        """
        dzero = {}
        for i in range(self._chains):
            if i == 0:
                dzero = check_csv(self.csv_files[i])
            else:
                d = check_csv(self.csv_files[i])
                for key in dzero:
                    if key != 'id' and dzero[key] != d[key]:
                        raise ValueError(
                            'csv file header mismatch, '
                            'file {}, key {} is {}, expected {}'.format(
                                self.csv_files[i], key, dzero[key], d[key]
                            )
                        )
        self._draws = dzero['draws']
        self._column_names = dzero['column_names']
        self._num_params = dzero['num_params']
        self._metric_type = dzero['metric']

    def assemble_sample(self) -> None:
        """
        Allocates and populates the stepsize, metric, and drawset arrays
        by parsing the validated stan_csv files.
        """
        if not (self._stepsize is None and self._metric is None and
                    self._sample is None):
            return
        self._stepsize = np.empty(self._chains, dtype=float)
        if self._metric_type == 'diag_e':
            self._metric = np.empty(
                (self._chains, self._num_params), dtype=float)
        else:
            self._metric = np.empty(
                (self._chains, self._num_params, self._num_params),
                dtype=float)
        self._sample = np.empty(
            (self._draws, self._chains, len(self._column_names)), dtype=float,
            order='F'
            )
        for chain in range(self._chains):
            with open(self.csv_files[chain], 'r') as fp:
                # read past initial comments, column header
                line = fp.readline().strip()
                while len(line) > 0 and line.startswith('#'):
                    line = fp.readline().strip()
                line = fp.readline().strip()  # adaptation header
                # stepsize
                line = fp.readline().strip()
                label, stepsize = line.split('=')
                self._stepsize[chain] = float(stepsize.strip())
                line = fp.readline().strip()  # metric header
                # metric
                if self._metric_type == 'diag_e':
                    line = fp.readline().lstrip(' #\t')
                    xs = line.split(',')
                    self._metric[chain, :] = [float(x) for x in xs]
                else:
                    for i in range(self._num_params):
                        line = fp.readline().lstrip(' #\t')
                        xs = line.split(',')
                        self._metric[chain, i, :] = [float(x) for x in xs]
                # draws
                for i in range(self._draws):
                    line = fp.readline().lstrip(' #\t')
                    xs = line.split(',')
                    self._sample[i, chain, :] = [float(x) for x in xs]
