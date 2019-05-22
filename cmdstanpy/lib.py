import platform
import re
import tempfile

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from cmdstanpy import TMPDIR
from cmdstanpy.utils import do_command, jsondump, rdump
from cmdstanpy.utils import check_csv, cmdstan_path


class Model(object):
    """Stan model."""

    def __init__(self, stan_file: str = None, exe_file: str = None) -> None:
        """Initialize object."""
        if stan_file is None:
            raise ValueError('must specify Stan program file')
        # Initialize object
        self.stan_file = Path(stan_file)
        if not stan_file.exists():
            raise ValueError('no such file {}'.format(self.stan_file))

        # full path to Stan program src
        if exe_file is not None:
            self.exe_file = Path(exe_file)
            if not exe_file.exists():
                raise ValueError('no such file {}'.format(self.exe_file))
        else:
            self.exe_file = exe_file
        filename = stan_file.name
        if len(filename) < 6 or not filename.endswith('.stan'):
            raise ValueError('invalid stan filename {}'.format(self.stan_file))
        self._name = stan_file.stem

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
        if data_file is not None:
            # path to on-disk cmdstan input datafile
            self._data_file = Path(data_file)
            if self._data_file.exists():
                try:
                    with open(self._data_file, 'w') as fd:
                        pass
                    try:
                        self._data_file.unlink()  # cleanup
                    except PermissionError:
                        # skip Windows error
                        pass
                except OSError:
                    raise Exception('invalid data_file name {}'.format(self._data_file))

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
    """Full set of arguments for NUTS/HMC sampler."""

    def __init__(
        self,
        model: Model,
        seed: int = None,
        data_file: str = None,
        init_param_values: str = None,
        output_file: str = None,
        refresh: int = None,
        post_warmup_draws: int = None,
        warmup_draws: int = None,
        save_warmup: bool = False,
        thin: int = None,
        do_adaptation: bool = True,
        adapt_gamma: float = None,
        adapt_delta: float = None,
        adapt_kappa: float = None,
        adapt_t0: float = None,
        nuts_max_depth: int = None,
        hmc_metric_file: str = None,
        hmc_stepsize: float = None,
        hmc_stepsize_jitter: float = None,
    ) -> None:
        """Initialize object."""
        self.model = model
        # Model object
        self.seed = seed
        # seed for pseudo-random number generator.
        self.data_file = None if data_file is None else Path(data_file)
        # full path to input data file name.
        self.init_param_values = None if init_param_values is None else Path(init_param_values)
        # full path to initial parameter values file name.
        self.output_file = None if output_file is None else Path(output_file)
        # full path to output file.
        self.refresh = refresh
        # number of iterations between progress message updates.
        self.post_warmup_draws = post_warmup_draws
        # number of post-warmup draws.
        self.warmup_draws = warmup_draws
        # number of wramup draws.
        self.save_warmup = save_warmup
        # boolean - include warmup iterations in output.
        self.thin = thin
        # period between draws.
        self.do_adaptation = do_adaptation
        # boolean - do adaptation during warmup.
        self.adapt_gamma = adapt_gamma
        # adaptation regularization scale.
        self.adapt_delta = adapt_delta
        # adaptation target acceptance statistic.
        self.adapt_kappa = adapt_kappa
        # adaptation relaxation exponent.
        self.adapt_t0 = adapt_t0
        # adaptation iteration offset.
        self.nuts_max_depth = nuts_max_depth
        # NUTS maximum tree depth.
        self.hmc_metric_file = None if hmc_metric_file is None else Path(hmc_metric_file)
        # initial value for HMC mass matrix.
        self.hmc_stepsize = hmc_stepsize
        # initial value for HMC stepsize.
        self.hmc_stepsize_jitter = hmc_stepsize_jitter
        # initial value for uniform random jitter of HMC stepsize.

    def validate(self) -> None:
        """Check arg consistency, correctness."""
        if self.model is None:
            raise ValueError('no stan model specified')
        if self.model.exe_file is None:
            raise ValueError(
                'stan model must be compiled first,'
                + ' run command compile_model("{}")'.format(
                    self.model.stan_file)
            )
        if not self.model.exe_file.exists():
            raise ValueError(
                'cannot access model executible "{}"'.format(
                    self.model.exe_file)
            )
        if self.output_file is not None:
            try:
                with open(self.output_file, 'w+') as fd:
                    pass
                self.output_file.unlink()  # cleanup
            except Exception:
                raise ValueError(
                    'invalid path for output files: {}'.format(
                        self.output_file)
                )
            if self.output_file.suffix.lower() == '.csv':
                self.output_file = self.output_file.stem
        if self.seed is None:
            rng = np.random.RandomState()
            self.seed = rng.randint(1, 99_999 + 1)
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
        if self.data_file is not None:
            if not self.data_file.exists():
                raise ValueError('no such file {}'.format(self.data_file))
        if self.init_param_values is not None:
            if not self.init_param_values.exists():
                raise ValueError('no such file {}'.format(
                    self.init_param_values))
        if self.hmc_metric_file is not None:
            if not self.hmc_metric_file.exists():
                raise ValueError('no such file {}'.format(
                    self.hmc_metric_file))
        if self.post_warmup_draws is not None:
            if self.post_warmup_draws < 0:
                raise ValueError(
                    'post_warmup_draws must be '
                    'a non-negative integer value'.format(
                        self.post_warmup_draws
                    )
                )
        if self.warmup_draws is not None:
            if self.warmup_draws < 0:
                raise ValueError(
                    'warmup_draws must be a '
                    'non-negative integer value'.format(
                        self.warmup_draws
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
        if self.data_file is not None:
            cmd = '{} data file={}'.format(cmd, self.data_file.as_posix())
        if self.init_param_values is not None:
            cmd = '{} init={}'.format(cmd, self.init_param_values.as_posix())
        cmd = '{} output file={}'.format(cmd, csv_file.as_posix())
        if self.refresh is not None:
            cmd = '{} refresh={}'.format(cmd, self.refresh)
        cmd = cmd + ' method=sample'
        if self.post_warmup_draws is not None:
            cmd = '{} num_samples={}'.format(cmd, self.post_warmup_draws)
        if self.warmup_draws is not None:
            cmd = '{} num_warmup={}'.format(cmd, self.warmup_draws)
        if self.save_warmup:
            cmd = cmd + ' save_warmup=1'
        if self.thin is not None:
            cmd = '{} thin={}'.format(cmd, self.thin)
        cmd = cmd + ' algorithm=hmc'
        if self.hmc_stepsize is not None:
            cmd = '{} stepsize={}'.format(cmd, self.hmc_stepsize)
        if self.hmc_stepsize_jitter is not None:
            cmd = '{} stepsize_jitter={}'.format(cmd, self.hmc_stepsize_jitter)
        if self.nuts_max_depth is not None:
            cmd = '{} engine=nuts max_depth={}'.format(
                cmd, self.nuts_max_depth)
        if self.do_adaptation and not (
            self.adapt_gamma is None
            and self.adapt_delta is None
            and self.adapt_kappa is None
            and self.adapt_t0 is None
        ):
            cmd = cmd + " adapt"
        if self.adapt_gamma is not None:
            cmd = '{} gamma={}'.format(cmd, self.adapt_gamma)
        if self.adapt_delta is not None:
            cmd = '{} delta={}'.format(cmd, self.adapt_delta)
        if self.adapt_kappa is not None:
            cmd = '{} kappa={}'.format(cmd, self.adapt_kappa)
        if self.adapt_t0 is not None:
            cmd = '{} t0={}'.format(cmd, self.adapt_t0)
        if self.hmc_metric_file is not None:
            cmd = '{} metric_file="{}"'.format(cmd, self.hmc_metric_file.as_posix())
        return cmd


# RunSet uses temp files - registers names of files, once created, not deleted
# TODO: add "save" operation - moves tempfiles to specified permanent dir
class RunSet(object):
    """Record of running NUTS sampler on a model."""

    def __init__(self, args: SamplerArgs, chains: int = 2) -> None:
        """Initialize object."""
        self._chains = chains
        # number of chains
        if chains < 1:
            raise ValueError(
                'chains must be positive integer value, '
                'found {i]}'.format(chains)
            )
        self.args = args
        # sampler args
        self.csv_files = []
        fmt = '{}{}{}'.format('{', ':>0{}d'.format(int(np.log10(chains))+1), '}') if chains >= 9 else '{}'
        prefix = '{}-' + fmt + '-'
        # per-chain sample csv files
        if args.output_file is None:
            csv_basename = 'stan-{}-draws'.format(args.model.name)
            for i in range(chains):
                fd = tempfile.NamedTemporaryFile(
                    mode='w+',
                    prefix=prefix.format(csv_basename, i + 1),
                    suffix='.csv',
                    dir=TMPDIR,
                    delete=False,
                )
                self.csv_files.append(Path(fd.name))
        else:
            prefix = prefix + '.csv'
            for i in range(chains):
                self.csv_files.append(Path(prefix.format(
                    args.output_file, i + 1)))
        self.console_files = []
        # per-chain sample console output files
        for csv_file in self.csv_files:
            txt_file = csv_file.stem + '.txt'
            self.console_files.append(txt_file)
        self.cmds = [
            args.compose_command(i + 1, csv_file)
            for csv_file in self.csv_files
        ]
        # per-chain sampler command
        self._retcodes = [-1 for _ in range(chains)]
        # per-chain return codes

    def __repr__(self) -> str:
        repr = 'RunSet(args={}, chains={}'.format(self.args, self._chains)
        repr = '{}\n csv_files={}\nconsole_files={})'.format(
            repr, '\n    '.join(self.csv_files), '\n    '.join(self.console_files)
        )
        return repr

    @property
    def retcodes(self) -> List[int]:
        """Get list of retcodes for all chains."""
        return self._retcodes

    def check_retcodes(self) -> bool:
        """Checks that all chains have retcode 0."""
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
    def chains(self) -> int:
        return self._chains

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

    def validate_csv_files(self) -> Dict:
        """
        Checks csv output files for all chains.
        Returns Dict with entries for sampler config and drawset .
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
        dzero['chains'] = self._chains
        return dzero


# TODO:  save sample - mv csv tempfiles to permanent location
class PosteriorSample(object):
    """Assembled draws from all chains in a RunSet."""

    def __init__(self, run: Dict = None, csv_files: Tuple[str] = None) -> None:
        """Initialize object."""
        self._run = run
        # sampler run info
        self._sample = None
        # assembled draws across all chains, stored column major
        if run is None:
            raise ValueError('missing sampler run info')
        if csv_files is None:
            raise ValueError('must specify sampler output csv files')
        self._csv_files = tuple(Path(csv_file) for csv_file in csv_files)
        # sampler output csv files
        for csv_file in csv_files:
            if not csv_file.exists():
                raise ValueError('no such file {}'.format(csv_file))
        self._chains = run['chains']
        self._draws = run['draws']
        self._column_names = run['column_names']

    def __repr__(self) -> str:
        return 'PosteriorSample(chains={},  draws={},  columns={})'.format(
            self._chains, self._draws, len(self._column_names)
        )

    def summary(self) -> pd.DataFrame:
        """
        Run cmdstan/bin/stansummary over all output csv files.
        Echo stansummary stdout/stderr to console.
        Assemble csv tempfile contents into pandasDataFrame.
        """
        names = self.column_names
        cmd_path = cmdstan_path() / 'bin' / 'stansummary'
        tmp_csv_file = 'stansummary-{}-{}-chains-'.format(
            self.model, self.chains)
        fd, tmp_csv_path = tempfile.mkstemp(
            suffix='.csv', prefix=tmp_csv_file, dir=TMPDIR, text=True
        )
        tmp_csv_path = Path(tmp_csv_path)
        csv_files = (csv_file.as_posix() for csv_file in self._csv_files)
        cmd = '{} --csv_file={} {}'.format(
            cmd_path, tmp_csv_path.as_posix(), ' '.join(csv_files)
        )
        do_command(cmd.split())  # breaks on all whitespace
        summary_data = pd.read_csv(
            tmp_csv_path, delimiter=',', header=0, index_col=0, comment='#'
        )
        mask = [
            x == 'lp__' or not x.endswith('__') for x in summary_data.index
            ]
        return summary_data[mask]

    def diagnose(self) -> None:
        """
        Run cmdstan/bin/diagnose over all output csv files.
        Echo diagnose stdout/stderr to console.
        """
        cmd_path = cmdstan_path() / 'bin' / 'diagnose'
        csv_files = ' '.join((csv_file.as_posix() for csv_file in self.csv_files))
        cmd = '{} {} '.format(cmd_path, csv_files)
        result = do_command(cmd=cmd.split())
        if result is None:
            print('No problems detected.')
        else:
            print(result)

    def extract(self, params: List[str] = None) -> pd.DataFrame:
        """
        Returns the assembled sample as a pandas DataFrame consisting of
        one column per parameter and one row per draw.
        """
        pnames_base = [name.split('.')[0] for name in self._column_names]
        if params is not None:
            for p in params:
                if not (p in self._column_names or p in pnames_base):
                    raise ValueError('unknown parameter: {}'.format(p))
        if self._sample is None:
            self._sample = self.get_sample()
        data = self._sample.reshape(
            (self._draws * self._chains), len(self._column_names), order='A'
        )
        df = pd.DataFrame(data=data, columns=self._column_names)
        if params is None:
            return df
        mask = []
        for p in params:
            for name in self._column_names:
                if p == name or p == name.split('.')[0]:
                    mask.append(name)
        return df[mask]

    @property
    def model(self) -> str:
        """Stan model name"""
        return self._run['model']

    @property
    def draws(self) -> int:
        """Number of draws per chain"""
        return self._draws

    @property
    def chains(self) -> int:
        """Number of chains"""
        return self._chains

    @property
    def columns(self) -> int:
        """
        Total number of information items returned by sampler for each draw.
        Consists of sampler state, model parameters and computed quantities.
        """
        return len(self._column_names)

    @property
    def column_names(self) -> (str, ...):
        """
        Names of information items returned by sampler for each draw.
        Includes for sampler state labels and
        names of model parameters and computed quantities.
        """
        return self._column_names

    @property
    def csv_files(self) -> (str, ...):
        """
        Full path name to stan_csv files returned by sampler.
        """
        return tuple(str(csv_file) for csv_file in self._csv_files)

    @property
    def sample(self) -> np.ndarray:
        """
        A 3-D numpy ndarray which contains all draws across all chain arranged
        as (draws, chains, columns) stored column major so that the values
        for each parameter are stored contiguously in memory, likewise
        all draws from a chain are contiguous.
        """
        if self._sample is None:
            self._sample = self.get_sample()
        return self._sample

    def get_sample(self) -> np.ndarray:
        """
        Returns posterior sample.
        The first time this function is called it assembles the sample
        from the stan_csv files; subsequent calls to this function
        return the assembled sample.
        """
        sample = np.empty(
            (self._draws, self._chains, len(self._column_names)), dtype=float,
            order='F'
        )
        for chain in range(self._chains):
            draw = 0
            with open(self._csv_files[chain], 'r') as fd:
                for line in fd:
                    if line.startswith('#') or line.startswith('lp__,'):
                        continue
                    vs = [float(x) for x in line.split(',')]
                    sample[draw, chain, :] = vs
                    draw += 1
        return sample
