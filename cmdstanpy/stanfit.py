import os
import re
import shutil
import tempfile
from typing import List, Tuple
from collections import OrderedDict
import numpy as np
import pandas as pd
import logging

from cmdstanpy import TMPDIR
from cmdstanpy.utils import (
    check_csv,
    EXTENSION,
    cmdstan_path,
    do_command, get_logger
)
from cmdstanpy.cmdstan_args import CmdStanArgs, OptimizeArgs


class StanFit(object):
    """Record of running NUTS sampler on a model."""

    def __init__(
            self,
            args: CmdStanArgs,
            chains: int = 4,
            logger: logging.Logger = None
    ) -> None:
        """Initialize object."""
        self._args = args
        self._is_optimizing = isinstance(self._args.method_args, OptimizeArgs)
        self._chains = chains
        self._logger = logger or get_logger()
        if chains < 1:
            raise ValueError(
                'chains must be positive integer value, '
                'found {i]}'.format(chains)
            )
        self.csv_files = []
        """per-chain sample csv files."""
        if args.output_basename is None:
            csv_basename = 'stan-{}-draws'.format(args.model_name)
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
                self.csv_files.append(
                    '{}-{}.csv'.format(args.output_basename, i + 1)
                )
        self.console_files = []
        """per-chain sample console output files."""
        for i in range(chains):
            txt_file = ''.join([os.path.splitext(self.csv_files[i])[0], '.txt'])
            self.console_files.append(txt_file)
        self.cmds = [
            args.compose_command(i, self.csv_files[i]) for i in range(chains)
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
        self._first_draw = None

    def __repr__(self) -> str:
        repr = 'StanFit(args={}, chains={}'.format(self._args, self._chains)
        repr = '{}\n csv_files={}\nconsole_files={})'.format(
            repr, '\n\t'.join(self.csv_files), '\n\t'.join(self.console_files)
        )
        return repr

    @property
    def model(self) -> str:
        """Stan model name"""
        return self._args.model_name

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
        """Metric used by sampler for each chain."""
        if self._metric is None:
            self._assemble_sample()
        return self._metric

    @property
    def stepsize(self) -> np.ndarray:
        """Stepsize used by sampler for each chain."""
        if self._stepsize is None:
            self._assemble_sample()
        return self._stepsize

    @property
    def is_optimizing(self) -> bool:
        """Returns true if we are optimizing rather than sampling"""
        return self._is_optimizing

    @property
    def optimized_params_np(self) -> np.array:
        """returns optimized params as numpy array"""
        return self._first_draw

    @property
    def optimized_params_pd(self) -> pd.DataFrame:
        """returns optimized params as pandas DataFrame"""
        return pd.DataFrame([self._first_draw], columns=self.column_names)

    @property
    def optimized_params_dict(self) -> OrderedDict:
        """returns optimized params as Dict"""
        return OrderedDict(zip(self.column_names, self._first_draw))

    def _sampling_only(self):
        if self.is_optimizing:
            raise RuntimeError("Method available only when sampling!")

    @property
    def sample(self) -> np.ndarray:
        """
        A 3-D numpy ndarray which contains all draws across all chain arranged
        as (draws, chains, columns) stored column major so that the values
        for each parameter are stored contiguously in memory, likewise
        all draws from a chain are contiguous.
        """
        self._sampling_only()

        if self._sample is None:
            self._assemble_sample()
        return self._sample

    def _check_retcodes(self) -> bool:
        """True when all chains have retcode 0."""
        for i in range(self._chains):
            if self._retcodes[i] != 0:
                return False
        return True

    def _retcode(self, idx: int) -> int:
        """get retcode for chain[idx]."""
        return self._retcodes[idx]

    def _set_retcode(self, idx: int, val: int) -> None:
        """Set retcode for chain[idx] to val."""
        self._retcodes[idx] = val

    def _check_console_msgs(self) -> bool:
        """Checks console messages for each chain."""
        valid = True
        msg = ''
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

    def _validate_csv_files(self) -> None:
        """
        Checks that csv output files for all chains are consistent.
        Populates attributes for draws, column_names, num_params, metric_type.
        Raises exception when inconsistencies detected.
        """
        dzero = {}
        for i in range(self._chains):
            if i == 0:
                dzero = check_csv(
                    self.csv_files[i],
                    is_optimizing=self.is_optimizing
                )
            else:
                d = check_csv(
                    self.csv_files[i],
                    is_optimizing=self.is_optimizing
                )
                for key in dzero:
                    if key not in ('id', 'first_draw') and dzero[key] != d[key]:
                        raise ValueError(
                            'csv file header mismatch, '
                            'file {}, key {} is {}, expected {}'.format(
                                self.csv_files[i], key, dzero[key], d[key]
                            )
                        )
        self._draws = dzero['draws']
        self._column_names = dzero['column_names']
        self._num_params = dzero['num_params']
        self._first_draw = dzero.get('first_draw')
        self._metric_type = dzero.get('metric')

    def _assemble_sample(self) -> None:
        """
        Allocates and populates the stepsize, metric, and sample arrays
        by parsing the validated stan_csv files.
        """
        if not (
            self._stepsize is None
            and self._metric is None
            and self._sample is None
        ):
            return
        self._stepsize = np.empty(self._chains, dtype=float)
        if self._metric_type == 'diag_e':
            self._metric = np.empty(
                (self._chains, self._num_params), dtype=float
            )
        else:
            self._metric = np.empty(
                (self._chains, self._num_params, self._num_params), dtype=float
            )
        self._sample = np.empty(
            (self._draws, self._chains, len(self._column_names)),
            dtype=float,
            order='F',
        )
        for chain in range(self._chains):
            with open(self.csv_files[chain], 'r') as fp:
                # skip initial comments, reads thru column header
                line = fp.readline().strip()
                while len(line) > 0 and line.startswith('#'):
                    line = fp.readline().strip()
                # skip warmup draws, if any, read to adaptation msg
                line = fp.readline().strip()
                if line != '# Adaptation terminated':
                    while line != '# Adaptation terminated':
                        line = fp.readline().strip()
                line = fp.readline().strip()  # stepsize
                label, stepsize = line.split('=')
                self._stepsize[chain] = float(stepsize.strip())
                line = fp.readline().strip()  # metric header
                # process metric
                if self._metric_type == 'diag_e':
                    line = fp.readline().lstrip(' #\t')
                    xs = line.split(',')
                    self._metric[chain, :] = [float(x) for x in xs]
                else:
                    for i in range(self._num_params):
                        line = fp.readline().lstrip(' #\t')
                        xs = line.split(',')
                        self._metric[chain, i, :] = [float(x) for x in xs]
                # process draws
                for i in range(self._draws):
                    line = fp.readline().lstrip(' #\t')
                    xs = line.split(',')
                    self._sample[i, chain, :] = [float(x) for x in xs]

    def summary(self) -> pd.DataFrame:
        """
        Run cmdstan/bin/stansummary over all output csv files.
        Echo stansummary stdout/stderr to console.
        Assemble csv tempfile contents into pandasDataFrame.
        """
        self._sampling_only()

        names = self.column_names
        cmd_path = os.path.join(
            cmdstan_path(), 'bin', 'stansummary' + EXTENSION
        )
        tmp_csv_file = 'stansummary-{}-{}-chains-'.format(
            self._args.model_name, self.chains
        )
        fd, tmp_csv_path = tempfile.mkstemp(
            suffix='.csv', prefix=tmp_csv_file, dir=TMPDIR, text=True
        )
        cmd = '{} --csv_file={} {}'.format(
            cmd_path, tmp_csv_path, ' '.join(self.csv_files)
        )
        # breaks on all whitespace
        do_command(cmd.split(), logger=self._logger)
        summary_data = pd.read_csv(
            tmp_csv_path, delimiter=',', header=0, index_col=0, comment='#'
        )
        mask = [x == 'lp__' or not x.endswith('__') for x in summary_data.index]
        return summary_data[mask]

    def diagnose(self) -> str:
        """
        Run cmdstan/bin/diagnose over all output csv files.
        Returns output of diagnose (stdout/stderr)

        The diagnose utility reads the outputs of all chains
        and checks for the following potential problems:

        + Transitions that hit the maximum treedepth
        + Divergent transitions
        + Low E-BFMI values (sampler transitions HMC potential energy)
        + Low effective sample sizes
        + High R-hat values

        :return str empty if no problems found
        """
        self._sampling_only()

        cmd_path = os.path.join(cmdstan_path(), 'bin', 'diagnose' + EXTENSION)
        csv_files = ' '.join(self.csv_files)
        cmd = '{} {} '.format(cmd_path, csv_files)
        result = do_command(cmd=cmd.split(), logger=self._logger)
        if result:
            self._logger.warning(result)
        return result

    def get_drawset(self, params: List[str] = None) -> pd.DataFrame:
        """
        Returns the assembled sample as a pandas DataFrame consisting of
        one column per parameter and one row per draw.

        :param params: list of model parameter names.
        """
        self._sampling_only()

        pnames_base = [name.split('.')[0] for name in self.column_names]
        if params is not None:
            for p in params:
                if not (p in self._column_names or p in pnames_base):
                    raise ValueError('unknown parameter: {}'.format(p))
        self._assemble_sample()
        data = self.sample.reshape(
            (self.draws * self.chains), len(self.column_names), order='A'
        )
        df = pd.DataFrame(data=data, columns=self.column_names)
        if params is None:
            return df
        mask = []
        for p in params:
            for name in self.column_names:
                if p == name or p == name.split('.')[0]:
                    mask.append(name)
        return df[mask]

    def save_csvfiles(self, dir: str = None, basename: str = None) -> None:
        """
        Moves csvfiles to specified directory using specified basename,
        appending suffix '-<id>.csv' to each.

        :param dir: directory path
        :param basename:  base filename
        """
        if dir is None:
            dir = '.'
        test_path = os.path.join(dir, '.{}-test.tmp'.format(basename))
        try:
            with open(test_path, 'w') as fd:
                pass
            os.remove(test_path)  # cleanup
        except OSError:
            raise Exception('cannot save to path: {}'.format(dir))

        for i in range(self.chains):
            if not os.path.exists(self.csv_files[i]):
                raise ValueError(
                    'cannot access csv file {}'.format(self.csv_files[i])
                )
            to_path = os.path.join(dir, '{}-{}.csv'.format(basename, i + 1))
            if os.path.exists(to_path):
                raise ValueError(
                    'file exists, not overwriting: {}'.format(to_path)
                )
            try:
                self._logger.debug(
                    'saving tmpfile: "%s" as: "%s"',
                    self.csv_files[i],
                    to_path
                )
                shutil.move(self.csv_files[i], to_path)
                self.csv_files[i] = to_path
            except (IOError, OSError) as e:
                raise ValueError(
                    'cannot save to file: {}'.format(to_path)
                ) from e
