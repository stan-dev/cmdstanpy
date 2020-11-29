"""Container objects for results of CmdStan run(s)."""

import copy
import logging
import math
import os
import re
import shutil
from collections import Counter, OrderedDict
from datetime import datetime
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import CmdStanArgs, Method
from cmdstanpy.utils import (
    EXTENSION,
    check_sampler_csv,
    cmdstan_path,
    create_named_text_file,
    do_command,
    get_logger,
    parse_var_dims,
    scan_generated_quantities_csv,
    scan_optimize_csv,
    scan_variational_csv,
)


class RunSet:
    """
    Record of CmdStan run for a specified configuration and number of chains.
    """

    def __init__(
        self,
        args: CmdStanArgs,
        chains: int = 4,
        chain_ids: List[int] = None,
        logger: logging.Logger = None,
    ) -> None:
        """Initialize object."""
        self._args = args
        self._chains = chains
        self._logger = logger or get_logger()
        if chains < 1:
            raise ValueError(
                'chains must be positive integer value, '
                'found {}'.format(chains)
            )
        if chain_ids is None:
            chain_ids = [x + 1 for x in range(chains)]
        elif len(chain_ids) != chains:
            raise ValueError(
                'mismatch between number of chains and chain_ids, '
                'found {} chains, but {} chain_ids'.format(
                    chains, len(chain_ids)
                )
            )
        self._chain_ids = chain_ids
        self._retcodes = [-1 for _ in range(chains)]

        # stdout, stderr are written to text files
        # prefix: ``<model_name>-<YYYYMMDDHHMM>-<chain_id>``
        # suffixes: ``-stdout.txt``, ``-stderr.txt``
        now = datetime.now()
        now_str = now.strftime('%Y%m%d%H%M')
        file_basename = '-'.join([args.model_name, now_str])
        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            output_dir = _TMPDIR
        self._csv_files = [None for _ in range(chains)]
        self._diagnostic_files = [None for _ in range(chains)]
        self._stdout_files = [None for _ in range(chains)]
        self._stderr_files = [None for _ in range(chains)]
        self._cmds = []
        for i in range(chains):
            if args.output_dir is None:
                csv_file = create_named_text_file(
                    dir=output_dir,
                    prefix='{}-{}-'.format(file_basename, str(chain_ids[i])),
                    suffix='.csv',
                )
            else:
                csv_file = os.path.join(
                    output_dir,
                    '{}-{}.{}'.format(file_basename, str(chain_ids[i]), 'csv'),
                )
            self._csv_files[i] = csv_file
            stdout_file = ''.join(
                [os.path.splitext(csv_file)[0], '-stdout.txt']
            )
            self._stdout_files[i] = stdout_file
            stderr_file = ''.join(
                [os.path.splitext(csv_file)[0], '-stderr.txt']
            )
            self._stderr_files[i] = stderr_file
            if args.save_diagnostics:
                if args.output_dir is None:
                    diag_file = create_named_text_file(
                        dir=_TMPDIR,
                        prefix='{}-diagnostic-{}-'.format(
                            file_basename, str(chain_ids[i])
                        ),
                        suffix='.csv',
                    )
                else:
                    diag_file = os.path.join(
                        output_dir,
                        '{}-diagnostic-{}.{}'.format(
                            file_basename, str(chain_ids[i]), 'csv'
                        ),
                    )
                self._diagnostic_files[i] = diag_file
                self._cmds.append(
                    args.compose_command(
                        i, self._csv_files[i], self._diagnostic_files[i]
                    )
                )
            else:
                self._cmds.append(args.compose_command(i, self._csv_files[i]))

    def __repr__(self) -> str:
        repr = 'RunSet: chains={}'.format(self._chains)
        repr = '{}\n cmd:\n\t{}'.format(repr, self._cmds[0])
        repr = '{}\n retcodes={}'.format(repr, self._retcodes)
        if os.path.exists(self._csv_files[0]):
            repr = '{}\n csv_files:\n\t{}'.format(
                repr, '\n\t'.join(self._csv_files)
            )
        if self._args.save_diagnostics and os.path.exists(
            self._diagnostic_files[0]
        ):
            repr = '{}\n diagnostics_files:\n\t{}'.format(
                repr, '\n\t'.join(self._diagnostic_files)
            )
        if os.path.exists(self._stdout_files[0]):
            repr = '{}\n console_msgs:\n\t{}'.format(
                repr, '\n\t'.join(self._stdout_files)
            )
        if os.path.exists(self._stderr_files[0]):
            repr = '{}\n error_msgs:\n\t{}'.format(
                repr, '\n\t'.join(self._stderr_files)
            )
        return repr

    @property
    def model(self) -> str:
        """Stan model name."""
        return self._args.model_name

    @property
    def method(self) -> Method:
        """Returns the CmdStan method used to generate this fit."""
        return self._args.method

    @property
    def chains(self) -> int:
        """Number of chains."""
        return self._chains

    @property
    def chain_ids(self) -> List[int]:
        """Chain ids."""
        return self._chain_ids

    @property
    def cmds(self) -> List[str]:
        """Per-chain call to CmdStan."""
        return self._cmds

    @property
    def csv_files(self) -> List[str]:
        """
        List of paths to CmdStan output files.
        """
        return self._csv_files

    @property
    def stdout_files(self) -> List[str]:
        """
        List of paths to CmdStan stdout transcripts.
        """
        return self._stdout_files

    @property
    def stderr_files(self) -> List[str]:
        """
        List of paths to CmdStan stderr transcripts.
        """
        return self._stderr_files

    def _check_retcodes(self) -> bool:
        """True when all chains have retcode 0."""
        for i in range(self._chains):
            if self._retcodes[i] != 0:
                return False
        return True

    @property
    def diagnostic_files(self) -> List[str]:
        """
        List of paths to CmdStan diagnostic output files.
        """
        return self._diagnostic_files

    def _retcode(self, idx: int) -> int:
        """Get retcode for chain[idx]."""
        return self._retcodes[idx]

    def _set_retcode(self, idx: int, val: int) -> None:
        """Set retcode for chain[idx] to val."""
        self._retcodes[idx] = val

    def get_err_msgs(self) -> List[str]:
        """Checks console messages for each chain."""
        msgs = []
        msgs.append(self.__repr__())
        for i in range(self._chains):
            if (
                os.path.exists(self._stderr_files[i])
                and os.stat(self._stderr_files[i]).st_size > 0
            ):
                with open(self._stderr_files[i], 'r') as fd:
                    msgs.append(
                        'chain_id {}:\n{}\n'.format(
                            self._chain_ids[i], fd.read()
                        )
                    )
            if (
                os.path.exists(self._stdout_files[i])
                and os.stat(self._stdout_files[i]).st_size > 0
            ):
                with open(self._stdout_files[i], 'r') as fd:
                    contents = fd.read()
                    # pattern matches initial "Exception" or "Error" msg
                    pat = re.compile(r'^E[rx].*$', re.M)
                errors = re.findall(pat, contents)
                if len(errors) > 0:
                    msgs.append(
                        'chain_id {}:\n\t{}\n'.format(
                            self._chain_ids[i], '\n\t'.join(errors)
                        )
                    )
        return '\n'.join(msgs)

    def save_csvfiles(self, dir: str = None) -> None:
        """
        Moves csvfiles to specified directory.

        :param dir: directory path
        """
        if dir is None:
            dir = os.path.realpath('.')
        test_path = os.path.join(dir, str(time()))
        try:
            os.makedirs(dir, exist_ok=True)
            with open(test_path, 'w'):
                pass
            os.remove(test_path)  # cleanup
        except (IOError, OSError, PermissionError) as exc:
            raise Exception('cannot save to path: {}'.format(dir)) from exc

        for i in range(self.chains):
            if not os.path.exists(self._csv_files[i]):
                raise ValueError(
                    'cannot access csv file {}'.format(self._csv_files[i])
                )

            path, filename = os.path.split(self._csv_files[i])
            if path == _TMPDIR:  # cleanup tmpstr in filename
                root, ext = os.path.splitext(filename)
                rlist = root.split('-')
                root = '-'.join(rlist[:-1])
                filename = ''.join([root, ext])

            to_path = os.path.join(dir, filename)
            if os.path.exists(to_path):
                raise ValueError(
                    'file exists, not overwriting: {}'.format(to_path)
                )
            try:
                self._logger.debug(
                    'saving tmpfile: "%s" as: "%s"', self._csv_files[i], to_path
                )
                shutil.move(self._csv_files[i], to_path)
                self._csv_files[i] = to_path
            except (IOError, OSError, PermissionError) as e:
                raise ValueError(
                    'cannot save to file: {}'.format(to_path)
                ) from e


class CmdStanMCMC:
    """
    Container for outputs from CmdStan sampler run.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods
    def __init__(
        self,
        runset: RunSet,
        validate_csv: bool = True,
        logger: logging.Logger = None,
    ) -> None:
        """Initialize object."""
        if not runset.method == Method.SAMPLE:
            raise ValueError(
                'Wrong runset method, expecting sample runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self._logger = logger or get_logger()
        # copy info from runset
        self._is_fixed_param = runset._args.method_args.fixed_param
        self._iter_sampling = runset._args.method_args.iter_sampling
        self._iter_warmup = runset._args.method_args.iter_warmup
        self._save_warmup = runset._args.method_args.save_warmup
        self._thin = runset._args.method_args.thin
        self._sig_figs = runset._args.sig_figs
        # parse the remainder from csv files
        self._draws_sampling = None
        self._draws_warmup = None
        self._column_names = ()
        self._num_params = None  # metric dim(s)
        self._metric_type = None
        self._metric = None
        self._stepsize = None
        self._draws = None
        self._draws_pd = None
        self._stan_variable_dims = {}
        self._validate_csv = validate_csv
        if validate_csv:
            self.validate_csv_files()

    def __repr__(self) -> str:
        repr = 'CmdStanMCMC: model={} chains={}{}'.format(
            self.runset.model,
            self.runset.chains,
            self.runset._args.method_args.compose(0, cmd=[]),
        )
        repr = '{}\n csv_files:\n\t{}\n output_files:\n\t{}'.format(
            repr,
            '\n\t'.join(self.runset.csv_files),
            '\n\t'.join(self.runset.stdout_files),
        )
        return repr

    @property
    def chains(self) -> int:
        """Number of chains."""
        return self.runset.chains

    @property
    def chain_ids(self) -> List[int]:
        """Chain ids."""
        return self.runset.chain_ids

    @property
    def num_draws(self) -> int:
        """Number of draws per chain."""
        if not self._validate_csv and self._draws_sampling is None:
            return int(
                math.ceil(
                    (self._iter_sampling + self._iter_warmup) / self._thin
                )
            )
        return self._draws_warmup + self._draws_sampling

    @property
    def num_draws_warmup(self) -> int:
        """Number of warmup draws per chain."""
        if not self._validate_csv and self._draws_sampling is None:
            return int(math.ceil((self._iter_warmup) / self._thin))
        return self._draws_warmup

    @property
    def num_draws_sampling(self) -> int:
        """Number of sampling (post-warmup) draws per chain."""
        if not self._validate_csv and self._draws_sampling is None:
            return int(math.ceil((self._iter_sampling) / self._thin))
        return self._draws_sampling

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of all per-draw outputs: all
        sampler and model parameters and quantities of interest
        """
        if not self._validate_csv and len(self._column_names) == 0:
            self._logger.warning(
                'csv files not yet validated, run method validate_csv_files()'
                ' in order to retrieve sample metadata.'
            )
            return None
        return self._column_names

    @property
    def stan_variable_dims(self) -> Dict:
        """
        Dict mapping Stan program variable names to variable dimensions.
        Scalar types have int value '1'.  Structured types have list of dims,
        e.g.,  program variable ``vector[10] foo`` has entry ``('foo', [10])``.
        """
        if not self._validate_csv and len(self._stan_variable_dims) == 0:
            self._logger.warning(
                'csv files not yet validated, run method validate_csv_files()'
                ' in order to retrieve sample metadata.'
            )
            return None
        return copy.deepcopy(self._stan_variable_dims)

    @property
    def metric_type(self) -> str:
        """
        Metric type used for adaptation, either 'diag_e' or 'dense_e'.
        When sampler algorithm 'fixed_param' is specified, metric_type is None.
        """
        if self._is_fixed_param:
            return None
        if not self._validate_csv and self._metric_type is None:
            self._logger.warning(
                'csv files not yet validated, run method validate_csv_files()'
                ' in order to retrieve sample metadata.'
            )
            return None
        return self._metric_type

    @property
    def metric(self) -> np.ndarray:
        """
        Metric used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, metric is None.
        """
        if self._is_fixed_param:
            return None
        if not self._validate_csv and self._metric is None:
            self._logger.warning(
                'csv files not yet validated, run method validate_csv_files()'
                ' in order to retrieve sample metadata.'
            )
            return None
        if self._draws is None:
            self._assemble_draws()
        return self._metric

    @property
    def stepsize(self) -> np.ndarray:
        """
        Stepsize used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, stepsize is None.
        """
        if self._is_fixed_param:
            return None
        if not self._validate_csv and self._stepsize is None:
            self._logger.warning(
                'csv files not yet validated, run method validate_csv_files()'
                ' in order to retrieve sample metadata.'
            )
            return None
        if self._draws is None:
            self._assemble_draws()
        return self._stepsize

    def draws(self, inc_warmup: bool = False) -> np.ndarray:
        """
        A 3-D numpy ndarray which contains all draws, from both warmup and
        sampling iterations, arranged as (draws, chains, columns) and stored
        column major, so that the values for each parameter are contiguous
        in memory, likewise all draws from a chain are contiguous.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.
        """
        if not self._validate_csv and self._draws is None:
            self.validate_csv_files()
        if self._draws is None:
            self._assemble_draws()
        if not inc_warmup:
            if self._save_warmup:
                return self._draws[self._draws_warmup :, :, :]
            return self._draws
        else:
            if not self._save_warmup:
                self._logger.warning(
                    'draws from warmup iterations not available,'
                    ' must run sampler with "save_warmup=True".'
                )
        return self._draws

    @property
    def sample(self) -> np.ndarray:
        """
        Deprecated - use method "draws()" instead.
        """
        self._logger.warning(
            'method "sample" will be deprecated, use method "draws" instead.'
        )
        return self.draws()

    @property
    def warmup(self) -> np.ndarray:
        """
        Deprecated - use "draws(inc_warmup=True)"
        """
        self._logger.warning(
            'method "warmup" has been deprecated, instead use method'
            ' "draws(inc_warmup=True)", returning draws from both'
            ' warmup and sampling iterations.'
        )
        return self.draws(inc_warmup=True)

    def validate_csv_files(self) -> None:
        """
        Checks that csv output files for all chains are consistent.
        Populates attributes for draws, column_names, num_params, metric_type.
        Raises exception when inconsistencies detected.
        """
        dzero = {}
        for i in range(self.runset.chains):
            if i == 0:
                dzero = check_sampler_csv(
                    path=self.runset.csv_files[i],
                    is_fixed_param=self._is_fixed_param,
                    iter_sampling=self._iter_sampling,
                    iter_warmup=self._iter_warmup,
                    save_warmup=self._save_warmup,
                    thin=self._thin,
                )
            else:
                drest = check_sampler_csv(
                    path=self.runset.csv_files[i],
                    is_fixed_param=self._is_fixed_param,
                    iter_sampling=self._iter_sampling,
                    iter_warmup=self._iter_warmup,
                    save_warmup=self._save_warmup,
                    thin=self._thin,
                )
                for key in dzero:
                    if (
                        key
                        not in [
                            'id',
                            'diagnostic_file',
                            'metric_file',
                            'stepsize',
                            'init',
                            'seed',
                        ]
                        and dzero[key] != drest[key]
                    ):
                        raise ValueError(
                            'csv file header mismatch, '
                            'file {}, key {} is {}, expected {}'.format(
                                self.runset.csv_files[i],
                                key,
                                dzero[key],
                                drest[key],
                            )
                        )
        self._draws_sampling = dzero['draws_sampling']
        if self._save_warmup:
            self._draws_warmup = dzero['draws_warmup']
        else:
            self._draws_warmup = 0
        self._column_names = dzero['column_names']
        if not self._is_fixed_param:
            self._num_params = dzero['num_params']
            self._metric_type = dzero.get('metric')
        self._stan_variable_dims = parse_var_dims(dzero['column_names'])

    def _assemble_draws(self) -> None:
        """
        Allocates and populates the stepsize, metric, and sample arrays
        by parsing the validated stan_csv files.
        """
        if self._draws is not None:
            return
        num_draws = self._draws_sampling
        if self._save_warmup:
            num_draws += self._draws_warmup
        self._draws = np.empty(
            (num_draws, self.runset.chains, len(self._column_names)),
            dtype=float,
            order='F',
        )
        if not self._is_fixed_param:
            self._stepsize = np.empty(self.runset.chains, dtype=float)
            if self._metric_type == 'diag_e':
                self._metric = np.empty(
                    (self.runset.chains, self._num_params), dtype=float
                )
            else:
                self._metric = np.empty(
                    (self.runset.chains, self._num_params, self._num_params),
                    dtype=float,
                )
        for chain in range(self.runset.chains):
            with open(self.runset.csv_files[chain], 'r') as fd:
                # skip initial comments, up to columns header
                line = fd.readline().strip()
                while len(line) > 0 and line.startswith('#'):
                    line = fd.readline().strip()
                # at columns header
                if not self._is_fixed_param:
                    if self._save_warmup:
                        for i in range(self._draws_warmup):
                            line = fd.readline().strip()
                            xs = line.split(',')
                            self._draws[i, chain, :] = [float(x) for x in xs]
                    # read to adaptation msg
                    if line != '# Adaptation terminated':
                        while line != '# Adaptation terminated':
                            line = fd.readline().strip()
                    line = fd.readline().strip()  # stepsize
                    _, stepsize = line.split('=')
                    self._stepsize[chain] = float(stepsize.strip())
                    line = fd.readline().strip()  # metric header
                    # process metric
                    if self._metric_type == 'diag_e':
                        line = fd.readline().lstrip(' #\t').strip()
                        xs = line.split(',')
                        self._metric[chain, :] = [float(x) for x in xs]
                    else:
                        for i in range(self._num_params):
                            line = fd.readline().lstrip(' #\t').strip()
                            xs = line.split(',')
                            self._metric[chain, i, :] = [float(x) for x in xs]
                # process draws
                for i in range(self._draws_warmup, num_draws):
                    line = fd.readline().strip()
                    xs = line.split(',')
                    self._draws[i, chain, :] = [float(x) for x in xs]

    def summary(
        self, percentiles: List[int] = None, sig_figs: int = None
    ) -> pd.DataFrame:
        """
        Run cmdstan/bin/stansummary over all output csv files, assemble
        summary into DataFrame object; first row contains summary statistics
        for total joint log probability `lp__`, remaining rows contain summary
        statistics for all parameters, transformed parameters, and generated
        quantities variables listed in the order in which they were declared
        in the Stan program.

        :param percentiles: Ordered non-empty list of percentiles to report.
            Must be integers from (1, 99), inclusive.

        :param sig_figs: Number of significant figures to report.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.
            If precision above 6 is requested, sample must have been produced
            by CmdStan version 2.25 or later and sampler output precision
            must equal to or greater than the requested summary precision.

        :return: pandas.DataFrame
        """
        percentiles_str = '--percentiles=5,50,95'
        if percentiles is not None:
            if len(percentiles) == 0:
                raise ValueError(
                    'invalid percentiles argument, must be ordered'
                    ' non-empty list from (1, 99), inclusive.'
                )
            cur_pct = 0
            for pct in percentiles:
                if pct > 99 or not pct > cur_pct:
                    raise ValueError(
                        'invalid percentiles spec, must be ordered'
                        ' non-empty list from (1, 99), inclusive.'
                    )
                cur_pct = pct
            percentiles_str = '='.join(
                ['--percentiles', ','.join([str(x) for x in percentiles])]
            )
        sig_figs_str = '--sig_figs=2'
        if sig_figs is not None:
            if not isinstance(sig_figs, int) or sig_figs < 1 or sig_figs > 18:
                raise ValueError(
                    'sig_figs must be an integer between 1 and 18,'
                    ' found {}'.format(sig_figs)
                )
            csv_sig_figs = self._sig_figs or 6
            if sig_figs > csv_sig_figs:
                self._logger.warning(
                    'Requesting %d significant digits of output, but CSV files'
                    ' only have %d digits of precision.',
                    sig_figs,
                    csv_sig_figs,
                )
            sig_figs_str = '--sig_figs=' + str(sig_figs)
        cmd_path = os.path.join(
            cmdstan_path(), 'bin', 'stansummary' + EXTENSION
        )
        tmp_csv_file = 'stansummary-{}-{}-chain-'.format(
            self.runset._args.model_name, self.runset.chains
        )
        tmp_csv_path = create_named_text_file(
            dir=_TMPDIR, prefix=tmp_csv_file, suffix='.csv'
        )
        cmd = [
            cmd_path,
            percentiles_str,
            sig_figs_str,
            '--csv_file={}'.format(tmp_csv_path),
        ] + self.runset.csv_files
        do_command(cmd, logger=self.runset._logger)
        with open(tmp_csv_path, 'rb') as fd:
            summary_data = pd.read_csv(
                fd,
                delimiter=',',
                header=0,
                index_col=0,
                comment='#',
                float_precision='high',
            )
        mask = [x == 'lp__' or not x.endswith('__') for x in summary_data.index]
        return summary_data[mask]

    def diagnose(self) -> str:
        """
        Run cmdstan/bin/diagnose over all output csv files.
        Returns output of diagnose (stdout/stderr).

        The diagnose utility reads the outputs of all chains
        and checks for the following potential problems:

        + Transitions that hit the maximum treedepth
        + Divergent transitions
        + Low E-BFMI values (sampler transitions HMC potential energy)
        + Low effective sample sizes
        + High R-hat values
        """
        cmd_path = os.path.join(cmdstan_path(), 'bin', 'diagnose' + EXTENSION)
        cmd = [cmd_path] + self.runset.csv_files
        result = do_command(cmd=cmd, logger=self.runset._logger)
        if result:
            self.runset._logger.info(result)
        return result

    def draws_pd(
        self, params: List[str] = None, inc_warmup: bool = False
    ) -> pd.DataFrame:
        """
        Returns the assembled draws as a pandas DataFrame consisting of
        one column per parameter and one row per draw.

        :param params: list of model parameter names.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.
        """
        pnames_base = [name.split('[')[0] for name in self.column_names]
        if params is not None:
            for param in params:
                if not (param in self._column_names or param in pnames_base):
                    raise ValueError('unknown parameter: {}'.format(param))
        self._assemble_draws()

        if inc_warmup and not self._save_warmup:
            self._logger.warning(
                'draws from warmup iterations not available,'
                ' must run sampler with "save_warmup=True".'
            )

        num_draws = self._draws_sampling
        if inc_warmup and self._save_warmup:
            num_draws += self._draws_warmup
        num_rows = num_draws * self.runset.chains
        if self._draws_pd is None or self._draws_pd.shape[0] != num_rows:
            # pylint: disable=redundant-keyword-arg
            data = self.draws(inc_warmup=inc_warmup).reshape(
                (num_rows, len(self.column_names)), order='A'
            )
            self._draws_pd = pd.DataFrame(data=data, columns=self.column_names)
        if params is None:
            return self._draws_pd
        mask = []
        params = set(params)
        for name in self.column_names:
            if any(item in params for item in (name, name.split('[')[0])):
                mask.append(name)
        return self._draws_pd[mask]

    def stan_variable(self, name: str) -> pd.DataFrame:
        """
        Return a new DataFrame which contains the set of post-warmup draws
        for the named Stan program variable.  Flattens the chains.
        Underlyingly draws are in chain order, i.e., for a sample
        consisting of N chains of M draws each, the first M array
        elements are from chain 1, the next M are from chain 2,
        and the last M elements are from chain N.

        * If the variable is a scalar variable, the shape of the DataFrame is
          ( draws X chains, 1).
        * If the variable is a vector, the shape of the DataFrame is
          ( draws X chains, len(vector))
        * If the variable is a matrix, the shape of the DataFrame is
          ( draws X chains, size(dim 1) X size(dim 2) )
        * If the variable is an array with N dimensions, the shape of the
          DataFrame is ( draws X chains, size(dim 1) X ... X size(dim N))

        :param name: variable name
        """
        if name not in self._stan_variable_dims:
            raise ValueError('unknown name: {}'.format(name))
        self._assemble_draws()
        dim0 = self._draws_sampling * self.runset.chains
        dims = np.prod(self._stan_variable_dims[name])
        pattern = r'^{}(\[[\d,]+\])?$'.format(name)
        names, idxs = [], []
        for i, column_name in enumerate(self.column_names):
            if re.search(pattern, column_name):
                names.append(column_name)
                idxs.append(i)
        return pd.DataFrame(
            self._draws[self._draws_warmup :, :, idxs].reshape(
                (dim0, dims), order='A'
            ),
            columns=names,
        )

    def stan_variables(self) -> Dict:
        """
        Return a dictionary of all Stan program variables.
        Creates copies of the data in the draws matrix.
        """
        result = {}
        for name in self.stan_variable_dims:
            result[name] = self.stan_variable(name)
        return result

    def sampler_diagnostics(self) -> Dict:
        """
        Returns the sampler diagnostics as a map from
        column name to draws X chains X 1 ndarray.
        """
        result = {}
        self._assemble_draws()
        diag_names = [x for x in self.column_names if x.endswith('__')]
        for idx, value in enumerate(diag_names):
            result[value] = self._draws[:, :, idx]
        return result

    def save_csvfiles(self, dir: str = None) -> None:
        """
        Move output csvfiles to specified directory.  If files were
        written to the temporary session directory, clean filename.
        E.g., save 'bernoulli-201912081451-1-5nm6as7u.csv' as
        'bernoulli-201912081451-1.csv'.

        :param dir: directory path
        """
        self.runset.save_csvfiles(dir)


class CmdStanMLE:
    """
    Container for outputs from CmdStan optimization.
    """

    def __init__(self, runset: RunSet) -> None:
        """Initialize object."""
        if not runset.method == Method.OPTIMIZE:
            raise ValueError(
                'Wrong runset method, expecting optimize runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self._column_names = ()
        self._mle = {}
        self._set_mle_attrs(runset.csv_files[0])

    def __repr__(self) -> str:
        repr = 'CmdStanMLE: model={}{}'.format(
            self.runset.model, self.runset._args.method_args.compose(0, cmd=[])
        )
        repr = '{}\n csv_file:\n\t{}\n output_file:\n\t{}'.format(
            repr,
            '\n\t'.join(self.runset.csv_files),
            '\n\t'.join(self.runset.stdout_files),
        )
        return repr

    def _set_mle_attrs(self, sample_csv_0: str) -> None:
        meta = scan_optimize_csv(sample_csv_0)
        self._column_names = meta['column_names']
        self._mle = meta['mle']

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of estimated quantities, includes joint log probability,
        and all parameters, transformed parameters, and generated quantitites.
        """
        return self._column_names

    @property
    def optimized_params_np(self) -> np.array:
        """Returns optimized params as numpy array."""
        return np.asarray(self._mle)

    @property
    def optimized_params_pd(self) -> pd.DataFrame:
        """Returns optimized params as pandas DataFrame."""
        return pd.DataFrame([self._mle], columns=self.column_names)

    @property
    def optimized_params_dict(self) -> OrderedDict:
        """Returns optimized params as Dict."""
        return OrderedDict(zip(self.column_names, self._mle))

    def save_csvfiles(self, dir: str = None) -> None:
        """
        Move output csvfiles to specified directory.  If files were
        written to the temporary session directory, clean filename.
        E.g., save 'bernoulli-201912081451-1-5nm6as7u.csv' as
        'bernoulli-201912081451-1.csv'.

        :param dir: directory path
        """
        self.runset.save_csvfiles(dir)


class CmdStanGQ:
    """
    Container for outputs from CmdStan generate_quantities run.
    """

    def __init__(self, runset: RunSet, mcmc_sample: pd.DataFrame) -> None:
        """Initialize object."""
        if not runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError(
                'Wrong runset method, expecting generate_quantities runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self.mcmc_sample = mcmc_sample
        self._generated_quantities = None
        self._column_names = scan_generated_quantities_csv(
            self.runset.csv_files[0]
        )['column_names']

    def __repr__(self) -> str:
        repr = 'CmdStanGQ: model={} chains={}{}'.format(
            self.runset.model,
            self.runset.chains,
            self.runset._args.method_args.compose(0, cmd=[]),
        )
        repr = '{}\n csv_files:\n\t{}\n output_files:\n\t{}'.format(
            repr,
            '\n\t'.join(self.runset.csv_files),
            '\n\t'.join(self.runset.stdout_files),
        )
        return repr

    @property
    def chains(self) -> int:
        """Number of chains."""
        return self.runset.chains

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of generated quantities of interest.
        """
        return self._column_names

    @property
    def generated_quantities(self) -> np.ndarray:
        """
        A 2-D numpy ndarray which contains generated quantities draws
        for all chains where the columns correspond to the generated quantities
        block variables and the rows correspond to the draws from all chains,
        where first M draws are the first M draws of chain 1 and the
        last M draws are the last M draws of chain N, i.e.,
        flattened chain, draw ordering.
        """
        if not self.runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError('Bad runset method {}.'.format(self.runset.method))
        if self._generated_quantities is None:
            self._assemble_generated_quantities()
        return self._generated_quantities

    @property
    def generated_quantities_pd(self) -> pd.DataFrame:
        """
        Returns the generated quantities as a pandas DataFrame consisting of
        one column per quantity of interest and one row per draw.
        """
        if not self.runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError('Bad runset method {}.'.format(self.runset.method))
        if self._generated_quantities is None:
            self._assemble_generated_quantities()
        return pd.DataFrame(
            data=self._generated_quantities, columns=self.column_names
        )

    @property
    def sample_plus_quantities(self) -> pd.DataFrame:
        """
        Returns the column-wise concatenation of the input drawset
        with generated quantities drawset.  If there are duplicate
        columns in both the input and the generated quantities,
        the input column is dropped in favor of the recomputed
        values in the generate quantities drawset.
        """
        if not self.runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError('Bad runset method {}.'.format(self.runset.method))
        if self._generated_quantities is None:
            self._assemble_generated_quantities()

        cols_1 = self.mcmc_sample.columns.tolist()
        cols_2 = self.generated_quantities_pd.columns.tolist()

        dups = [
            item
            for item, count in Counter(cols_1 + cols_2).items()
            if count > 1
        ]
        return pd.concat(
            [self.mcmc_sample.drop(columns=dups), self.generated_quantities_pd],
            axis=1,
        )

    def _assemble_generated_quantities(self) -> None:
        drawset_list = []
        for chain in range(self.runset.chains):
            drawset_list.append(
                pd.read_csv(
                    self.runset.csv_files[chain],
                    comment='#',
                    float_precision='high',
                )
            )
        self._generated_quantities = pd.concat(drawset_list).values

    def save_csvfiles(self, dir: str = None) -> None:
        """
        Move output csvfiles to specified directory.  If files were
        written to the temporary session directory, clean filename.
        E.g., save 'bernoulli-201912081451-1-5nm6as7u.csv' as
        'bernoulli-201912081451-1.csv'.

        :param dir: directory path
        """
        self.runset.save_csvfiles(dir)


class CmdStanVB:
    """
    Container for outputs from CmdStan variational run.
    """

    def __init__(self, runset: RunSet) -> None:
        """Initialize object."""
        if not runset.method == Method.VARIATIONAL:
            raise ValueError(
                'Wrong runset method, expecting variational inference, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self._column_names = ()
        self._variational_mean = {}
        self._variational_sample = None
        self._set_variational_attrs(runset.csv_files[0])

    def __repr__(self) -> str:
        repr = 'CmdStanVB: model={}{}'.format(
            self.runset.model, self.runset._args.method_args.compose(0, cmd=[])
        )
        repr = '{}\n csv_file:\n\t{}\n output_file:\n\t{}'.format(
            repr,
            '\n\t'.join(self.runset.csv_files),
            '\n\t'.join(self.runset.stdout_files),
        )
        return repr

    def _set_variational_attrs(self, sample_csv_0: str) -> None:
        meta = scan_variational_csv(sample_csv_0)
        self._column_names = meta['column_names']
        self._variational_mean = meta['variational_mean']
        self._variational_sample = meta['variational_sample']

    @property
    def columns(self) -> int:
        """
        Total number of information items returned by sampler.
        Includes approximation information and names of model parameters
        and computed quantities.
        """
        return len(self._column_names)

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of information items returned by sampler for each draw.
        Includes approximation information and names of model parameters
        and computed quantities.
        """
        return self._column_names

    @property
    def variational_params_np(self) -> np.array:
        """Returns inferred parameter means as numpy array."""
        return self._variational_mean

    @property
    def variational_params_pd(self) -> pd.DataFrame:
        """Returns inferred parameter means as pandas DataFrame."""
        return pd.DataFrame([self._variational_mean], columns=self.column_names)

    @property
    def variational_params_dict(self) -> OrderedDict:
        """Returns inferred parameter means as Dict."""
        return OrderedDict(zip(self.column_names, self._variational_mean))

    @property
    def variational_sample(self) -> np.array:
        """Returns the set of approximate posterior output draws."""
        return self._variational_sample

    def save_csvfiles(self, dir: str = None) -> None:
        """
        Move output csvfiles to specified directory.  If files were
        written to the temporary session directory, clean filename.
        E.g., save 'bernoulli-201912081451-1-5nm6as7u.csv' as
        'bernoulli-201912081451-1.csv'.

        :param dir: directory path
        """
        self.runset.save_csvfiles(dir)
