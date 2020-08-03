"""Container objects for results of CmdStan run(s)."""

import os
import re
import shutil
import copy
import logging
from typing import List, Tuple, Dict
from collections import Counter, OrderedDict
from datetime import datetime
from time import time
import numpy as np
import pandas as pd

from cmdstanpy import _TMPDIR
from cmdstanpy.utils import (
    check_sampler_csv,
    scan_optimize_csv,
    scan_generated_quantities_csv,
    scan_variational_csv,
    create_named_text_file,
    EXTENSION,
    cmdstan_path,
    do_command,
    get_logger,
    parse_var_dims,
)
from cmdstanpy.cmdstan_args import Method, CmdStanArgs


class RunSet:
    """
    Record of CmdStan run for a specified configuration and number of chains.
    """

    def __init__(
        self, args: CmdStanArgs, chains: int = 4, logger: logging.Logger = None
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
                    prefix='{}-{}-'.format(file_basename, i + 1),
                    suffix='.csv',
                )
            else:
                csv_file = os.path.join(
                    output_dir, '{}-{}.{}'.format(file_basename, i + 1, 'csv')
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
                        prefix='{}-diagnostic-{}-'.format(file_basename, i + 1),
                        suffix='.csv',
                    )
                else:
                    diag_file = os.path.join(
                        output_dir,
                        '{}-diagnostic-{}.{}'.format(
                            file_basename, i + 1, 'csv'
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
        repr = '{}\n csv_files:\n\t{}\n output_files:\n\t{}'.format(
            repr, '\n\t'.join(self._csv_files), '\n\t'.join(self._stdout_files)
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
        """Number of sampler chains."""
        return self._chains

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

    def _get_err_msgs(self) -> List[str]:
        """Checks console messages for each chain."""
        msgs = []
        for i in range(self._chains):
            if (
                os.path.exists(self._stderr_files[i])
                and os.stat(self._stderr_files[i]).st_size > 0
            ):
                with open(self._stderr_files[i], 'r') as fd:
                    msgs.append('chain {}:\n{}\n'.format(i + 1, fd.read()))
            if (
                os.path.exists(self._stdout_files[i])
                and os.stat(self._stdout_files[i]).st_size > 0
            ):
                with open(self._stdout_files[i], 'r') as fd:
                    contents = fd.read()
                    pat = re.compile(r'^Exception.*$', re.M)
                errors = re.findall(pat, contents)
                if len(errors) > 0:
                    msgs.append('chain {}: {}\n'.format(i + 1, errors))
        return msgs

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
        except OSError:
            raise Exception('cannot save to path: {}'.format(dir))

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

    def __init__(self, runset: RunSet) -> None:
        """Initialize object."""
        if not runset.method == Method.SAMPLE:
            raise ValueError(
                'Wrong runset method, expecting sample runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        # copy info from runset
        self._is_fixed_param = runset._args.method_args.fixed_param
        self._iter_sampling = runset._args.method_args.iter_sampling
        self._iter_warmup = runset._args.method_args.iter_warmup
        self._save_warmup = runset._args.method_args.save_warmup
        self._thin = runset._args.method_args.thin
        # parse the remainder from csv files
        self._draws_sampling = None
        self._draws_warmup = None
        self._column_names = ()
        self._num_params = None  # metric dim(s)
        self._metric_type = None
        self._metric = None
        self._stepsize = None
        self._sample = None
        self._warmup = None
        self._drawset = None
        self._stan_variable_dims = {}
        self._validate_csv_files()

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
    def num_draws(self) -> int:
        """Number of post-warmup draws per chain."""
        return self._draws_sampling

    @property
    def num_draws_warmup(self) -> int:
        """Number of warmup draws per chain."""
        return self._draws_warmup

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of all per-draw outputs: all
        sampler and model parameters and quantities of interest
        """
        return self._column_names

    @property
    def stan_variable_dims(self) -> Dict:
        """
        Dict mapping Stan program variable names to variable dimensions.
        Scalar types have int value '1'.  Structured types have list of dims,
        e.g.,  program variable ``vector[10] foo`` has entry ``('foo', [10])``.
        """
        return copy.deepcopy(self._stan_variable_dims)

    @property
    def metric_type(self) -> str:
        """
        Metric type used for adaptation, either 'diag_e' or 'dense_e'.
        When sampler algorithm 'fixed_param' is specified, metric_type is None.
        """
        return self._metric_type

    @property
    def metric(self) -> np.ndarray:
        """
        Metric used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, metric is None.
        """
        if not self._is_fixed_param and self._metric is None:
            self._assemble_sample()
        return self._metric

    @property
    def stepsize(self) -> np.ndarray:
        """
        Stepsize used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, stepsize is None.
        """
        if not self._is_fixed_param and self._stepsize is None:
            self._assemble_sample()
        return self._stepsize

    @property
    def sample(self) -> np.ndarray:
        """
        A 3-D numpy ndarray which contains all draws across all chains
        arranged as (draws, chains, columns) stored column major
        so that the values for each parameter are stored contiguously
        in memory, likewise all draws from a chain are contiguous.
        """
        if self._sample is None:
            self._assemble_sample()
        return self._sample

    @property
    def warmup(self) -> np.ndarray:
        """
        A 3-D numpy ndarray which contains all warmup draws across all chains
        arranged as (draws, chains, columns) stored column major
        so that the values for each parameter are stored contiguously
        in memory, likewise all draws from a chain are contiguous.
        """
        if not self._save_warmup:
            return None
        if self._sample is None:
            self._assemble_sample()
        return self._warmup

    def _validate_csv_files(self) -> None:
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
                        key not in ['id', 'diagnostic_file']
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

    def _assemble_sample(self) -> None:
        """
        Allocates and populates the stepsize, metric, and sample arrays
        by parsing the validated stan_csv files.
        """
        if self._sample is not None:
            return
        self._sample = np.empty(
            (self._draws_sampling, self.runset.chains, len(self._column_names)),
            dtype=float,
            order='F',
        )
        if self._save_warmup:
            self._warmup = np.empty(
                (
                    self._draws_warmup,
                    self.runset.chains,
                    len(self._column_names),
                ),
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
                            self._warmup[i, chain, :] = [float(x) for x in xs]
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
                for i in range(self._draws_sampling):
                    line = fd.readline().strip()
                    xs = line.split(',')
                    self._sample[i, chain, :] = [float(x) for x in xs]

    def summary(self) -> pd.DataFrame:
        """
        Run cmdstan/bin/stansummary over all output csv files.
        Echo stansummary stdout/stderr to console.
        Assemble csv tempfile contents into pandasDataFrame.
        """
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
            '--csv_file={}'.format(tmp_csv_path),
        ] + self.runset.csv_files
        do_command(cmd, logger=self.runset._logger)
        with open(tmp_csv_path, 'rb') as fd:
            summary_data = pd.read_csv(
                fd, delimiter=',', header=0, index_col=0, comment='#'
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

    def get_drawset(self, params: List[str] = None) -> pd.DataFrame:
        """
        Returns the assembled sample as a pandas DataFrame consisting of
        one column per parameter and one row per draw.

        :param params: list of model parameter names.
        """
        pnames_base = [name.split('.')[0] for name in self.column_names]
        if params is not None:
            for param in params:
                if not (param in self._column_names or param in pnames_base):
                    raise ValueError('unknown parameter: {}'.format(param))
        self._assemble_sample()
        if self._drawset is None:
            # pylint: disable=redundant-keyword-arg
            data = self.sample.reshape(
                (self.num_draws * self.runset.chains),
                len(self.column_names),
                order='A',
            )
            self._drawset = pd.DataFrame(data=data, columns=self.column_names)
        if params is None:
            return self._drawset
        mask = []
        params = set(params)
        for name in self.column_names:
            if any(item in params for item in (name, name.split('.')[0])):
                mask.append(name)
        return self._drawset[mask]

    def stan_variable(self, name: str) -> np.ndarray:
        """
        Return a new ndarray which contains the set of draws
        for the named Stan program variable.  Flattens the chains.
        Underlyingly draws are in chain order, i.e., for a sample
        consisting of N chains of M draws each, the first M array
        elements are from chain 1, the next M are from chain 2,
        and the last M elements are from chain N.

        * If the variable is a scalar variable, this returns a 1-d array,
          length(draws X chains).
        * If the variable is a vector, this is a 2-d array,
          shape ( draws X chains, len(vector))
        * If the variable is a matrix, this is a 3-d array,
          shape ( draws X chains, matrix nrows, matrix ncols ).
        * If the variable is an array with N dimensions, this is an N+1-d array,
          shape ( draws X chains, size(dim 1), ... size(dim N)).

        :param name: variable name
        """
        if name not in self._stan_variable_dims:
            raise ValueError('unknown name: {}'.format(name))
        self._assemble_sample()
        dim0 = self.num_draws * self.runset.chains
        dims = self._stan_variable_dims[name]
        if dims == 1:
            idx = self.column_names.index(name)
            return self.sample[:, :, idx].reshape((dim0,), order='A')
        else:
            idxs = [
                x[0]
                for x in enumerate(self.column_names)
                if x[1].startswith(name + '.')
            ]
            var_dims = [dim0]
            var_dims.extend(dims)
            return self.sample[:, :, idxs[0] : idxs[-1] + 1].reshape(
                tuple(var_dims), order='A'
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
        self._assemble_sample()
        diag_names = [x for x in self.column_names if x.endswith('__')]
        for idx, value in enumerate(diag_names):
            result[value] = self.sample[:, :, idx]
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
                pd.read_csv(self.runset.csv_files[chain], comment='#')
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
