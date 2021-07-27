"""Container objects for results of CmdStan run(s)."""

import copy
import glob
import logging
import math
import os
import re
import shutil
from collections import Counter, OrderedDict
from datetime import datetime
from time import time
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

try:
    import xarray as xr

    XARRAY_INSTALLED = True
except ImportError:
    XARRAY_INSTALLED = False

from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP, _TMPDIR
from cmdstanpy.cmdstan_args import (
    CmdStanArgs,
    Method,
    OptimizeArgs,
    SamplerArgs,
    VariationalArgs,
)
from cmdstanpy.utils import (
    EXTENSION,
    check_sampler_csv,
    cmdstan_path,
    cmdstan_version_at,
    create_named_text_file,
    do_command,
    get_logger,
    flatten_chains,
    parse_method_vars,
    parse_stan_vars,
    scan_config,
    scan_generated_quantities_csv,
    scan_optimize_csv,
    scan_variational_csv,
)


class RunSet:
    """
    Encapsulates the configuration and results of a call to any CmdStan
    inference method. Records the method return code and locations of
    all console, error, and output files.
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
        self._profile_files = [None for _ in range(chains)]
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
            # optional output files:  diagnostics, profiling
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
            if args.save_profile:
                if args.output_dir is None:
                    profile_file = create_named_text_file(
                        dir=_TMPDIR,
                        prefix='{}-profile-{}-'.format(
                            file_basename, str(chain_ids[i])
                        ),
                        suffix='.csv',
                    )
                else:
                    profile_file = os.path.join(
                        output_dir,
                        '{}-profile-{}.{}'.format(
                            file_basename, str(chain_ids[i]), 'csv'
                        ),
                    )
                self._profile_files[i] = profile_file
            if args.save_diagnostics and args.save_profile:
                self._cmds.append(
                    args.compose_command(
                        i,
                        self._csv_files[i],
                        diagnostic_file=self._diagnostic_files[i],
                        profile_file=self._profile_files[i],
                    )
                )
            elif args.save_diagnostics:
                self._cmds.append(
                    args.compose_command(
                        i,
                        self._csv_files[i],
                        diagnostic_file=self._diagnostic_files[i],
                    )
                )
            elif args.save_profile:
                self._cmds.append(
                    args.compose_command(
                        i,
                        self._csv_files[i],
                        profile_file=self._profile_files[i],
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
        if self._args.save_profile and os.path.exists(self._profile_files[0]):
            repr = '{}\n profile_files:\n\t{}'.format(
                repr, '\n\t'.join(self._profile_files)
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
        """CmdStan method used to generate this fit."""
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
        """List of call(s) to CmdStan, one call per-chain."""
        return self._cmds

    @property
    def csv_files(self) -> List[str]:
        """List of paths to CmdStan output files."""
        return self._csv_files

    @property
    def stdout_files(self) -> List[str]:
        """List of paths to CmdStan stdout transcripts."""
        return self._stdout_files

    @property
    def stderr_files(self) -> List[str]:
        """List of paths to CmdStan stderr transcripts."""
        return self._stderr_files

    def _check_retcodes(self) -> bool:
        """Returns ``True`` when all chains have retcode 0."""
        for i in range(self._chains):
            if self._retcodes[i] != 0:
                return False
        return True

    @property
    def diagnostic_files(self) -> List[str]:
        """List of paths to CmdStan hamiltonian diagnostic files."""
        return self._diagnostic_files

    @property
    def profile_files(self) -> List[str]:
        """List of paths to CmdStan profiler files."""
        return self._profile_files

    def _retcode(self, idx: int) -> int:
        """Get retcode for chain[idx]."""
        return self._retcodes[idx]

    def _set_retcode(self, idx: int, val: int) -> None:
        """Set retcode for chain[idx] to val."""
        self._retcodes[idx] = val

    def get_err_msgs(self) -> List[str]:
        """Checks console messages for each chain."""
        msgs = []
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
            # pre 2.27, all msgs sent to stdout, including errors
            if (
                not cmdstan_version_at(2, 27)
                and os.path.exists(self._stdout_files[i])
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


class InferenceMetadata:
    """
    CmdStan configuration and contents of output file parsed out of
    the Stan CSV file header comments and column headers.
    Assumes valid CSV files.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize object from CSV headers"""
        self._cmdstan_config = config
        self._method_vars_cols = parse_method_vars(names=config['column_names'])
        stan_vars_dims, stan_vars_cols = parse_stan_vars(
            names=config['column_names']
        )
        self._stan_vars_dims = stan_vars_dims
        self._stan_vars_cols = stan_vars_cols

    def __repr__(self) -> str:
        return 'Metadata:\n{}\n'.format(self._cmdstan_config)

    @property
    def cmdstan_config(self) -> Dict:
        return copy.deepcopy(self._cmdstan_config)

    @property
    def method_vars_cols(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns a map from a Stan inference method variable to
        a tuple of column indices in inference engine's output array.
        Sampler variable names always end in `__`, e.g. `lp__`.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._method_vars_cols)

    @property
    def stan_vars_cols(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns a map from a Stan program variable name to a
        tuple of the column indices in the vector or matrix of
        estimates produced by a CmdStan inference method.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._stan_vars_cols)

    @property
    def stan_vars_dims(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns map from Stan program variable names to variable dimensions.
        Scalar types are mapped to the empty tuple, e.g.,
        program variable ``int foo`` has dimesion ``()`` and
        program variable ``vector[10] bar`` has single dimension ``(10)``.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._stan_vars_dims)


class CmdStanMCMC:
    """
    Container for outputs from CmdStan sampler run.
    Provides methods to summarize and diagnose the model fit
    and accessor methods to access the entire sample or
    individual items.

    The sample is lazily instantiated on first access of either
    the resulting sample or the HMC tuning parameters, i.e., the
    step size and metric.  The sample can viewed either as a 2D array
    of draws from all chains by sampler and model variables, or as a
    3D array of draws by chains by variables.
    """

    # pylint: disable=too-many-public-methods
    def __init__(
        self,
        runset: RunSet,
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
        # info from runset to be exposed
        self._iter_sampling = runset._args.method_args.iter_sampling
        if self._iter_sampling is None:
            self._iter_sampling = _CMDSTAN_SAMPLING
        self._iter_warmup = runset._args.method_args.iter_warmup
        if self._iter_warmup is None:
            self._iter_warmup = _CMDSTAN_WARMUP
        self._thin = runset._args.method_args.thin
        if self._thin is None:
            self._thin = _CMDSTAN_THIN
        self._is_fixed_param = runset._args.method_args.fixed_param
        self._save_warmup = runset._args.method_args.save_warmup
        self._sig_figs = runset._args.sig_figs
        # info from CSV values, instantiated lazily
        self._metric = None
        self._step_size = None
        self._draws = None
        self._draws_pd = None
        # info from CSV initial comments and header
        config = self._validate_csv_files()
        self._metadata = InferenceMetadata(config)

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
        # TODO - hamiltonian, profiling files
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
    def num_draws_warmup(self) -> int:
        """Number of warmup draws per chain, i.e., thinned warmup iterations."""
        return int(math.ceil((self._iter_warmup) / self._thin))

    @property
    def num_draws_sampling(self) -> int:
        """
        Number of sampling (post-warmup) draws per chain, i.e.,
        thinned sampling iterations.
        """
        return int(math.ceil((self._iter_sampling) / self._thin))

    @property
    def metadata(self) -> InferenceMetadata:
        """
        Returns object which contains CmdStan configuration as well as
        information about the names and structure of the inference method
        and model output variables.
        """
        return self._metadata

    @property
    def sampler_vars_cols(self) -> Dict:
        """
        Deprecated - use "metadata.method_vars_cols" instead
        """
        self._logger.warning(
            'property "sampler_vars_cols" has been deprecated, '
            'use "metadata.method_vars_cols" instead.'
        )
        return self.metadata.method_vars_cols

    @property
    def stan_vars_cols(self) -> Dict:
        """
        Deprecated - use "metadata.stan_vars_cols" instead
        """
        self._logger.warning(
            'property "stan_vars_cols" has been deprecated, '
            'use "metadata.stan_vars_cols" instead.'
        )
        return self.metadata.method_vars_cols

    @property
    def stan_vars_dims(self) -> Dict:
        """
        Deprecated - use "metadata.stan_vars_dims" instead
        """
        self._logger.warning(
            'property "stan_vars_dims" has been deprecated, '
            'use "metadata.stan_vars_dims" instead.'
        )
        return self.metadata.stan_vars_dims

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of all outputs from the sampler, comprising sampler parameters
        and all components of all model parameters, transformed parameters,
        and quantities of interest. Corresponds to Stan CSV file header row,
        with names munged to array notation, e.g. `beta[1]` not `beta.1`.
        """
        return self._metadata.cmdstan_config['column_names']

    @property
    def num_unconstrained_params(self) -> int:
        """
        Count of _unconstrained_ model parameters. This is the metric size;
        for metric `diag_e`, the length of the diagonal vector, for metric
        `dense_e` this is the size of the full covariance matrix.

        If the parameter variables in a model are
        constrained parameter types, the number of constrained and
        unconstrained parameters may differ.  The sampler reports the
        constrained parameters and computes with the unconstrained parameters.
        E.g. a model with 2 parameter variables, ``real alpha`` and
        ``vector[3] beta`` has 4 constrained and 4 unconstrained parameters,
        however a model with variables ``real alpha`` and ``simplex[3] beta``
        has 4 constrained and 3 unconstrained parameters.
        """
        return self._metadata.cmdstan_config['num_unconstrained_params']

    @property
    def metric_type(self) -> str:
        """
        Metric type used for adaptation, either 'diag_e' or 'dense_e'.
        When sampler algorithm 'fixed_param' is specified, metric_type is None.
        """
        if self._is_fixed_param:
            return None
        return self._metadata.cmdstan_config['metric']  # cmdstan arg name

    @property
    def metric(self) -> np.ndarray:
        """
        Metric used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, metric is None.
        """
        if self._is_fixed_param:
            return None
        if self._metric is None:
            self._assemble_draws()
        return self._metric

    @property
    def step_size(self) -> np.ndarray:
        """
        Step size used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, step size is None.
        """
        if self._is_fixed_param:
            return None
        if self._step_size is None:
            self._assemble_draws()
        return self._step_size

    @property
    def thin(self) -> int:
        """
        Period between recorded iterations.  (Default is 1).
        """
        return self._thin

    def draws(
        self, *, inc_warmup: bool = False, concat_chains: bool = False
    ) -> np.ndarray:
        """
        Returns a numpy.ndarray over all draws from all chains which is
        stored column major so that the values for a parameter are contiguous
        in memory, likewise all draws from a chain are contiguous.
        By default, returns a 3D array arranged (draws, chains, columns);
        parameter ``concat_chains=True`` will return a 2D array where all
        chains are flattened into a single column, although underlyingly,
        given M chains of N draws, the first N draws are from chain 1, up
        through the last N draws from chain M.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        :param concat_chains: When ``True`` return a 2D array flattening all
            all draws from all chains.  Default value is ``False``.
        """
        if self._draws is None:
            self._assemble_draws()

        if inc_warmup and not self._save_warmup:
            self._logger.warning(
                'draws from warmup iterations not available,'
                ' must run sampler with "save_warmup=True".'
            )

        num_rows = self._draws.shape[0]
        start_idx = 0
        if not inc_warmup and self._save_warmup:
            start_idx = self.num_draws_warmup
            num_rows -= start_idx

        if concat_chains:
            return flatten_chains(self._draws[start_idx:, :, :])
        return self._draws[start_idx:, :, :]

    @property
    def sample(self) -> np.ndarray:
        """
        Deprecated - use method "draws()" instead.
        """
        self._logger.warning(
            'method "sample" has been deprecated, use method "draws" instead.'
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

    def _validate_csv_files(self) -> dict:
        """
        Checks that Stan CSV output files for all chains are consistent
        and returns dict containing config and column names.

        Raises exception when inconsistencies detected.
        """
        dzero = {}
        for i in range(self.chains):
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
                            'profile_file',
                            'stepsize',
                            'init',
                            'seed',
                            'start_datetime',
                        ]
                        and dzero[key] != drest[key]
                    ):
                        raise ValueError(
                            'CmdStan config mismatch in Stan CSV file {}: '
                            'arg {} is {}, expected {}'.format(
                                self.runset.csv_files[i],
                                key,
                                dzero[key],
                                drest[key],
                            )
                        )
        return dzero

    def _assemble_draws(self) -> None:
        """
        Allocates and populates the step size, metric, and sample arrays
        by parsing the validated stan_csv files.
        """
        if self._draws is not None:
            return

        num_draws = self.num_draws_sampling
        sampling_iter_start = 0
        if self._save_warmup:
            num_draws += self.num_draws_warmup
            sampling_iter_start = self.num_draws_warmup
        self._draws = np.empty(
            (num_draws, self.chains, len(self.column_names)),
            dtype=float,
            order='F',
        )
        if not self._is_fixed_param:
            self._step_size = np.empty(self.chains, dtype=float)
            if self.metric_type == 'diag_e':
                self._metric = np.empty(
                    (self.chains, self.num_unconstrained_params), dtype=float
                )
            else:
                self._metric = np.empty(
                    (
                        self.chains,
                        self.num_unconstrained_params,
                        self.num_unconstrained_params,
                    ),
                    dtype=float,
                )
        for chain in range(self.chains):
            with open(self.runset.csv_files[chain], 'r') as fd:
                # skip initial comments, up to columns header
                line = fd.readline().strip()
                while len(line) > 0 and line.startswith('#'):
                    line = fd.readline().strip()
                # at columns header
                if not self._is_fixed_param:
                    if self._save_warmup:
                        for i in range(self.num_draws_warmup):
                            line = fd.readline().strip()
                            xs = line.split(',')
                            self._draws[i, chain, :] = [float(x) for x in xs]
                    # read to adaptation msg
                    line = fd.readline().strip()
                    if line != '# Adaptation terminated':
                        while line != '# Adaptation terminated':
                            line = fd.readline().strip()
                    line = fd.readline().strip()  # step_size
                    _, step_size = line.split('=')
                    self._step_size[chain] = float(step_size.strip())
                    line = fd.readline().strip()  # metric header
                    # process metric
                    if self.metric_type == 'diag_e':
                        line = fd.readline().lstrip(' #\t').strip()
                        xs = line.split(',')
                        self._metric[chain, :] = [float(x) for x in xs]
                    else:
                        for i in range(self.num_unconstrained_params):
                            line = fd.readline().lstrip(' #\t').strip()
                            xs = line.split(',')
                            self._metric[chain, i, :] = [float(x) for x in xs]
                # process draws
                for i in range(sampling_iter_start, num_draws):
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
        tmp_csv_file = 'stansummary-{}-'.format(self.runset._args.model_name)
        tmp_csv_path = create_named_text_file(
            dir=_TMPDIR, prefix=tmp_csv_file, suffix='.csv', name_only=True
        )
        csv_str = '--csv_filename={}'.format(tmp_csv_path)
        if not cmdstan_version_at(2, 24):
            csv_str = '--csv_file={}'.format(tmp_csv_path)
        cmd = [
            cmd_path,
            percentiles_str,
            sig_figs_str,
            csv_str,
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
        Returns the sampler draws as a pandas DataFrame.  Flattens all
        chains into single column.

        :param params: optional list of variable names.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.
        """
        if inc_warmup and not self._save_warmup:
            self._logger.warning(
                'draws from warmup iterations not available,'
                ' must run sampler with "save_warmup=True".'
            )
        self._assemble_draws()
        mask = []
        if params is not None:
            for param in set(params):
                if (
                    param not in self.metadata.method_vars_cols
                    and param not in self.metadata.stan_vars_cols
                ):
                    raise ValueError('unknown parameter: {}'.format(param))
                if param in self.metadata.method_vars_cols:
                    mask.append(param)
                else:
                    for idx in self.metadata.stan_vars_cols[param]:
                        mask.append(self.column_names[idx])
        num_draws = self.num_draws_sampling
        if inc_warmup and self._save_warmup:
            num_draws += self.num_draws_warmup
        num_rows = num_draws * self.chains
        if self._draws_pd is None or self._draws_pd.shape[0] != num_rows:
            # pylint: disable=redundant-keyword-arg
            self._draws_pd = pd.DataFrame(
                data=flatten_chains(self.draws(inc_warmup=inc_warmup)),
                columns=self.column_names,
            )
        if params is None:
            return self._draws_pd
        return self._draws_pd[mask]

    def draws_xr(
        self, vars: List[str] = None, inc_warmup: bool = False
    ) -> "xr.Dataset":
        """
        Returns the sampler draws as a xarray Dataset.
        :param vars: optional list of variable names.
        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.
        """
        if not XARRAY_INSTALLED:
            raise RuntimeError(
                "xarray is not installed, cannot produce draws array"
            )
        if inc_warmup and not self._save_warmup:
            self._logger.warning(
                "draws from warmup iterations not available,"
                ' must run sampler with "save_warmup=True".'
            )

        if vars is None:
            vars = self._metadata.stan_vars_dims.keys()
        self._assemble_draws()

        num_draws = self.num_draws_sampling
        meta = self._metadata.cmdstan_config
        attrs = {
            "stan_version": f"{meta['stan_version_major']}."
            f"{meta['stan_version_minor']}.{meta['stan_version_patch']}",
            "model": meta["model"],
            "num_unconstrained_params": self.num_unconstrained_params,
            "num_draws_sampling": num_draws,
        }
        if inc_warmup and self._save_warmup:
            num_draws += self.num_draws_warmup
            attrs["num_draws_warmup"] = self.num_draws_warmup

        data = {}
        coordinates = {"chain": self.chain_ids, "draw": np.arange(num_draws)}
        for var in vars:
            start_row = 0
            if not inc_warmup and self._save_warmup:
                start_row = self.num_draws_warmup
            add_var_to_xarray(
                data,
                var,
                self._metadata.stan_vars_dims[var],
                self._metadata.stan_vars_cols[var],
                start_row,
                self._draws,
            )

        return xr.Dataset(data, coords=coordinates, attrs=attrs).transpose(
            'chain', 'draw', ...
        )

    def stan_variable(self, name: str, inc_warmup: bool = False) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the set of draws
        for the named Stan program variable.  Flattens the chains,
        leaving the draws in chain order.  The first array dimension,
        corresponds to number of draws or post-warmup draws in the sample,
        per argument ``inc_warmup``.  The remaining dimensions correspond to
        the shape of the Stan program variable.

        Underlyingly draws are in chain order, i.e., for a sample with
        N chains of M draws each, the first M array elements are from chain 1,
        the next M are from chain 2, and the last M elements are from chain N.

        * If the variable is a scalar variable, the return array has shape
          ( draws X chains, 1).
        * If the variable is a vector, the return array has shape
          ( draws X chains, len(vector))
        * If the variable is a matrix, the return array has shape
          ( draws X chains, size(dim 1) X size(dim 2) )
        * If the variable is an array with N dimensions, the return array
          has shape ( draws X chains, size(dim 1) X ... X size(dim N))

        For example, if the Stan program variable ``theta`` is a 3x3 matrix,
        and the sample consists of 4 chains with 1000 post-warmup draws,
        this function will return a numpy.ndarray with shape (4000,3,3).

        :param name: variable name

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.
        """
        if name not in self._metadata.stan_vars_dims:
            raise ValueError('unknown name: {}'.format(name))
        self._assemble_draws()
        draw1 = 0
        if not inc_warmup and self._save_warmup:
            draw1 = self.num_draws_warmup
        num_draws = self.num_draws_sampling
        if inc_warmup and self._save_warmup:
            num_draws += self.num_draws_warmup
        dims = [num_draws * self.chains]
        col_idxs = self._metadata.stan_vars_cols[name]
        if len(col_idxs) > 0:
            dims.extend(self._metadata.stan_vars_dims[name])
        # pylint: disable=redundant-keyword-arg
        return self._draws[draw1:, :, col_idxs].reshape(dims, order='F')

    def stan_variables(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.
        """
        result = {}
        for name in self._metadata.stan_vars_dims.keys():
            result[name] = self.stan_variable(name)
        return result

    def method_variables(self) -> Dict:
        """
        Returns a dictionary of all sampler variables, i.e., all
        output column names ending in `__`.  Assumes that all variables
        are scalar variables where column name is variable name.
        Maps each column name to a numpy.ndarray (draws x chains x 1)
        containing per-draw diagnostic values.
        """
        result = {}
        self._assemble_draws()
        for idxs in self.metadata.method_vars_cols.values():
            for idx in idxs:
                result[self.column_names[idx]] = self._draws[:, :, idx]
        return result

    def sampler_variables(self) -> Dict:
        """
        Deprecated, use "method_variables" instead
        """
        self._logger.warning(
            'method "sampler_variables" has been deprecated, '
            'use method "method_variables" instead.'
        )
        return self.method_variables()

    def sampler_diagnostics(self) -> Dict:
        """
        Deprecated, use "method_variables" instead
        """
        self._logger.warning(
            'method "sampler_diagnostics" has been deprecated, '
            'use method "method_variables" instead.'
        )
        return self.method_variables()

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
        self._metadata = None
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
        # TODO - profiling files
        return repr

    def _set_mle_attrs(self, sample_csv_0: str) -> None:
        meta = scan_optimize_csv(sample_csv_0)
        self._metadata = InferenceMetadata(meta)
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
    def metadata(self) -> InferenceMetadata:
        """
        Returns object which contains CmdStan configuration as well as
        information about the names and structure of the inference method
        and model output variables.
        """
        return self._metadata

    @property
    def optimized_params_np(self) -> np.ndarray:
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

    def stan_variable(self, name: str) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        :param name: variable name
        """
        if name not in self._metadata.stan_vars_dims:
            raise ValueError('unknown name: {}'.format(name))
        col_idxs = list(self._metadata.stan_vars_cols[name])
        vals = list(self._mle)
        xs = [vals[x] for x in col_idxs]
        shape = ()
        if len(col_idxs) > 0:
            shape = self._metadata.stan_vars_dims[name]
        return np.array(xs).reshape(shape)

    def stan_variables(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.
        """
        result = {}
        for name in self._metadata.stan_vars_dims.keys():
            result[name] = self.stan_variable(name)
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


class CmdStanGQ:
    """
    Container for outputs from CmdStan generate_quantities run.
    """

    def __init__(
        self,
        runset: RunSet,
        mcmc_sample: CmdStanMCMC,
        logger: logging.Logger = None,
    ) -> None:
        """Initialize object."""
        if not runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError(
                'Wrong runset method, expecting generate_quantities runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self._logger = logger or get_logger()
        self.mcmc_sample = mcmc_sample
        self._generated_quantities = None
        self._generated_quantities_pd = None
        config = self._validate_csv_files()
        self._metadata = InferenceMetadata(config)

    def __repr__(self) -> str:
        repr = 'CmdStanGQ: model={} chains={}{}'.format(
            self.runset.model,
            self.chains,
            self.runset._args.method_args.compose(0, cmd=[]),
        )
        repr = '{}\n csv_files:\n\t{}\n output_files:\n\t{}'.format(
            repr,
            '\n\t'.join(self.runset.csv_files),
            '\n\t'.join(self.runset.stdout_files),
        )
        return repr

    def _validate_csv_files(self) -> dict:
        """
        Checks that Stan CSV output files for all chains are consistent
        and returns dict containing config and column names.

        Raises exception when inconsistencies detected.
        """
        dzero = {}
        for i in range(self.chains):
            if i == 0:
                dzero = scan_generated_quantities_csv(
                    path=self.runset.csv_files[i],
                )
            else:
                drest = scan_generated_quantities_csv(
                    path=self.runset.csv_files[i],
                )
                for key in dzero:
                    if (
                        key
                        not in [
                            'id',
                            'fitted_params',
                            'diagnostic_file',
                            'metric_file',
                            'profile_file',
                            'init',
                            'seed',
                            'start_datetime',
                        ]
                        and dzero[key] != drest[key]
                    ):
                        raise ValueError(
                            'CmdStan config mismatch in Stan CSV file {}: '
                            'arg {} is {}, expected {}'.format(
                                self.runset.csv_files[i],
                                key,
                                dzero[key],
                                drest[key],
                            )
                        )
        return dzero

    @property
    def chains(self) -> int:
        """Number of chains."""
        return self.runset.chains

    @property
    def chain_ids(self) -> List[int]:
        """Chain ids."""
        return self.runset.chain_ids

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of generated quantities of interest.
        """
        return self._metadata.cmdstan_config['column_names']

    @property
    def metadata(self) -> InferenceMetadata:
        """
        Returns object which contains CmdStan configuration as well as
        information about the names and structure of the inference method
        and model output variables.
        """
        return self._metadata

    @property
    def generated_quantities(self) -> np.ndarray:
        """
        A 2D numpy ndarray which contains generated quantities draws
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
        return flatten_chains(self._generated_quantities)

    @property
    def generated_quantities_pd(self) -> pd.DataFrame:
        """
        Returns the generated quantities as a pandas DataFrame.  Flattens all
        chains into single column.
        """
        if self._generated_quantities is None:
            self._assemble_generated_quantities()
        if self._generated_quantities_pd is None:
            self._generated_quantities_pd = pd.DataFrame(
                data=flatten_chains(self._generated_quantities),
                columns=self.column_names,
            )
        return self._generated_quantities_pd

    def generated_quantities_xr(
        self, vars: List[str] = None, inc_warmup: bool = False
    ) -> "xr.Dataset":
        """
        Returns the generated quantities draws as a xarray Dataset.
        :param vars: optional list of variable names.
        :param inc_warmup: When ``True`` and the warmup draws are present in
            the MCMC sample, then the warmup draws are included.
            Default value is ``False``.
        """
        if not XARRAY_INSTALLED:
            raise RuntimeError(
                "xarray is not installed, cannot produce draws array"
            )
        if vars is None:
            vars = self.metadata.stan_vars_cols.keys()
        self._assemble_generated_quantities()

        num_draws = self.mcmc_sample.num_draws_sampling
        sample_config = self.mcmc_sample.metadata.cmdstan_config
        attrs = {
            "stan_version": f"{sample_config['stan_version_major']}."
            f"{sample_config['stan_version_minor']}."
            f"{sample_config['stan_version_patch']}",
            "model": sample_config["model"],
            "num_unconstrained_params":
            self.mcmc_sample.num_unconstrained_params,
            "num_draws_sampling": num_draws,
        }
        if inc_warmup and sample_config['save_warmup']:
            num_draws += self.mcmc_sample.num_draws_warmup
            attrs["num_draws_warmup"] = self.mcmc_sample.num_draws_warmup

        data = {}
        coordinates = {"chain": self.chain_ids, "draw": np.arange(num_draws)}
        for var in vars:
            start_row = 0
            if not inc_warmup and sample_config['save_warmup']:
                start_row = self.mcmc_sample.num_draws_warmup
            add_var_to_xarray(
                data,
                var,
                self._metadata.stan_vars_dims[var],
                self._metadata.stan_vars_cols[var],
                start_row,
                self._generated_quantities,
            )

        return xr.Dataset(data, coords=coordinates, attrs=attrs).transpose(
            'chain', 'draw', ...
        )

    @property
    def sample_plus_quantities(self) -> pd.DataFrame:
        """
        Deprecated - use method "sample_plus_quantities_pd" instead
        """
        self._logger.warning(
            'property "sample_plus_quantities" has been deprecated, '
            'use method "sample_plus_quantities_pd" instead.'
        )
        return self.sample_plus_quantities_pd()

    def sample_plus_quantities_pd(
        self, inc_warmup: bool = False
    ) -> pd.DataFrame:
        """
        Returns the column-wise concatenation of the input drawset
        with generated quantities drawset.  If there are duplicate
        columns in both the input and the generated quantities,
        the input column is dropped in favor of the recomputed
        values in the generate quantities drawset.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the MCMC sample, then the warmup draws are included.
            Default value is ``False``.
        """
        if not self.runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError('Bad runset method {}.'.format(self.runset.method))
        if self._generated_quantities is None:
            self._assemble_generated_quantities()
        cols_1 = self.mcmc_sample.column_names
        cols_2 = self.column_names
        dups = [
            item
            for item, count in Counter(cols_1 + cols_2).items()
            if count > 1
        ]
        if (
            self.mcmc_sample.metadata.cmdstan_config['save_warmup']
            and not inc_warmup
        ):
            draw1 = self.mcmc_sample.num_draws_warmup * self.chains
            return pd.concat(
                [
                    self.mcmc_sample.draws_pd().drop(columns=dups),
                    self.generated_quantities_pd[draw1:].reset_index(),
                ],
                axis='columns',
            )
        return pd.concat(
            [
                self.mcmc_sample.draws_pd(inc_warmup=inc_warmup).drop(
                    columns=dups
                ),
                self.generated_quantities_pd,
            ],
            axis='columns',
        )

    def sample_plus_quantities_xr(
        self, inc_warmup: bool = False
    ) -> "xr.Dataset":
        """
        Returns xarray object over variables in mcmc sample and
        generated quantitites.  De-duplicates variables in both drawsets,
        using values from generated quantities drawset.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the MCMC sample, then the warmup draws are included.
            Default value is ``False``.
        """
        if not XARRAY_INSTALLED:
            raise RuntimeError(
                "xarray is not installed, cannot produce draws array"
            )

        self._assemble_generated_quantities()
        num_draws = self.mcmc_sample.num_draws_sampling
        sample_config = self.mcmc_sample.metadata.cmdstan_config
        attrs = {
            "stan_version": f"{sample_config['stan_version_major']}."
            f"{sample_config['stan_version_minor']}."
            f"{sample_config['stan_version_patch']}",
            "model": sample_config["model"],
            "num_unconstrained_params":
            self.mcmc_sample.num_unconstrained_params,
            "num_draws_sampling": num_draws,
        }
        if inc_warmup and sample_config['save_warmup']:
            num_draws += self.mcmc_sample.num_draws_warmup
            attrs["num_draws_warmup"] = self.mcmc_sample.num_draws_warmup

        data = {}
        coordinates = {"chain": self.chain_ids, "draw": np.arange(num_draws)}

        start_row = 0
        if not inc_warmup and sample_config['save_warmup']:
            start_row = self.mcmc_sample.num_draws_warmup
        for var in self._metadata.stan_vars_cols.keys():
            add_var_to_xarray(
                data,
                var,
                self._metadata.stan_vars_dims[var],
                self._metadata.stan_vars_cols[var],
                start_row,
                self._generated_quantities,
            )
        mcmc_vars = [
            x
            for x in self.mcmc_sample.metadata.stan_vars_cols.keys()
            if x not in self._metadata.stan_vars_cols.keys()
        ]
        start_row = 0
        for var in mcmc_vars:
            add_var_to_xarray(
                data,
                var,
                self.mcmc_sample.metadata.stan_vars_dims[var],
                self.mcmc_sample.metadata.stan_vars_cols[var],
                start_row,
                self.mcmc_sample.draws(
                    inc_warmup=inc_warmup, concat_chains=False
                ),
            )
        return xr.Dataset(data, coords=coordinates, attrs=attrs).transpose(
            'chain', 'draw', ...
        )

    def stan_variable(self, name: str, inc_warmup: bool = False) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the set of draws
        for the named Stan program variable.  Flattens the chains,
        leaving the draws in chain order.  The first array dimension,
        corresponds to number of draws in the sample.
        The remaining dimensions correspond to
        the shape of the Stan program variable.

        Underlyingly draws are in chain order, i.e., for a sample with
        N chains of M draws each, the first M array elements are from chain 1,
        the next M are from chain 2, and the last M elements are from chain N.

        * If the variable is a scalar variable, the return array has shape
          ( draws X chains, 1).
        * If the variable is a vector, the return array has shape
          ( draws X chains, len(vector))
        * If the variable is a matrix, the return array has shape
          ( draws X chains, size(dim 1) X size(dim 2) )
        * If the variable is an array with N dimensions, the return array
          has shape ( draws X chains, size(dim 1) X ... X size(dim N))

        For example, if the Stan program variable ``theta`` is a 3x3 matrix,
        and the sample consists of 4 chains with 1000 post-warmup draws,
        this function will return a numpy.ndarray with shape (4000,3,3).

        :param name: variable name

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the MCMC sample, then the warmup draws are included.
            Default value is ``False``.
        """
        model_var_names = self.mcmc_sample.metadata.stan_vars_cols.keys()
        gq_var_names = self.metadata.stan_vars_cols.keys()
        if not (name in model_var_names or name in gq_var_names):
            raise ValueError('unknown name: {}'.format(name))
        if name not in gq_var_names:
            return self.mcmc_sample.stan_variable(name, inc_warmup=inc_warmup)
        else:  # is gq variable
            self._assemble_generated_quantities()
            col_idxs = self._metadata.stan_vars_cols[name]
            if (
                not inc_warmup
                and self.mcmc_sample.metadata.cmdstan_config['save_warmup']
            ):
                draw1 = self.mcmc_sample.num_draws_warmup * self.chains
                return flatten_chains(self._generated_quantities)[
                    draw1:, col_idxs
                ]
            return flatten_chains(self._generated_quantities)[:, col_idxs]

    def stan_variables(self, inc_warmup: bool = False) -> Dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.
        """
        result = {}
        sample_var_names = self.mcmc_sample.metadata.stan_vars_cols.keys()
        gq_var_names = self.metadata.stan_vars_cols.keys()
        for name in gq_var_names:
            result[name] = self.stan_variable(name, inc_warmup)
        for name in sample_var_names:
            if name not in gq_var_names:
                result[name] = self.stan_variable(name, inc_warmup)
        return result

    def _assemble_generated_quantities(self) -> None:
        # use numpy genfromtext
        warmup = self.mcmc_sample.metadata.cmdstan_config['save_warmup']
        num_draws = self.mcmc_sample.draws(inc_warmup=warmup).shape[0]
        gq_sample = np.empty(
            (num_draws, self.chains, len(self.column_names)),
            dtype=float,
            order='F',
        )
        for chain in range(self.chains):
            with open(self.runset.csv_files[chain], 'r') as fd:
                lines = (line for line in fd if not line.startswith('#'))
                gq_sample[:, chain, :] = np.loadtxt(
                    lines, dtype=np.ndarray, ndmin=2, skiprows=1, delimiter=','
                )
        self._generated_quantities = gq_sample

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
        self._metadata = None
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
        # TODO - diagnostic, profiling files
        return repr

    def _set_variational_attrs(self, sample_csv_0: str) -> None:
        meta = scan_variational_csv(sample_csv_0)
        self._metadata = InferenceMetadata(meta)
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
    def variational_params_np(self) -> np.ndarray:
        """
        Returns inferred parameter means as numpy array.
        """
        return self._variational_mean

    @property
    def variational_params_pd(self) -> pd.DataFrame:
        """
        Returns inferred parameter means as pandas DataFrame.
        """
        return pd.DataFrame([self._variational_mean], columns=self.column_names)

    @property
    def variational_params_dict(self) -> OrderedDict:
        """
        Returns inferred parameter means as Dict.
        """
        return OrderedDict(zip(self.column_names, self._variational_mean))

    @property
    def metadata(self) -> InferenceMetadata:
        """
        Returns object which contains CmdStan configuration as well as
        information about the names and structure of the inference method
        and model output variables.
        """
        return self._metadata

    def stan_variable(self, name: str) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        :param name: variable name
        """
        if name not in self._metadata.stan_vars_dims:
            raise ValueError('unknown name: {}'.format(name))
        col_idxs = list(self._metadata.stan_vars_cols[name])
        vals = list(self._variational_mean)
        xs = [vals[x] for x in col_idxs]
        shape = ()
        if len(col_idxs) > 0:
            shape = self._metadata.stan_vars_dims[name]
        return np.array(xs).reshape(shape)

    def stan_variables(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.
        """
        result = {}
        for name in self._metadata.stan_vars_dims.keys():
            result[name] = self.stan_variable(name)
        return result

    @property
    def variational_sample(self) -> np.ndarray:
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


def from_csv(
    path: Union[str, List[str]] = None, method: str = None
) -> Union[CmdStanMCMC, CmdStanMLE, CmdStanVB]:
    """
    Instantiate a CmdStan object from a the Stan CSV files from a CmdStan run.
    CSV files are specified from either a list of Stan CSV files or a single
    filepath which can be either a directory name, a Stan CSV filename, or
    a pathname pattern (i.e., a Python glob).  The optional argument 'method'
    checks that the CSV files were produced by that method.
    Stan CSV files from CmdStan methods 'sample', 'optimize', and 'variational'
    result in objects of class CmdStanMCMC, CmdStanMLE, and CmdStanVB,
    respectively.

    :param path: directory path
    :param method: method name (optional)

    :return: either a CmdStanMCMC, CmdStanMLE, or CmdStanVB object
    """
    if path is None:
        raise ValueError('Must specify path to Stan CSV files.')
    if method is not None and method not in [
        'sample',
        'optimize',
        'variational',
    ]:
        raise ValueError(
            'Bad method argument {}, must be one of: '
            '"sample", "optimize", "variational"'.format(method)
        )

    csvfiles = []
    if isinstance(path, list):
        csvfiles = path
    elif isinstance(path, str):
        if '*' in path:
            splits = os.path.split(path)
            if splits[0] is not None:
                if not (os.path.exists(splits[0]) and os.path.isdir(splits[0])):
                    raise ValueError(
                        'Invalid path specification, {} '
                        ' unknown directory: {}'.format(path, splits[0])
                    )
            csvfiles = glob.glob(path)
        elif os.path.exists(path) and os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(".csv"):
                    csvfiles.append(os.path.join(path, file))
        elif os.path.exists(path):
            csvfiles.append(path)
        else:
            raise ValueError('Invalid path specification: {}'.format(path))
    else:
        raise ValueError('Invalid path specification: {}'.format(path))

    if len(csvfiles) == 0:
        raise ValueError('No CSV files found in directory {}'.format(path))
    for file in csvfiles:
        if not (os.path.exists(file) and file.endswith('.csv')):
            raise ValueError(
                'Bad CSV file path spec,'
                ' includes non-csv file: {}'.format(file)
            )

    config_dict = {}
    try:
        with open(csvfiles[0], 'r') as fd:
            scan_config(fd, config_dict, 0)
    except (IOError, OSError, PermissionError) as e:
        raise ValueError('Cannot read CSV file: {}'.format(csvfiles[0])) from e
    if 'model' not in config_dict or 'method' not in config_dict:
        raise ValueError("File {} is not a Stan CSV file.".format(csvfiles[0]))
    if method is not None and method != config_dict['method']:
        raise ValueError(
            'Expecting Stan CSV output files from method {}, '
            ' found outputs from method {}'.format(
                method, config_dict['method']
            )
        )
    fit = None
    try:
        if config_dict['method'] == 'sample':
            chains = len(csvfiles)
            sampler_args = SamplerArgs(
                iter_sampling=config_dict['num_samples'],
                iter_warmup=config_dict['num_warmup'],
                thin=config_dict['thin'],
                save_warmup=config_dict['save_warmup'],
            )
            cmdstan_args = CmdStanArgs(
                model_name=config_dict['model'],
                model_exe=config_dict['model'],
                chain_ids=[x + 1 for x in range(chains)],
                method_args=sampler_args,
            )
            runset = RunSet(args=cmdstan_args, chains=chains)
            runset._csv_files = csvfiles
            for i in range(len(runset._retcodes)):
                runset._set_retcode(i, 0)
            fit = CmdStanMCMC(runset)
            fit.draws()
        elif config_dict['method'] == 'optimize':
            if 'algorithm' not in config_dict:
                raise ValueError(
                    "Cannot find optimization algorithm"
                    " in file {}.".format(csvfiles[0])
                )
            optimize_args = OptimizeArgs(
                algorithm=config_dict['algorithm'],
            )
            cmdstan_args = CmdStanArgs(
                model_name=config_dict['model'],
                model_exe=config_dict['model'],
                chain_ids=None,
                method_args=optimize_args,
            )
            runset = RunSet(args=cmdstan_args)
            runset._csv_files = csvfiles
            for i in range(len(runset._retcodes)):
                runset._set_retcode(i, 0)
            fit = CmdStanMLE(runset)
        elif config_dict['method'] == 'variational':
            if 'algorithm' not in config_dict:
                raise ValueError(
                    "Cannot find variational algorithm"
                    " in file {}.".format(csvfiles[0])
                )
            variational_args = VariationalArgs(
                algorithm=config_dict['algorithm'],
                iter=config_dict['iter'],
                grad_samples=config_dict['grad_samples'],
                elbo_samples=config_dict['elbo_samples'],
                eta=config_dict['eta'],
                tol_rel_obj=config_dict['tol_rel_obj'],
                eval_elbo=config_dict['eval_elbo'],
                output_samples=config_dict['output_samples'],
            )
            cmdstan_args = CmdStanArgs(
                model_name=config_dict['model'],
                model_exe=config_dict['model'],
                chain_ids=None,
                method_args=variational_args,
            )
            runset = RunSet(args=cmdstan_args)
            runset._csv_files = csvfiles
            for i in range(len(runset._retcodes)):
                runset._set_retcode(i, 0)
            fit = CmdStanVB(runset)
        else:
            get_logger().info(
                'Unable to process CSV output files from method %s.',
                (config_dict['method']),
            )
    except (IOError, OSError, PermissionError) as e:
        raise ValueError(
            'An error occured processing the CSV files:\n\t{}'.format(str(e))
        ) from e
    return fit


def add_var_to_xarray(
    data: Dict[str, xr.DataArray],
    var_name: str,
    dims: Tuple[int, ...],
    col_idxs: Tuple[int, ...],
    start_row: int,
    drawset: np.ndarray,
) -> None:
    """
    Adds Stan variable values and labels to an xarray DataArray
    """
    var_dims = ('draw', 'chain')
    if dims:
        var_dims = ("draw", "chain") + tuple(
            f"{var_name}_dim_{i}" for i in range(len(dims))
        )
        data[var_name] = (var_dims, drawset[start_row:, :, col_idxs])
    else:
        data[var_name] = (
            var_dims,
            np.squeeze(drawset[start_row:, :, col_idxs], axis=2),
        )
