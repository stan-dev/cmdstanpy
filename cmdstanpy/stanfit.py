"""Container objects for results of CmdStan run(s)."""

import copy
import glob
import math
import os
import re
import shutil
from collections import Counter, OrderedDict
from datetime import datetime
from io import StringIO
from time import time
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

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
    cmdstan_version_before,
    create_named_text_file,
    do_command,
    flatten_chains,
    get_logger,
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

    RunSet objects are instantiated by the CmdStanModel class inference methods
    which validate all inputs, therefore "__init__" method skips input checks.
    """

    def __init__(
        self,
        args: CmdStanArgs,
        chains: int = 1,
        *,
        chain_ids: Optional[List[int]] = None,
        time_fmt: str = "%Y%m%d%H%M%S",
        one_process_per_chain: bool = True,
    ) -> None:
        """Initialize object (no input arg checks)."""
        self._args = args
        self._chains = chains
        self._one_process_per_chain = one_process_per_chain
        if one_process_per_chain:
            self._num_procs = chains
        else:
            self._num_procs = 1
        self._retcodes = [-1 for _ in range(self._num_procs)]
        if chain_ids is None:
            chain_ids = [i + 1 for i in range(chains)]
        self._chain_ids = chain_ids

        if args.output_dir is not None:
            self._output_dir = args.output_dir
        else:
            self._output_dir = _TMPDIR

        # output files prefix: ``<model_name>-<YYYYMMDDHHMM>_<chain_id>``
        self._base_outfile = (
            f'{args.model_name}-{datetime.now().strftime(time_fmt)}'
        )
        # per-process console messages
        self._stdout_files = [''] * self._num_procs
        if one_process_per_chain:
            for i in range(chains):
                self._stdout_files[i] = self.file_path("-stdout.txt", id=i)
        else:
            self._stdout_files[0] = self.file_path("-stdout.txt")

        # per-chain output files
        self._csv_files = [''] * chains
        self._diagnostic_files = [''] * chains  # optional
        self._profile_files = [''] * chains  # optional

        if chains == 1:
            self._csv_files[0] = self.file_path(".csv")
            if args.save_latent_dynamics:
                self._diagnostic_files[0] = self.file_path(
                    ".csv", extra="-diagnostic"
                )
            if args.save_profile:
                self._profile_files[0] = self.file_path(
                    ".csv", extra="-profile"
                )
        else:
            for i in range(chains):
                self._csv_files[i] = self.file_path(".csv", id=chain_ids[i])
                if args.save_latent_dynamics:
                    self._diagnostic_files[i] = self.file_path(
                        ".csv", extra="-diagnostic", id=chain_ids[i]
                    )
                if args.save_profile:
                    self._profile_files[i] = self.file_path(
                        ".csv", extra="-profile", id=chain_ids[i]
                    )

    def __repr__(self) -> str:
        repr = 'RunSet: chains={}, chain_ids={}, num_processes={}'.format(
            self._chains, self._chain_ids, self._num_procs
        )
        repr = '{}\n cmd (chain 1):\n\t{}'.format(repr, self.cmd(0))
        repr = '{}\n retcodes={}'.format(repr, self._retcodes)
        repr = f'{repr}\n per-chain output files (showing chain 1 only):'
        repr = '{}\n csv_file:\n\t{}'.format(repr, self._csv_files[0])
        if self._args.save_latent_dynamics:
            repr = '{}\n diagnostics_file:\n\t{}'.format(
                repr, self._diagnostic_files[0]
            )
        if self._args.save_profile:
            repr = '{}\n profile_file:\n\t{}'.format(
                repr, self._profile_files[0]
            )
        repr = '{}\n console_msgs (if any):\n\t{}'.format(
            repr, self._stdout_files[0]
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
    def num_procs(self) -> int:
        """Number of processes run."""
        return self._num_procs

    @property
    def one_process_per_chain(self) -> bool:
        """
        When True, for each chain, call CmdStan in its own subprocess.
        When False, use CmdStan's `num_chains` arg to run parallel chains.
        Always True if CmdStan < 2.28.
        For CmdStan 2.28 and up, `sample` method determines value.
        """
        return self._one_process_per_chain

    @property
    def chains(self) -> int:
        """Number of chains."""
        return self._chains

    @property
    def chain_ids(self) -> List[int]:
        """Chain ids."""
        return self._chain_ids

    def cmd(self, idx: int) -> List[str]:
        """
        Assemble CmdStan invocation.
        When running parallel chains from single process (2.28 and up),
        specify CmdStan arg `num_chains` and leave chain idx off CSV files.
        """
        if self._one_process_per_chain:
            return self._args.compose_command(
                idx,
                csv_file=self.csv_files[idx],
                diagnostic_file=self.diagnostic_files[idx]
                if self._args.save_latent_dynamics
                else None,
                profile_file=self.profile_files[idx]
                if self._args.save_profile
                else None,
            )
        else:
            return self._args.compose_command(
                idx,
                csv_file=self.file_path('.csv'),
                diagnostic_file=self.file_path(".csv", extra="-diagnostic")
                if self._args.save_latent_dynamics
                else None,
                profile_file=self.file_path(".csv", extra="-profile")
                if self._args.save_profile
                else None,
                num_chains=self._chains,
            )

    @property
    def csv_files(self) -> List[str]:
        """List of paths to CmdStan output files."""
        return self._csv_files

    @property
    def stdout_files(self) -> List[str]:
        """
        List of paths to transcript of CmdStan messages sent to the console.
        Transcripts include config information, progress, and error messages.
        """
        return self._stdout_files

    def _check_retcodes(self) -> bool:
        """Returns ``True`` when all chains have retcode 0."""
        for code in self._retcodes:
            if code != 0:
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

    # pylint: disable=invalid-name
    def file_path(
        self, suffix: str, *, extra: str = "", id: Optional[int] = None
    ) -> str:
        if id is not None:
            suffix = f"_{id}{suffix}"
        file = os.path.join(
            self._output_dir, f"{self._base_outfile}{extra}{suffix}"
        )
        return file

    def _retcode(self, idx: int) -> int:
        """Get retcode for process[idx]."""
        return self._retcodes[idx]

    def _set_retcode(self, idx: int, val: int) -> None:
        """Set retcode at process[idx] to val."""
        self._retcodes[idx] = val

    def get_err_msgs(self) -> str:
        """Checks console messages for each CmdStan run."""
        msgs = []
        for i in range(self._num_procs):
            if (
                os.path.exists(self._stdout_files[i])
                and os.stat(self._stdout_files[i]).st_size > 0
            ):
                if self._args.method == Method.OPTIMIZE:
                    msgs.append('console log output:\n')
                    with open(self._stdout_files[0], 'r') as fd:
                        msgs.append(fd.read())
                else:
                    with open(self._stdout_files[i], 'r') as fd:
                        contents = fd.read()
                        # pattern matches initial "Exception" or "Error" msg
                        pat = re.compile(r'^E[rx].*$', re.M)
                        errors = re.findall(pat, contents)
                        if len(errors) > 0:
                            msgs.append('\n\t'.join(errors))
        return '\n'.join(msgs)

    def save_csvfiles(self, dir: Optional[str] = None) -> None:
        """
        Moves CSV files to specified directory.

        :param dir: directory path

        See Also
        --------
        cmdstanpy.from_csv
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
            raise Exception('Cannot save to path: {}'.format(dir)) from exc

        for i in range(self.chains):
            if not os.path.exists(self._csv_files[i]):
                raise ValueError(
                    'Cannot access CSV file {}'.format(self._csv_files[i])
                )

            to_path = os.path.join(dir, os.path.basename(self._csv_files[i]))
            if os.path.exists(to_path):
                raise ValueError(
                    'File exists, not overwriting: {}'.format(to_path)
                )
            try:
                get_logger().debug(
                    'saving tmpfile: "%s" as: "%s"', self._csv_files[i], to_path
                )
                shutil.move(self._csv_files[i], to_path)
                self._csv_files[i] = to_path
            except (IOError, OSError, PermissionError) as e:
                raise ValueError(
                    'Cannot save to file: {}'.format(to_path)
                ) from e


class InferenceMetadata:
    """
    CmdStan configuration and contents of output file parsed out of
    the Stan CSV file header comments and column headers.
    Assumes valid CSV files.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
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
    def cmdstan_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing a set of name, value pairs
        parsed out of the Stan CSV file header.  These include the
        command configuration and the CSV file header row information.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._cmdstan_config)

    @property
    def method_vars_cols(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns a map from a Stan inference method variable to
        a tuple of column indices in inference engine's output array.
        Method variable names always end in `__`, e.g. `lp__`.
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
        program variable ``int foo`` has dimension ``()`` and
        program variable ``vector[10] bar`` has single dimension ``(10)``.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._stan_vars_dims)


class CmdStanMCMC:
    """
    Container for outputs from CmdStan sampler run.
    Provides methods to summarize and diagnose the model fit
    and accessor methods to access the entire sample or
    individual items. Created by :meth:`CmdStanModel.sample`

    The sample is lazily instantiated on first access of either
    the resulting sample or the HMC tuning parameters, i.e., the
    step size and metric.
    """

    # pylint: disable=too-many-public-methods
    def __init__(
        self,
        runset: RunSet,
    ) -> None:
        """Initialize object."""
        if not runset.method == Method.SAMPLE:
            raise ValueError(
                'Wrong runset method, expecting sample runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset

        # info from runset to be exposed
        sampler_args = self.runset._args.method_args
        assert isinstance(
            sampler_args, SamplerArgs
        )  # make the typechecker happy
        iter_sampling = sampler_args.iter_sampling
        if iter_sampling is None:
            self._iter_sampling = _CMDSTAN_SAMPLING
        else:
            self._iter_sampling = iter_sampling
        iter_warmup = sampler_args.iter_warmup
        if iter_warmup is None:
            self._iter_warmup = _CMDSTAN_WARMUP
        else:
            self._iter_warmup = iter_warmup
        thin = sampler_args.thin
        if thin is None:
            self._thin: int = _CMDSTAN_THIN
        else:
            self._thin = thin
        self._is_fixed_param = sampler_args.fixed_param
        self._save_warmup = sampler_args.save_warmup
        self._sig_figs = runset._args.sig_figs
        # info from CSV values, instantiated lazily
        self._metric = np.array(())
        self._step_size = np.array(())
        self._draws = np.array(())
        # info from CSV initial comments and header
        config = self._validate_csv_files()
        self._metadata: InferenceMetadata = InferenceMetadata(config)

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
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of all outputs from the sampler, comprising sampler parameters
        and all components of all model parameters, transformed parameters,
        and quantities of interest. Corresponds to Stan CSV file header row,
        with names munged to array notation, e.g. `beta[1]` not `beta.1`.
        """
        return self._metadata.cmdstan_config['column_names']  # type: ignore

    @property
    def metric_type(self) -> Optional[str]:
        """
        Metric type used for adaptation, either 'diag_e' or 'dense_e'.
        When sampler algorithm 'fixed_param' is specified, metric_type is None.
        """
        if self._is_fixed_param:
            return None
        # cmdstan arg name
        return self._metadata.cmdstan_config['metric']  # type: ignore

    @property
    def metric(self) -> Optional[np.ndarray]:
        """
        Metric used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, metric is None.
        """
        if self._is_fixed_param:
            return None
        if self._metadata.cmdstan_config['metric'] == 'unit_e':
            get_logger().info(
                'Unit diagnonal metric, inverse mass matrix size unknown.'
            )
            return None
        if self._draws.shape == (0,):
            self._assemble_draws()
        return self._metric

    @property
    def step_size(self) -> Optional[np.ndarray]:
        """
        Step size used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, step size is None.
        """
        if self._is_fixed_param:
            return None
        if self._step_size.shape == (0,):
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
        chains are flattened into a single column, preserving chain order,
        so that given M chains of N draws, the first N draws are from chain 1,
        up through the last N draws from chain M.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        :param concat_chains: When ``True`` return a 2D array flattening all
            all draws from all chains.  Default value is ``False``.

        See Also
        --------
        CmdStanMCMC.draws_pd
        CmdStanMCMC.draws_xr
        CmdStanGQ.draws
        """
        if self._draws.size == 0:
            self._assemble_draws()

        if inc_warmup and not self._save_warmup:
            get_logger().warning(
                "Sample doesn't contain draws from warmup iterations,"
                ' rerun sampler with "save_warmup=True".'
            )

        start_idx = 0
        if not inc_warmup and self._save_warmup:
            start_idx = self.num_draws_warmup

        if concat_chains:
            return flatten_chains(self._draws[start_idx:, :, :])
        return self._draws[start_idx:, :, :]  # type: ignore

    def _validate_csv_files(self) -> Dict[str, Any]:
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
                    # check args that matter for parsing, plus name, version
                    if (
                        key
                        in [
                            'stan_version_major',
                            'stan_version_minor',
                            'stan_version_patch',
                            'stanc_version',
                            'model',
                            'num_samples',
                            'num_warmup',
                            'save_warmup',
                            'thin',
                            'refresh',
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
        if self._draws.shape != (0,):
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
        self._step_size = np.empty(self.chains, dtype=float)
        for chain in range(self.chains):
            with open(self.runset.csv_files[chain], 'r') as fd:
                line = fd.readline().strip()
                # read initial comments, CSV header row
                while len(line) > 0 and line.startswith('#'):
                    line = fd.readline().strip()
                if not self._is_fixed_param:
                    # handle warmup draws, if any
                    if self._save_warmup:
                        for i in range(self.num_draws_warmup):
                            line = fd.readline().strip()
                            xs = line.split(',')
                            self._draws[i, chain, :] = [float(x) for x in xs]
                    line = fd.readline().strip()
                    if line != '# Adaptation terminated':  # shouldn't happen?
                        while line != '# Adaptation terminated':
                            line = fd.readline().strip()
                    # step_size, metric (diag_e and dense_e only)
                    line = fd.readline().strip()
                    _, step_size = line.split('=')
                    self._step_size[chain] = float(step_size.strip())
                    if self._metadata.cmdstan_config['metric'] != 'unit_e':
                        line = fd.readline().strip()  # metric type
                        line = fd.readline().lstrip(' #\t')
                        num_unconstrained_params = len(line.split(','))
                        if chain == 0:  # can't allocate w/o num params
                            if self.metric_type == 'diag_e':
                                self._metric = np.empty(
                                    (self.chains, num_unconstrained_params),
                                    dtype=float,
                                )
                            else:
                                self._metric = np.empty(
                                    (
                                        self.chains,
                                        num_unconstrained_params,
                                        num_unconstrained_params,
                                    ),
                                    dtype=float,
                                )
                        if self.metric_type == 'diag_e':
                            xs = line.split(',')
                            self._metric[chain, :] = [float(x) for x in xs]
                        else:
                            xs = line.split(',')
                            self._metric[chain, 0, :] = [float(x) for x in xs]
                            for i in range(1, num_unconstrained_params):
                                line = fd.readline().lstrip(' #\t').strip()
                                xs = line.split(',')
                                self._metric[chain, i, :] = [
                                    float(x) for x in xs
                                ]
                # process draws
                for i in range(sampling_iter_start, num_draws):
                    line = fd.readline().strip()
                    xs = line.split(',')
                    self._draws[i, chain, :] = [float(x) for x in xs]
        assert self._draws is not None

    def summary(
        self,
        percentiles: Optional[List[int]] = None,
        sig_figs: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run cmdstan/bin/stansummary over all output CSV files, assemble
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
                    'Invalid percentiles argument, must be ordered'
                    ' non-empty list from (1, 99), inclusive.'
                )
            cur_pct = 0
            for pct in percentiles:
                if pct > 99 or not pct > cur_pct:
                    raise ValueError(
                        'Invalid percentiles spec, must be ordered'
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
                    'Keyword "sig_figs" must be an integer between 1 and 18,'
                    ' found {}'.format(sig_figs)
                )
            csv_sig_figs = self._sig_figs or 6
            if sig_figs > csv_sig_figs:
                get_logger().warning(
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
        # TODO: remove at some future release
        if cmdstan_version_before(2, 24):
            csv_str = '--csv_file={}'.format(tmp_csv_path)
        cmd = [
            cmd_path,
            percentiles_str,
            sig_figs_str,
            csv_str,
        ] + self.runset.csv_files
        do_command(cmd, fd_out=None)
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

    def diagnose(self) -> Optional[str]:
        """
        Run cmdstan/bin/diagnose over all output CSV files,
        return console output.

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
        result = StringIO()
        do_command(cmd=cmd, fd_out=result)
        return result.getvalue()

    def draws_pd(
        self,
        vars: Union[List[str], str, None] = None,
        inc_warmup: bool = False,
    ) -> pd.DataFrame:
        """
        Returns the sample draws as a pandas DataFrame.
        Flattens all chains into single column.  Container variables
        (array, vector, matrix) will span multiple columns, one column
        per element. E.g. variable 'matrix[2,2] foo' spans 4 columns:
        'foo[1,1], ... foo[2,2]'.

        :param vars: optional list of variable names.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        See Also
        --------
        CmdStanMCMC.draws
        CmdStanMCMC.draws_xr
        CmdStanGQ.draws_pd
        """
        if vars is not None:
            if isinstance(vars, str):
                vars_list = [vars]
            else:
                vars_list = vars

        if inc_warmup and not self._save_warmup:
            get_logger().warning(
                'Draws from warmup iterations not available,'
                ' must run sampler with "save_warmup=True".'
            )

        self._assemble_draws()
        cols = []
        if vars is not None:
            for var in set(vars_list):
                if (
                    var not in self.metadata.method_vars_cols
                    and var not in self.metadata.stan_vars_cols
                ):
                    raise ValueError('Unknown variable: {}'.format(var))
                if var in self.metadata.method_vars_cols:
                    cols.append(var)
                else:
                    for idx in self.metadata.stan_vars_cols[var]:
                        cols.append(self.column_names[idx])
        else:
            cols = list(self.column_names)

        return pd.DataFrame(
            data=flatten_chains(self.draws(inc_warmup=inc_warmup)),
            columns=self.column_names,
        )[cols]

    def draws_xr(
        self, vars: Union[str, List[str], None] = None, inc_warmup: bool = False
    ) -> "xr.Dataset":
        """
        Returns the sampler draws as a xarray Dataset.

        :param vars: optional list of variable names.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        See Also
        --------
        CmdStanMCMC.draws
        CmdStanMCMC.draws_pd
        CmdStanGQ.draws_xr
        """
        if not XARRAY_INSTALLED:
            raise RuntimeError(
                'Package "xarray" is not installed, cannot produce draws array.'
            )
        if inc_warmup and not self._save_warmup:
            get_logger().warning(
                "Draws from warmup iterations not available,"
                ' must run sampler with "save_warmup=True".'
            )
        if vars is None:
            vars_list = list(self.metadata.stan_vars_cols.keys())
        elif isinstance(vars, str):
            vars_list = [vars]
        else:
            vars_list = vars

        self._assemble_draws()

        num_draws = self.num_draws_sampling
        meta = self._metadata.cmdstan_config
        attrs: MutableMapping[Hashable, Any] = {
            "stan_version": f"{meta['stan_version_major']}."
            f"{meta['stan_version_minor']}.{meta['stan_version_patch']}",
            "model": meta["model"],
            "num_draws_sampling": num_draws,
        }
        if inc_warmup and self._save_warmup:
            num_draws += self.num_draws_warmup
            attrs["num_draws_warmup"] = self.num_draws_warmup

        data: MutableMapping[Hashable, Any] = {}
        coordinates: MutableMapping[Hashable, Any] = {
            "chain": self.chain_ids,
            "draw": np.arange(num_draws),
        }

        for var in vars_list:
            build_xarray_data(
                data,
                var,
                self._metadata.stan_vars_dims[var],
                self._metadata.stan_vars_cols[var],
                0,
                self.draws(inc_warmup=inc_warmup),
            )
        return xr.Dataset(data, coords=coordinates, attrs=attrs).transpose(
            'chain', 'draw', ...
        )

    def stan_variable(
        self,
        var: Optional[str] = None,
        inc_warmup: bool = False,
    ) -> np.ndarray:
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

        :param var: variable name

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        See Also
        --------
        CmdStanMCMC.stan_variables
        CmdStanMLE.stan_variable
        CmdStanVB.stan_variable
        CmdStanGQ.stan_variable
        """
        if var is None:
            raise ValueError('No variable name specified.')
        if var not in self._metadata.stan_vars_dims:
            raise ValueError('Unknown variable name: {}'.format(var))
        self._assemble_draws()
        draw1 = 0
        if not inc_warmup and self._save_warmup:
            draw1 = self.num_draws_warmup
        num_draws = self.num_draws_sampling
        if inc_warmup and self._save_warmup:
            num_draws += self.num_draws_warmup
        dims = [num_draws * self.chains]
        col_idxs = self._metadata.stan_vars_cols[var]
        if len(col_idxs) > 0:
            dims.extend(self._metadata.stan_vars_dims[var])
        # pylint: disable=redundant-keyword-arg
        return self._draws[draw1:, :, col_idxs].reshape(  # type: ignore
            dims, order='F'
        )

    def stan_variables(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        See Also
        --------
        CmdStanMCMC.stan_variable
        CmdStanMLE.stan_variables
        CmdStanVB.stan_variables
        CmdStanGQ.stan_variables
        """
        result = {}
        for name in self._metadata.stan_vars_dims.keys():
            result[name] = self.stan_variable(name)
        return result

    def method_variables(self) -> Dict[str, np.ndarray]:
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

    def save_csvfiles(self, dir: Optional[str] = None) -> None:
        """
        Move output CSV files to specified directory.  If files were
        written to the temporary session directory, clean filename.
        E.g., save 'bernoulli-201912081451-1-5nm6as7u.csv' as
        'bernoulli-201912081451-1.csv'.

        :param dir: directory path

        See Also
        --------
        stanfit.RunSet.save_csvfiles
        cmdstanpy.from_csv
        """
        self.runset.save_csvfiles(dir)


class CmdStanMLE:
    """
    Container for outputs from CmdStan optimization.
    Created by :meth:`CmdStanModel.optimize`.
    """

    def __init__(self, runset: RunSet) -> None:
        """Initialize object."""
        if not runset.method == Method.OPTIMIZE:
            raise ValueError(
                'Wrong runset method, expecting optimize runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        # info from runset to be exposed
        self.converged = runset._check_retcodes()
        optimize_args = self.runset._args.method_args
        assert isinstance(
            optimize_args, OptimizeArgs
        )  # make the typechecker happy
        self._save_iterations = optimize_args.save_iterations
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
        if not self.converged:
            repr = '{}\n Warning: invalid estimate, '.format(repr)
            repr = '{} optimization failed to converge.'.format(repr)
        return repr

    def _set_mle_attrs(self, sample_csv_0: str) -> None:
        meta = scan_optimize_csv(sample_csv_0, self._save_iterations)
        self._metadata = InferenceMetadata(meta)
        self._column_names: Tuple[str, ...] = meta['column_names']
        assert isinstance(meta['mle'], np.ndarray)  # make the typechecker happy
        self._mle = meta['mle']
        if self._save_iterations:
            assert isinstance(
                meta['all_iters'], np.ndarray
            )  # make the typechecker happy
            self._all_iters = meta['all_iters']

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of estimated quantities, includes joint log probability,
        and all parameters, transformed parameters, and generated quantities.
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
        """
        Returns all final estimates from the optimizer as a numpy.ndarray
        which contains all optimizer outputs, i.e., the value for `lp__`
        as well as all Stan program variables.
        """
        if not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        return self._mle

    @property
    def optimized_iterations_np(self) -> Optional[np.ndarray]:
        """
        Returns all saved iterations from the optimizer and final estimate
        as a numpy.ndarray which contains all optimizer outputs, i.e.,
        the value for `lp__` as well as all Stan program variables.

        """
        if not self._save_iterations:
            get_logger().warning(
                'Intermediate iterations not saved to CSV output file. '
                'Rerun the optimize method with "save_iterations=True".'
            )
            return None
        if not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        return self._all_iters

    @property
    def optimized_params_pd(self) -> pd.DataFrame:
        """
        Returns all final estimates from the optimizer as a pandas.DataFrame
        which contains all optimizer outputs, i.e., the value for `lp__`
        as well as all Stan program variables.
        """
        if not self.runset._check_retcodes():
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        return pd.DataFrame([self._mle], columns=self.column_names)

    @property
    def optimized_iterations_pd(self) -> Optional[pd.DataFrame]:
        """
        Returns all saved iterations from the optimizer and final estimate
        as a pandas.DataFrame which contains all optimizer outputs, i.e.,
        the value for `lp__` as well as all Stan program variables.

        """
        if not self._save_iterations:
            get_logger().warning(
                'Intermediate iterations not saved to CSV output file. '
                'Rerun the optimize method with "save_iterations=True".'
            )
            return None
        if not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        return pd.DataFrame(self._all_iters, columns=self.column_names)

    @property
    def optimized_params_dict(self) -> Dict[str, float]:
        """
        Returns all estimates from the optimizer, including `lp__` as a
        Python Dict.  Only returns estimate from final iteration.
        """
        if not self.runset._check_retcodes():
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        return OrderedDict(zip(self.column_names, self._mle))

    def stan_variable(
        self,
        var: Optional[str] = None,
        *,
        inc_iterations: bool = False,
        warn: bool = True,
    ) -> Union[np.ndarray, float]:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        :param var: variable name

        :param inc_iterations: When ``True`` and the intermediate estimates
            are included in the output, i.e., the optimizer was run with
            ``save_iterations=True``, then intermediate estimates are included.
            Default value is ``False``.

        See Also
        --------
        CmdStanMLE.stan_variables
        CmdStanMCMC.stan_variable
        CmdStanVB.stan_variable
        CmdStanGQ.stan_variable
        """
        if var is None:
            raise ValueError('no variable name specified.')
        if var not in self._metadata.stan_vars_dims:
            raise ValueError('unknown variable name: {}'.format(var))
        if warn and inc_iterations and not self._save_iterations:
            get_logger().warning(
                'Intermediate iterations not saved to CSV output file. '
                'Rerun the optimize method with "save_iterations=True".'
            )
        if warn and not self.runset._check_retcodes():
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )

        col_idxs = list(self._metadata.stan_vars_cols[var])
        if inc_iterations and self._save_iterations:
            num_rows = self._all_iters.shape[0]
        else:
            num_rows = 1
        if len(col_idxs) > 1:  # container var
            dims = (num_rows,) + self._metadata.stan_vars_dims[var]
            # pylint: disable=redundant-keyword-arg
            if num_rows > 1:
                result = self._all_iters[:, col_idxs].reshape(  # type: ignore
                    dims, order='F'
                )
            else:
                result = self._mle[col_idxs].reshape(dims[1:], order="F")
        else:  # scalar var
            col_idx = col_idxs[0]
            if num_rows > 1:
                result = self._all_iters[:, col_idx]
            else:
                result = float(self._mle[col_idx])
        assert isinstance(
            result, (np.ndarray, float)
        )  # make the typechecker happy
        return result

    def stan_variables(
        self, inc_iterations: bool = False
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        :param inc_iterations: When ``True`` and the intermediate estimates
            are included in the output, i.e., the optimizer was run with
            ``save_iterations=True``, then intermediate estimates are included.
            Default value is ``False``.


        See Also
        --------
        CmdStanMLE.stan_variable
        CmdStanMCMC.stan_variables
        CmdStanVB.stan_variables
        CmdStanGQ.stan_variables
        """
        if not self.runset._check_retcodes():
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        result = {}
        for name in self._metadata.stan_vars_dims.keys():
            result[name] = self.stan_variable(
                name, inc_iterations=inc_iterations, warn=False
            )
        return result

    def save_csvfiles(self, dir: Optional[str] = None) -> None:
        """
        Move output CSV files to specified directory.  If files were
        written to the temporary session directory, clean filename.
        E.g., save 'bernoulli-201912081451-1-5nm6as7u.csv' as
        'bernoulli-201912081451-1.csv'.

        :param dir: directory path

        See Also
        --------
        stanfit.RunSet.save_csvfiles
        cmdstanpy.from_csv
        """
        self.runset.save_csvfiles(dir)


class CmdStanGQ:
    """
    Container for outputs from CmdStan generate_quantities run.
    Created by :meth:`CmdStanModel.generate_quantities`.
    """

    def __init__(
        self,
        runset: RunSet,
        mcmc_sample: CmdStanMCMC,
    ) -> None:
        """Initialize object."""
        if not runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError(
                'Wrong runset method, expecting generate_quantities runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self.mcmc_sample = mcmc_sample
        self._draws = np.array(())
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
        return self._metadata.cmdstan_config['column_names']  # type: ignore

    @property
    def metadata(self) -> InferenceMetadata:
        """
        Returns object which contains CmdStan configuration as well as
        information about the names and structure of the inference method
        and model output variables.
        """
        return self._metadata

    def draws(
        self,
        *,
        inc_warmup: bool = False,
        concat_chains: bool = False,
        inc_sample: bool = False,
    ) -> np.ndarray:
        """
        Returns a numpy.ndarray over the generated quantities draws from
        all chains which is stored column major so that the values
        for a parameter are contiguous in memory, likewise all draws from
        a chain are contiguous.  By default, returns a 3D array arranged
        (draws, chains, columns); parameter ``concat_chains=True`` will
        return a 2D array where all chains are flattened into a single column,
        preserving chain order, so that given M chains of N draws,
        the first N draws are from chain 1, ..., and the the last N draws
        are from chain M.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        :param concat_chains: When ``True`` return a 2D array flattening all
            all draws from all chains.  Default value is ``False``.

        :param inc_sample: When ``True`` include all columns in the mcmc_sample
            draws array as well, excepting columns for variables already present
            in the generated quantities drawset. Default value is ``False``.

        See Also
        --------
        CmdStanGQ.draws_pd
        CmdStanGQ.draws_xr
        CmdStanMCMC.draws
        """
        if self._draws.size == 0:
            self._assemble_generated_quantities()
        if (
            inc_warmup
            and not self.mcmc_sample.metadata.cmdstan_config['save_warmup']
        ):
            get_logger().warning(
                "Sample doesn't contain draws from warmup iterations,"
                ' rerun sampler with "save_warmup=True".'
            )
        if inc_sample:
            cols_1 = self.mcmc_sample.column_names
            cols_2 = self.column_names
            dups = [
                item
                for item, count in Counter(cols_1 + cols_2).items()
                if count > 1
            ]
            drop_cols: List[int] = []
            for dup in dups:
                drop_cols.extend(self.mcmc_sample.metadata.stan_vars_cols[dup])

        start_idx = 0
        if (
            not inc_warmup
            and self.mcmc_sample.metadata.cmdstan_config['save_warmup']
        ):
            start_idx = self.mcmc_sample.num_draws_warmup

        if concat_chains and inc_sample:
            return flatten_chains(
                np.dstack(
                    (
                        np.delete(self.mcmc_sample.draws(), drop_cols, axis=1),
                        self._draws,
                    )
                )[start_idx:, :, :]
            )
        if concat_chains:
            return flatten_chains(self._draws[start_idx:, :, :])
        if inc_sample:
            return np.dstack(  # type: ignore
                (
                    np.delete(self.mcmc_sample.draws(), drop_cols, axis=1),
                    self._draws,
                )
            )[start_idx:, :, :]
        return self._draws[start_idx:, :, :]  # type: ignore

    def draws_pd(
        self,
        vars: Union[List[str], str, None] = None,
        inc_warmup: bool = False,
        inc_sample: bool = False,
    ) -> pd.DataFrame:
        """
        Returns the generated quantities draws as a pandas DataFrame.
        Flattens all chains into single column.  Container variables
        (array, vector, matrix) will span multiple columns, one column
        per element. E.g. variable 'matrix[2,2] foo' spans 4 columns:
        'foo[1,1], ... foo[2,2]'.

        :param vars: optional list of variable names.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        See Also
        --------
        CmdStanGQ.draws
        CmdStanGQ.draws_xr
        CmdStanMCMC.draws_pd
        """
        if vars is not None:
            if isinstance(vars, str):
                vars_list = [vars]
            else:
                vars_list = vars
        if (
            inc_warmup
            and not self.mcmc_sample.metadata.cmdstan_config['save_warmup']
        ):
            get_logger().warning(
                'Draws from warmup iterations not available,'
                ' must run sampler with "save_warmup=True".'
            )
        self._assemble_generated_quantities()

        gq_cols = []
        mcmc_vars = []
        if vars is not None:
            for var in set(vars_list):
                if var in self.metadata.stan_vars_cols:
                    for idx in self.metadata.stan_vars_cols[var]:
                        gq_cols.append(self.column_names[idx])
                elif (
                    inc_sample
                    and var in self.mcmc_sample.metadata.stan_vars_cols
                ):
                    mcmc_vars.append(var)
                else:
                    raise ValueError('Unknown variable: {}'.format(var))
        else:
            gq_cols = list(self.column_names)

        if inc_sample and mcmc_vars:
            if gq_cols:
                return pd.concat(
                    [
                        self.mcmc_sample.draws_pd(
                            vars=mcmc_vars, inc_warmup=inc_warmup
                        ).reset_index(drop=True),
                        pd.DataFrame(
                            data=flatten_chains(
                                self.draws(inc_warmup=inc_warmup)
                            ),
                            columns=self.column_names,
                        )[gq_cols],
                    ],
                    axis='columns',
                )
            else:
                return self.mcmc_sample.draws_pd(
                    vars=mcmc_vars, inc_warmup=inc_warmup
                )
        elif inc_sample and vars is None:
            cols_1 = self.mcmc_sample.column_names
            cols_2 = self.column_names
            dups = [
                item
                for item, count in Counter(cols_1 + cols_2).items()
                if count > 1
            ]
            return pd.concat(
                [
                    self.mcmc_sample.draws_pd(inc_warmup=inc_warmup)
                    .drop(columns=dups)
                    .reset_index(drop=True),
                    pd.DataFrame(
                        data=flatten_chains(self.draws(inc_warmup=inc_warmup)),
                        columns=self.column_names,
                    ),
                ],
                axis='columns',
                ignore_index=True,
            )
        elif gq_cols:
            return pd.DataFrame(
                data=flatten_chains(self.draws(inc_warmup=inc_warmup)),
                columns=self.column_names,
            )[gq_cols]

        return pd.DataFrame(
            data=flatten_chains(self.draws(inc_warmup=inc_warmup)),
            columns=self.column_names,
        )

    def draws_xr(
        self,
        vars: Union[str, List[str], None] = None,
        inc_warmup: bool = False,
        inc_sample: bool = False,
    ) -> "xr.Dataset":
        """
        Returns the generated quantities draws as a xarray Dataset.

        :param vars: optional list of variable names.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the MCMC sample, then the warmup draws are included.
            Default value is ``False``.

        See Also
        --------
        CmdStanGQ.draws
        CmdStanGQ.draws_pd
        CmdStanMCMC.draws_xr
        """
        if not XARRAY_INSTALLED:
            raise RuntimeError(
                'Package "xarray" is not installed, cannot produce draws array.'
            )
        mcmc_vars_list = []
        dup_vars = []
        if vars is not None:
            if isinstance(vars, str):
                vars_list = [vars]
            else:
                vars_list = vars
            for var in vars_list:
                if var not in self.metadata.stan_vars_cols:
                    if inc_sample and (
                        var in self.mcmc_sample.metadata.stan_vars_cols
                    ):
                        mcmc_vars_list.append(var)
                        dup_vars.append(var)
                    else:
                        raise ValueError('Unknown variable: {}'.format(var))
        else:
            vars_list = list(self.metadata.stan_vars_cols.keys())
            if inc_sample:
                for var in self.mcmc_sample.metadata.stan_vars_cols.keys():
                    if var not in vars_list and var not in mcmc_vars_list:
                        mcmc_vars_list.append(var)
        for var in dup_vars:
            vars_list.remove(var)

        self._assemble_generated_quantities()

        num_draws = self.mcmc_sample.num_draws_sampling
        sample_config = self.mcmc_sample.metadata.cmdstan_config
        attrs: MutableMapping[Hashable, Any] = {
            "stan_version": f"{sample_config['stan_version_major']}."
            f"{sample_config['stan_version_minor']}."
            f"{sample_config['stan_version_patch']}",
            "model": sample_config["model"],
            "num_draws_sampling": num_draws,
        }
        if inc_warmup and sample_config['save_warmup']:
            num_draws += self.mcmc_sample.num_draws_warmup
            attrs["num_draws_warmup"] = self.mcmc_sample.num_draws_warmup

        data: MutableMapping[Hashable, Any] = {}
        coordinates: MutableMapping[Hashable, Any] = {
            "chain": self.chain_ids,
            "draw": np.arange(num_draws),
        }

        for var in vars_list:
            build_xarray_data(
                data,
                var,
                self._metadata.stan_vars_dims[var],
                self._metadata.stan_vars_cols[var],
                0,
                self.draws(inc_warmup=inc_warmup),
            )
        if inc_sample:
            for var in mcmc_vars_list:
                build_xarray_data(
                    data,
                    var,
                    self.mcmc_sample.metadata.stan_vars_dims[var],
                    self.mcmc_sample.metadata.stan_vars_cols[var],
                    0,
                    self.mcmc_sample.draws(inc_warmup=inc_warmup),
                )

        return xr.Dataset(data, coords=coordinates, attrs=attrs).transpose(
            'chain', 'draw', ...
        )

    def stan_variable(
        self,
        var: Optional[str] = None,
        inc_warmup: bool = False,
    ) -> np.ndarray:
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

        :param var: variable name

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the MCMC sample, then the warmup draws are included.
            Default value is ``False``.

        See Also
        --------
        CmdStanGQ.stan_variables
        CmdStanMCMC.stan_variable
        CmdStanMLE.stan_variable
        CmdStanVB.stan_variable
        """
        if var is None:
            raise ValueError('No variable name specified.')
        model_var_names = self.mcmc_sample.metadata.stan_vars_cols.keys()
        gq_var_names = self.metadata.stan_vars_cols.keys()
        if not (var in model_var_names or var in gq_var_names):
            raise ValueError('Unknown variable name: {}'.format(var))
        if var not in gq_var_names:
            return self.mcmc_sample.stan_variable(var, inc_warmup=inc_warmup)
        else:  # is gq variable
            self._assemble_generated_quantities()
            draw1 = 0
            if (
                not inc_warmup
                and self.mcmc_sample.metadata.cmdstan_config['save_warmup']
            ):
                draw1 = self.mcmc_sample.num_draws_warmup
            num_draws = self.mcmc_sample.num_draws_sampling
            if (
                inc_warmup
                and self.mcmc_sample.metadata.cmdstan_config['save_warmup']
            ):
                num_draws += self.mcmc_sample.num_draws_warmup
            dims = [num_draws * self.chains]
            col_idxs = self._metadata.stan_vars_cols[var]
            if len(col_idxs) > 0:
                dims.extend(self._metadata.stan_vars_dims[var])
            # pylint: disable=redundant-keyword-arg
            return self._draws[draw1:, :, col_idxs].reshape(  # type: ignore
                dims, order='F'
            )

    def stan_variables(self, inc_warmup: bool = False) -> Dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the MCMC sample, then the warmup draws are included.
            Default value is ``False``

        See Also
        --------
        CmdStanGQ.stan_variable
        CmdStanMCMC.stan_variables
        CmdStanMLE.stan_variables
        CmdStanVB.stan_variables
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
        self._draws = gq_sample

    def save_csvfiles(self, dir: Optional[str] = None) -> None:
        """
        Move output CSV files to specified directory.  If files were
        written to the temporary session directory, clean filename.
        E.g., save 'bernoulli-201912081451-1-5nm6as7u.csv' as
        'bernoulli-201912081451-1.csv'.

        :param dir: directory path

        See Also
        --------
        stanfit.RunSet.save_csvfiles
        cmdstanpy.from_csv
        """
        self.runset.save_csvfiles(dir)


class CmdStanVB:
    """
    Container for outputs from CmdStan variational run.
    Created by :meth:`CmdStanModel.variational`.
    """

    def __init__(self, runset: RunSet) -> None:
        """Initialize object."""
        if not runset.method == Method.VARIATIONAL:
            raise ValueError(
                'Wrong runset method, expecting variational inference, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
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
        # these three assignments don't grant type information
        self._column_names: Tuple[str, ...] = meta['column_names']
        self._eta: float = meta['eta']
        self._variational_mean: np.ndarray = meta['variational_mean']
        self._variational_sample: np.ndarray = meta['variational_sample']

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
    def eta(self) -> float:
        """
        Step size scaling parameter 'eta'
        """
        return self._eta

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
    def variational_params_dict(self) -> Dict[str, np.ndarray]:
        """Returns inferred parameter means as Dict."""
        return OrderedDict(zip(self.column_names, self._variational_mean))

    @property
    def metadata(self) -> InferenceMetadata:
        """
        Returns object which contains CmdStan configuration as well as
        information about the names and structure of the inference method
        and model output variables.
        """
        return self._metadata

    def stan_variable(
        self, var: Optional[str] = None
    ) -> Union[np.ndarray, float]:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        :param var: variable name

        See Also
        --------
        CmdStanVB.stan_variables
        CmdStanMCMC.stan_variable
        CmdStanMLE.stan_variable
        CmdStanGQ.stan_variable
        """
        if var is None:
            raise ValueError('No variable name specified.')
        if var not in self._metadata.stan_vars_dims:
            raise ValueError('Unknown variable name: {}'.format(var))
        col_idxs = list(self._metadata.stan_vars_cols[var])
        shape: Tuple[int, ...] = ()
        if len(col_idxs) > 1:
            shape = self._metadata.stan_vars_dims[var]
            result = np.asarray(self._variational_mean)[col_idxs].reshape(
                shape, order="F"
            )
        else:
            result = float(self._variational_mean[col_idxs[0]])
        assert isinstance(result, (np.ndarray, float))
        return result

    def stan_variables(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        See Also
        --------
        CmdStanVB.stan_variable
        CmdStanMCMC.stan_variables
        CmdStanMLE.stan_variables
        CmdStanGQ.stan_variables
        """
        result = {}
        for name in self._metadata.stan_vars_dims.keys():
            result[name] = self.stan_variable(name)
        return result

    @property
    def variational_sample(self) -> np.ndarray:
        """Returns the set of approximate posterior output draws."""
        return self._variational_sample

    def save_csvfiles(self, dir: Optional[str] = None) -> None:
        """
        Move output CSV files to specified directory.  If files were
        written to the temporary session directory, clean filename.
        E.g., save 'bernoulli-201912081451-1-5nm6as7u.csv' as
        'bernoulli-201912081451-1.csv'.

        :param dir: directory path

        See Also
        --------
        stanfit.RunSet.save_csvfiles
        cmdstanpy.from_csv
        """
        self.runset.save_csvfiles(dir)


def from_csv(
    path: Union[str, List[str], None] = None, method: Optional[str] = None
) -> Union[CmdStanMCMC, CmdStanMLE, CmdStanVB, None]:
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

    config_dict: Dict[str, Any] = {}
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
    try:
        if config_dict['method'] == 'sample':
            chains = len(csvfiles)
            sampler_args = SamplerArgs(
                iter_sampling=config_dict['num_samples'],
                iter_warmup=config_dict['num_warmup'],
                thin=config_dict['thin'],
                save_warmup=config_dict['save_warmup'],
            )
            # bugfix 425, check for fixed_params output
            try:
                check_sampler_csv(
                    csvfiles[0],
                    iter_sampling=config_dict['num_samples'],
                    iter_warmup=config_dict['num_warmup'],
                    thin=config_dict['thin'],
                    save_warmup=config_dict['save_warmup'],
                )
            except ValueError:
                try:
                    check_sampler_csv(
                        csvfiles[0],
                        is_fixed_param=True,
                        iter_sampling=config_dict['num_samples'],
                        iter_warmup=config_dict['num_warmup'],
                        thin=config_dict['thin'],
                        save_warmup=config_dict['save_warmup'],
                    )
                    sampler_args = SamplerArgs(
                        iter_sampling=config_dict['num_samples'],
                        iter_warmup=config_dict['num_warmup'],
                        thin=config_dict['thin'],
                        save_warmup=config_dict['save_warmup'],
                        fixed_param=True,
                    )
                except (ValueError) as e:
                    raise ValueError(
                        'Invalid or corrupt Stan CSV output file, '
                    ) from e

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
            return fit
        elif config_dict['method'] == 'optimize':
            if 'algorithm' not in config_dict:
                raise ValueError(
                    "Cannot find optimization algorithm"
                    " in file {}.".format(csvfiles[0])
                )
            optimize_args = OptimizeArgs(
                algorithm=config_dict['algorithm'],
                save_iterations=config_dict['save_iterations'],
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
            return CmdStanMLE(runset)
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
            return CmdStanVB(runset)
        else:
            get_logger().info(
                'Unable to process CSV output files from method %s.',
                (config_dict['method']),
            )
            return None
    except (IOError, OSError, PermissionError) as e:
        raise ValueError(
            'An error occured processing the CSV files:\n\t{}'.format(str(e))
        ) from e


def build_xarray_data(
    data: MutableMapping[Hashable, Tuple[Tuple[str, ...], np.ndarray]],
    var_name: str,
    dims: Tuple[int, ...],
    col_idxs: Tuple[int, ...],
    start_row: int,
    drawset: np.ndarray,
) -> None:
    """
    Adds Stan variable name, labels, and values to a dictionary
    that will be used to construct an xarray DataSet.
    """
    var_dims: Tuple[str, ...] = ('draw', 'chain')
    if dims:
        var_dims += tuple(f"{var_name}_dim_{i}" for i in range(len(dims)))
        data[var_name] = (
            var_dims,
            drawset[start_row:, :, col_idxs].reshape(
                *drawset.shape[:2], *dims, order="F"
            ),
        )
    else:
        data[var_name] = (
            var_dims,
            np.squeeze(drawset[start_row:, :, col_idxs], axis=2),
        )
