"""
Container for the result of running the sample (MCMC) method
or generate quantities (GQ) method
"""

import math
import os
from collections import Counter
from io import StringIO
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
from cmdstanpy.cmdstan_args import Method, SamplerArgs
from cmdstanpy.utils import (
    EXTENSION,
    check_sampler_csv,
    cmdstan_path,
    cmdstan_version_before,
    create_named_text_file,
    do_command,
    flatten_chains,
    get_logger,
    scan_generated_quantities_csv,
)

from .metadata import InferenceMetadata
from .runset import RunSet


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
