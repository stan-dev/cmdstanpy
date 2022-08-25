"""
Container for the result of running the
generate quantities (GQ) method
"""

from collections import Counter
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

from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils import (
    BaseType,
    build_xarray_data,
    flatten_chains,
    get_logger,
    scan_generated_quantities_csv,
)

from .mcmc import CmdStanMCMC
from .metadata import InferenceMetadata
from .runset import RunSet


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
        self._draws: np.ndarray = np.array(())
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

    def __getattr__(self, attr: str) -> np.ndarray:
        """Synonymous with ``fit.stan_variable(attr)"""
        if attr.startswith("_"):
            raise AttributeError(f"Unknown variable name {attr}")
        try:
            return self.stan_variable(attr)
        except ValueError as e:
            # pylint: disable=raise-missing-from
            raise AttributeError(*e.args)

    def _validate_csv_files(self) -> Dict[str, Any]:
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
        if self._draws.shape == (0,):
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
            return np.dstack(
                (
                    np.delete(self.mcmc_sample.draws(), drop_cols, axis=1),
                    self._draws,
                )
            )[start_idx:, :, :]
        return self._draws[start_idx:, :, :]

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
            vars_list = list(dict.fromkeys(vars_list))
        if (
            inc_warmup
            and not self.mcmc_sample.metadata.cmdstan_config['save_warmup']
        ):
            get_logger().warning(
                'Draws from warmup iterations not available,'
                ' must run sampler with "save_warmup=True".'
            )
        if self._draws.shape == (0,):
            self._assemble_generated_quantities()

        gq_cols = []
        mcmc_vars = []
        if vars is not None:
            for var in vars_list:
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
            vars_list = gq_cols

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
                )[vars_list]
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

        if self._draws.shape == (0,):
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
                self._metadata.stan_vars_types[var],
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
                    self.mcmc_sample._metadata.stan_vars_types[var],
                )

        return xr.Dataset(data, coords=coordinates, attrs=attrs).transpose(
            'chain', 'draw', ...
        )

    def stan_variable(
        self,
        var: str,
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

        This functionaltiy is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

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
        model_var_names = self.mcmc_sample.metadata.stan_vars_cols.keys()
        gq_var_names = self.metadata.stan_vars_cols.keys()
        if not (var in model_var_names or var in gq_var_names):
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are '
                + ", ".join(model_var_names | gq_var_names)
            )
        if var not in gq_var_names:
            return self.mcmc_sample.stan_variable(var, inc_warmup=inc_warmup)
        else:  # is gq variable
            if self._draws.shape == (0,):
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

            draws = self._draws[draw1:, :, col_idxs]

            if self._metadata.stan_vars_types[var] == BaseType.COMPLEX:
                draws = draws[..., ::2] + 1j * draws[..., 1::2]
                dims = dims[:-1]

            draws = draws.reshape(dims, order='F')

            return draws

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
        # use numpy loadtxt
        warmup = self.mcmc_sample.metadata.cmdstan_config['save_warmup']
        num_draws = self.mcmc_sample.draws(inc_warmup=warmup).shape[0]
        gq_sample: np.ndarray = np.empty(
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
