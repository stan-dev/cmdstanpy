"""
Container for the result of running the
generate quantities (GQ) method
"""

from __future__ import annotations

import os
from collections.abc import Hashable, Sequence, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Generic, NoReturn, TypeVar, overload

import numpy as np
import pandas as pd

try:
    import xarray as xr

    XARRAY_INSTALLED = True
except ImportError:
    XARRAY_INSTALLED = False


from cmdstanpy.utils import (
    build_xarray_data,
    flatten_chains,
    get_logger,
    stancsv,
)

from .base import MultiChainFit, StanFit
from .laplace import CmdStanLaplace
from .mcmc import CmdStanMCMC
from .metadata import GeneratedQuantitiesConfig, GeneratedQuantitiesRunConfig
from .mle import CmdStanMLE
from .pathfinder import CmdStanPathfinder
from .vb import CmdStanVB

PrevFit = TypeVar(
    'PrevFit',
    CmdStanMCMC,
    CmdStanMLE,
    CmdStanVB,
    CmdStanLaplace,
    CmdStanPathfinder,
)


@dataclass(kw_only=True)
class CmdStanGQ(MultiChainFit[GeneratedQuantitiesConfig], Generic[PrevFit]):
    """
    Container for outputs from CmdStan generate_quantities run.
    Created by :meth:`CmdStanModel.generate_quantities`.
    """

    previous_fit: PrevFit

    _draws: np.ndarray = field(default_factory=lambda: np.array(()), init=False)

    @classmethod
    def from_files(
        cls,
        csv_files: Sequence[str | os.PathLike],
        config_files: Sequence[str | os.PathLike] | str | os.PathLike,
        previous_fit: PrevFit,
        stdout_files: Sequence[str | os.PathLike] | None = None,
        chain_ids: Sequence[int] | None = None,
    ) -> CmdStanGQ[PrevFit]:
        """Build a CmdStanGQ from output files.

        ``config_files`` may be a single path (when CmdStan ran multiple chains
        in one process) or a per-chain list.
        """
        return cls(
            previous_fit=previous_fit,
            **cls._from_files_kwargs(
                csv_files,
                config_files,
                stdout_files,
                chain_ids,
                GeneratedQuantitiesRunConfig,
            ),
        )

    def __repr__(self) -> str:
        lines = [
            f'CmdStanGQ: model={self.model_name} chains={self.chains}'
            f' method={self.config.method_config.method}',
            ' csv_files:\n\t' + '\n\t'.join(self.csv_files),
        ]
        if self.stdout_files is not None:
            lines.append(' output_files:\n\t' + '\n\t'.join(self.stdout_files))
        return '\n'.join(lines)

    def draws(
        self,
        *,
        inc_warmup: bool = False,
        inc_iterations: bool = False,
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

        :param inc_sample: When ``True`` include all columns in the previous_fit
            draws array as well, excepting columns for variables already present
            in the generated quantities drawset. Default value is ``False``.

        See Also
        --------
        CmdStanGQ.draws_pd
        CmdStanGQ.draws_xr
        CmdStanMCMC.draws
        """
        self._assemble()
        inc_warmup |= inc_iterations
        if inc_warmup:
            if (
                isinstance(self.previous_fit, CmdStanMCMC)
                and not self.previous_fit._save_warmup
            ):
                get_logger().warning(
                    "Sample doesn't contain draws from warmup iterations,"
                    ' rerun sampler with "save_warmup=True".'
                )
            elif (
                isinstance(self.previous_fit, CmdStanMLE)
                and not self.previous_fit.config.method_config.save_iterations
            ):
                get_logger().warning(
                    "MLE doesn't contain draws from pre-convergence iterations,"
                    ' rerun optimization with "save_iterations=True".'
                )
            elif isinstance(self.previous_fit, CmdStanVB):
                get_logger().warning(
                    "Variational fit doesn't make sense with argument "
                    '"inc_warmup=True"'
                )

        start_idx = self._draws_start(inc_warmup)
        draws = self._draws[start_idx:]
        if inc_sample:
            previous_draws = self._previous_draws(True)[start_idx:]
            draws = np.concatenate(
                (previous_draws[:, :, self._previous_column_indices()], draws),
                axis=2,
            )

        return flatten_chains(draws) if concat_chains else draws

    def draws_pd(
        self,
        vars: list[str] | str | None = None,
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
        identifiers = ['chain__', 'iter__', 'draw__']
        selected_columns: list[str] = []
        include_previous = inc_sample and vars is None
        if vars is not None:
            vars_list = [vars] if isinstance(vars, str) else vars
            for var in dict.fromkeys(vars_list):
                fit: StanFit[Any]
                if var in self.metadata.stan_vars:
                    fit = self
                elif inc_sample and var in self.previous_fit.metadata.stan_vars:
                    fit = self.previous_fit
                    include_previous = True
                elif var in identifiers:
                    selected_columns.append(var)
                    continue
                else:
                    raise ValueError(f'Unknown variable: {var}')
                info = fit.metadata.stan_vars[var]
                selected_columns.extend(
                    fit.column_names[info.start_idx : info.end_idx]
                )

        previous_columns = (
            [
                self.previous_fit.column_names[idx]
                for idx in self._previous_column_indices()
            ]
            if include_previous
            else []
        )
        draws = self.draws(inc_warmup=inc_warmup, inc_sample=include_previous)
        n_draws, n_chains, _ = draws.shape
        frame = pd.DataFrame(
            flatten_chains(draws),
            columns=previous_columns + list(self.column_names),
        )
        frame['chain__'] = np.repeat(
            np.arange(1, n_chains + 1, dtype=float), n_draws
        )
        frame['iter__'] = np.tile(
            np.arange(1, n_draws + 1, dtype=float), n_chains
        )
        frame['draw__'] = np.arange(1, n_draws * n_chains + 1, dtype=float)

        # An empty variable list returns all GQ columns and IDs.
        columns = selected_columns or (
            previous_columns + identifiers + list(self.column_names)
        )
        return frame[columns]

    @overload
    def draws_xr(
        self: (
            CmdStanGQ[CmdStanMLE]
            | CmdStanGQ[CmdStanVB]
            | CmdStanGQ[CmdStanLaplace]
            | CmdStanGQ[CmdStanPathfinder]
        ),
        vars: str | list[str] | None = None,
        inc_warmup: bool = False,
        inc_sample: bool = False,
    ) -> NoReturn: ...

    @overload
    def draws_xr(
        self: CmdStanGQ[CmdStanMCMC],
        vars: str | list[str] | None = None,
        inc_warmup: bool = False,
        inc_sample: bool = False,
    ) -> xr.Dataset: ...

    def draws_xr(
        self,
        vars: str | list[str] | None = None,
        inc_warmup: bool = False,
        inc_sample: bool = False,
    ) -> xr.Dataset:
        """
        Returns the generated quantities draws as a xarray Dataset.

        This method can only be called when the underlying fit was made
        through sampling, it cannot be used on MLE or VB outputs.

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
        if not isinstance(self.previous_fit, CmdStanMCMC):
            raise RuntimeError(
                'Method "draws_xr" is only available when '
                'original fit is done via Sampling.'
            )
        prev = self.previous_fit
        if vars is None:
            requested = list(self.metadata.stan_vars)
            if inc_sample:
                requested.extend(prev.metadata.stan_vars)
        else:
            requested = [vars] if isinstance(vars, str) else vars

        gq_vars: list[str] = []
        previous_vars: list[str] = []
        for var in dict.fromkeys(requested):
            if var in self.metadata.stan_vars:
                gq_vars.append(var)
            elif inc_sample and var in prev.metadata.stan_vars:
                previous_vars.append(var)
            else:
                raise ValueError(f'Unknown variable: {var}')

        if inc_warmup and not prev._save_warmup:
            get_logger().warning(
                "Sample doesn't contain draws from warmup iterations,"
                ' rerun sampler with "save_warmup=True".'
            )
        include_warmup = inc_warmup and prev._save_warmup
        num_draws = prev.num_draws_sampling
        attrs: MutableMapping[Hashable, Any] = {
            "stan_version": f"{prev.config.stan_major_version}."
            f"{prev.config.stan_minor_version}."
            f"{prev.config.stan_patch_version}",
            "model": prev.model_name,
            "num_draws_sampling": num_draws,
        }
        if include_warmup:
            num_draws += prev.num_draws_warmup
            attrs["num_draws_warmup"] = prev.num_draws_warmup

        data: MutableMapping[Hashable, Any] = {}
        coordinates: MutableMapping[Hashable, Any] = {
            "chain": self.chain_ids,
            "draw": np.arange(num_draws),
        }

        if gq_vars:
            gq_draws = self.draws(inc_warmup=include_warmup)
            for var in gq_vars:
                build_xarray_data(data, self.metadata.stan_vars[var], gq_draws)
        if previous_vars:
            previous_draws = prev.draws(inc_warmup=include_warmup)
            for var in previous_vars:
                build_xarray_data(
                    data, prev.metadata.stan_vars[var], previous_draws
                )

        return xr.Dataset(data, coords=coordinates, attrs=attrs).transpose(
            'chain', 'draw', ...
        )

    def stan_variable(self, var: str, **kwargs: bool) -> np.ndarray:
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
          ( draws * chains, 1).
        * If the variable is a vector, the return array has shape
          ( draws * chains, len(vector))
        * If the variable is a matrix, the return array has shape
          ( draws * chains, size(dim 1), size(dim 2) )
        * If the variable is an array with N dimensions, the return array
          has shape ( draws * chains, size(dim 1), ..., size(dim N))

        For example, if the Stan program variable ``theta`` is a 3x3 matrix,
        and the sample consists of 4 chains with 1000 post-warmup draws,
        this function will return a numpy.ndarray with shape (4000,3,3).

        This functionaltiy is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        :param kwargs: Additional keyword arguments are passed to the underlying
            fit's ``stan_variable`` method if the variable is not a generated
            quantity.

        See Also
        --------
        CmdStanGQ.stan_variables
        CmdStanMCMC.stan_variable
        CmdStanMLE.stan_variable
        CmdStanPathfinder.stan_variable
        CmdStanVB.stan_variable
        CmdStanLaplace.stan_variable
        """
        model_var_names = self.previous_fit.metadata.stan_vars.keys()
        gq_var_names = self.metadata.stan_vars.keys()
        if not (var in model_var_names or var in gq_var_names):
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are '
                + ", ".join(model_var_names | gq_var_names)
            )
        if var not in gq_var_names:
            return self.previous_fit.stan_variable(var, **kwargs)

        # is gq variable
        self._assemble()

        draw1 = self._draws_start(
            inc_warmup=kwargs.get('inc_warmup', False)
            or kwargs.get('inc_iterations', False)
        )
        draws = flatten_chains(self._draws[draw1:])
        out: np.ndarray = self.metadata.stan_vars[var].extract_reshape(draws)
        return out

    def stan_variables(self, **kwargs: bool) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        :param kwargs: Additional keyword arguments are passed to the underlying
            fit's ``stan_variable`` method if the variable is not a generated
            quantity.

        See Also
        --------
        CmdStanGQ.stan_variable
        CmdStanMCMC.stan_variables
        CmdStanMLE.stan_variables
        CmdStanPathfinder.stan_variables
        CmdStanVB.stan_variables
        CmdStanLaplace.stan_variables
        """
        result = {}
        sample_var_names = self.previous_fit.metadata.stan_vars.keys()
        gq_var_names = self.metadata.stan_vars.keys()
        for name in gq_var_names:
            result[name] = self.stan_variable(name, **kwargs)
        for name in sample_var_names:
            if name not in gq_var_names:
                result[name] = self.stan_variable(name, **kwargs)
        return result

    def _assemble(self) -> None:
        if self._draws.shape != (0,):
            return
        num_draws = self._num_draws_total()

        gq_sample: np.ndarray = np.empty(
            (num_draws, self.chains, len(self.column_names)),
            dtype=float,
            order='F',
        )
        for chain in range(self.chains):
            csv_file = self.csv_files[chain]
            try:
                *_, draws = stancsv.parse_comments_header_and_draws(
                    self.csv_files[chain]
                )
                gq_sample[:, chain, :] = stancsv.csv_bytes_list_to_numpy(draws)
            except Exception as exc:
                raise ValueError(
                    f"An error occurred when parsing Stan csv {csv_file}"
                    f" for chain {chain}"
                ) from exc
        self._draws = gq_sample

    def _draws_start(self, inc_warmup: bool) -> int:
        """Start of the returned rows; -1 selects the final optimizer row."""
        p_fit = self.previous_fit
        if isinstance(p_fit, CmdStanMCMC):
            if p_fit._save_warmup and not inc_warmup:
                return p_fit.num_draws_warmup
        elif isinstance(p_fit, CmdStanMLE):
            if not inc_warmup:
                return -1
        elif isinstance(p_fit, CmdStanVB):
            return 1  # Always skip the variational mean.
        return 0

    def _num_draws_total(self) -> int:
        """Number of GQ CSV rows, including warmup, iterations, or VB mean."""
        p_fit = self.previous_fit
        if isinstance(p_fit, CmdStanMCMC):
            return p_fit.num_draws_sampling + (
                p_fit.num_draws_warmup if p_fit._save_warmup else 0
            )
        if isinstance(p_fit, CmdStanMLE):
            if p_fit.config.method_config.save_iterations:
                return len(p_fit.optimized_iterations_np)  # type: ignore
            return 1
        if isinstance(p_fit, CmdStanVB):
            return int(p_fit.variational_sample.shape[0]) + 1
        return int(p_fit.draws().shape[0])

    def _previous_draws(self, inc_warmup: bool) -> np.ndarray:
        """
        Extract the draws from self.previous_fit.
        Return is always 3-d
        """
        p_fit = self.previous_fit
        if isinstance(p_fit, CmdStanMCMC):
            return p_fit.draws(inc_warmup=inc_warmup and p_fit._save_warmup)
        elif isinstance(p_fit, CmdStanMLE):
            if inc_warmup and p_fit.config.method_config.save_iterations:
                return p_fit.optimized_iterations_np[:, None]  # type: ignore

            return np.atleast_2d(  # type: ignore
                p_fit.optimized_params_np,
            )[:, None]
        elif isinstance(p_fit, CmdStanVB):
            if inc_warmup:
                return np.vstack(
                    [p_fit.variational_params_np, p_fit.variational_sample]
                )[:, None]
            return p_fit.variational_sample[:, None]
        else:  # CmdStanLaplace, CmdStanPathfinder
            return p_fit.draws()[:, None, :]

    def _previous_column_indices(self) -> list[int]:
        """Previous-fit columns retained when merging with generated quantities."""
        gq_columns = set(self.column_names)
        return [
            idx
            for idx, name in enumerate(self.previous_fit.column_names)
            if name not in gq_columns
        ]
