"""
    Container for the result of running a laplace approximation.
"""

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
from cmdstanpy.utils.data_munging import build_xarray_data
from cmdstanpy.utils.stancsv import scan_generic_csv

from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet

# TODO list:
# - docs and example notebook
# - make sure features like standalone GQ are updated/working


class CmdStanLaplace:
    def __init__(self, runset: RunSet, mode: CmdStanMLE) -> None:
        """Initialize object."""
        if not runset.method == Method.LAPLACE:
            raise ValueError(
                'Wrong runset method, expecting laplace runset, '
                'found method {}'.format(runset.method)
            )
        self._runset = runset
        self._mode = mode

        self._draws: np.ndarray = np.array(())

        config = scan_generic_csv(runset.csv_files[0])
        self._metadata = InferenceMetadata(config)

    def _assemble_draws(self) -> None:
        if self._draws.shape != (0,):
            return

        with open(self._runset.csv_files[0], 'r') as fd:
            while (fd.readline()).startswith("#"):
                pass
            self._draws = np.loadtxt(
                fd,
                dtype=float,
                ndmin=2,
                delimiter=',',
                comments="#",
            )

    def stan_variable(self, var: str) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        This functionaltiy is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        See Also
        --------
        CmdStanMLE.stan_variables
        CmdStanMCMC.stan_variable
        CmdStanPathfinder.stan_variable
        CmdStanVB.stan_variable
        CmdStanGQ.stan_variable
        """
        self._assemble_draws()
        try:
            out: np.ndarray = self._metadata.stan_vars[var].extract_reshape(
                self._draws
            )
            return out
        except KeyError:
            # pylint: disable=raise-missing-from
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are '
                + ", ".join(self._metadata.stan_vars.keys())
            )

    def stan_variables(self) -> Dict[str, np.ndarray]:
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
        CmdStanPathfinder.stan_variables
        CmdStanVB.stan_variables
        """
        result = {}
        for name in self._metadata.stan_vars:
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
        self._assemble_draws()
        return {
            name: var.extract_reshape(self._draws)
            for name, var in self._metadata.method_vars.items()
        }

    def draws(self) -> np.ndarray:
        """
        Return a numpy.ndarray containing the draws from the
        approximate posterior distribution. This is a 2-D array
        of shape (draws, parameters).
        """
        self._assemble_draws()
        return self._draws

    def draws_pd(
        self,
        vars: Union[List[str], str, None] = None,
    ) -> pd.DataFrame:
        if vars is not None:
            if isinstance(vars, str):
                vars_list = [vars]
            else:
                vars_list = vars

        self._assemble_draws()
        cols = []
        if vars is not None:
            for var in dict.fromkeys(vars_list):
                if var in self._metadata.method_vars:
                    cols.append(var)
                elif var in self._metadata.stan_vars:
                    info = self._metadata.stan_vars[var]
                    cols.extend(
                        self.column_names[info.start_idx : info.end_idx]
                    )
                else:
                    raise ValueError(f'Unknown variable: {var}')

        else:
            cols = list(self.column_names)

        return pd.DataFrame(self._draws, columns=self.column_names)[cols]

    def draws_xr(
        self,
        vars: Union[str, List[str], None] = None,
    ) -> "xr.Dataset":
        """
        Returns the sampler draws as a xarray Dataset.

        :param vars: optional list of variable names.

        See Also
        --------
        CmdStanMCMC.draws_xr
        CmdStanGQ.draws_xr
        """
        if not XARRAY_INSTALLED:
            raise RuntimeError(
                'Package "xarray" is not installed, cannot produce draws array.'
            )

        if vars is None:
            vars_list = list(self._metadata.stan_vars.keys())
        elif isinstance(vars, str):
            vars_list = [vars]
        else:
            vars_list = vars

        self._assemble_draws()

        meta = self._metadata.cmdstan_config
        attrs: MutableMapping[Hashable, Any] = {
            "stan_version": f"{meta['stan_version_major']}."
            f"{meta['stan_version_minor']}.{meta['stan_version_patch']}",
            "model": meta["model"],
        }

        data: MutableMapping[Hashable, Any] = {}
        coordinates: MutableMapping[Hashable, Any] = {
            "draw": np.arange(self._draws.shape[0]),
        }

        for var in vars_list:
            build_xarray_data(
                data,
                self._metadata.stan_vars[var],
                self._draws[:, np.newaxis, :],
            )
        return (
            xr.Dataset(data, coords=coordinates, attrs=attrs)
            .transpose('draw', ...)
            .squeeze()
        )

    @property
    def mode(self) -> CmdStanMLE:
        """
        Return the maximum a posteriori estimate (mode)
        as a :class:`CmdStanMLE` object.
        """
        return self._mode

    @property
    def metadata(self) -> InferenceMetadata:
        """
        Returns object which contains CmdStan configuration as well as
        information about the names and structure of the inference method
        and model output variables.
        """
        return self._metadata

    def __repr__(self) -> str:
        mode = '\n'.join(
            ['\t' + line for line in repr(self.mode).splitlines()]
        )[1:]
        rep = 'CmdStanLaplace: model={} \nmode=({})\n{}'.format(
            self._runset.model,
            mode,
            self._runset._args.method_args.compose(0, cmd=[]),
        )
        rep = '{}\n csv_files:\n\t{}\n output_files:\n\t{}'.format(
            rep,
            '\n\t'.join(self._runset.csv_files),
            '\n\t'.join(self._runset.stdout_files),
        )
        return rep

    def __getattr__(self, attr: str) -> np.ndarray:
        """Synonymous with ``fit.stan_variable(attr)"""
        if attr.startswith("_"):
            raise AttributeError(f"Unknown variable name {attr}")
        try:
            return self.stan_variable(attr)
        except ValueError as e:
            # pylint: disable=raise-missing-from
            raise AttributeError(*e.args)

    def __getstate__(self) -> dict:
        # This function returns the mapping of objects to serialize with pickle.
        # See https://docs.python.org/3/library/pickle.html#object.__getstate__
        # for details. We call _assemble_draws to ensure posterior samples have
        # been loaded prior to serialization.
        self._assemble_draws()
        return self.__dict__

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of all outputs from the sampler, comprising sampler parameters
        and all components of all model parameters, transformed parameters,
        and quantities of interest. Corresponds to Stan CSV file header row,
        with names munged to array notation, e.g. `beta[1]` not `beta.1`.
        """
        return self._metadata.cmdstan_config['column_names']  # type: ignore

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
        self._runset.save_csvfiles(dir)
