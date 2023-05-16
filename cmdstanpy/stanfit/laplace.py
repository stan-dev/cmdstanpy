"""
    Container for the result of running a laplace approximation.
"""

from typing import Dict, Optional

import numpy as np

from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils.data_munging import extract_reshape
from cmdstanpy.utils.stancsv import scan_laplace_csv

from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet

# TODO list:
# - remaining methods
# - tests
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

        config = scan_laplace_csv(runset.csv_files[0])
        self._metadata = InferenceMetadata(config)

    def _assemble_draws(self) -> None:
        if self._draws.shape != (0,):
            return

        # TODO: should we fake a chain dimension?
        with open(self._runset.csv_files[0], 'r') as fd:
            lines = (line for line in fd if not line.startswith('#'))
            self._draws = np.loadtxt(
                lines,
                dtype=float,
                ndmin=2,
                skiprows=1,
                delimiter=',',
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
        CmdStanVB.stan_variable
        CmdStanGQ.stan_variable
        """
        self._assemble_draws()
        draws = self._draws
        dims = (draws.shape[0],)
        col_idxs = self._metadata.stan_vars_cols[var]
        return extract_reshape(
            dims=dims + self._metadata.stan_vars_dims[var],
            col_idxs=col_idxs,
            var_type=self._metadata.stan_vars_types[var],
            start_row=0,
            draws_in=draws,
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
        CmdStanVB.stan_variables
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
        for name, idxs in self._metadata.method_vars_cols.items():
            result[name] = self._draws[..., idxs[0]]
        return result

    # def draws
    # def draws_pd
    # def draws_xr

    @property
    def mode(self) -> CmdStanMLE:
        """
        Return the maximum a posteriori estimate (mode)
        as a :class:`CmdStanMLE` object.
        """
        return self._mode

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
