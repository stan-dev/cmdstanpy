"""
    Container for the result of running a laplace approximation.
"""

from typing import Optional

import numpy as np

from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils.data_munging import extract_reshape
from cmdstanpy.utils.stancsv import scan_laplace_csv

# from cmdstanpy.cmdstan_args import LaplaceArgs,
from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet


class CmdStanLaplace:
    def __init__(self, runset: RunSet, mode: CmdStanMLE) -> None:
        """Initialize object."""
        if not runset.method == Method.LAPLACE:
            raise ValueError(
                'Wrong runset method, expecting laplace runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self.mode = mode

        self._draws: np.ndarray = np.array(())

        config = scan_laplace_csv(runset.csv_files[0])
        self._metadata = InferenceMetadata(config)

    def _assemble_draws(self) -> None:
        if self._draws.shape != (0,):
            return

        with open(self.runset.csv_files[0], 'r') as fd:
            lines = (line for line in fd if not line.startswith('#'))
            self._draws = np.loadtxt(
                lines,
                dtype=float,
                ndmin=2,
                skiprows=1,
                delimiter=',',
            )

    def stan_variable(self, var: str) -> np.ndarray:

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
