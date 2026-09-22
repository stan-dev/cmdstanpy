"""Container for the results of running autodiff variational inference"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cmdstanpy.utils import stancsv

from .base import SingleFileFit
from .metadata import VariationalConfig, VariationalRunConfig


@dataclass(kw_only=True)
class CmdStanVB(SingleFileFit[VariationalConfig]):
    """
    Container for outputs from CmdStan variational run.
    Created by :meth:`CmdStanModel.variational`.

    The first row of the output CSV holds the inferred variational means;
    the remaining rows hold the approximate posterior sample.
    """

    _eta: float | None = field(default=None, init=False)

    @classmethod
    def from_files(
        cls,
        csv_file: str | os.PathLike,
        config_file: str | os.PathLike,
        stdout_file: str | os.PathLike | None = None,
    ) -> CmdStanVB:
        return cls(
            **cls._from_files_kwargs(
                csv_file, config_file, stdout_file, VariationalRunConfig
            )
        )

    def _assemble(self) -> None:
        if self._draws.shape != (0,):
            return

        try:
            (
                comment_lines,
                _,
                draw_lines,
            ) = stancsv.parse_comments_header_and_draws(self.csv_file)
            self._eta = stancsv.parse_variational_eta(comment_lines)
            self._draws = stancsv.csv_bytes_list_to_numpy(draw_lines)
        except Exception as exc:
            raise ValueError(
                f"An error occurred when parsing Stan csv {self.csv_file}"
            ) from exc

    def _draws_for_inits(self) -> np.ndarray:
        """Exclude the variational mean stored in the first CSV row."""
        return self.variational_sample

    def __repr__(self) -> str:
        mc = self.config.method_config
        lines = [
            f'CmdStanVB: model={self.model_name}'
            f' method={mc.method} algorithm={mc.algorithm}',
            f' csv_file:\n\t{self.csv_file}',
        ]
        if self.config_file is not None:
            lines.append(f' config_file:\n\t{self.config_file}')
        if self.stdout_file is not None:
            lines.append(f' output_file:\n\t{self.stdout_file}')
        return '\n'.join(lines)

    @property
    def columns(self) -> int:
        """
        Total number of information items returned by sampler.
        Includes approximation information and names of model parameters
        and computed quantities.
        """
        return len(self.column_names)

    @property
    def eta(self) -> float:
        """
        Step size scaling parameter 'eta'
        """
        self._assemble()
        return self._eta  # type: ignore[return-value]

    @property
    def variational_params_np(self) -> np.ndarray:
        """
        Returns inferred parameter means as numpy array.
        """
        self._assemble()
        mean: np.ndarray = self._draws[0]
        return mean

    @property
    def variational_params_pd(self) -> pd.DataFrame:
        """
        Returns inferred parameter means as pandas DataFrame.
        """
        return pd.DataFrame(
            [self.variational_params_np], columns=self.column_names
        )

    @property
    def variational_params_dict(self) -> dict[str, np.ndarray]:
        """Returns inferred parameter means as Dict."""
        return OrderedDict(zip(self.column_names, self.variational_params_np))

    def stan_variable(self, var: str, *, mean: bool = False) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable, with
        a leading axis added for the number of draws from the variational
        approximation.

        * If the variable is a scalar variable, the return array has shape
          ( draws, ).
        * If the variable is a vector, the return array has shape
          ( draws, len(vector))
        * If the variable is a matrix, the return array has shape
          ( draws, size(dim 1), size(dim 2) )
        * If the variable is an array with N dimensions, the return array
          has shape ( draws, size(dim 1), ..., size(dim N))

        This functionality is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        :param mean: if True, return the variational mean. Otherwise,
            return the variational sample. Defaults to False.

        See Also
        --------
        CmdStanVB.stan_variables
        """
        if mean:
            draws = self.variational_params_np
        else:
            draws = self.variational_sample
        return self._extract_stan_var(var, draws)

    def stan_variables(self, *, mean: bool = False) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        :param mean: if True, return the variational mean. Otherwise,
            return the variational sample. Defaults to False.

        See Also
        --------
        CmdStanVB.stan_variable
        """
        result = {}
        for name in self.metadata.stan_vars:
            result[name] = self.stan_variable(name, mean=mean)
        return result

    @property
    def variational_sample(self) -> np.ndarray:
        """Returns the set of approximate posterior output draws."""
        self._assemble()
        return self._draws[1:]

    @property
    def variational_sample_pd(self) -> pd.DataFrame:
        """
        Returns the set of approximate posterior output draws as
        a pandas DataFrame.
        """
        return pd.DataFrame(self.variational_sample, columns=self.column_names)
