"""Container for the result of running optimization"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cmdstanpy.utils import get_logger

from .base import SingleFileFit
from .metadata import OptimizeConfig, OptimizeRunConfig

# Codes indicating successful convergence of MLE
# See stan::optimize::TerminationCondition in stan-dev/stan for definitions
_CONVERGED_CODES = frozenset((10, 20, 21, 30, 31))


@dataclass(kw_only=True)
class CmdStanMLE(SingleFileFit[OptimizeConfig]):
    """
    Container for outputs from CmdStan optimization.
    Created by :meth:`CmdStanModel.optimize`.

    The last row of the output CSV holds the final estimate; when the
    optimizer was run with ``save_iterations=True`` the preceding rows
    hold the intermediate iterations.
    """

    converged: bool = True

    @classmethod
    def from_files(
        cls,
        csv_file: str | os.PathLike,
        config_file: str | os.PathLike,
        stdout_file: str | os.PathLike | None = None,
        converged: bool = True,
    ) -> CmdStanMLE:
        fit = cls(
            converged=converged,
            **cls._from_files_kwargs(
                csv_file, config_file, stdout_file, OptimizeRunConfig
            ),
        )
        # Below conditional only true in CmdStan 2.39+
        if 'converged__' in fit.metadata.method_vars:
            # Intermediate rows have status 0, so saved iterations require
            # using the termination condition from the final row
            status = fit.method_variables()['converged__'][-1]
            fit.converged = status in _CONVERGED_CODES
        return fit

    def _warn_if_not_converged(self) -> None:
        if not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )

    def _check_for_saved_iterations(self) -> bool:
        """Whether saved iterations are available, warning when not."""
        if not self.config.method_config.save_iterations:
            get_logger().warning(
                'Intermediate iterations not saved to CSV output file. '
                'Rerun the optimize method with "save_iterations=True".'
            )
            return False
        return True

    def create_inits(
        self, seed: int | None = None, chains: int = 4
    ) -> dict[str, np.ndarray]:
        """
        Create initial values for the parameters of the model
        from the MLE.

        :param seed: Unused. Kept for compatibility with other
        create_inits methods.
        :param chains: Unused. Kept for compatibility with other
        create_inits methods.
        :return: The initial values for the parameters of the model.

        Returns a dictionary of MLE estimates in the format expected
        for the ``inits`` argument of :meth:`CmdStanModel.sample`.
        When running multi-chain sampling, all chains will be initialized
        at the same points.
        """
        # pylint: disable=unused-argument

        return self.stan_variables()

    def __repr__(self) -> str:
        mc = self.config.method_config
        lines = [
            f'CmdStanMLE: model={self.model_name}'
            f' method={mc.method} algorithm={mc.algorithm}',
            f' csv_file:\n\t{self.csv_file}',
        ]
        if self.config_file is not None:
            lines.append(f' config_file:\n\t{self.config_file}')
        if self.stdout_file is not None:
            lines.append(f' output_file:\n\t{self.stdout_file}')
        if not self.converged:
            lines.append(
                ' Warning: invalid estimate, optimization failed to converge.'
            )
        return '\n'.join(lines)

    @property
    def optimized_params_np(self) -> np.ndarray:
        """
        Returns all final estimates from the optimizer as a numpy.ndarray
        which contains all optimizer outputs, i.e., the value for `lp__`
        as well as all Stan program variables.
        """
        self._warn_if_not_converged()
        self._assemble()
        mle: np.ndarray = self._draws[-1]
        return mle

    @property
    def optimized_iterations_np(self) -> np.ndarray | None:
        """
        Returns all saved iterations from the optimizer and final estimate
        as a numpy.ndarray which contains all optimizer outputs, i.e.,
        the value for `lp__` as well as all Stan program variables.

        """
        if not self._check_for_saved_iterations():
            return None
        self._warn_if_not_converged()
        self._assemble()
        return self._draws

    @property
    def optimized_params_pd(self) -> pd.DataFrame:
        """
        Returns all final estimates from the optimizer as a pandas.DataFrame
        which contains all optimizer outputs, i.e., the value for `lp__`
        as well as all Stan program variables.
        """
        return pd.DataFrame(
            [self.optimized_params_np], columns=self.column_names
        )

    @property
    def optimized_iterations_pd(self) -> pd.DataFrame | None:
        """
        Returns all saved iterations from the optimizer and final estimate
        as a pandas.DataFrame which contains all optimizer outputs, i.e.,
        the value for `lp__` as well as all Stan program variables.

        """
        iters = self.optimized_iterations_np
        if iters is None:
            return None
        return pd.DataFrame(iters, columns=self.column_names)

    @property
    def optimized_params_dict(self) -> dict[str, np.float64]:
        """
        Returns all estimates from the optimizer, including `lp__` as a
        Python Dict.  Only returns estimate from final iteration.
        """
        return OrderedDict(zip(self.column_names, self.optimized_params_np))

    def stan_variable(
        self,
        var: str,
        *,
        inc_iterations: bool = False,
        warn: bool = True,
    ) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        This functionality is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        :param inc_iterations: When ``True`` and the intermediate estimates
            are included in the output, i.e., the optimizer was run with
            ``save_iterations=True``, then intermediate estimates are included.
            Default value is ``False``.

        See Also
        --------
        CmdStanMLE.stan_variables
        """
        if var not in self.metadata.stan_vars:
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are ' + ", ".join(self.metadata.stan_vars)
            )
        save_iterations = self.config.method_config.save_iterations
        if warn and inc_iterations and not save_iterations:
            self._check_for_saved_iterations()
        if warn:
            self._warn_if_not_converged()
        self._assemble()
        if inc_iterations and save_iterations:
            data = self._draws
        else:
            data = self._draws[-1]

        return self._extract_stan_var(var, data)

    def stan_variables(
        self, inc_iterations: bool = False
    ) -> dict[str, np.ndarray]:
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
        """
        self._warn_if_not_converged()
        result = {}
        for name in self.metadata.stan_vars:
            result[name] = self.stan_variable(
                name, inc_iterations=inc_iterations, warn=False
            )
        return result
