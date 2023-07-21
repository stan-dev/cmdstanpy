"""Container for the result of running optimization"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cmdstanpy.cmdstan_args import Method, OptimizeArgs
from cmdstanpy.utils import BaseType, get_logger, scan_optimize_csv

from .metadata import InferenceMetadata
from .runset import RunSet


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
        self._save_iterations: bool = optimize_args.save_iterations
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

    def __getattr__(self, attr: str) -> Union[np.ndarray, float]:
        """Synonymous with ``fit.stan_variable(attr)"""
        if attr.startswith("_"):
            raise AttributeError(f"Unknown variable name {attr}")
        try:
            return self.stan_variable(attr)
        except ValueError as e:
            # pylint: disable=raise-missing-from
            raise AttributeError(*e.args)

    def _set_mle_attrs(self, sample_csv_0: str) -> None:
        meta = scan_optimize_csv(sample_csv_0, self._save_iterations)
        self._metadata = InferenceMetadata(meta)
        self._column_names: Tuple[str, ...] = meta['column_names']
        self._mle: np.ndarray = meta['mle']
        if self._save_iterations:
            self._all_iters: np.ndarray = meta['all_iters']

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
    def optimized_params_dict(self) -> Dict[str, np.float64]:
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
        var: str,
        *,
        inc_iterations: bool = False,
        warn: bool = True,
    ) -> Union[np.ndarray, float]:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        This functionaltiy is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

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
        if var not in self._metadata.stan_vars_dims:
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are '
                + ", ".join(self._metadata.stan_vars_dims)
            )
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
                result = self._all_iters[:, col_idxs]
            else:
                result = self._mle[col_idxs]
                dims = dims[1:]

            if self._metadata.stan_vars_types[var] == BaseType.COMPLEX:
                result = result[..., ::2] + 1j * result[..., 1::2]
                dims = dims[:-1]

            result = result.reshape(dims, order='F')

            return result

        else:  # scalar var
            col_idx = col_idxs[0]
            if num_rows > 1:
                return self._all_iters[:, col_idx]
            else:
                return float(self._mle[col_idx])

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
