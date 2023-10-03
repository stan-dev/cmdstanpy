"""Container for the results of running autodiff variational inference"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils import scan_variational_csv
from cmdstanpy.utils.logging import get_logger

from .metadata import InferenceMetadata
from .runset import RunSet


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

    def __getattr__(self, attr: str) -> Union[np.ndarray, float]:
        """Synonymous with ``fit.stan_variable(attr)"""
        if attr.startswith("_"):
            raise AttributeError(f"Unknown variable name {attr}")
        try:
            return self.stan_variable(attr)
        except ValueError as e:
            # pylint: disable=raise-missing-from
            raise AttributeError(*e.args)

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
        self, var: str, *, mean: Optional[bool] = None
    ) -> Union[np.ndarray, float]:
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

        This functionaltiy is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        :param mean: if True, return the variational mean. Otherwise,
            return the variational sample.  The default behavior will
            change in a future release to return the variational sample.

        See Also
        --------
        CmdStanVB.stan_variables
        CmdStanMCMC.stan_variable
        CmdStanMLE.stan_variable
        CmdStanPathfinder.stan_variable
        CmdStanGQ.stan_variable
        CmdStanLaplace.stan_variable
        """
        # TODO(2.0): remove None case, make default `False`
        if mean is None:
            get_logger().warning(
                "The default behavior of CmdStanVB.stan_variable() "
                "will change in a future release to return the "
                "variational sample, rather than the mean.\n"
                "To maintain the current behavior, pass the argument "
                "mean=True"
            )
            mean = True
        if mean:
            draws = self._variational_mean
        else:
            draws = self._variational_sample

        try:
            out: np.ndarray = self._metadata.stan_vars[var].extract_reshape(
                draws
            )
            # TODO(2.0): remove
            if out.shape == () or out.shape == (1,):
                if mean:
                    get_logger().warning(
                        "The default behavior of "
                        "CmdStanVB.stan_variable(mean=True) will change in a "
                        "future release to always return a numpy.ndarray, even "
                        "for scalar variables."
                    )
                return out.item()  # type: ignore
            return out
        except KeyError:
            # pylint: disable=raise-missing-from
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are '
                + ", ".join(self._metadata.stan_vars.keys())
            )

    def stan_variables(
        self, *, mean: Optional[bool] = None
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        See Also
        --------
        CmdStanVB.stan_variable
        CmdStanMCMC.stan_variables
        CmdStanMLE.stan_variables
        CmdStanGQ.stan_variables
        CmdStanPathfinder.stan_variables
        CmdStanLaplace.stan_variables
        """
        result = {}
        for name in self._metadata.stan_vars:
            result[name] = self.stan_variable(name, mean=mean)
        return result

    @property
    def variational_sample(self) -> np.ndarray:
        """Returns the set of approximate posterior output draws."""
        return self._variational_sample

    @property
    def variational_sample_pd(self) -> pd.DataFrame:
        """
        Returns the set of approximate posterior output draws as
        a pandas DataFrame.
        """
        return pd.DataFrame(self._variational_sample, columns=self.column_names)

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
