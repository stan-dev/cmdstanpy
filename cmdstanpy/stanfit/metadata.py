"""Container for metadata parsed from the output of a CmdStan run"""

import copy
from typing import Any, Dict, Tuple

from cmdstanpy.utils import BaseType, parse_method_vars, parse_stan_vars


class InferenceMetadata:
    """
    CmdStan configuration and contents of output file parsed out of
    the Stan CSV file header comments and column headers.
    Assumes valid CSV files.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize object from CSV headers"""
        self._cmdstan_config = config
        self._method_vars_cols = parse_method_vars(names=config['column_names'])
        stan_vars_dims, stan_vars_cols, stan_vars_types = parse_stan_vars(
            names=config['column_names']
        )
        self._stan_vars_dims = stan_vars_dims
        self._stan_vars_cols = stan_vars_cols
        self._stan_vars_types = stan_vars_types

    def __repr__(self) -> str:
        return 'Metadata:\n{}\n'.format(self._cmdstan_config)

    @property
    def cmdstan_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing a set of name, value pairs
        parsed out of the Stan CSV file header.  These include the
        command configuration and the CSV file header row information.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._cmdstan_config)

    @property
    def method_vars_cols(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns a map from a Stan inference method variable to
        a tuple of column indices in inference engine's output array.
        Method variable names always end in `__`, e.g. `lp__`.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._method_vars_cols)

    @property
    def stan_vars_cols(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns a map from a Stan program variable name to a
        tuple of the column indices in the vector or matrix of
        estimates produced by a CmdStan inference method.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._stan_vars_cols)

    @property
    def stan_vars_dims(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns map from Stan program variable names to variable dimensions.
        Scalar types are mapped to the empty tuple, e.g.,
        program variable ``int foo`` has dimension ``()`` and
        program variable ``vector[10] bar`` has single dimension ``(10)``.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._stan_vars_dims)

    @property
    def stan_vars_types(self) -> Dict[str, BaseType]:
        """
        Returns map from Stan program variable names to variable base type.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._stan_vars_types)
