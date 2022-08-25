"""
Common functions for reshaping numpy arrays
"""
from typing import Hashable, MutableMapping, Tuple

import numpy as np

from .stancsv import BaseType


def flatten_chains(draws_array: np.ndarray) -> np.ndarray:
    """
    Flatten a 3D array of draws X chains X variable into 2D array
    where all chains are concatenated into a single column.

    :param draws_array: 3D array of draws
    """
    if len(draws_array.shape) != 3:
        raise ValueError(
            'Expecting 3D array, found array with {} dims'.format(
                len(draws_array.shape)
            )
        )

    num_rows = draws_array.shape[0] * draws_array.shape[1]
    num_cols = draws_array.shape[2]
    return draws_array.reshape((num_rows, num_cols), order='F')


def build_xarray_data(
    data: MutableMapping[Hashable, Tuple[Tuple[str, ...], np.ndarray]],
    var_name: str,
    dims: Tuple[int, ...],
    col_idxs: Tuple[int, ...],
    start_row: int,
    drawset: np.ndarray,
    var_type: BaseType,
) -> None:
    """
    Adds Stan variable name, labels, and values to a dictionary
    that will be used to construct an xarray DataSet.
    """
    var_dims: Tuple[str, ...] = ('draw', 'chain')
    if dims:
        var_dims += tuple(f"{var_name}_dim_{i}" for i in range(len(dims)))

        draws = drawset[start_row:, :, col_idxs]

        if var_type == BaseType.COMPLEX:
            draws = draws[..., ::2] + 1j * draws[..., 1::2]
            var_dims = var_dims[:-1]
            dims = dims[:-1]

        draws = draws.reshape(*drawset.shape[:2], *dims, order="F")

        data[var_name] = (
            var_dims,
            draws,
        )

    else:
        data[var_name] = (
            var_dims,
            np.squeeze(drawset[start_row:, :, col_idxs], axis=2),
        )
