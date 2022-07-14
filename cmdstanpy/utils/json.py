"""
Utilities for writing Stan Json files
"""
import json
import math
from collections.abc import Collection
from typing import Any, List, Mapping, Union

import numpy as np
import ujson

from .logging import get_logger


def rewrite_inf_nan(
    data: Union[float, int, List[Any]]
) -> Union[str, int, float, List[Any]]:
    """Replaces NaN and Infinity with string representations"""
    if isinstance(data, float):
        if math.isnan(data):
            return 'NaN'
        if math.isinf(data):
            return ('+' if data > 0 else '-') + 'inf'
        return data
    elif isinstance(data, list):
        return [rewrite_inf_nan(item) for item in data]
    else:
        return data


def serialize_complex(c: Any) -> List[float]:
    if isinstance(c, complex):
        return [c.real, c.imag]
    else:
        raise TypeError(f"Unserializable type: {type(c)}")


def write_stan_json(path: str, data: Mapping[str, Any]) -> None:
    """
    Dump a mapping of strings to data to a JSON file.

    Values can be any numeric type, a boolean (converted to int),
    or any collection compatible with :func:`numpy.asarray`, e.g a
    :class:`pandas.Series`.

    Produces a file compatible with the
    `Json Format for Cmdstan
    <https://mc-stan.org/docs/cmdstan-guide/json.html>`__

    :param path: File path for the created json. Will be overwritten if
        already in existence.

    :param data: A mapping from strings to values. This can be a dictionary
        or something more exotic like an :class:`xarray.Dataset`. This will be
        copied before type conversion, not modified
    """
    data_out = {}
    for key, val in data.items():
        handle_nan_inf = False
        if val is not None:
            if isinstance(val, (str, bytes)) or (
                type(val).__module__ != 'numpy'
                and not isinstance(val, (Collection, bool, int, float))
            ):
                raise TypeError(
                    f"Invalid type '{type(val)}' provided to "
                    + f"write_stan_json for key '{key}'"
                )
            try:
                handle_nan_inf = not np.all(np.isfinite(val))
            except TypeError:
                # handles cases like val == ['hello']
                # pylint: disable=raise-missing-from
                raise ValueError(
                    "Invalid type provided to "
                    f"write_stan_json for key '{key}' "
                    f"as part of collection {type(val)}"
                )

        if type(val).__module__ == 'numpy':
            data_out[key] = val.tolist()
        elif isinstance(val, Collection):
            data_out[key] = np.asarray(val).tolist()
        elif isinstance(val, bool):
            data_out[key] = int(val)
        else:
            data_out[key] = val

        if handle_nan_inf:
            data_out[key] = rewrite_inf_nan(data_out[key])

    with open(path, 'w') as fd:
        try:
            ujson.dump(data_out, fd)
        except TypeError as e:
            get_logger().debug(e)
            json.dump(data_out, fd, default=serialize_complex)
