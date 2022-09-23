"""
Utilities for writing Stan Json files
"""
import json
from collections.abc import Collection
from typing import Any, List, Mapping

import numpy as np


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
                # handles cases like val == ['hello']
                np.isfinite(val)
            except TypeError:
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

    with open(path, 'w') as fd:
        json.dump(data_out, fd, default=serialize_complex)
