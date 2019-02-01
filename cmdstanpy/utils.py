"""
I/O functions for working with CmdStan executables.

"""

import numpy as np


def _rdump_array(key, val):
    c = 'c(' + ', '.join(map(str, val.T.flat)) + ')'
    if (val.size, ) == val.shape:
        return '{key} <- {c}'.format(key=key, c=c)
    else:
        dim = '.Dim = c{0}'.format(val.shape)
        struct = '{key} <- structure({c}, {dim})'.format(key=key, c=c, dim=dim)
        return struct


def rdump(path, data):
    """Dump a dict of data to a R dump format file.
    """
    with open(path, 'w') as fd:
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.size > 1:
                line = _rdump_array(key, val)
            else:
                try:
                    val = val.flat[0]
                except AttributeError:
                    pass
                line = '%s <- %s' % (key, val)
            fd.write(line)
            fd.write('\n')

