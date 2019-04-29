"""
Utility functions
"""

from typing import Dict
import numpy as np


def _rdump_array(key:str, val:np.ndarray) -> str:
    """Flatten numpy ndarray, format as Rdump variable declaration."""
    c = 'c(' + ', '.join(map(str, val.T.flat)) + ')'
    if (val.size, ) == val.shape:
        return '{key} <- {c}'.format(key=key, c=c)
    else:
        dim = '.Dim = c{}'.format(val.shape)
        struct = '{key} <- structure({c}, {dim})'.format(key=key, c=c, dim=dim)
        return struct


def rdump(path:str, data:Dict) -> None:
    """Dump a dict of data to a R dump format file."""
    with open(path, 'w') as fd:
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.size > 1:
                line = _rdump_array(key, val)
            elif isinstance(val, list) and len(val) > 1:
                line = _rdump_array(key, np.asarray(val))
            else:

                try:
                    val = val.flat[0]
                except AttributeError:
                    pass
                line = '%s <- %s' % (key, val)
            fd.write(line)
            fd.write('\n')


def scan_stan_csv(filename:str) -> Dict:
    """Capture essential config, shape from stan_csv file."""
    dict = {}
    draws_found = 0
    file_is_data = False
    with open(filename) as fp:
        line = fp.readline().strip()
        while len(line) > 0 and line.startswith("#"):
            if line == "# data":
                file_is_data = True
            elif line == "# random":
                file_is_data = False
            if not line.endswith("(Default)"):
                line = line.lstrip(' #\t')
                key_val = line.split('=')
                if len(key_val) == 2:
                    if key_val[0].strip() == "file":
                        if file_is_data:
                            dict['data_file'] = key_val[1].strip()
                    else:
                        dict[key_val[0].strip()] = key_val[1].strip()
            line = fp.readline().strip()
        dict['col_headers'] = line.split(',')
        cols_header = len(dict['col_headers'])
        line = fp.readline().strip()
        while len(line) > 0 and line.startswith("#"):
            line = fp.readline().strip()
        while len(line) > 0 and not line.startswith("#"):
            draws_found += 1
            cols_draw = len(line.split(','))
            if cols_header != cols_draw:
                raise ValueError(
                    'bad csv file {}, expected {} columns, found {}'.format(
                        filename, cols_header, cols_draw))
            line = fp.readline().strip()
    # check draws against spec
    draws_spec = 1000
    num_warmup = 1000
    if 'num_samples' in dict:
        draws_spec = int(dict['num_samples'])
    if 'num_warmup' in dict:
        num_warmup = int(dict['num_warmup'])
    if 'save_warmup' in dict and dict['save_warmup'] == '1':
        draws_spec = draws_spec + num_warmup
    if draws_found != draws_spec:
        raise ValueError('bad csv file {}, expected {} draws, found {}'.format(
            filename, draws_spec, draws_found))
    dict['draws'] = draws_found
    return dict
