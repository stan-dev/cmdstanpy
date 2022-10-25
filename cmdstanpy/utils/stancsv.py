"""
Utility functions for reading the Stan CSV format
"""
import json
import math
import re
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    MutableMapping,
    Optional,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP


class BaseType(Enum):
    """Stan langauge base type"""

    COMPLEX = auto()
    PRIM = auto()  # future: int / real

    def __repr__(self) -> str:
        return '<%s.%s>' % (self.__class__.__name__, self.name)


def check_sampler_csv(
    path: str,
    is_fixed_param: bool = False,
    iter_sampling: Optional[int] = None,
    iter_warmup: Optional[int] = None,
    save_warmup: bool = False,
    thin: Optional[int] = None,
) -> Dict[str, Any]:
    """Capture essential config, shape from stan_csv file."""
    meta = scan_sampler_csv(path, is_fixed_param)
    if thin is None:
        thin = _CMDSTAN_THIN
    elif thin > _CMDSTAN_THIN:
        if 'thin' not in meta:
            raise ValueError(
                'bad Stan CSV file {}, '
                'config error, expected thin = {}'.format(path, thin)
            )
        if meta['thin'] != thin:
            raise ValueError(
                'bad Stan CSV file {}, '
                'config error, expected thin = {}, found {}'.format(
                    path, thin, meta['thin']
                )
            )
    draws_sampling = iter_sampling
    if draws_sampling is None:
        draws_sampling = _CMDSTAN_SAMPLING
    draws_warmup = iter_warmup
    if draws_warmup is None:
        draws_warmup = _CMDSTAN_WARMUP
    draws_warmup = int(math.ceil(draws_warmup / thin))
    draws_sampling = int(math.ceil(draws_sampling / thin))
    if meta['draws_sampling'] != draws_sampling:
        raise ValueError(
            'bad Stan CSV file {}, expected {} draws, found {}'.format(
                path, draws_sampling, meta['draws_sampling']
            )
        )
    if save_warmup:
        if not ('save_warmup' in meta and meta['save_warmup'] == 1):
            raise ValueError(
                'bad Stan CSV file {}, '
                'config error, expected save_warmup = 1'.format(path)
            )
        if meta['draws_warmup'] != draws_warmup:
            raise ValueError(
                'bad Stan CSV file {}, '
                'expected {} warmup draws, found {}'.format(
                    path, draws_warmup, meta['draws_warmup']
                )
            )
    return meta


def scan_sampler_csv(path: str, is_fixed_param: bool = False) -> Dict[str, Any]:
    """Process sampler stan_csv output file line by line."""
    dict: Dict[str, Any] = {}
    lineno = 0
    with open(path, 'r') as fd:
        try:
            lineno = scan_config(fd, dict, lineno)
            lineno = scan_column_names(fd, dict, lineno)
            if not is_fixed_param:
                lineno = scan_warmup_iters(fd, dict, lineno)
                lineno = scan_hmc_params(fd, dict, lineno)
            lineno = scan_sampling_iters(fd, dict, lineno, is_fixed_param)
        except ValueError as e:
            raise ValueError("Error in reading csv file: " + path) from e
    return dict


def scan_optimize_csv(path: str, save_iters: bool = False) -> Dict[str, Any]:
    """Process optimizer stan_csv output file line by line."""
    dict: Dict[str, Any] = {}
    lineno = 0
    # scan to find config, header, num saved iters
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
        iters = 0
        for line in fd:
            iters += 1
    if save_iters:
        all_iters: np.ndarray = np.empty(
            (iters, len(dict['column_names'])), dtype=float, order='F'
        )
    # rescan to capture estimates
    with open(path, 'r') as fd:
        for i in range(lineno):
            fd.readline()
        for i in range(iters):
            line = fd.readline().strip()
            if len(line) < 1:
                raise ValueError(
                    'cannot parse CSV file {}, error at line {}'.format(
                        path, lineno + i
                    )
                )
            xs = line.split(',')
            if save_iters:
                all_iters[i, :] = [float(x) for x in xs]
            if i == iters - 1:
                mle: np.ndarray = np.array(xs, dtype=float)
    dict['mle'] = mle
    if save_iters:
        dict['all_iters'] = all_iters
    return dict


def scan_generated_quantities_csv(path: str) -> Dict[str, Any]:
    """
    Process standalone generated quantities stan_csv output file line by line.
    """
    dict: Dict[str, Any] = {}
    lineno = 0
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
    return dict


def scan_variational_csv(path: str) -> Dict[str, Any]:
    """Process advi stan_csv output file line by line."""
    dict: Dict[str, Any] = {}
    lineno = 0
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
        line = fd.readline().lstrip(' #\t').rstrip()
        lineno += 1
        if line.startswith('Stepsize adaptation complete.'):
            line = fd.readline().lstrip(' #\t\n')
            lineno += 1
            if not line.startswith('eta'):
                raise ValueError(
                    'line {}: expecting eta, found:\n\t "{}"'.format(
                        lineno, line
                    )
                )
            _, eta = line.split('=')
            dict['eta'] = float(eta)
            line = fd.readline().lstrip(' #\t\n')
            lineno += 1
        xs = line.split(',')
        variational_mean = [float(x) for x in xs]
        dict['variational_mean'] = np.array(variational_mean)
        dict['variational_sample'] = pd.read_csv(
            path,
            comment='#',
            skiprows=lineno,
            header=None,
            float_precision='high',
        ).to_numpy()
    return dict


def scan_config(fd: TextIO, config_dict: Dict[str, Any], lineno: int) -> int:
    """
    Scan initial stan_csv file comments lines and
    save non-default configuration information to config_dict.
    """
    cur_pos = fd.tell()
    line = fd.readline().strip()
    while len(line) > 0 and line.startswith('#'):
        lineno += 1
        if line.endswith('(Default)'):
            line = line.replace('(Default)', '')
        line = line.lstrip(' #\t')
        key_val = line.split('=')
        if len(key_val) == 2:
            if key_val[0].strip() == 'file' and not key_val[1].endswith('csv'):
                config_dict['data_file'] = key_val[1].strip()
            elif key_val[0].strip() != 'file':
                raw_val = key_val[1].strip()
                val: Union[int, float, str]
                try:
                    val = int(raw_val)
                except ValueError:
                    try:
                        val = float(raw_val)
                    except ValueError:
                        val = raw_val
                config_dict[key_val[0].strip()] = val
        cur_pos = fd.tell()
        line = fd.readline().strip()
    fd.seek(cur_pos)
    return lineno


def scan_warmup_iters(
    fd: TextIO, config_dict: Dict[str, Any], lineno: int
) -> int:
    """
    Check warmup iterations, if any.
    """
    if 'save_warmup' not in config_dict:
        return lineno
    cur_pos = fd.tell()
    line = fd.readline().strip()
    draws_found = 0
    while len(line) > 0 and not line.startswith('#'):
        lineno += 1
        draws_found += 1
        cur_pos = fd.tell()
        line = fd.readline().strip()
    fd.seek(cur_pos)
    config_dict['draws_warmup'] = draws_found
    return lineno


def scan_column_names(
    fd: TextIO, config_dict: MutableMapping[str, Any], lineno: int
) -> int:
    """
    Process columns header, add to config_dict as 'column_names'
    """
    line = fd.readline().strip()
    lineno += 1
    names = line.split(',')
    config_dict['column_names_raw'] = tuple(names)
    config_dict['column_names'] = tuple(munge_varnames(names))
    return lineno


def munge_varnames(names: List[str]) -> List[str]:
    """
    Change formatting for indices of container var elements
    from use of dot separator to array-like notation, e.g.,
    rewrite label ``y_forecast.2.4`` to ``y_forecast[2,4]``.
    """
    if names is None:
        raise ValueError('missing argument "names"')
    result = []
    for name in names:
        if '.' not in name:
            result.append(name)
        else:
            head, *rest = name.split('.')
            result.append(''.join([head, '[', ','.join(rest), ']']))
    return result


def parse_method_vars(names: Tuple[str, ...]) -> Dict[str, Tuple[int, ...]]:
    """
    Parses out names ending in `__` from list of CSV file column names.
    Return a dict mapping sampler variable name to Stan CSV file column, using
    zero-based column indexing.
    Currently, (Stan 2.X) all CmdStan inference method vars are scalar,
    the map entries are tuples of int to allow for structured variables.
    """
    if names is None:
        raise ValueError('missing argument "names"')
    # note: method vars are currently all scalar so not checking for structure
    return {v: (k,) for (k, v) in enumerate(names) if v.endswith('__')}


def parse_stan_vars(
    names: Tuple[str, ...]
) -> Tuple[
    Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]], Dict[str, BaseType]
]:
    """
    Parses out Stan variable names (i.e., names not ending in `__`)
    from list of CSV file column names.
    Returns three dicts which map variable names to base type, dimensions and
    CSV file columns, respectively, using zero-based column indexing.
    Note: assumes: (a) munged varnames and (b) container vars are non-ragged
    and dense; no checks on size, indices.
    """
    if names is None:
        raise ValueError('missing argument "names"')
    dims_map: Dict[str, Tuple[int, ...]] = {}
    cols_map: Dict[str, Tuple[int, ...]] = {}
    types_map: Dict[str, BaseType] = {}
    idxs = []
    dims: Union[List[str], List[int]]
    for (idx, name) in enumerate(names):
        if name.endswith('real]') or name.endswith('imag]'):
            basetype = BaseType.COMPLEX
        else:
            basetype = BaseType.PRIM
        idxs.append(idx)
        var, *dims = name.split('[')
        if var.endswith('__'):
            idxs = []
        elif len(dims) == 0:
            dims_map[var] = ()
            cols_map[var] = tuple(idxs)
            types_map[var] = basetype
            idxs = []
        else:
            if idx < len(names) - 1 and names[idx + 1].split('[')[0] == var:
                continue
            coords = dims[0][:-1].split(',')
            if coords[-1] == 'imag':
                dims = [int(x) for x in coords[:-1]]
                dims.append(2)
            else:
                dims = [int(x) for x in coords]
            dims_map[var] = tuple(dims)
            cols_map[var] = tuple(idxs)
            types_map[var] = basetype
            idxs = []
    return (dims_map, cols_map, types_map)


def scan_hmc_params(
    fd: TextIO, config_dict: Dict[str, Any], lineno: int
) -> int:
    """
    Scan step size, metric from  stan_csv file comment lines.
    """
    metric = config_dict['metric']
    line = fd.readline().strip()
    lineno += 1
    if not line == '# Adaptation terminated':
        raise ValueError(
            'line {}: expecting metric, found:\n\t "{}"'.format(lineno, line)
        )
    line = fd.readline().strip()
    lineno += 1
    label, step_size = line.split('=')
    if not label.startswith('# Step size'):
        raise ValueError(
            'line {}: expecting step size, '
            'found:\n\t "{}"'.format(lineno, line)
        )
    try:
        float(step_size.strip())
    except ValueError as e:
        raise ValueError(
            'line {}: invalid step size: {}'.format(lineno, step_size)
        ) from e
    if metric == 'unit_e':
        return lineno
    line = fd.readline().strip()
    lineno += 1
    if not (
        (
            metric == 'diag_e'
            and line == '# Diagonal elements of inverse mass matrix:'
        )
        or (
            metric == 'dense_e' and line == '# Elements of inverse mass matrix:'
        )
    ):
        raise ValueError(
            'line {}: invalid or missing mass matrix '
            'specification'.format(lineno)
        )
    line = fd.readline().lstrip(' #\t')
    lineno += 1
    num_unconstrained_params = len(line.split(','))
    if metric == 'diag_e':
        return lineno
    else:
        for _ in range(1, num_unconstrained_params):
            line = fd.readline().lstrip(' #\t')
            lineno += 1
            if len(line.split(',')) != num_unconstrained_params:
                raise ValueError(
                    'line {}: invalid or missing mass matrix '
                    'specification'.format(lineno)
                )
        return lineno


def scan_sampling_iters(
    fd: TextIO, config_dict: Dict[str, Any], lineno: int, is_fixed_param: bool
) -> int:
    """
    Parse sampling iteration, save number of iterations to config_dict.
    Also save number of divergences, max_treedepth hits
    """
    draws_found = 0
    num_cols = len(config_dict['column_names'])
    if not is_fixed_param:
        idx_divergent = config_dict['column_names'].index('divergent__')
        idx_treedepth = config_dict['column_names'].index('treedepth__')
        max_treedepth = config_dict['max_depth']
        ct_divergences = 0
        ct_max_treedepth = 0

    cur_pos = fd.tell()
    line = fd.readline().strip()
    while len(line) > 0 and not line.startswith('#'):
        lineno += 1
        draws_found += 1
        data = line.split(',')
        if len(data) != num_cols:
            raise ValueError(
                'line {}: bad draw, expecting {} items, found {}\n'.format(
                    lineno, num_cols, len(line.split(','))
                )
                + 'This error could be caused by running out of disk space.\n'
                'Try clearing up TEMP or setting output_dir to a path'
                ' on another drive.',
            )
        cur_pos = fd.tell()
        line = fd.readline().strip()
        if not is_fixed_param:
            ct_divergences += int(data[idx_divergent])  # type: ignore
            if int(data[idx_treedepth]) == max_treedepth:  # type: ignore
                ct_max_treedepth += 1

    fd.seek(cur_pos)
    config_dict['draws_sampling'] = draws_found
    if not is_fixed_param:
        config_dict['ct_divergences'] = ct_divergences
        config_dict['ct_max_treedepth'] = ct_max_treedepth
    return lineno


def read_metric(path: str) -> List[int]:
    """
    Read metric file in JSON or Rdump format.
    Return dimensions of entry "inv_metric".
    """
    if path.endswith('.json'):
        with open(path, 'r') as fd:
            metric_dict = json.load(fd)
        if 'inv_metric' in metric_dict:
            dims_np: np.ndarray = np.asarray(metric_dict['inv_metric'])
            return list(dims_np.shape)
        else:
            raise ValueError(
                'metric file {}, bad or missing'
                ' entry "inv_metric"'.format(path)
            )
    else:
        dims = list(read_rdump_metric(path))
        if dims is None:
            raise ValueError(
                'metric file {}, bad or missing'
                ' entry "inv_metric"'.format(path)
            )
        return dims


def read_rdump_metric(path: str) -> List[int]:
    """
    Find dimensions of variable named 'inv_metric' in Rdump data file.
    """
    metric_dict = rload(path)
    if metric_dict is None or not (
        'inv_metric' in metric_dict
        and isinstance(metric_dict['inv_metric'], np.ndarray)
    ):
        raise ValueError(
            'metric file {}, bad or missing entry "inv_metric"'.format(path)
        )
    return list(metric_dict['inv_metric'].shape)


def rload(fname: str) -> Optional[Dict[str, Union[int, float, np.ndarray]]]:
    """Parse data and parameter variable values from an R dump format file.
    This parser only supports the subset of R dump data as described
    in the "Dump Data Format" section of the CmdStan manual, i.e.,
    scalar, vector, matrix, and array data types.
    """
    data_dict = {}
    with open(fname, 'r') as fd:
        lines = fd.readlines()
    # Variable data may span multiple lines, parse accordingly
    idx = 0
    while idx < len(lines) and '<-' not in lines[idx]:
        idx += 1
    if idx == len(lines):
        return None
    start_idx = idx
    idx += 1
    while True:
        while idx < len(lines) and '<-' not in lines[idx]:
            idx += 1
        next_var = idx
        var_data = ''.join(lines[start_idx:next_var]).replace('\n', '')
        lhs, rhs = [item.strip() for item in var_data.split('<-')]
        lhs = lhs.replace('"', '')  # strip optional Jags double quotes
        rhs = rhs.replace('L', '')  # strip R long int qualifier
        data_dict[lhs] = parse_rdump_value(rhs)
        if idx == len(lines):
            break
        start_idx = next_var
        idx += 1
    return data_dict


def parse_rdump_value(rhs: str) -> Union[int, float, np.ndarray]:
    """Process right hand side of Rdump variable assignment statement.
    Value is either scalar, vector, or multi-dim structure.
    Use regex to capture structure values, dimensions.
    """
    pat = re.compile(
        r'structure\(\s*c\((?P<vals>[^)]*)\)'
        r'(,\s*\.Dim\s*=\s*c\s*\((?P<dims>[^)]*)\s*\))?\)'
    )
    val: Union[int, float, np.ndarray]
    try:
        if rhs.startswith('structure'):
            parse = pat.match(rhs)
            if parse is None or parse.group('vals') is None:
                raise ValueError(rhs)
            vals = [float(v) for v in parse.group('vals').split(',')]
            val = np.array(vals, order='F')
            if parse.group('dims') is not None:
                dims = [int(v) for v in parse.group('dims').split(',')]
                val = np.array(vals).reshape(dims, order='F')
        elif rhs.startswith('c(') and rhs.endswith(')'):
            val = np.array([float(item) for item in rhs[2:-1].split(',')])
        elif '.' in rhs or 'e' in rhs:
            val = float(rhs)
        else:
            val = int(rhs)
    except TypeError as e:
        raise ValueError('bad value in Rdump file: {}'.format(rhs)) from e
    return val
