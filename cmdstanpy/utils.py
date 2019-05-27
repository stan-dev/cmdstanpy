"""
Utility functions
"""
import os
import os.path
import json
import subprocess
import numpy as np
from typing import Dict, TextIO


def validate_cmdstan_path(path: str) -> None:
    """
    Validate that CmdStan directory exists and binaries have been built.
    Throws exception if specified path is invalid.
    """
    if not os.path.isdir(path):
        raise ValueError('no such CmdStan directory {}'.format(path))
    if not os.path.exists(os.path.join(path, 'bin', 'stanc')):
        raise ValueError(
            "no CmdStan binaries found, "
            "do 'make build' in directory {}".format(path)
        )


def set_cmdstan_path(path: str) -> None:
    """
    Validate, then set CmdStan directory path.
    """
    validate_cmdstan_path(path)
    os.environ['CMDSTAN'] = path


def cmdstan_path() -> str:
    """
    Validate, then return CmdStan directory path.
    """
    cmdstan_path = ''
    if 'CMDSTAN' in os.environ:
        cmdstan_path = os.environ['CMDSTAN']
    else:
        cmdstan_path = os.path.expanduser(
            os.path.join('~', '.cmdstanpy', 'cmdstan'))
    validate_cmdstan_path(cmdstan_path)
    return cmdstan_path


def do_command(cmd: str, cwd: str = None) -> str:
    """
    Spawn process, print stdout/stderr to console.
    Throws exception on non-zero returncode.
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    proc.wait()
    stdout, stderr = proc.communicate()
    if proc.returncode:
        if stderr:
            msg = 'ERROR\n {} '.format(stderr.decode('ascii').strip())
        raise Exception(msg)
    if stdout:
        return stdout.decode('ascii').strip()
    return None


def _rdump_array(key: str, val: np.ndarray) -> str:
    """Flatten numpy ndarray, format as Rdump variable declaration."""
    c = 'c(' + ', '.join(map(str, val.T.flat)) + ')'
    if (val.size,) == val.shape:
        return '{key} <- {c}'.format(key=key, c=c)
    else:
        dim = '.Dim = c{}'.format(val.shape)
        struct = '{key} <- structure({c}, {dim})'.format(key=key, c=c, dim=dim)
        return struct


def jsondump(path: str, data: Dict) -> None:
    """Dump a dict of data to a JSON file."""
    for key, val in data.items():
        if isinstance(val, np.ndarray) and val.size > 1:
            data[key] = val.tolist()
    with open(path, 'w') as fd:
        json.dump(data, fd)


def rdump(path: str, data: Dict) -> None:
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
                line = '{} <- {}'.format(key, val)
            fd.write(line)
            fd.write('\n')


def check_csv(filename: str) -> Dict:
    """Capture essential config, shape from stan_csv file."""
    dict = scan_stan_csv(filename)
    # check draws against spec
    if 'num_samples' in dict:
        draws_spec = int(dict['num_samples'])
    else:
        draws_spec = 1000
    if 'num_warmup' in dict:
        num_warmup = int(dict['num_warmup'])
    else:
        num_warmup = 1000
    if 'save_warmup' in dict and dict['save_warmup'] == '1':
        draws_spec = draws_spec + num_warmup
    if dict['draws'] != draws_spec:
        raise ValueError(
            'bad csv file {}, expected {} draws, found {}'.format(
                filename, draws_spec, dict['draws']
            )
        )
    return dict


def scan_stan_csv(filename: str) -> Dict:
    """Process stan_csv file line by line."""
    dict = {}
    lineno = 0
    try:
        fp = open(filename, 'r')
        try:
            lineno = scan_config(fp, dict, lineno)
            lineno = scan_column_names(fp, dict, lineno)
            lineno = scan_metric(fp, dict, lineno)
            lineno = scan_draws(fp, dict, lineno)
        finally:
            fp.close()
    except IOError:
        raise IOError()
    return dict


def scan_config(fp: TextIO, dict: Dict, lineno: int) -> int:
    """
    Scan initial stan_csv file comments lines and
    save non-default configuration information to dict.
    """
    cur_pos = fp.tell()
    line = fp.readline().strip()
    while len(line) > 0 and line.startswith('#'):
        lineno += 1
        if not line.endswith('(Default)'):
            line = line.lstrip(' #\t')
            key_val = line.split('=')
            if len(key_val) == 2:
                if key_val[0].strip() == 'file' and not key_val[1].endswith(
                        'csv'):
                    dict['data_file'] = key_val[1].strip()
                elif key_val[0].strip() != 'file':
                    dict[key_val[0].strip()] = key_val[1].strip()
        cur_pos = fp.tell()
        line = fp.readline().strip()
    fp.seek(cur_pos)
    return lineno


def scan_column_names(fp: TextIO, dict: Dict, lineno: int) -> int:
    """
    Parse column header into dict entries column_names
    """
    line = fp.readline().strip()
    lineno += 1
    names = line.split(',')
    dict['column_names'] = tuple(names)
    return lineno


def scan_metric(fp: TextIO, dict: Dict, lineno: int) -> int:
    """
    Scan stepsize, metric from  stan_csv file comment lines,
    set dict entries 'metric' and 'num_params'
    """
    if 'metric' not in dict:
        dict['metric'] = 'diag_e'
    metric = dict['metric']
    line = fp.readline().strip()
    lineno += 1
    if not line == '# Adaptation terminated':
        raise ValueError(
            'line {}: expecting metric, '
            'found:\n\t "{}"'.format(lineno, line))
    line = fp.readline().strip()
    lineno += 1
    label, stepsize = line.split('=')
    if not label.startswith('# Step size'):
        raise ValueError(
            'line {}: expecting stepsize, '
            'found:\n\t "{}"'.format(lineno, line))
    try:
        float(stepsize.strip())
    except ValueError:
        raise ValueError(
            'line {}: invalid stepsize: {}'.format(
                lineno, stepsize))
    line = fp.readline().strip()
    lineno += 1
    if not ((metric == 'diag_e' and
                 line == '# Diagonal elements of inverse mass matrix:') or
                 (metric == 'dense_e' and
                      line == '# Elements of inverse mass matrix:')):
        raise ValueError(
            'line {}: invalid or missing mass matrix '
            'specification'.format(lineno))
    line = fp.readline().lstrip(' #\t')
    lineno += 1
    num_params = len(line.split(','))
    dict['num_params'] = num_params
    if metric == 'diag_e':
        return lineno
    else:
        for i in range(1,num_params):
            line = fp.readline().lstrip(' #\t')
            lineno += 1
            if len(line.split(',')) != num_params:
                raise ValueError(
                    'line {}: invalid or missing mass matrix '
                    'specification'.format(lineno))
        return lineno


def scan_draws(fp: TextIO, dict: Dict, lineno: int) -> int:
    """
    Parse draws, check elements per draw, save num draws to dict.
    """
    draws_found = 0
    num_cols = len(dict['column_names'])
    cur_pos = fp.tell()
    line = fp.readline().strip()
    while len(line) > 0 and not line.startswith('#'):
        lineno += 1
        draws_found += 1
        if len(line.split(',')) != num_cols:
            raise ValueError(
                'line {}: bad draw, expecting {} items, found {}'.format(
                    lineno, num_cols, len(line.split(',')))
                )
        cur_pos = fp.tell()
        line = fp.readline().strip()
    dict['draws'] = draws_found
    fp.seek(cur_pos)
    return lineno
