"""
Utility functions
"""
import os
from pathlib import Path
import json
import platform
import subprocess
import numpy as np
from typing import Dict


def validate_cmdstan_path(path: str) -> None:
    """
    Validate that CmdStan directory exists and binaries have been built.
    Throws exception if specified path is invalid.
    """
    if path is not None:
        path = Path(path)
    if path is None or not path.is_dir():
        raise ValueError('no such CmdStan directory {}'.format(path))
    stanc = 'stanc'
    if platform.system() == "Windows":
        stanc += '.exe'
    if not (path / 'bin' / stanc).exists():
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
    cmdstan_path = os.getenv('CMDSTAN', Path('.') / 'releases' / 'cmdstan')
    cmdstan_path = Path(cmdstan_path).absolute()
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
    draws_found = 0
    lineno = 0
    with open(filename) as fp:
        line = fp.readline().strip()
        lineno += 1
        while len(line) > 0 and line.startswith('#'):
            parse_header(line, dict)
            line = fp.readline().strip()
            lineno += 1
        dict['column_names'] = tuple(line.split(','))
        num_cols = len(dict['column_names'])
        line = fp.readline().strip()
        lineno += 1
        while len(line) > 0 and line.startswith('#'):
            line = fp.readline().strip()
            lineno += 1
        while len(line) > 0 and not line.startswith('#'):
            draws_found += 1
            if len(line.split(',')) != num_cols:
                raise ValueError(
                    'file {}, at line {}: bad draw, expecting {} items, '
                    'found {}'.format(
                        filename, lineno, num_cols, len(line.split(',')))
                )
            line = fp.readline().strip()
            lineno += 1
    dict['draws'] = draws_found
    return dict


def parse_header(line: str, dict: Dict) -> None:
    """
    Parse initial stan_csv file comments lines and
    save non-default configuration information to dict.
    """
    if not line.endswith('(Default)'):
        line = line.lstrip(' #\t')
        key_val = line.split('=')
        if len(key_val) == 2:
            if key_val[0].strip() == 'file' and not key_val[1].endswith('csv'):
                dict['data_file'] = key_val[1].strip()
            elif key_val[0].strip() != 'file':
                dict[key_val[0].strip()] = key_val[1].strip()
