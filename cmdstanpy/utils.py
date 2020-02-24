"""
Utility functions
"""
import os
from collections import OrderedDict
from collections.abc import Sequence
from numbers import Integral, Real

try:
    import ujson as json
except ImportError:
    import json
import math
import platform
import re
import subprocess
import shutil
import tempfile
import logging
import sys
from typing import Dict, TextIO, List, Union, Tuple

import numpy as np
import pandas as pd


from cmdstanpy import _TMPDIR

EXTENSION = '.exe' if platform.system() == 'Windows' else ''


def get_logger():
    """cmdstanpy logger"""
    logger = logging.getLogger('cmdstanpy')
    if len(logger.handlers) == 0:
        logging.basicConfig(level=logging.INFO)
    return logger


def get_latest_cmdstan(dot_dir: str) -> str:
    """
    Given a valid directory path, find all installed CmdStan versions
    and return highest (i.e., latest) version number.
    Assumes directory populated via script `install_cmdstan`.
    """
    versions = [
        name.split('-')[1]
        for name in os.listdir(dot_dir)
        if os.path.isdir(os.path.join(dot_dir, name))
        and name.startswith('cmdstan-')
        and name[8].isdigit()
    ]
    versions.sort(key=lambda s: list(map(int, s.split('.'))))
    if len(versions) == 0:
        return None
    latest = 'cmdstan-{}'.format(versions[len(versions) - 1])
    return latest


class MaybeDictToFilePath:
    """Context manager for json files."""

    def __init__(self, *objs: Union[str, dict], logger: logging.Logger = None):
        self._unlink = [False] * len(objs)
        self._paths = [''] * len(objs)
        self._logger = logger or get_logger()
        i = 0
        for obj in objs:
            if isinstance(obj, dict):
                data_file = create_named_text_file(
                    dir=_TMPDIR, prefix='', suffix='.json'
                )
                self._logger.debug('input tempfile: %s', data_file)
                if any(
                    not item
                    for item in obj
                    if isinstance(item, (Sequence, np.ndarray))
                ):
                    rdump(data_file, obj)
                else:
                    jsondump(data_file, obj)
                self._paths[i] = data_file
                self._unlink[i] = True
            elif isinstance(obj, str):
                if not os.path.exists(obj):
                    raise ValueError("File doesn't exist {}".format(obj))
                self._paths[i] = obj
            elif obj is None:
                self._paths[i] = None
            elif i == 1 and isinstance(obj, (Integral, Real)):
                self._paths[i] = obj
            else:
                raise ValueError('data must be string or dict')
            i += 1

    def __enter__(self):
        return self._paths

    def __exit__(self, exc_type, exc_val, exc_tb):
        for can_unlink, path in zip(self._unlink, self._paths):
            if can_unlink and path:
                try:
                    os.remove(path)
                except PermissionError:
                    pass


def validate_cmdstan_path(path: str) -> None:
    """
    Validate that CmdStan directory exists and binaries have been built.
    Throws exception if specified path is invalid.
    """
    if not os.path.isdir(path):
        raise ValueError('no such CmdStan directory {}'.format(path))
    if not os.path.exists(os.path.join(path, 'bin', 'stanc' + EXTENSION)):
        raise ValueError(
            'no CmdStan binaries found, '
            'run command line script "install_cmdstan"'
        )


class TemporaryCopiedFile:
    """Context manager for tmpfiles, handles spaces in filepath."""

    def __init__(self, file_path: str):
        self._path = None
        self._tmpdir = None
        if ' ' in os.path.abspath(file_path) and platform.system() == 'Windows':
            base_path, file_name = os.path.split(os.path.abspath(file_path))
            os.makedirs(base_path, exist_ok=True)
            try:
                short_base_path = windows_short_path(base_path)
                if os.path.exists(short_base_path):
                    file_path = os.path.join(short_base_path, file_name)
            except RuntimeError:
                pass

        if ' ' in os.path.abspath(file_path):
            tmpdir = tempfile.mkdtemp()
            if ' ' in tmpdir:
                raise RuntimeError(
                    'Unable to generate temporary path without spaces! \n'
                    + 'Please move your stan file to location without spaces.'
                )

            _, path = tempfile.mkstemp(suffix='.stan', dir=tmpdir)

            shutil.copy(file_path, path)
            self._path = path
            self._tmpdir = tmpdir
        else:
            self._path = file_path

    def __enter__(self):
        return self._path, self._tmpdir is not None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)


def set_cmdstan_path(path: str) -> None:
    """
    Validate, then set CmdStan directory path.
    """
    validate_cmdstan_path(path)
    os.environ['CMDSTAN'] = path


def set_make_env(make: str) -> None:
    """
    set MAKE environmental variable.
    """
    os.environ['MAKE'] = make


def cmdstan_path() -> str:
    """
    Validate, then return CmdStan directory path.
    """
    cmdstan = ''
    if 'CMDSTAN' in os.environ:
        cmdstan = os.environ['CMDSTAN']
    else:
        cmdstan_dir = os.path.expanduser(os.path.join('~', '.cmdstanpy'))
        if not os.path.exists(cmdstan_dir):
            raise ValueError(
                'no CmdStan installation found, '
                'run command line script "install_cmdstan"'
            )
        latest_cmdstan = get_latest_cmdstan(cmdstan_dir)
        if latest_cmdstan is None:
            raise ValueError(
                'no CmdStan installation found, '
                'run command line script "install_cmdstan"'
            )
        cmdstan = os.path.join(cmdstan_dir, latest_cmdstan)
        os.environ['CMDSTAN'] = cmdstan
    validate_cmdstan_path(cmdstan)
    return cmdstan


def cxx_toolchain_path(version: str = None) -> Tuple[str]:
    """
    Validate, then activate C++ toolchain directory path.
    """
    if platform.system() != 'Windows':
        raise RuntimeError(
            'Functionality is currently only supported on Windows'
        )
    if version is not None and not isinstance(version, str):
        raise TypeError('Format version number as a string')
    logger = get_logger()
    toolchain_root = ''
    if 'CMDSTAN_TOOLCHAIN' in os.environ:
        toolchain_root = os.environ['CMDSTAN_TOOLCHAIN']
        if os.path.exists(os.path.join(toolchain_root, 'mingw_64')):
            compiler_path = os.path.join(
                toolchain_root,
                'mingw_64' if (sys.maxsize > 2 ** 32) else 'mingw_32',
                'bin',
            )
            if os.path.exists(compiler_path):
                tool_path = os.path.join(toolchain_root, 'bin')
                if not os.path.exists(tool_path):
                    tool_path = ''
                    compiler_path = ''
                    logger.warning(
                        'Found invalid installion for RTools35 on %s',
                        toolchain_root,
                    )
                    toolchain_root = ''
            else:
                compiler_path = ''
                logger.warning(
                    'Found invalid installion for RTools35 on %s',
                    toolchain_root,
                )
                toolchain_root = ''
        elif os.path.exists(os.path.join(toolchain_root, 'mingw64')):
            compiler_path = os.path.join(
                toolchain_root,
                'mingw64' if (sys.maxsize > 2 ** 32) else 'mingw32',
                'bin',
            )
            if os.path.exists(compiler_path):
                tool_path = os.path.join(toolchain_root, 'usr', 'bin')
                if not os.path.exists(tool_path):
                    tool_path = ''
                    compiler_path = ''
                    logger.warning(
                        'Found invalid installion for RTools40 on %s',
                        toolchain_root,
                    )
                    toolchain_root = ''
            else:
                compiler_path = ''
                logger.warning(
                    'Found invalid installion for RTools40 on %s',
                    toolchain_root,
                )
                toolchain_root = ''
    else:
        rtools_dir = os.path.expanduser(
            os.path.join('~', '.cmdstanpy', 'RTools')
        )
        if not os.path.exists(rtools_dir):
            raise ValueError(
                'no RTools installation found, '
                'run command line script "install_cxx_toolchain"'
            )
        compiler_path = ''
        tool_path = ''
        if version not in ('4', '40', '4.0') and os.path.exists(
            os.path.join(rtools_dir, 'RTools35')
        ):
            toolchain_root = os.path.join(rtools_dir, 'RTools35')
            compiler_path = os.path.join(
                toolchain_root,
                'mingw_64' if (sys.maxsize > 2 ** 32) else 'mingw_32',
                'bin',
            )
            if os.path.exists(compiler_path):
                tool_path = os.path.join(toolchain_root, 'bin')
                if not os.path.exists(tool_path):
                    tool_path = ''
                    compiler_path = ''
                    logger.warning(
                        'Found invalid installion for RTools35 on %s',
                        toolchain_root,
                    )
                    toolchain_root = ''
            else:
                compiler_path = ''
                logger.warning(
                    'Found invalid installion for RTools35 on %s',
                    toolchain_root,
                )
                toolchain_root = ''
        if (
            not toolchain_root or version in ('4', '40', '4.0')
        ) and os.path.exists(os.path.join(rtools_dir, 'RTools40')):
            toolchain_root = os.path.join(rtools_dir, 'RTools40')
            compiler_path = os.path.join(
                toolchain_root,
                'mingw64' if (sys.maxsize > 2 ** 32) else 'mingw32',
                'bin',
            )
            if os.path.exists(compiler_path):
                tool_path = os.path.join(toolchain_root, 'usr', 'bin')
                if not os.path.exists(tool_path):
                    tool_path = ''
                    compiler_path = ''
                    logger.warning(
                        'Found invalid installion for RTools40 on %s',
                        toolchain_root,
                    )
                    toolchain_root = ''
            else:
                compiler_path = ''
                logger.warning(
                    'Found invalid installion for RTools40 on %s',
                    toolchain_root,
                )
                toolchain_root = ''
    if not toolchain_root:
        raise ValueError(
            'no C++ toolchain installation found, '
            'run command line script "install_cxx_toolchain"'
        )
    logger.info('Adds C++ toolchain to $PATH: %s', toolchain_root)
    os.environ['PATH'] = ';'.join(
        list(
            OrderedDict.fromkeys(
                [compiler_path, tool_path] + os.getenv('PATH', '').split(';')
            )
        )
    )
    return compiler_path, tool_path


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
        if isinstance(val, np.ndarray):
            val = val.tolist()
            data[key] = val
    with open(path, 'w') as fd:
        json.dump(data, fd)


def rdump(path: str, data: Dict) -> None:
    """Dump a dict of data to a R dump format file."""
    with open(path, 'w') as fd:
        for key, val in data.items():
            if isinstance(val, (np.ndarray, Sequence)):
                line = _rdump_array(key, np.asarray(val))
            else:
                line = '{} <- {}'.format(key, val)
            print(line, file=fd)


def rload(fname: str) -> dict:
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


def parse_rdump_value(rhs: str) -> Union[int, float, np.array]:
    """Process right hand side of Rdump variable assignment statement.
       Value is either scalar, vector, or multi-dim structure.
       Use regex to capture structure values, dimensions.
    """
    pat = re.compile(
        r'structure\(\s*c\((?P<vals>[^)]*)\)'
        r'(,\s*\.Dim\s*=\s*c\s*\((?P<dims>[^)]*)\s*\))?\)'
    )
    val = None
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
    except TypeError:
        raise ValueError('bad value in Rdump file: {}'.format(rhs))
    return val


def check_sampler_csv(path: str, is_fixed_param: bool = False) -> Dict:
    """Capture essential config, shape from stan_csv file."""
    meta = scan_sampler_csv(path, is_fixed_param)
    draws_spec = int(meta.get('num_samples', 1000))
    if 'thin' in meta:
        draws_spec = int(math.ceil(draws_spec / meta['thin']))
    if meta['draws'] != draws_spec:
        raise ValueError(
            'bad csv file {}, expected {} draws, found {}'.format(
                path, draws_spec, meta['draws']
            )
        )
    return meta


def scan_sampler_csv(path: str, is_fixed_param: bool = False) -> Dict:
    """Process sampler stan_csv output file line by line."""
    dict = {}
    lineno = 0
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
        if not is_fixed_param:
            lineno = scan_warmup(fd, dict, lineno)
            lineno = scan_metric(fd, dict, lineno)
        lineno = scan_draws(fd, dict, lineno)
    return dict


def scan_optimize_csv(path: str) -> Dict:
    """Process optimizer stan_csv output file line by line."""
    dict = {}
    lineno = 0
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
        line = fd.readline().lstrip(' #\t')
        xs = line.split(',')
        mle = [float(x) for x in xs]
        dict['mle'] = mle
    return dict


def scan_generated_quantities_csv(path: str) -> Dict:
    """
    Process standalone generated quantities stan_csv output file line by line.
    """
    dict = {}
    lineno = 0
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
    return dict


def scan_variational_csv(path: str) -> Dict:
    """Process advi stan_csv output file line by line."""
    dict = {}
    lineno = 0
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
        line = fd.readline().lstrip(' #\t')
        lineno += 1
        if not line.startswith('Stepsize adaptation complete.'):
            raise ValueError(
                'line {}: expecting adaptation msg, found:\n\t "{}"'.format(
                    lineno, line
                )
            )
        line = fd.readline().lstrip(' #\t\n')
        lineno += 1
        if not line.startswith('eta = 1'):
            raise ValueError(
                'line {}: expecting eta = 1, found:\n\t "{}"'.format(
                    lineno, line
                )
            )
        line = fd.readline().lstrip(' #\t\n')
        lineno += 1
        xs = line.split(',')
        variational_mean = [float(x) for x in xs]
        dict['variational_mean'] = variational_mean
        dict['variational_sample'] = pd.read_csv(
            path, comment='#', skiprows=lineno, header=None
        )
    return dict


def scan_config(fd: TextIO, config_dict: Dict, lineno: int) -> int:
    """
    Scan initial stan_csv file comments lines and
    save non-default configuration information to config_dict.
    """
    cur_pos = fd.tell()
    line = fd.readline().strip()
    while len(line) > 0 and line.startswith('#'):
        lineno += 1
        if not line.endswith('(Default)'):
            line = line.lstrip(' #\t')
            key_val = line.split('=')
            if len(key_val) == 2:
                if key_val[0].strip() == 'file' and not key_val[1].endswith(
                    'csv'
                ):
                    config_dict['data_file'] = key_val[1].strip()
                elif key_val[0].strip() != 'file':
                    raw_val = key_val[1].strip()
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


def scan_warmup(fd: TextIO, config_dict: Dict, lineno: int) -> int:
    """
    Check warmup iterations, if any.
    """
    if 'save_warmup' not in config_dict:
        return lineno
    cur_pos = fd.tell()
    line = fd.readline().strip()
    while len(line) > 0 and not line.startswith('#'):
        lineno += 1
        cur_pos = fd.tell()
        line = fd.readline().strip()
    fd.seek(cur_pos)
    return lineno


def scan_column_names(fd: TextIO, config_dict: Dict, lineno: int) -> int:
    """
    Process columns header, add to config_dict as 'column_names'
    """
    line = fd.readline().strip()
    lineno += 1
    names = line.split(',')
    config_dict['column_names'] = tuple(names)
    config_dict['num_params'] = len(names) - 1
    return lineno


def scan_metric(fd: TextIO, config_dict: Dict, lineno: int) -> int:
    """
    Scan stepsize, metric from  stan_csv file comment lines,
    set config_dict entries 'metric' and 'num_params'
    """
    if 'metric' not in config_dict:
        config_dict['metric'] = 'diag_e'
    metric = config_dict['metric']
    line = fd.readline().strip()
    lineno += 1
    if not line == '# Adaptation terminated':
        raise ValueError(
            'line {}: expecting metric, found:\n\t "{}"'.format(lineno, line)
        )
    line = fd.readline().strip()
    lineno += 1
    label, stepsize = line.split('=')
    if not label.startswith('# Step size'):
        raise ValueError(
            'line {}: expecting stepsize, '
            'found:\n\t "{}"'.format(lineno, line)
        )
    try:
        float(stepsize.strip())
    except ValueError:
        raise ValueError(
            'line {}: invalid stepsize: {}'.format(lineno, stepsize)
        )
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
    num_params = len(line.split(','))
    config_dict['num_params'] = num_params
    if metric == 'diag_e':
        return lineno
    else:
        for _ in range(1, num_params):
            line = fd.readline().lstrip(' #\t')
            lineno += 1
            if len(line.split(',')) != num_params:
                raise ValueError(
                    'line {}: invalid or missing mass matrix '
                    'specification'.format(lineno)
                )
        return lineno


def scan_draws(fd: TextIO, config_dict: Dict, lineno: int) -> int:
    """
    Parse draws, check elements per draw, save num draws to config_dict.
    """
    draws_found = 0
    num_cols = len(config_dict['column_names'])
    cur_pos = fd.tell()
    line = fd.readline().strip()
    while len(line) > 0 and not line.startswith('#'):
        lineno += 1
        draws_found += 1
        data = line.split(',')
        if len(data) != num_cols:
            raise ValueError(
                'line {}: bad draw, expecting {} items, found {}'.format(
                    lineno, num_cols, len(line.split(','))
                )
            )
        cur_pos = fd.tell()
        line = fd.readline().strip()
    config_dict['draws'] = draws_found
    fd.seek(cur_pos)
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
            dims = np.asarray(metric_dict['inv_metric'])
            return list(dims.shape)
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
    if not (
        'inv_metric' in metric_dict
        and isinstance(metric_dict['inv_metric'], np.ndarray)
    ):
        raise ValueError(
            'metric file {}, bad or missing entry "inv_metric"'.format(path)
        )
    return list(metric_dict['inv_metric'].shape)


def do_command(cmd: str, cwd: str = None, logger: logging.Logger = None) -> str:
    """
    Spawn process, print stdout/stderr to console.
    Throws RuntimeError on non-zero returncode.
    """
    if logger:
        logger.debug('cmd: %s', cmd)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode:
        if stderr:
            msg = 'ERROR\n {} '.format(stderr.decode('utf-8').strip())
        raise RuntimeError(msg)
    if stdout:
        return stdout.decode('utf-8').strip()
    return None


def windows_short_path(path: str) -> str:
    """
    Gets the short path name of a given long path.
    http://stackoverflow.com/a/23598461/200291

    On non-Windows platforms, returns the path

    If (base)path does not exist, function raises RuntimeError
    """
    if platform.system() != 'Windows':
        return path

    if os.path.isfile(path) or (
        not os.path.isdir(path) and os.path.splitext(path)[1] != ''
    ):
        base_path, file_name = os.path.split(path)
    else:
        base_path, file_name = path, ''

    if not os.path.exists(base_path):
        raise RuntimeError(
            'Windows short path function needs a valid directory. '
            'Base directory does not exist: "{}"'.format(base_path)
        )

    import ctypes
    from ctypes import wintypes

    # pylint: disable=invalid-name
    _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    _GetShortPathNameW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    _GetShortPathNameW.restype = wintypes.DWORD

    output_buf_size = 0
    while True:
        output_buf = ctypes.create_unicode_buffer(output_buf_size)
        needed = _GetShortPathNameW(base_path, output_buf, output_buf_size)
        if output_buf_size >= needed:
            short_base_path = output_buf.value
            break
        else:
            output_buf_size = needed

    short_path = (
        os.path.join(short_base_path, file_name)
        if file_name
        else short_base_path
    )
    return short_path


def create_named_text_file(dir: str, prefix: str, suffix: str) -> str:
    """
    Create a named unique file.
    """
    fd = tempfile.NamedTemporaryFile(
        mode='w+', prefix=prefix, suffix=suffix, dir=dir, delete=False
    )
    path = fd.name
    fd.close()
    return path


def install_cmdstan(version: str = None, dir: str = None) -> bool:
    """
    Run 'install_cmdstan' -script
    """
    logger = get_logger()
    python = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'install_cmdstan.py')
    cmd = [python, path]
    if version is not None:
        cmd.extend(['--version', version])
    if dir is not None:
        cmd.extend(['--dir', dir])
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ
    )
    while proc.poll() is None:
        output = proc.stdout.readline().decode('utf-8').strip()
        if output:
            logger.info(output)
    proc.communicate()
    if proc.returncode:
        logger.warning('CmdStan installation failed')
        return False
    return True
