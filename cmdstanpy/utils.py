"""
Utility functions
"""
import contextlib
import functools
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Collection
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import ujson as json
from tqdm.auto import tqdm

from cmdstanpy import (
    _CMDSTAN_SAMPLING,
    _CMDSTAN_THIN,
    _CMDSTAN_WARMUP,
    _DOT_CMDSTAN,
    _TMPDIR,
)

from . import progress as progbar

EXTENSION = '.exe' if platform.system() == 'Windows' else ''


@functools.lru_cache(maxsize=None)
def get_logger() -> logging.Logger:
    """cmdstanpy logger"""
    logger = logging.getLogger('cmdstanpy')
    if len(logger.handlers) == 0:
        logging.basicConfig(level=logging.INFO)
    return logger


def validate_dir(install_dir: str) -> None:
    """Check that specified install directory exists, can write."""
    if not os.path.exists(install_dir):
        try:
            os.makedirs(install_dir)
        except (IOError, OSError, PermissionError) as e:
            raise ValueError(
                'Cannot create directory: {}'.format(install_dir)
            ) from e
    else:
        if not os.path.isdir(install_dir):
            raise ValueError(
                'File exists, should be a directory: {}'.format(install_dir)
            )
        try:
            with open('tmp_test_w', 'w'):
                pass
            os.remove('tmp_test_w')  # cleanup
        except OSError as e:
            raise ValueError(
                'Cannot write files to directory {}'.format(install_dir)
            ) from e


def get_latest_cmdstan(cmdstan_dir: str) -> Optional[str]:
    """
    Given a valid directory path, find all installed CmdStan versions
    and return highest (i.e., latest) version number.

    Assumes directory consists of CmdStan releases, created by
    function `install_cmdstan`, and therefore dirnames have format
    "cmdstan-<maj>.<min>.<patch>" or "cmdstan-<maj>.<min>.<patch>-rc<num>",
    which is CmdStan release practice as of v 2.24.
    """
    versions = [
        name[8:]
        for name in os.listdir(cmdstan_dir)
        if os.path.isdir(os.path.join(cmdstan_dir, name))
        and name.startswith('cmdstan-')
        and name[8].isdigit()
        and len(name[8:].split('.')) == 3
    ]
    if len(versions) == 0:
        return None
    # munge rc for sort, e.g. 2.25.0-rc1 -> 2.25.-99
    for i in range(len(versions)):  # # pylint: disable=C0200
        if '-rc' in versions[i]:
            comps = versions[i].split('-rc')
            mmp = comps[0].split('.')
            rc_num = comps[1]
            patch = str(int(rc_num) - 100)
            versions[i] = '.'.join([mmp[0], mmp[1], patch])

    versions.sort(key=lambda s: list(map(int, s.split('.'))))
    latest = versions[len(versions) - 1]

    # unmunge as needed
    mmp = latest.split('.')
    if int(mmp[2]) < 0:
        print("here")
        rc_num = str(int(mmp[2]) + 100)
        mmp[2] = "0-rc" + rc_num
        latest = '.'.join(mmp)

    return 'cmdstan-' + latest


def validate_cmdstan_path(path: str) -> None:
    """
    Validate that CmdStan directory exists and binaries have been built.
    Throws exception if specified path is invalid.
    """
    if not os.path.isdir(path):
        raise ValueError(f'No CmdStan directory, path {path} does not exist.')
    if not os.path.exists(os.path.join(path, 'bin', 'stanc' + EXTENSION)):
        raise ValueError(
            'CmdStan installataion missing binaries, run "install_cmdstan"'
        )


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
        cmdstan_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))
        if not os.path.exists(cmdstan_dir):
            raise ValueError(
                'No CmdStan installation found, run "install_cmdstan" or'
                ' (re)activate your conda environment!'
            )
        latest_cmdstan = get_latest_cmdstan(cmdstan_dir)
        if latest_cmdstan is None:
            raise ValueError(
                'No CmdStan installation found, run "install_cmdstan" or'
                ' (re)activate your conda environment!'
            )
        cmdstan = os.path.join(cmdstan_dir, latest_cmdstan)
        os.environ['CMDSTAN'] = cmdstan
    validate_cmdstan_path(cmdstan)
    return os.path.normpath(cmdstan)


def cmdstan_version() -> Optional[Tuple[int, ...]]:
    """
    Parses version string out of CmdStan makefile variable CMDSTAN_VERSION,
    returns Tuple(Major, minor).

    If CmdStan installation is not found or cannot parse version from makefile
    logs warning and returns None.  Lenient behavoir required for CI tests,
    per comment:
    https://github.com/stan-dev/cmdstanpy/pull/321#issuecomment-733817554
    """
    try:
        makefile = os.path.join(cmdstan_path(), 'makefile')
    except ValueError:
        get_logger().info('No CmdStan installation found.')
        return None

    if not os.path.exists(makefile):
        get_logger().info(
            'CmdStan installation %s missing makefile, cannot get version.',
            cmdstan_path(),
        )
        return None

    with open(makefile, 'r') as fd:
        contents = fd.read()

    start_idx = contents.find('CMDSTAN_VERSION := ')
    if start_idx < 0:
        get_logger().info(
            'Cannot parse version from makefile: %s.',
            makefile,
        )
        return None

    start_idx += len('CMDSTAN_VERSION := ')
    end_idx = contents.find('\n', start_idx)

    version = contents[start_idx:end_idx]
    splits = version.split('.')
    if len(splits) != 3:
        get_logger().info(
            'Cannot parse version, expected "<major>.<minor>.<patch>", '
            'found: "%s".',
            version,
        )
        return None
    return tuple(int(x) for x in splits[0:2])


def cmdstan_version_before(
    major: int, minor: int, info: Optional[Dict[str, str]] = None
) -> bool:
    """
    Check that CmdStan version is less than Major.minor version.

    :param major: Major version number
    :param minor: Minor version number

    :return: True if version at or above major.minor, else False.
    """
    cur_version = None
    if info is None or 'stan_version_major' not in info:
        cur_version = cmdstan_version()
    else:
        cur_version = (
            int(info['stan_version_major']),
            int(info['stan_version_minor']),
        )
    if cur_version is None:
        get_logger().info(
            'Cannot determine whether version is before %d.%d.', major, minor
        )
        return False
    if cur_version[0] < major or (
        cur_version[0] == major and cur_version[1] < minor
    ):
        return True
    return False


def cxx_toolchain_path(
    version: Optional[str] = None, install_dir: Optional[str] = None
) -> Tuple[str, ...]:
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
    if 'CMDSTAN_TOOLCHAIN' in os.environ:
        toolchain_root = os.environ['CMDSTAN_TOOLCHAIN']
        if os.path.exists(os.path.join(toolchain_root, 'mingw64')):
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

        elif os.path.exists(os.path.join(toolchain_root, 'mingw_64')):
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
    else:
        rtools40_home = os.environ.get('RTOOLS40_HOME')
        cmdstan_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))
        for toolchain_root in (
            ([rtools40_home] if rtools40_home is not None else [])
            + (
                [
                    os.path.join(install_dir, 'RTools40'),
                    os.path.join(install_dir, 'RTools35'),
                    os.path.join(install_dir, 'RTools30'),
                    os.path.join(install_dir, 'RTools'),
                ]
                if install_dir is not None
                else []
            )
            + [
                os.path.join(cmdstan_dir, 'RTools40'),
                os.path.join(os.path.abspath("/"), "RTools40"),
                os.path.join(cmdstan_dir, 'RTools35'),
                os.path.join(os.path.abspath("/"), "RTools35"),
                os.path.join(cmdstan_dir, 'RTools'),
                os.path.join(os.path.abspath("/"), "RTools"),
                os.path.join(os.path.abspath("/"), "RBuildTools"),
            ]
        ):
            compiler_path = ''
            tool_path = ''

            if os.path.exists(toolchain_root):
                if version not in ('35', '3.5', '3'):
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
                                'Found invalid installation for RTools40 on %s',
                                toolchain_root,
                            )
                            toolchain_root = ''
                        else:
                            break
                    else:
                        compiler_path = ''
                        logger.warning(
                            'Found invalid installation for RTools40 on %s',
                            toolchain_root,
                        )
                        toolchain_root = ''
                else:
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
                                'Found invalid installation for RTools35 on %s',
                                toolchain_root,
                            )
                            toolchain_root = ''
                        else:
                            break
                    else:
                        compiler_path = ''
                        logger.warning(
                            'Found invalid installation for RTools35 on %s',
                            toolchain_root,
                        )
                        toolchain_root = ''
            else:
                toolchain_root = ''

    if not toolchain_root:
        raise ValueError(
            'no RTools toolchain installation found, '
            'run command line script '
            '"python -m cmdstanpy.install_cxx_toolchain"'
        )
    logger.info('Add C++ toolchain to $PATH: %s', toolchain_root)
    os.environ['PATH'] = ';'.join(
        list(
            OrderedDict.fromkeys(
                [compiler_path, tool_path] + os.getenv('PATH', '').split(';')
            )
        )
    )
    return compiler_path, tool_path


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


def write_stan_json(path: str, data: Mapping[str, Any]) -> None:
    """
    Dump a mapping of strings to data to a JSON file.

    Values can be any numeric type, a boolean (converted to int),
    or any collection compatible with :func:`numpy.asarray`, e.g a
    :class:`pandas.Series`.

    Produces a file compatible with the
    `Json Format for Cmdstan
    <https://mc-stan.org/docs/2_27/cmdstan-guide/json.html>`__

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
        json.dump(data_out, fd)


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
            lineno = scan_sampling_iters(fd, dict, lineno)
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
        all_iters = np.empty(
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
                mle = np.array(xs, dtype=float)
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
        dict['variational_mean'] = variational_mean
        dict['variational_sample'] = pd.read_csv(
            path,
            comment='#',
            skiprows=lineno,
            header=None,
            float_precision='high',
        )
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
    return [
        re.sub(r',([\d,]+)$', r'[\1]', column.replace('.', ','))
        for column in names
    ]


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
    return {v: tuple([k]) for (k, v) in enumerate(names) if v.endswith('__')}


def parse_stan_vars(
    names: Tuple[str, ...]
) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]]]:
    """
    Parses out Stan variable names (i.e., names not ending in `__`)
    from list of CSV file column names.
    Returns a pair of dicts which map variable names to dimensions and
    variable names to columns, respectively, using zero-based column indexing.
    Note: assumes: (a) munged varnames and (b) container vars are non-ragged
    and dense; no checks size, indices.
    """
    if names is None:
        raise ValueError('missing argument "names"')
    dims_map: Dict[str, Tuple[int, ...]] = {}
    cols_map: Dict[str, Tuple[int, ...]] = {}
    idxs = []
    dims: Union[List[str], List[int]]
    for (idx, name) in enumerate(names):
        idxs.append(idx)
        var, *dims = name.split('[')
        if var.endswith('__'):
            idxs = []
        elif len(dims) == 0:
            dims_map[var] = ()
            cols_map[var] = tuple(idxs)
            idxs = []
        else:
            if idx < len(names) - 1 and names[idx + 1].split('[')[0] == var:
                continue
            dims = [int(x) for x in dims[0][:-1].split(',')]
            dims_map[var] = tuple(dims)
            cols_map[var] = tuple(idxs)
            idxs = []
    return (dims_map, cols_map)


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
    fd: TextIO, config_dict: Dict[str, Any], lineno: int
) -> int:
    """
    Parse sampling iteration, save number of iterations to config_dict.
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
                'line {}: bad draw, expecting {} items, found {}\n'.format(
                    lineno, num_cols, len(line.split(','))
                )
                + 'This error could be caused by running out of disk space.\n'
                'Try clearing up TEMP or setting output_dir to a path'
                ' on another drive.',
            )
        cur_pos = fd.tell()
        line = fd.readline().strip()
    config_dict['draws_sampling'] = draws_found
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
            dims_np = np.asarray(metric_dict['inv_metric'])
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


def do_command(
    cmd: List[str],
    cwd: Optional[str] = None,
    *,
    fd_out: Optional[TextIO] = sys.stdout,
    pbar: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Run command as subprocess, polls process output pipes and
    either streams outputs to supplied output stream or sends
    each line (stripped) to the supplied progress bar callback hook.

    Raises ``RuntimeError`` on non-zero return code or execption ``OSError``.

    :param cmd: command and args.
    :param cwd: directory in which to run command, if unspecified,
        run command in the current working directory.
    :param fd_out: when supplied, streams to this output stream,
        else writes to sys.stdout.
    :param pbar: optional callback hook to tqdm, which takes
       single ``str`` arguent, see:
       https://github.com/tqdm/tqdm#hooks-and-callbacks.

    """
    get_logger().debug('cmd: %s\ncwd: %s', ' '.join(cmd), cwd)
    try:
        # NB: Using this rather than cwd arg to Popen due to windows behavior
        with pushd(cwd if cwd is not None else '.'):
            # TODO: replace with subprocess.run in later Python versions?
            proc = subprocess.Popen(
                cmd,
                bufsize=1,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # avoid buffer overflow
                env=os.environ,
                universal_newlines=True,
            )
            while proc.poll() is None:
                if proc.stdout is not None:
                    line = proc.stdout.readline()
                    if fd_out is not None:
                        fd_out.write(line)
                    if pbar is not None:
                        pbar(line.strip())

            stdout, _ = proc.communicate()
            if stdout:
                if len(stdout) > 0:
                    if fd_out is not None:
                        fd_out.write(stdout)
                    if pbar is not None:
                        pbar(stdout.strip())

            if proc.returncode != 0:  # throw RuntimeError + msg
                serror = ''
                try:
                    serror = os.strerror(proc.returncode)
                except (ArithmeticError, ValueError):
                    pass
                msg = 'Command {}\n\t{} {}'.format(
                    cmd, returncode_msg(proc.returncode), serror
                )
                raise RuntimeError(msg)
    except OSError as e:
        msg = 'Command: {}\nfailed with error {}\n'.format(cmd, str(e))
        raise RuntimeError(msg) from e


def returncode_msg(retcode: int) -> str:
    """interpret retcode"""
    if retcode < 0:
        sig = -1 * retcode
        return f'terminated by signal {sig}'
    if retcode <= 125:
        return 'error during processing'
    if retcode == 126:  # shouldn't happen
        return ''
    if retcode == 127:
        return 'program not found'
    sig = retcode - 128
    return f'terminated by signal {sig}'


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
    _GetShortPathNameW = (
        ctypes.windll.kernel32.GetShortPathNameW  # type: ignore
    )

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


def create_named_text_file(
    dir: str, prefix: str, suffix: str, name_only: bool = False
) -> str:
    """
    Create a named unique file, return filename.
    Flag 'name_only' will create then delete the tmp file;
    this lets us create filename args for commands which
    disallow overwriting existing files (e.g., 'stansummary').
    """
    fd = tempfile.NamedTemporaryFile(
        mode='w+', prefix=prefix, suffix=suffix, dir=dir, delete=name_only
    )
    path = fd.name
    fd.close()
    return path


def show_versions(output: bool = True) -> str:
    """Prints out system and dependency information for debugging"""

    import importlib
    import locale
    import struct

    deps_info = []
    try:
        (sysname, _, release, _, machine, processor) = platform.uname()
        deps_info.extend(
            [
                ("python", sys.version),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", f"{sysname}"),
                ("OS-release", f"{release}"),
                ("machine", f"{machine}"),
                ("processor", f"{processor}"),
                ("byteorder", f"{sys.byteorder}"),
                ("LC_ALL", f'{os.environ.get("LC_ALL", "None")}'),
                ("LANG", f'{os.environ.get("LANG", "None")}'),
                ("LOCALE", f"{locale.getlocale()}"),
            ]
        )
    # pylint: disable=broad-except
    except Exception:
        pass

    try:
        deps_info.append(('cmdstan_folder', cmdstan_path()))
        deps_info.append(('cmdstan', str(cmdstan_version())))
    # pylint: disable=broad-except
    except Exception:
        deps_info.append(('cmdstan', 'NOT FOUND'))

    deps = ['cmdstanpy', 'pandas', 'xarray', 'tdqm', 'numpy', 'ujson']
    for module in deps:
        try:
            if module in sys.modules:
                mod = sys.modules[module]
            else:
                mod = importlib.import_module(module)
        # pylint: disable=broad-except
        except Exception:
            deps_info.append((module, None))
        else:
            try:
                ver = mod.__version__  # type: ignore
                deps_info.append((module, ver))
            # pylint: disable=broad-except
            except Exception:
                deps_info.append((module, "installed"))

    out = 'INSTALLED VERSIONS\n---------------------\n'
    for k, info in deps_info:
        out += f'{k}: {info}\n'
    if output:
        print(out)
        return " "
    else:
        return out


def install_cmdstan(
    version: Optional[str] = None,
    dir: Optional[str] = None,
    overwrite: bool = False,
    compiler: bool = False,
    progress: bool = False,
    verbose: bool = False,
    cores: int = 1,
) -> bool:
    """
    Download and install a CmdStan release from GitHub. Downloads the release
    tar.gz file to temporary storage.  Retries GitHub requests in order
    to allow for transient network outages. Builds CmdStan executables
    and tests the compiler by building example model ``bernoulli.stan``.

    :param version: CmdStan version string, e.g. "2.24.1".
        Defaults to latest CmdStan release.

    :param dir: Path to install directory.  Defaults to hidden directory
        ``$HOME/.cmdstan``.
        If no directory is specified and the above directory does not
        exist, directory ``$HOME/.cmdstan`` will be created and populated.

    :param overwrite:  Boolean value; when ``True``, will overwrite and
        rebuild an existing CmdStan installation.  Default is ``False``.

    :param compiler: Boolean value; when ``True`` on WINDOWS ONLY, use the
        C++ compiler from the ``install_cxx_toolchain`` command or install
        one if none is found.

    :param progress: Boolean value; when ``True``, show a progress bar for
        downloading and unpacking CmdStan.  Default is ``False``.

    :param verbose: Boolean value; when ``True``, show console output from all
        intallation steps, i.e., download, build, and test CmdStan release.
        Default is ``False``.
    :param cores: Integer, number of cores to use in the ``make`` command.
        Default is 1 core.

    :return: Boolean value; ``True`` for success.
    """
    logger = get_logger()
    args = {
        "version": version,
        "overwrite": overwrite,
        "verbose": verbose,
        "compiler": compiler,
        "progress": progress,
        "dir": dir,
        "cores": cores,
    }

    try:
        from .install_cmdstan import main

        main(args)
    # pylint: disable=broad-except
    except Exception as e:
        logger.warning('CmdStan installation failed.')
        logger.warning(str(e))
        return False

    if dir is not None:
        if version is not None:
            set_cmdstan_path(os.path.join(dir, 'cmdstan-' + version))
        else:
            set_cmdstan_path(
                os.path.join(dir, get_latest_cmdstan(dir))  # type: ignore
            )
    return True


@progbar.wrap_callback
def wrap_url_progress_hook() -> Optional[Callable[[int, int, int], None]]:
    """Sets up tqdm callback for url downloads."""
    pbar: tqdm = tqdm(
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        colour='blue',
        leave=False,
    )

    def download_progress_hook(
        count: int, block_size: int, total_size: int
    ) -> None:
        if pbar.total is None:
            pbar.total = total_size
            pbar.reset()
        downloaded_size = count * block_size
        pbar.update(downloaded_size - pbar.n)
        if pbar.n >= total_size:
            pbar.close()

    return download_progress_hook


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


@contextlib.contextmanager
def pushd(new_dir: str) -> Iterator[None]:
    """Acts like pushd/popd."""
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(previous_dir)


def report_signal(sig: int) -> None:
    """Provide info for processes terminated by a system signal."""
    print('terminated by signal: {}'.format(sig))


class MaybeDictToFilePath:
    """Context manager for json files."""

    def __init__(
        self,
        *objs: Union[str, Mapping[str, Any], List[Any], int, float, None],
    ):
        self._unlink = [False] * len(objs)
        self._paths: List[Any] = [''] * len(objs)
        i = 0
        # pylint: disable=isinstance-second-argument-not-valid-type
        for obj in objs:
            if isinstance(obj, Mapping):
                data_file = create_named_text_file(
                    dir=_TMPDIR, prefix='', suffix='.json'
                )
                get_logger().debug('input tempfile: %s', data_file)
                write_stan_json(data_file, obj)
                self._paths[i] = data_file
                self._unlink[i] = True
            elif isinstance(obj, str):
                if not os.path.exists(obj):
                    raise ValueError("File doesn't exist {}".format(obj))
                self._paths[i] = obj
            elif isinstance(obj, list):
                err_msgs = []
                missing_obj_items = []
                for j, obj_item in enumerate(obj):
                    if not isinstance(obj_item, str):
                        err_msgs.append(
                            (
                                'List element {} must be a filename string,'
                                ' found {}'
                            ).format(j, obj_item)
                        )
                    elif not os.path.exists(obj_item):
                        missing_obj_items.append(
                            "File doesn't exist: {}".format(obj_item)
                        )
                if err_msgs:
                    raise ValueError('\n'.join(err_msgs))
                if missing_obj_items:
                    raise ValueError('\n'.join(missing_obj_items))
                self._paths[i] = obj
            elif obj is None:
                self._paths[i] = None
            elif i == 1 and isinstance(obj, (int, float)):
                self._paths[i] = obj
            else:
                raise ValueError('data must be string or dict')
            i += 1

    def __enter__(self) -> List[str]:
        return self._paths

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        for can_unlink, path in zip(self._unlink, self._paths):
            if can_unlink and path:
                try:
                    os.remove(path)
                except PermissionError:
                    pass


class SanitizedOrTmpFilePath:
    """Context manager for tmpfiles, handles spaces in filepath."""

    def __init__(self, file_path: str):
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

    def __enter__(self) -> Tuple[str, bool]:
        return self._path, self._tmpdir is not None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
