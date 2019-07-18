"""
Utility functions
"""
import os
import json
import math
import numpy as np
import platform
import re
import subprocess
import shutil
import tempfile
import logging

from typing import Dict, TextIO, List, Union
from cmdstanpy import TMPDIR

EXTENSION = '.exe' if platform.system() == 'Windows' else ''


def get_logger():
    logger = logging.getLogger('cmdstanpy')
    if len(logger.handlers) == 0:
        logging.basicConfig(level=logging.INFO)
    return logger


def get_latest_cmdstan(dot_dir: str) -> str:
    """
    Given a valid directory path, find all installed CmdStan versions
    and return highest (i.e., latest) version number.
    Assumes directory populated via script `bin/install_cmdstan`.
    """
    versions = [
        name.split('-')[1]
        for name in os.listdir(dot_dir)
        if os.path.isdir(os.path.join(dot_dir, name))
        and name.startswith('cmdstan-')
    ]
    versions.sort(key=lambda s: list(map(int, s.split('.'))))
    if len(versions) == 0:
        return None
    latest = 'cmdstan-{}'.format(versions[len(versions) - 1])
    return latest


class MaybeDictToFilePath(object):
    def __init__(
            self,
            *objs: Union[str, dict],
            logger: logging.Logger = None):
        self._unlink = [False] * len(objs)
        self._paths = [""] * len(objs)
        self._logger = logger or get_logger()
        i = 0
        for o in objs:
            if isinstance(o, dict):
                with tempfile.NamedTemporaryFile(
                        mode='w+', suffix='.json', dir=TMPDIR, delete=False
                ) as fd:
                    data_file = fd.name
                    self._logger.debug('input tempfile: %s', fd.name)
                    jsondump(data_file, o)
                self._paths[i] = data_file
                self._unlink[i] = True
            elif isinstance(o, str):
                if not os.path.exists(o):
                    raise ValueError("File doesn't exists {}".format(o))
                self._paths[i] = o
            elif o is None:
                self._paths[i] = None
            else:
                raise ValueError("data must be string or dict")
            i += 1

    def __enter__(self):
        return self._paths

    def __exit__(self, exc_type, exc_val, exc_tb):
        for can_unlink, path in zip(self._unlink, self._paths):
            if can_unlink and path:
                os.remove(path)


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


class TemporaryCopiedFile(object):
    def __init__(self, file_path: str):
        self._path = None
        self._tmpdir = None
        if " " in file_path:
            tmpdir = tempfile.mkdtemp()
            if " " in tmpdir:
                raise RuntimeError(
                    "Unable to generate temporary path without spaces! \n" +
                    "Please move your stan file to location without spaces."
                )

            _, path = tempfile.mkstemp(suffix=".stan", dir=tmpdir)

            shutil.copy(file_path, path)
            self._path = path
            self._tmpdir = tmpdir
        else:
            self._changed = False
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
    cmdstan_path = ''
    if 'CMDSTAN' in os.environ:
        cmdstan_path = os.environ['CMDSTAN']
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
        cmdstan_path = os.path.join(cmdstan_dir, latest_cmdstan)
    validate_cmdstan_path(cmdstan_path)
    return cmdstan_path


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


def check_csv(path: str, is_optimizing: bool = False) -> Dict:
    """Capture essential config, shape from stan_csv file."""
    meta = scan_stan_csv(path, is_optimizing=is_optimizing)
    # check draws against spec
    if is_optimizing:
        draws_spec = 1
    else:
        draws_spec = int(meta.get('num_samples',1000))
        if 'thin' in meta:
            draws_spec = int(math.ceil(draws_spec / meta['thin']))
    if meta['draws'] != draws_spec:
        raise ValueError(
            'bad csv file {}, expected {} draws, found {}'.format(
                path, draws_spec, meta['draws']
            )
        )
    return meta


def scan_stan_csv(path: str, is_optimizing: bool = False) -> Dict:
    """Process stan_csv file line by line."""
    dict = {}
    lineno = 0
    with open(path, 'r') as fp:
        lineno = scan_config(fp, dict, lineno)
        lineno = scan_column_names(fp, dict, lineno)
        lineno = scan_warmup(fp, dict, lineno)
        if not is_optimizing:
            lineno = scan_metric(fp, dict, lineno)
        lineno = scan_draws(fp, dict, lineno)
    return dict


def scan_config(fp: TextIO, config_dict: Dict, lineno: int) -> int:
    """
    Scan initial stan_csv file comments lines and
    save non-default configuration information to config_dict.
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
                    'csv'
                ):
                    config_dict['data_file'] = key_val[1].strip()
                elif key_val[0].strip() != 'file':
                    config_dict[key_val[0].strip()] = key_val[1].strip()
        cur_pos = fp.tell()
        line = fp.readline().strip()
    fp.seek(cur_pos)
    return lineno


def scan_warmup(fp: TextIO, config_dict: Dict, lineno: int) -> int:
    """
    Check warmup iterations, if any.
    """
    if 'save_warmup' not in config_dict:
        return lineno
    cur_pos = fp.tell()
    line = fp.readline().strip()
    while len(line) > 0 and not line.startswith('#'):
        lineno += 1
        cur_pos = fp.tell()
        line = fp.readline().strip()
    fp.seek(cur_pos)
    return lineno


def scan_column_names(fp: TextIO, config_dict: Dict, lineno: int) -> int:
    """
    Process columns header, add to config_dict as 'column_names'
    """
    line = fp.readline().strip()
    lineno += 1
    names = line.split(',')
    config_dict['column_names'] = tuple(names)
    config_dict['num_params'] = len(names) - 1
    return lineno


def scan_metric(fp: TextIO, config_dict: Dict, lineno: int) -> int:
    """
    Scan stepsize, metric from  stan_csv file comment lines,
    set config_dict entries 'metric' and 'num_params'
    """
    if 'metric' not in config_dict:
        config_dict['metric'] = 'diag_e'
    metric = config_dict['metric']
    line = fp.readline().strip()
    lineno += 1
    if not line == '# Adaptation terminated':
        raise ValueError(
            'line {}: expecting metric, found:\n\t "{}"'.format(lineno, line)
        )
    line = fp.readline().strip()
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
    line = fp.readline().strip()
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
    line = fp.readline().lstrip(' #\t')
    lineno += 1
    num_params = len(line.split(','))
    config_dict['num_params'] = num_params
    if metric == 'diag_e':
        return lineno
    else:
        for i in range(1, num_params):
            line = fp.readline().lstrip(' #\t')
            lineno += 1
            if len(line.split(',')) != num_params:
                raise ValueError(
                    'line {}: invalid or missing mass matrix '
                    'specification'.format(lineno)
                )
        return lineno


def scan_draws(
        fp: TextIO,
        config_dict: Dict,
        lineno: int
) -> int:
    """
    Parse draws, check elements per draw, save num draws to config_dict.
    """
    draws_found = 0
    num_cols = len(config_dict['column_names'])
    cur_pos = fp.tell()
    line = fp.readline().strip()
    first_draw = None
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
        if first_draw is None:
            first_draw = np.array(data, dtype=np.float64)
        cur_pos = fp.tell()
        line = fp.readline().strip()
    config_dict['draws'] = draws_found
    config_dict['first_draw'] = first_draw
    fp.seek(cur_pos)
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
        dims = read_rdump_metric(path)
        if dims is None:
            raise ValueError(
                'metric file {}, bad or missing'
                ' entry "inv_metric"'.format(path)
            )
        return dims


def read_rdump_metric(path: str) -> List[int]:
    """
    Find dimensions of variable named 'inv_metric' using regex search.
    """
    with open(path, 'r') as fp:
        data = fp.read().replace('\n', '')
        m1 = re.search(r'inv_metric\s*<-\s*structure\(\s*c\(', data)
        if not m1:
            return None
        m2 = re.search(r'\.Dim\s*=\s*c\(([^)]+)\)', data, m1.end())
        if not m2:
            return None
        dims = m2.group(1).split(',')
        return [int(d) for d in dims]


def do_command(cmd: str, cwd: str = None, logger: logging.Logger = None) -> str:
    """
    Spawn process, print stdout/stderr to console.
    Throws exception on non-zero returncode.
    """
    if logger:
        logger.debug('cmd: %s', cmd)
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    proc.wait()
    stdout, stderr = proc.communicate()
    if proc.returncode:
        if stderr:
            msg = 'ERROR\n {} '.format(stderr.decode('utf-8').strip())
        raise Exception(msg)
    if stdout:
        return stdout.decode('utf-8').strip()
    return None
