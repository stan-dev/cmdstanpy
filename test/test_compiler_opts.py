"""Compiler options tests"""

import logging
import os
from test import check_present

import pytest

from cmdstanpy.compilation import CompilerOptions

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_opts_empty() -> None:
    opts = CompilerOptions()
    opts.validate()
    assert opts.compose() == []
    assert repr(opts) == 'stanc_options={}, cpp_options={}'

    stanc_opts = {}
    opts = CompilerOptions(stanc_options=stanc_opts)
    opts.validate()
    assert opts.compose() == []

    cpp_opts = {}
    opts = CompilerOptions(cpp_options=cpp_opts)
    opts.validate()
    assert opts.compose() == []

    opts = CompilerOptions(stanc_options=stanc_opts, cpp_options=cpp_opts)
    opts.validate()
    assert opts.compose() == []
    assert repr(opts) == 'stanc_options={}, cpp_options={}'


def test_opts_stanc(caplog: pytest.LogCaptureFixture) -> None:
    stanc_opts = {}
    opts = CompilerOptions()
    opts.validate()
    assert opts.compose() == []

    opts = CompilerOptions(stanc_options=stanc_opts)
    opts.validate()
    assert opts.compose() == []

    stanc_opts['warn-uninitialized'] = True
    opts = CompilerOptions(stanc_options=stanc_opts)
    opts.validate()
    assert opts.compose() == ['STANCFLAGS+=--warn-uninitialized']

    stanc_opts['name'] = 'foo'
    opts = CompilerOptions(stanc_options=stanc_opts)
    opts.validate()
    assert opts.compose() == [
        'STANCFLAGS+=--warn-uninitialized',
        'STANCFLAGS+=--name=foo',
    ]

    stanc_opts['O1'] = True
    opts = CompilerOptions(stanc_options=stanc_opts)
    opts.validate()
    assert opts.compose() == [
        'STANCFLAGS+=--warn-uninitialized',
        'STANCFLAGS+=--name=foo',
        'STANCFLAGS+=--O1',
    ]

    # should add to logger
    stanc_opts['Oexperimental'] = True
    opts = CompilerOptions(stanc_options=stanc_opts)
    with caplog.at_level(logging.WARNING):
        logging.getLogger()
        opts.validate()

    expect = (
        'More than one of (O, O1, O2, Oexperimental)'
        'optimizations passed. Only the last one will'
        'be used'
    )

    check_present(caplog, ('cmdstanpy', 'WARNING', expect))

    assert opts.compose() == [
        'STANCFLAGS+=--warn-uninitialized',
        'STANCFLAGS+=--name=foo',
        'STANCFLAGS+=--O1',
        'STANCFLAGS+=--Oexperimental',
    ]


def test_opts_stanc_opencl() -> None:
    stanc_opts = {}
    stanc_opts['use-opencl'] = 'foo'
    opts = CompilerOptions(stanc_options=stanc_opts)
    opts.validate()
    assert opts.compose() == ['STANCFLAGS+=--use-opencl', 'STAN_OPENCL=TRUE']


def test_opts_stanc_ignore() -> None:
    stanc_opts = {}
    stanc_opts['auto-format'] = True
    opts = CompilerOptions(stanc_options=stanc_opts)
    opts.validate()
    assert opts.compose() == []


def test_opts_stanc_includes() -> None:
    path2 = os.path.join(HERE, 'data', 'optimize')
    paths_str = ','.join([DATAFILES_PATH, path2]).replace('\\', '/')
    expect = 'STANCFLAGS+=--include-paths=' + paths_str

    stanc_opts = {'include-paths': paths_str}
    opts = CompilerOptions(stanc_options=stanc_opts)
    opts.validate()
    opts_list = opts.compose()
    assert expect in opts_list

    stanc_opts = {'include-paths': [DATAFILES_PATH, path2]}
    opts = CompilerOptions(stanc_options=stanc_opts)
    opts.validate()
    opts_list = opts.compose()
    assert expect in opts_list


def test_opts_add_include_paths() -> None:
    expect = 'STANCFLAGS+=--include-paths=' + DATAFILES_PATH.replace('\\', '/')
    stanc_opts = {'warn-uninitialized': True}
    opts = CompilerOptions(stanc_options=stanc_opts)
    opts.validate()
    opts_list = opts.compose()
    assert expect not in opts_list

    opts.add_include_path(DATAFILES_PATH)
    opts.validate()
    opts_list = opts.compose()
    assert expect in opts_list

    path2 = os.path.join(HERE, 'data', 'optimize')
    paths_str = ','.join([DATAFILES_PATH, path2]).replace('\\', '/')
    expect = 'STANCFLAGS+=--include-paths=' + paths_str
    opts.add_include_path(path2)
    opts.validate()
    opts_list = opts.compose()
    assert expect in opts_list


def test_opts_cpp() -> None:
    cpp_opts = {}
    opts = CompilerOptions(cpp_options=cpp_opts)
    opts.validate()
    assert opts.compose() == []

    cpp_opts['STAN_MPI'] = 'TRUE'
    opts = CompilerOptions(cpp_options=cpp_opts)
    opts.validate()
    assert opts.compose() == ['STAN_MPI=TRUE']


def test_opts_cpp_opencl() -> None:
    cpp_opts = {'OPENCL_DEVICE_ID': 1}
    opts = CompilerOptions(cpp_options=cpp_opts)
    opts.validate()
    opts_list = opts.compose()
    assert 'STAN_OPENCL=TRUE' in opts_list
    assert 'OPENCL_DEVICE_ID=1' in opts_list

    cpp_opts = {'OPENCL_DEVICE_ID': 'BAD'}
    opts = CompilerOptions(cpp_options=cpp_opts)
    with pytest.raises(ValueError):
        opts.validate()

    cpp_opts = {'OPENCL_DEVICE_ID': -1}
    opts = CompilerOptions(cpp_options=cpp_opts)
    with pytest.raises(ValueError):
        opts.validate()

    cpp_opts = {'OPENCL_PLATFORM_ID': 'BAD'}
    opts = CompilerOptions(cpp_options=cpp_opts)
    with pytest.raises(ValueError):
        opts.validate()

    cpp_opts = {'OPENCL_PLATFORM_ID': -1}
    opts = CompilerOptions(cpp_options=cpp_opts)
    with pytest.raises(ValueError):
        opts.validate()


def test_user_header() -> None:
    header_file = os.path.join(DATAFILES_PATH, 'return_one.hpp')
    opts = CompilerOptions(user_header=header_file)
    opts.validate()
    assert opts.stanc_options['allow-undefined']

    bad = os.path.join(DATAFILES_PATH, 'nonexistant.hpp')
    opts = CompilerOptions(user_header=bad)
    with pytest.raises(ValueError, match="cannot be found"):
        opts.validate()

    bad_dir = os.path.join(DATAFILES_PATH, 'optimize')
    opts = CompilerOptions(user_header=bad_dir)
    with pytest.raises(ValueError, match="cannot be found"):
        opts.validate()

    non_header = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    opts = CompilerOptions(user_header=non_header)
    with pytest.raises(ValueError, match="must end in .hpp"):
        opts.validate()

    header_file = os.path.join(DATAFILES_PATH, 'return_one.hpp')
    opts = CompilerOptions(
        user_header=header_file, cpp_options={'USER_HEADER': 'foo'}
    )
    with pytest.raises(ValueError, match="Disagreement"):
        opts.validate()
