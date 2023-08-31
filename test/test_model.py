"""CmdStanModel tests"""

import contextlib
import io
import logging
import os
import re
import shutil
import tempfile
from glob import glob
from test import check_present, raises_nested
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from cmdstanpy.model import CmdStanModel
from cmdstanpy.utils import EXTENSION, cmdstan_version_before

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')

CODE = """data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(1, 1); // uniform prior on interval 0,1
  y ~ bernoulli(theta);
}
"""

BERN_STAN = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
BERN_DATA = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
BERN_EXE = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
BERN_BASENAME = 'bernoulli'


def test_model_good() -> None:
    # compile on instantiation, override model name
    model = CmdStanModel(model_name='bern', stan_file=BERN_STAN)
    assert BERN_STAN == model.stan_file
    assert os.path.samefile(model.exe_file, BERN_EXE)
    assert 'bern' == model.name

    # compile with external header
    model = CmdStanModel(
        stan_file=os.path.join(DATAFILES_PATH, "external.stan"),
        user_header=os.path.join(DATAFILES_PATH, 'return_one.hpp'),
    )

    # default model name
    model = CmdStanModel(stan_file=BERN_STAN)
    assert BERN_BASENAME == model.name

    # instantiate with existing exe
    model = CmdStanModel(stan_file=BERN_STAN, exe_file=BERN_EXE)
    assert BERN_STAN == model.stan_file
    assert os.path.samefile(model.exe_file, BERN_EXE)


def test_ctor_compile_arg() -> None:
    # instantiate, don't compile
    if os.path.exists(BERN_EXE):
        os.remove(BERN_EXE)
    model = CmdStanModel(stan_file=BERN_STAN, compile=False)
    assert BERN_STAN == model.stan_file
    assert model.exe_file is None

    model = CmdStanModel(stan_file=BERN_STAN, compile=True)
    assert os.path.samefile(model.exe_file, BERN_EXE)
    exe_time = os.path.getmtime(model.exe_file)

    model = CmdStanModel(stan_file=BERN_STAN)
    assert exe_time == os.path.getmtime(model.exe_file)

    model = CmdStanModel(stan_file=BERN_STAN, compile='force')
    assert exe_time < os.path.getmtime(model.exe_file)


def test_exe_only() -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    assert BERN_EXE == model.exe_file
    exe_only = os.path.join(DATAFILES_PATH, 'exe_only')
    shutil.copyfile(model.exe_file, exe_only)

    model2 = CmdStanModel(exe_file=exe_only)
    with pytest.raises(RuntimeError):
        model2.code()
    with pytest.raises(RuntimeError):
        model2.compile()
    assert not model2._fixed_param


def test_fixed_param() -> None:
    stan = os.path.join(DATAFILES_PATH, 'datagen_poisson_glm.stan')
    model = CmdStanModel(stan_file=stan)
    assert model._fixed_param


def test_model_pedantic(caplog: pytest.LogCaptureFixture) -> None:
    stan_file = os.path.join(DATAFILES_PATH, 'bernoulli_pedantic.stan')
    with caplog.at_level(logging.WARNING):
        logging.getLogger()
        model = CmdStanModel(model_name='bern', stan_file=stan_file)
        model.compile(force=True, stanc_options={'warn-pedantic': True})

    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            re.compile(r'(?s).*The parameter theta has no priors.*'),
        ),
    )


@pytest.mark.order(before="test_model_good")
def test_model_bad() -> None:
    with pytest.raises(ValueError):
        CmdStanModel(stan_file=None, exe_file=None)
    with pytest.raises(ValueError):
        CmdStanModel(model_name='bad')
    with pytest.raises(ValueError):
        CmdStanModel(model_name='', stan_file=BERN_STAN)
    with pytest.raises(ValueError):
        CmdStanModel(model_name='   ', stan_file=BERN_STAN)
    with pytest.raises(ValueError):
        CmdStanModel(stan_file=os.path.join(DATAFILES_PATH, "external.stan"))
    CmdStanModel(stan_file=BERN_STAN)
    os.remove(BERN_EXE)
    with pytest.raises(ValueError):
        CmdStanModel(stan_file=BERN_STAN, exe_file=BERN_EXE)


def test_stanc_options() -> None:
    allowed_optims = ("", "0", "1", "experimental")
    for optim in allowed_optims:
        opts = {
            f'O{optim}': True,
            'allow-undefined': True,
            'use-opencl': True,
            'name': 'foo',
        }
        model = CmdStanModel(
            stan_file=BERN_STAN, compile=False, stanc_options=opts
        )
        stanc_opts = model.stanc_options
        assert stanc_opts[f'O{optim}']
        assert stanc_opts['allow-undefined']
        assert stanc_opts['use-opencl']
        assert stanc_opts['name'] == 'foo'

        cpp_opts = model.cpp_options
        assert cpp_opts['STAN_OPENCL'] == 'TRUE'

    with pytest.raises(ValueError):
        bad_opts = {'X': True}
        model = CmdStanModel(
            stan_file=BERN_STAN, compile=False, stanc_options=bad_opts
        )
    with pytest.raises(ValueError):
        bad_opts = {'include-paths': True}
        model = CmdStanModel(
            stan_file=BERN_STAN, compile=False, stanc_options=bad_opts
        )
    with pytest.raises(ValueError):
        bad_opts = {'include-paths': 'lkjdf'}
        model = CmdStanModel(
            stan_file=BERN_STAN, compile=False, stanc_options=bad_opts
        )


def test_cpp_options() -> None:
    opts = {
        'STAN_OPENCL': 'TRUE',
        'STAN_MPI': 'TRUE',
        'STAN_THREADS': 'TRUE',
    }
    model = CmdStanModel(stan_file=BERN_STAN, compile=False, cpp_options=opts)
    cpp_opts = model.cpp_options
    assert cpp_opts['STAN_OPENCL'] == 'TRUE'
    assert cpp_opts['STAN_MPI'] == 'TRUE'
    assert cpp_opts['STAN_THREADS'] == 'TRUE'


def test_model_info() -> None:
    model = CmdStanModel(stan_file=BERN_STAN, compile=False)
    model.compile(force=True)
    info_dict = model.exe_info()
    assert info_dict['STAN_THREADS'].lower() == 'false'

    if model.exe_file is not None and os.path.exists(model.exe_file):
        os.remove(model.exe_file)
    empty_dict = model.exe_info()
    assert len(empty_dict) == 0

    model_info = model.src_info()
    assert model_info != {}
    assert 'theta' in model_info['parameters']

    model_include = CmdStanModel(
        stan_file=os.path.join(DATAFILES_PATH, "bernoulli_include.stan"),
        compile=False,
    )
    model_info_include = model_include.src_info()
    assert model_info_include != {}
    assert 'theta' in model_info_include['parameters']
    assert 'included_files' in model_info_include


def test_compile_with_bad_includes(caplog: pytest.LogCaptureFixture) -> None:
    # Ensure compilation fails if we break an included file.
    stan_file = os.path.join(DATAFILES_PATH, "add_one_model.stan")
    exe_file = os.path.splitext(stan_file)[0] + EXTENSION
    if os.path.isfile(exe_file):
        os.unlink(exe_file)
    with tempfile.TemporaryDirectory() as include_path:
        include_source = os.path.join(
            DATAFILES_PATH, "include-path", "add_one_function.stan"
        )
        include_target = os.path.join(include_path, "add_one_function.stan")
        shutil.copy(include_source, include_target)
        model = CmdStanModel(
            stan_file=stan_file,
            compile=False,
            stanc_options={"include-paths": [include_path]},
        )
        with caplog.at_level(logging.INFO):
            model.compile()
        check_present(
            caplog, ('cmdstanpy', 'INFO', re.compile('compiling stan file'))
        )
        with open(include_target, "w") as fd:
            fd.write("gobbledygook")
        with pytest.raises(ValueError, match="Failed to get source info"):
            model.compile()


@pytest.mark.parametrize(
    "stan_file, include_paths",
    [
        ('add_one_model.stan', ['include-path']),
        ('bernoulli_include.stan', []),
    ],
)
def test_compile_with_includes(
    caplog: pytest.LogCaptureFixture, stan_file: str, include_paths: List[str]
) -> None:
    getmtime = os.path.getmtime
    stan_file = os.path.join(DATAFILES_PATH, stan_file)
    exe_file = os.path.splitext(stan_file)[0] + EXTENSION
    if os.path.isfile(exe_file):
        os.unlink(exe_file)
    include_paths = [
        os.path.join(DATAFILES_PATH, path) for path in include_paths
    ]

    # Compile for the first time.
    model = CmdStanModel(
        stan_file=stan_file,
        compile=False,
        stanc_options={"include-paths": include_paths},
    )
    with caplog.at_level(logging.INFO):
        model.compile()
    check_present(
        caplog, ('cmdstanpy', 'INFO', re.compile('compiling stan file'))
    )

    # Compile for the second time, ensuring cache is used.
    with caplog.at_level(logging.DEBUG):
        model.compile()
    check_present(
        caplog, ('cmdstanpy', 'DEBUG', re.compile('found newer exe file'))
    )

    # Compile after modifying included file, ensuring cache is not used.
    def _patched_getmtime(filename: str) -> float:
        includes = ['divide_real_by_two.stan', 'add_one_function.stan']
        if any(filename.endswith(include) for include in includes):
            return float('inf')
        return getmtime(filename)

    caplog.clear()
    with caplog.at_level(logging.INFO), patch(
        'os.path.getmtime', side_effect=_patched_getmtime
    ):
        model.compile()
    check_present(
        caplog, ('cmdstanpy', 'INFO', re.compile('compiling stan file'))
    )


def test_compile_force() -> None:
    if os.path.exists(BERN_EXE):
        os.remove(BERN_EXE)
    model = CmdStanModel(stan_file=BERN_STAN, compile=False, cpp_options={})
    assert model.exe_file is None

    model.compile(force=True)
    assert model.exe_file is not None
    assert os.path.exists(model.exe_file)

    info_dict = model.exe_info()
    assert info_dict['STAN_THREADS'].lower() == 'false'

    more_opts = {'STAN_THREADS': 'TRUE'}

    model.compile(force=True, cpp_options=more_opts)
    assert model.exe_file is not None
    assert os.path.exists(model.exe_file)

    info_dict2 = model.exe_info()
    assert info_dict2['STAN_THREADS'].lower() == 'true'

    override_opts = {'STAN_NO_RANGE_CHECKS': 'TRUE'}

    model.compile(force=True, cpp_options=override_opts, override_options=True)
    info_dict3 = model.exe_info()
    assert info_dict3['STAN_THREADS'].lower() == 'false'
    assert info_dict3['STAN_NO_RANGE_CHECKS'].lower() == 'true'

    model.compile(force=True, cpp_options=more_opts)
    info_dict4 = model.exe_info()
    assert info_dict4['STAN_THREADS'].lower() == 'true'

    # test compile='force' in constructor
    model2 = CmdStanModel(stan_file=BERN_STAN, compile='force')
    info_dict5 = model2.exe_info()
    assert info_dict5['STAN_THREADS'].lower() == 'false'


def test_model_paths() -> None:
    # pylint: disable=unused-variable
    model = CmdStanModel(stan_file=BERN_STAN)  # instantiates exe
    assert os.path.exists(BERN_EXE)

    dotdot_stan = os.path.realpath(os.path.join('..', 'bernoulli.stan'))
    dotdot_exe = os.path.realpath(os.path.join('..', 'bernoulli' + EXTENSION))
    shutil.copyfile(BERN_STAN, dotdot_stan)
    shutil.copyfile(BERN_EXE, dotdot_exe)
    model1 = CmdStanModel(
        stan_file=os.path.join('..', 'bernoulli.stan'),
        exe_file=os.path.join('..', 'bernoulli' + EXTENSION),
    )
    assert model1.stan_file == dotdot_stan
    assert model1.exe_file == dotdot_exe
    os.remove(dotdot_stan)
    os.remove(dotdot_exe)

    tilde_stan = os.path.realpath(
        os.path.join(os.path.expanduser('~'), 'bernoulli.stan')
    )
    tilde_exe = os.path.realpath(
        os.path.join(os.path.expanduser('~'), 'bernoulli' + EXTENSION)
    )
    shutil.copyfile(BERN_STAN, tilde_stan)
    shutil.copyfile(BERN_EXE, tilde_exe)
    model2 = CmdStanModel(
        stan_file=os.path.join('~', 'bernoulli.stan'),
        exe_file=os.path.join('~', 'bernoulli' + EXTENSION),
    )
    assert model2.stan_file == tilde_stan
    assert model2.exe_file == tilde_exe
    os.remove(tilde_stan)
    os.remove(tilde_exe)


def test_model_none() -> None:
    with pytest.raises(ValueError):
        _ = CmdStanModel(exe_file=None, stan_file=None)


def test_model_file_does_not_exist() -> None:
    with pytest.raises(ValueError):
        CmdStanModel(stan_file='xdlfkjx', exe_file='sdfndjsds')

    stan = os.path.join(DATAFILES_PATH, 'b')
    with pytest.raises(ValueError):
        CmdStanModel(stan_file=stan)


def test_model_syntax_error() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bad_syntax.stan')
    with pytest.raises(ValueError, match=r'.*Syntax error.*'):
        CmdStanModel(stan_file=stan)


def test_model_syntax_error_without_compile():
    stan = os.path.join(DATAFILES_PATH, 'bad_syntax.stan')
    CmdStanModel(stan_file=stan, compile=False)


def test_repr() -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    model_repr = repr(model)
    assert 'name=bernoulli' in model_repr


def test_print() -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    assert CODE == model.code()


def test_model_compile() -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    assert os.path.samefile(model.exe_file, BERN_EXE)

    model = CmdStanModel(stan_file=BERN_STAN)
    assert os.path.samefile(model.exe_file, BERN_EXE)
    old_exe_time = os.path.getmtime(model.exe_file)
    os.remove(BERN_EXE)
    model.compile()
    new_exe_time = os.path.getmtime(model.exe_file)
    assert new_exe_time > old_exe_time

    # test compile with existing exe - timestamp on exe unchanged
    exe_time = os.path.getmtime(model.exe_file)
    model2 = CmdStanModel(stan_file=BERN_STAN)
    assert exe_time == os.path.getmtime(model2.exe_file)


@pytest.mark.parametrize("path", ["space in path", "tilde~in~path"])
def test_model_compile_special_char(path: str) -> None:
    with tempfile.TemporaryDirectory(
        prefix="cmdstanpy_testfolder_"
    ) as tmp_path:
        path_with_special_char = os.path.join(tmp_path, path)
        os.makedirs(path_with_special_char, exist_ok=True)
        bern_stan_new = os.path.join(
            path_with_special_char, os.path.split(BERN_STAN)[1]
        )
        bern_exe_new = os.path.join(
            path_with_special_char, os.path.split(BERN_EXE)[1]
        )
        shutil.copyfile(BERN_STAN, bern_stan_new)
        model = CmdStanModel(stan_file=bern_stan_new)

        old_exe_time = os.path.getmtime(model.exe_file)
        os.remove(bern_exe_new)
        model.compile()
        new_exe_time = os.path.getmtime(model.exe_file)
        assert new_exe_time > old_exe_time

        # test compile with existing exe - timestamp on exe unchanged
        exe_time = os.path.getmtime(model.exe_file)
        model2 = CmdStanModel(stan_file=bern_stan_new)
        assert exe_time == os.path.getmtime(model2.exe_file)


@pytest.mark.parametrize("path", ["space in path", "tilde~in~path"])
def test_model_includes_special_char(path: str) -> None:
    """Test model with include file in path with spaces."""
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_include.stan')
    stan_divide = os.path.join(DATAFILES_PATH, 'divide_real_by_two.stan')

    with tempfile.TemporaryDirectory(
        prefix="cmdstanpy_testfolder_"
    ) as tmp_path:
        path_with_special_char = os.path.join(tmp_path, path)
        os.makedirs(path_with_special_char, exist_ok=True)
        bern_stan_new = os.path.join(
            path_with_special_char, os.path.split(stan)[1]
        )
        stan_divide_new = os.path.join(
            path_with_special_char, os.path.split(stan_divide)[1]
        )
        shutil.copyfile(stan, bern_stan_new)
        shutil.copyfile(stan_divide, stan_divide_new)

        model = CmdStanModel(
            stan_file=bern_stan_new,
            stanc_options={'include-paths': path_with_special_char},
        )
        assert path in str(model.exe_file)

        assert path in model.src_info()['included_files'][0]
        assert (
            "divide_real_by_two.stan" in model.src_info()['included_files'][0]
        )


def test_model_includes_explicit() -> None:
    if os.path.exists(BERN_EXE):
        os.remove(BERN_EXE)
    model = CmdStanModel(
        stan_file=BERN_STAN, stanc_options={'include-paths': DATAFILES_PATH}
    )
    assert BERN_STAN == model.stan_file
    assert os.path.samefile(model.exe_file, BERN_EXE)


def test_model_compile_with_explicit_includes() -> None:
    stan_file = os.path.join(DATAFILES_PATH, "add_one_model.stan")
    exe_file = os.path.splitext(stan_file)[0] + EXTENSION
    if os.path.isfile(exe_file):
        os.unlink(exe_file)

    model = CmdStanModel(stan_file=stan_file, compile=False)
    include_paths = [os.path.join(DATAFILES_PATH, "include-path")]
    stanc_options = {"include-paths": include_paths}
    model.compile(stanc_options=stanc_options)


def test_model_includes_implicit() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_include.stan')
    exe = os.path.join(DATAFILES_PATH, 'bernoulli_include' + EXTENSION)
    if os.path.exists(exe):
        os.remove(exe)
    model2 = CmdStanModel(stan_file=stan)
    assert os.path.samefile(model2.exe_file, exe)


@pytest.mark.skipif(
    not cmdstan_version_before(2, 32),
    reason="Deprecated syntax removed in Stan 2.32",
)
def test_model_format_deprecations() -> None:
    stan = os.path.join(DATAFILES_PATH, 'format_me_deprecations.stan')

    model = CmdStanModel(stan_file=stan, compile=False)

    sys_stdout = io.StringIO()
    with contextlib.redirect_stdout(sys_stdout):
        model.format(canonicalize=True)

    formatted = sys_stdout.getvalue()
    assert "//" in formatted
    assert "#" not in formatted
    assert "<-" not in formatted
    assert formatted.count('(') == 0

    shutil.copy(stan, stan + '.testbak')
    try:
        model.format(overwrite_file=True, canonicalize=True)
        assert len(glob(stan + '.bak-*')) == 1
    finally:
        shutil.copy(stan + '.testbak', stan)


@pytest.mark.skipif(
    cmdstan_version_before(2, 29), reason='Options only available later'
)
def test_model_format_options() -> None:
    stan = os.path.join(DATAFILES_PATH, 'format_me.stan')

    model = CmdStanModel(stan_file=stan, compile=False)

    sys_stdout = io.StringIO()
    with contextlib.redirect_stdout(sys_stdout):
        model.format(max_line_length=10)
    formatted = sys_stdout.getvalue()
    assert len(formatted.splitlines()) > 11

    sys_stdout = io.StringIO()
    with contextlib.redirect_stdout(sys_stdout):
        model.format(canonicalize='braces')
    formatted = sys_stdout.getvalue()
    assert formatted.count('{') == 3
    assert formatted.count('(') == 4

    sys_stdout = io.StringIO()
    with contextlib.redirect_stdout(sys_stdout):
        model.format(canonicalize=['parentheses'])
    formatted = sys_stdout.getvalue()
    assert formatted.count('{') == 1
    assert formatted.count('(') == 1

    sys_stdout = io.StringIO()
    with contextlib.redirect_stdout(sys_stdout):
        model.format(canonicalize=True)
    formatted = sys_stdout.getvalue()
    assert formatted.count('{') == 3
    assert formatted.count('(') == 1


@patch(
    'cmdstanpy.utils.cmdstan.cmdstan_version',
    MagicMock(return_value=(2, 27)),
)
def test_format_old_version() -> None:
    assert cmdstan_version_before(2, 28)

    stan = os.path.join(DATAFILES_PATH, 'format_me.stan')
    model = CmdStanModel(stan_file=stan, compile=False)
    with raises_nested(RuntimeError, r"--canonicalize"):
        model.format(canonicalize='braces')
    with raises_nested(RuntimeError, r"--max-line"):
        model.format(max_line_length=88)

    model.format(canonicalize=True)
