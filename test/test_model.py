"""CmdStanModel tests"""

import logging
import os
import re
import shutil
import tempfile
from test import check_present
from unittest.mock import patch

import numpy as np
import pytest

from cmdstanpy.model import CmdStanModel
from cmdstanpy.utils import EXTENSION

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
    model = CmdStanModel(stan_file=BERN_STAN)
    assert BERN_STAN == model.stan_file
    assert os.path.samefile(model.exe_file, BERN_EXE)
    assert 'bernoulli' == model.name

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
    if os.path.exists(BERN_EXE):
        os.remove(BERN_EXE)

    model = CmdStanModel(stan_file=BERN_STAN)
    assert model.stan_file is not None
    assert os.path.samefile(model.stan_file, BERN_STAN)
    assert os.path.samefile(model.exe_file, BERN_EXE)
    exe_time = os.path.getmtime(model.exe_file)

    model = CmdStanModel(stan_file=BERN_STAN)
    assert exe_time == os.path.getmtime(model.exe_file)

    model = CmdStanModel(stan_file=BERN_STAN, force_compile=True)
    assert exe_time < os.path.getmtime(model.exe_file)


def test_exe_only() -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    assert BERN_EXE == model.exe_file
    exe_only = os.path.join(DATAFILES_PATH, 'exe_only')
    shutil.copyfile(model.exe_file, exe_only)

    model2 = CmdStanModel(exe_file=exe_only)
    with pytest.raises(RuntimeError):
        model2.code()

    assert not model2._fixed_param


def test_legacy_fixed_param() -> None:
    stan = os.path.join(DATAFILES_PATH, 'datagen_poisson_glm.stan')
    model = CmdStanModel(stan_file=stan)
    assert not model._fixed_param


def test_model_pedantic(caplog: pytest.LogCaptureFixture) -> None:
    stan_file = os.path.join(DATAFILES_PATH, 'bernoulli_pedantic.stan')
    with caplog.at_level(logging.WARNING):
        CmdStanModel(
            stan_file=stan_file,
            stanc_options={'warn-pedantic': True},
            force_compile=True,
        )

    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            re.compile(r'(?s).*The parameter theta has no priors.*'),
        ),
    )


def test_model_bad() -> None:
    with pytest.raises(ValueError):
        CmdStanModel(stan_file=None, exe_file=None)
    with pytest.raises(ValueError):
        CmdStanModel()
    with pytest.raises(ValueError):
        CmdStanModel(stan_file=os.path.join(DATAFILES_PATH, "external.stan"))

    CmdStanModel(stan_file=BERN_STAN)
    exe2 = os.path.join(DATAFILES_PATH, 'bern2oulli' + EXTENSION)

    shutil.copyfile(BERN_EXE, exe2)
    with pytest.raises(ValueError):
        CmdStanModel(stan_file=BERN_STAN, exe_file=exe2)
    os.remove(exe2)
    os.remove(BERN_EXE)
    with pytest.raises(ValueError):
        CmdStanModel(stan_file=BERN_STAN, exe_file=BERN_EXE)


def test_bad_stanc_options() -> None:
    with pytest.raises(ValueError):
        bad_opts = {'X': True}
        CmdStanModel(stan_file=BERN_STAN, stanc_options=bad_opts)
    with pytest.raises(ValueError):
        bad_opts = {'include-paths': True}
        CmdStanModel(stan_file=BERN_STAN, stanc_options=bad_opts)
    with pytest.raises(ValueError):
        bad_opts = {'include-paths': 'lkjdf'}  # type: ignore
        CmdStanModel(stan_file=BERN_STAN, stanc_options=bad_opts)


def test_model_info() -> None:
    model = CmdStanModel(stan_file=BERN_STAN, force_compile=True)
    info_dict = model.exe_info()
    assert info_dict['STAN_THREADS'].lower() == 'false'

    os.remove(model.exe_file)
    with pytest.raises(RuntimeError):
        model.exe_info()

    model_info = model.src_info()
    assert model_info != {}
    assert 'theta' in model_info['parameters']

    model_include = CmdStanModel(
        stan_file=os.path.join(DATAFILES_PATH, "bernoulli_include.stan"),
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
        with caplog.at_level(logging.INFO):
            CmdStanModel(
                stan_file=stan_file,
                stanc_options={"include-paths": [include_path]},
                force_compile=True,
            )

        check_present(
            caplog, ('cmdstanpy', 'INFO', re.compile('compiling stan file'))
        )
        with open(include_target, "w") as fd:
            fd.write("gobbledygook")
        with pytest.raises(ValueError, match="Failed to get source info"):
            CmdStanModel(
                stan_file=stan_file,
                stanc_options={"include-paths": [include_path]},
                force_compile=True,
            )


@pytest.mark.parametrize(
    "stan_file, include_paths",
    [
        ('add_one_model.stan', ['include-path']),
        ('bernoulli_include.stan', []),
    ],
)
def test_compile_with_includes(
    caplog: pytest.LogCaptureFixture, stan_file: str, include_paths: list[str]
) -> None:
    getmtime = os.path.getmtime
    stan_file = os.path.join(DATAFILES_PATH, stan_file)
    exe_file = os.path.splitext(stan_file)[0] + EXTENSION
    if os.path.isfile(exe_file):
        os.unlink(exe_file)
    include_paths = [
        os.path.join(DATAFILES_PATH, path) for path in include_paths
    ]

    with caplog.at_level(logging.INFO):
        # Compile for the first time.
        CmdStanModel(
            stan_file=stan_file,
            stanc_options={"include-paths": include_paths},
        )
    check_present(
        caplog, ('cmdstanpy', 'INFO', re.compile('compiling stan file'))
    )

    # Compile for the second time, ensuring cache is used.
    with caplog.at_level(logging.DEBUG):
        CmdStanModel(
            stan_file=stan_file,
            stanc_options={"include-paths": include_paths},
        )
    check_present(
        caplog, ('cmdstanpy', 'DEBUG', re.compile('found newer exe file'))
    )

    # Compile after modifying included file, ensuring cache is not used.
    def _patched_getmtime(filename: str) -> float:
        includes = ['divide_real_by_two.stan', 'add_one_function.stan']
        if any(str(filename).endswith(include) for include in includes):
            return float('inf')
        return getmtime(filename)

    caplog.clear()
    with (
        caplog.at_level(logging.INFO),
        patch('os.path.getmtime', side_effect=_patched_getmtime),
    ):
        CmdStanModel(
            stan_file=stan_file,
            stanc_options={"include-paths": include_paths},
        )
    check_present(
        caplog, ('cmdstanpy', 'INFO', re.compile('compiling stan file'))
    )


def test_compile_force() -> None:
    if os.path.exists(BERN_EXE):
        os.remove(BERN_EXE)
    more_opts = {'STAN_THREADS': 'TRUE'}

    model = CmdStanModel(
        stan_file=BERN_STAN, cpp_options=more_opts, force_compile=True
    )

    assert os.path.exists(model.exe_file)

    info_dict = model.exe_info()
    assert info_dict['STAN_THREADS'].lower() == 'true'

    # test compile='force' in constructor
    model2 = CmdStanModel(stan_file=BERN_STAN, force_compile=True)
    info_dict2 = model2.exe_info()
    assert info_dict2['STAN_THREADS'].lower() == 'false'


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


def test_repr() -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    assert repr(BERN_STAN) in repr(model)


def test_print() -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    assert CODE == model.code()


def test_model_compile() -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    assert os.path.samefile(model.exe_file, BERN_EXE)

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
        shutil.copyfile(BERN_STAN, bern_stan_new)
        model = CmdStanModel(stan_file=bern_stan_new)

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

    include_paths = [os.path.join(DATAFILES_PATH, "include-path")]
    stanc_options = {"include-paths": include_paths}
    model = CmdStanModel(stan_file=stan_file, stanc_options=stanc_options)
    assert os.path.exists(model.exe_file)


def test_model_includes_implicit() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_include.stan')
    exe = os.path.join(DATAFILES_PATH, 'bernoulli_include' + EXTENSION)
    if os.path.exists(exe):
        os.remove(exe)
    model2 = CmdStanModel(stan_file=stan)
    assert os.path.samefile(model2.exe_file, exe)


def test_diagnose() -> None:
    # Check the gradients.
    model = CmdStanModel(stan_file=BERN_STAN)
    gradients = model.diagnose(data=BERN_DATA)

    # Check we have the right columns.
    assert set(gradients) == {
        "param_idx",
        "value",
        "model",
        "finite_diff",
        "error",
    }

    # Check gradients against the same value as in `log_prob`.
    inits = {"theta": 0.34903938392023830482}
    gradients = model.diagnose(data=BERN_DATA, inits=inits)
    np.testing.assert_allclose(gradients.model.iloc[0], -1.18847)

    # Simulate bad gradients by using large finite difference.
    with pytest.raises(RuntimeError, match="may exceed the error threshold"):
        model.diagnose(data=BERN_DATA, epsilon=3)

    # Check we get the results if we set require_gradients_ok=False.
    gradients = model.diagnose(
        data=BERN_DATA,
        epsilon=3,
        require_gradients_ok=False,
    )
    assert np.abs(gradients["error"]).max() > 1e-3
