"""Tests for deprecated positional arguments to public functions"""

# positional calls below are deliberate
# pylint: disable=too-many-function-args

import inspect
import logging
import os
import re
import sys
from test import check_present
from typing import Any, Callable
from unittest import mock

import pytest

from cmdstanpy.model import CmdStanModel
from cmdstanpy.utils import install_cmdstan
from cmdstanpy.utils.deprecation import deprecate_positional_args

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class Dummy:
    @deprecate_positional_args(last_deprecated="c")
    def method(
        self, a: Any = None, *, b: Any = None, c: Any = None, d: Any = None
    ) -> tuple[Any, Any, Any, Any]:
        return a, b, c, d


def test_keywords_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert Dummy().method(1, b=2, c=3, d=4) == (1, 2, 3, 4)
        assert Dummy().method() == (None, None, None, None)
    assert not caplog.records


def test_positional_maps_to_keywords(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert Dummy().method(1, 2, 3, d=4) == (1, 2, 3, 4)
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            re.compile(r"Passing `b`, `c` to Dummy\.method\(\) by position"),
        ),
    )


def test_positional_and_keyword_collision() -> None:
    with pytest.raises(TypeError, match="multiple values for argument 'b'"):
        Dummy().method(1, 2, b=3)


def test_too_many_positional() -> None:
    # `d` was never positional, so it must not be filled by position
    with pytest.raises(TypeError):
        Dummy().method(1, 2, 3, 4)


def test_signature_is_keyword_only() -> None:
    params = inspect.signature(Dummy.method).parameters
    assert params['a'].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params['b'].kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    'func, positional, last_deprecated',
    [
        (CmdStanModel.sample, ['self', 'data'], 'timeout'),
        (CmdStanModel.optimize, ['self', 'data'], 'jacobian'),
        (CmdStanModel.variational, ['self', 'data'], 'timeout'),
        (
            CmdStanModel.generate_quantities,
            ['self', 'data', 'previous_fit'],
            'timeout',
        ),
        (install_cmdstan, ['version'], 'cores'),
    ],
)
def test_public_signatures(
    func: Callable[..., Any], positional: list[str], last_deprecated: str
) -> None:
    params = inspect.signature(func).parameters
    kinds = {name: p.kind for name, p in params.items()}
    assert [
        n
        for n, k in kinds.items()
        if k == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ] == positional
    # guards against a rename silently shifting the positional mapping
    assert kinds[last_deprecated] == inspect.Parameter.KEYWORD_ONLY


def test_optimize_positional_seed(caplog: pytest.LogCaptureFixture) -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    model = CmdStanModel(stan_file=stan)
    with caplog.at_level(logging.WARNING):
        mle_pos = model.optimize(jdata, 1234)
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            re.compile(r"Passing `seed` to CmdStanModel\.optimize\(\)"),
        ),
    )
    mle_kw = model.optimize(jdata, seed=1234)
    assert mle_pos.metadata.cmdstan_config['seed'] == 1234
    assert mle_pos.optimized_params_dict == mle_kw.optimized_params_dict


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="mock.patch resolves `cmdstanpy.install_cmdstan` to the "
    "re-exported function, not the module, before Python 3.11",
)
def test_install_cmdstan_positional(caplog: pytest.LogCaptureFixture) -> None:
    # stop right after argument handling; nothing is downloaded
    with mock.patch(
        'cmdstanpy.install_cmdstan.run_install', side_effect=RuntimeError
    ) as run_install:
        with caplog.at_level(logging.WARNING):
            install_cmdstan("2.36.0", "some_dir", True)
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            re.compile(r"Passing `dir`, `overwrite` to install_cmdstan\(\)"),
        ),
    )
    settings = run_install.call_args.args[0]
    assert settings.version == "2.36.0"
    assert settings.dir == "some_dir"
    assert settings.overwrite
