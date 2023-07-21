"""Tests for the `log_prob` method new in CmdStan 2.31.0"""

import logging
import os
import re
from test import check_present

import pytest

from cmdstanpy.model import CmdStanModel
from cmdstanpy.utils import EXTENSION

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')

BERN_STAN = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
BERN_DATA = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
BERN_EXE = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
BERN_BASENAME = 'bernoulli'


def test_lp_good() -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    x = model.log_prob({"theta": 0.1}, data=BERN_DATA)
    assert "lp__" in x.columns


def test_lp_bad(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = CmdStanModel(stan_file=BERN_STAN)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="failed with return code"):
            model.log_prob({"not_here": 0.1}, data=BERN_DATA)

    check_present(
        caplog,
        (
            'cmdstanpy',
            'ERROR',
            re.compile(r"(?s).*parameter theta not found.*"),
        ),
    )
