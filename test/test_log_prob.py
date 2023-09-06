"""Tests for the `log_prob` method new in CmdStan 2.31.0"""

import logging
import os
import re
from test import check_present
from typing import Optional

import numpy as np
import pytest

from cmdstanpy.model import CmdStanModel
from cmdstanpy.utils import EXTENSION

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')

BERN_STAN = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
BERN_DATA = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
BERN_EXE = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
BERN_BASENAME = 'bernoulli'


@pytest.mark.parametrize("sig_figs", [15, 3, None])
def test_lp_good(sig_figs: Optional[int]) -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    params = {"theta": 0.34903938392023830482}
    out = model.log_prob(params, data=BERN_DATA, sig_figs=sig_figs)
    assert "lp_" in out.columns[0]

    # Check the number of digits.
    expected_values = {
        None: ["-7.02147", "-1.18847"],
        3: ["-7.02", "-1.19"],
        15: ["-7.02146677130525","-1.18847260704286"],
    }[sig_figs]
    for actual, expected in zip(out.values[0], expected_values):
        assert str(actual) == expected

    out_unadjusted = model.log_prob(
        params, data=BERN_DATA, jacobian=False, sig_figs=sig_figs
    )
    assert "lp_" in out_unadjusted.columns[0]
    assert not np.allclose(out.to_numpy(), out_unadjusted.to_numpy())

    expected_values = {
        None: ["-5.53959", "-1.49039"],
        3: ["-5.54", "-1.49"],
        15: ["-5.53959011989073", "-1.49039383920238"],
    }[sig_figs]
    for actual, expected in zip(out_unadjusted.values[0], expected_values):
        assert str(actual) == expected


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
            re.compile(r"(?s).*variable does not exist.*name=theta.*"),
        ),
    )
