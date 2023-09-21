"""Tests for the `log_prob` method new in CmdStan 2.31.0"""

import logging
import os
import re
from test import check_present
from typing import List, Optional

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


@pytest.mark.parametrize("sig_figs, expected, expected_unadjusted", [
    (11, ["-7.0214667713","-1.188472607"], ["-5.5395901199", "-1.4903938392"]),
    (3, ["-7.02", "-1.19"], ["-5.54", "-1.49"]),
    (None, ["-7.02147", "-1.18847"], ["-5.53959", "-1.49039"])
])
def test_lp_good(sig_figs: Optional[int], expected: List[str],
                 expected_unadjusted: List[str]) -> None:
    model = CmdStanModel(stan_file=BERN_STAN)
    params = {"theta": 0.34903938392023830482}
    out = model.log_prob(params, data=BERN_DATA, sig_figs=sig_figs)
    assert "lp_" in out.columns[0]

    # Check the number of digits.
    for actual, value in zip(out.values[0], expected):
        assert str(actual) == value

    out_unadjusted = model.log_prob(
        params, data=BERN_DATA, jacobian=False, sig_figs=sig_figs
    )
    assert "lp_" in out_unadjusted.columns[0]
    assert not np.allclose(out.to_numpy(), out_unadjusted.to_numpy())

    for actual, value in zip(out_unadjusted.values[0], expected_unadjusted):
        assert str(actual) == value


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
