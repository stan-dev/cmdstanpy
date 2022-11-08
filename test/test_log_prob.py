"""Tests for the `log_prob` method new in CmdStan 2.31.0"""

import logging
import os
from test import CustomTestCase

from testfixtures import LogCapture, StringComparison

from cmdstanpy.model import CmdStanModel
from cmdstanpy.utils import EXTENSION

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')

BERN_STAN = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
BERN_DATA = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
BERN_EXE = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
BERN_BASENAME = 'bernoulli'


class CmdStanLogProb(CustomTestCase):
    def test_lp_good(self):
        model = CmdStanModel(stan_file=BERN_STAN)
        x = model.log_prob({"theta": 0.1}, data=BERN_DATA)
        assert "lp_" in x.columns

    def test_lp_bad(self):
        model = CmdStanModel(stan_file=BERN_STAN)

        with LogCapture(level=logging.ERROR) as log:
            with self.assertRaisesRegex(
                RuntimeError, "failed with return code"
            ):
                model.log_prob({"not_here": 0.1}, data=BERN_DATA)

        log.check_present(
            (
                'cmdstanpy',
                'ERROR',
                StringComparison(r"(?s).*parameter theta not found.*"),
            )
        )
