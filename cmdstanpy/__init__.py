# pylint: disable=wrong-import-position
"""CmdStanPy Module"""

import atexit
import shutil
import tempfile

_STANSUMMARY_STATS = [
    'Mean',
    'MCSE',
    'StdDev',
    '5%',
    '50%',
    '95%',
    'N_Eff',
    'N_Eff/s',
    'R_hat',
]

_TMPDIR = tempfile.mkdtemp()
_CMDSTAN_WARMUP = 1000
_CMDSTAN_SAMPLING = 1000
_CMDSTAN_THIN = 1


def _cleanup_tmpdir():
    """Force deletion of _TMPDIR."""
    print('deleting tmpfiles dir: {}'.format(_TMPDIR))
    shutil.rmtree(_TMPDIR, ignore_errors=True)
    print('done')


atexit.register(_cleanup_tmpdir)


from .utils import (
    set_cmdstan_path,
    cmdstan_path,
    set_make_env,
    install_cmdstan,
)  # noqa
from .stanfit import CmdStanMCMC, CmdStanMLE, CmdStanGQ, CmdStanVB  # noqa
from .model import CmdStanModel  # noqa
from ._version import __version__  # noqa

__all__ = [
    'set_cmdstan_path',
    'cmdstan_path',
    'set_make_env',
    'install_cmdstan',
    'CmdStanMCMC',
    'CmdStanMLE',
    'CmdStanGQ',
    'CmdStanVB',
    'CmdStanModel',
]
