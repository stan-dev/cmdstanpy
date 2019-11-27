import atexit
import shutil
import tempfile

STANSUMMARY_STATS = [
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

TMPDIR = tempfile.mkdtemp()


def cleanup_tmpdir():
    print('deleting tmpfiles dir: {}'.format(TMPDIR))
    shutil.rmtree(TMPDIR, ignore_errors=True)
    print('done')


atexit.register(cleanup_tmpdir)

from .utils import set_cmdstan_path, cmdstan_path, set_make_env, install_cmdstan # noqa
from .stanfit import CmdStanMCMC, CmdStanMLE, CmdStanGQ, CmdStanVB # noqa
from .model import CmdStanModel # noqa
from ._version import __version__ # noqa
