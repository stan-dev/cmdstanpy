# pylint: disable=wrong-import-position
"""CmdStanPy Module"""

import atexit
import shutil
import tempfile

_TMPDIR = tempfile.mkdtemp()
_CMDSTAN_WARMUP = 1000
_CMDSTAN_SAMPLING = 1000
_CMDSTAN_THIN = 1
_CMDSTAN_REFRESH = 100
_DOT_CMDSTAN = '.cmdstan'


def _cleanup_tmpdir() -> None:
    """Force deletion of _TMPDIR."""
    shutil.rmtree(_TMPDIR, ignore_errors=True)


atexit.register(_cleanup_tmpdir)


from ._version import __version__  # noqa
from .install_cmdstan import rebuild_cmdstan
from .model import CmdStanModel
from .stanfit import (
    CmdStanGQ,
    CmdStanMCMC,
    CmdStanMLE,
    CmdStanVB,
    InferenceMetadata,
    from_csv,
)
from .utils import (
    cmdstan_path,
    cmdstan_version,
    install_cmdstan,
    set_cmdstan_path,
    set_make_env,
    show_versions,
    write_stan_json,
)

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
    'InferenceMetadata',
    'from_csv',
    'write_stan_json',
    'show_versions',
    'rebuild_cmdstan',
    'cmdstan_version',
]
