import atexit
import os.path
import shutil
import tempfile

CMDSTAN_PATH = os.path.abspath(os.path.join('releases', 'cmdstan'))
TMPDIR = tempfile.mkdtemp()
STANSUMMARY_STATS = ['Mean', 'MCSE', 'StdDev', '5%', '50%', '95%',
                         'N_Eff', 'N_Eff/s', 'R_hat']


def cleanup_tmpdir():
    print('deleting tmpfiles dir: {}'.format(TMPDIR))
    shutil.rmtree(TMPDIR, ignore_errors=True)
    print("done")


atexit.register(cleanup_tmpdir)
