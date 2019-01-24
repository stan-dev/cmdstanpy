import os
import subprocess
import hashlib
import logging
from backports import tempfile
import threading
import filelock
import numpy as np
from . import io

logger = logging.getLogger('cmdstanpy.model')


class CmdStanNotFound(RuntimeError):
    pass


def _find_cmdstan(path=''):
    if path:
        path = os.path.expanduser(os.path.expandvars(path))
        os.environ['CMDSTAN'] = path
    path = os.environ.get('CMDSTAN', 'cmdstan')
    if not os.path.exists(os.path.join(path, 'runCmdStanTests.py')):
        raise CmdStanNotFound(
            'please provide CmdStan path, e.g. lib.cmdstan_path("/path/to/")')
    return path


def _run(cmd, cwd=None, check=True):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = proc.stdout.read().decode('ascii').strip()
    if stdout:
        print(stdout)
    stderr = proc.stderr.read().decode('ascii').strip()
    if stderr:
        print(stderr)


def preprocess_model(stan_fname, hpp_fname=None, overwrite=False):
    """Invoke stanc on the given Stan model file.
    """
    hpp_fname = hpp_fname or stan_fname.replace('.stan', '.hpp')
    if os.path.exists(hpp_fname):
        return
    stanc_path = os.path.join(_find_cmdstan(), 'bin', 'stanc')
    cmd = [stanc_path, '--o={}'.format(hpp_fname), stan_fname]
    cwd = os.path.abspath(os.path.dirname(stan_fname))
    _run(cmd, cwd=cwd, check=False)


def compile_model(stan_fname, opt_lvl=3):
    """Compile the given Stan model file to an executable.
    """
    preprocess_model(stan_fname, overwrite=True)
    path = os.path.abspath(os.path.dirname(stan_fname))
    name = stan_fname[:-5]
    target = os.path.join(path, name)
    cmd = ['make', 'O={}'.format(opt_lvl), target]
    _run(cmd, _find_cmdstan())


def stansummary_csv(csv_in):
    """Invoke the stansummary command on a list of CSV files,
    and return the resulting summary.
    """
    if not isinstance(csv_in, list):
        csv_in = [csv_in]
    exe = os.path.join(_find_cmdstan(), 'bin', 'stansummary')
    with tempfile.TemporaryDirectory() as td:
        csv_out = os.path.join(td, 'summary.csv')
        cmd = [exe, '--csv_file={}'.format(csv_out)] + csv_in
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
        summary = io.parse_summary_csv(csv_out)
    return summary


def model_path():
    """Returns the directory where compiled models should be
    cached.
    """
    key = 'PYCMDSTAN_MODEL_PATH'
    if key not in os.environ:
        os.environ[key] = os.path.join(
            os.path.expanduser('~'), '.cache', 'cmdstanpy')
    if not os.path.exists(os.environ[key]):
        logger.debug('creating cache dir {}'.format(os.environ[key]))
        try:
            os.makedirs(os.environ[key])
        except OSError:
            pass
    logger.debug('have model path {}'.format(os.environ[key]))
    return os.environ[key]


class Model:
    """Stan model.
    """

    def __init__(self, code=None, fname=None, opt_lvl=3):
        """Create a new Stan model from code or filename.
        """
        self.code = code
        self.opt_lvl = opt_lvl
        self.fname = fname
        if fname and code is None:
            with open(fname, 'r') as fd:
                self.code = fd.read()

    @property
    def model_id(self):
        """SHA256 checksum of model code.
        """
        sha = hashlib.sha256()
        sha.update(self.code.encode('ascii'))
        return 'model-{}'.format(sha.hexdigest()[:8])

    def _compile(self, stan_fname):
        with open(stan_fname, 'w') as fd:
            fd.write(self.code)
        compile_model(stan_fname, opt_lvl=self.opt_lvl)

    def compile(self, path=None):
        """Compile the model.
        """
        self.path = path or model_path()
        self.stan_fname = os.path.join(self.path,
                                       '{}.stan'.format(self.model_id))
        self.exe, _ = self.stan_fname.rsplit('.stan')
        self._lock = filelock.FileLock('{}.lock'.format(self.exe))
        with self._lock:
            if not os.path.exists(self.exe):
                self._compile(self.stan_fname)

    def sample(self, **kwargs):
        """Run the model with the sample (NUTS/HMC) method.
        """
        return self.run(method='sample', **kwargs)

    def variational(self, **kwargs):
        """Run the model with the variational (ADVI) method.
        """
        return self.run(method='variational', **kwargs)

    def optimize(self, **kwargs):
        """Run the model with the optimize (L-BFGS) method.
        """
        return self.run(method='optimize', **kwargs)

    def run(self, method, chains=1, wait=False, seed=0, **kwargs):
        """Run chains instances of the model with the given method.
        Extra keyword arguments are passed to Run.__init__.
        """
        rng = np.random.RandomState()
        rng.seed(seed)
        runs = [
            Run(model=self,
                method=method,
                id=rng.randint(1, 99999 + 1),
                **kwargs) for _ in range(chains)
        ]
        if len(runs) == 1:
            return runs[0]
        else:
            return RunSet(*runs)


class Run:
    def __init__(self,
                 model,
                 method,
                 data=None,
                 id=None,
                 start=True,
                 wait=False,
                 tmp_dir='',
                 **method_args):
        """Create a new run of the given model, for a given method.
        """
        self.model = model
        self.id = id
        self.method = method
        self.method_args = method_args
        self.data = data
        self.tmp_dir = tmp_dir or tempfile.TemporaryDirectory()
        self.output_csv_fname = os.path.join(self.tmp_dir.name, 'output.csv')
        self.output_fname = os.path.join(self.tmp_dir.name, 'output.txt')
        self._output_fd = None
        self._cmd = None
        self.proc = None
        self.stdout = None
        self._csv = None
        if data:
            self.data_R_fname = os.path.join(self.tmp_dir.name, 'data.R')
            io.rdump(self.data_R_fname, data)
        if start:
            self.start(wait=wait)

    def __del__(self):
        self.tmp_dir.cleanup()

    def _complex_args(self, cmd, args):
        for key, val in args.items():
            # num_warmup=3
            sep = '='
            # adapt_='delta=0.8' -> adapt delta=0.8
            if key.endswith('_'):
                val = '{0} {1}'.format(key[:-1], val)
                sep = key = ''
            more = []
            if isinstance(val, str):
                # "adapt delta=0.8" -> ['adapt', 'delta=0.8']
                parts = val.split(' ')
                val = parts[0]
                more = parts[1:]
            cmd.append('{0}{1}{2}'.format(key, sep, val))
            cmd.extend(more)

    @property
    def cmd(self):
        """Generate the command line for this run.
        """
        if self._cmd is not None:
            return self._cmd
        if not hasattr(self.model, 'exe'):
            self.model.compile()
        self._cmd = cmd = [self.model.exe]
        if self.id is not None:
            cmd.append('id={}'.format(self.id))
        cmd.append(self.method)
        self._complex_args(cmd, self.method_args)
        if self.data:
            cmd.extend(['data', 'file={}'.format(self.data_R_fname)])
        cmd.extend(['output',
                    'file={}'.format(self.output_csv_fname),
                    'refresh=1'])
        return cmd

    def start(self, wait=True):
        """Start the run; invokes executable in subprocess.
        """
        if self.proc is not None:
            raise RuntimeError('run has already started')
        logger.info('starting %s', ' '.join(self.cmd))
        logger.debug('starting %r', self.cmd)
        self._output_fd = open(self.output_fname, 'w')
        self.proc = subprocess.Popen(
            self.cmd, stdout=self._output_fd, stderr=subprocess.STDOUT)
        if wait:
            self.wait()

    def _wait_loop(self):
        while True:
            try:
                self.proc.wait(1)
                break
            except subprocess.TimeoutExpired:
                continue
            except KeyboardInterrupt:
                break

    def wait(self):
        """Wait for run to complete.
        """
        if self.proc is None:
            self.start(wait=False)
        self._wait_loop()
        self._output_fd.close()
        with open(self.output_fname, 'r') as fd:
            self.stdout = fd.read()
        logger.info('finished %s', ' '.join(self.cmd))
        if self.proc.returncode != 0:
            msg = 'Stan model exited with error %d\n%s'
            msg %= self.proc.returncode, self.stdout
            raise RuntimeError(msg)

    @property
    def csv(self):
        """Return output samples.
        """
        if self._csv is None:
            self.wait()
            self._csv = io.parse_csv(self.output_csv_fname)
        return self._csv

    def __getitem__(self, key):
        """Retrieve a parameter from output.
        """
        return self.csv[key]


class RunSet:
    """A set of runs.
    """

    def __init__(self, *runs):
        """Create a new RunSet from a sequence of Runs.
        """
        self.runs = runs
        self._summary = None
        self._csv = None
        self._summary_thread = threading.Thread(target=self._build_summary)
        self._summary_thread.start()

    def _build_summary(self):
        run_csvs = []
        for run in self.runs:
            run.wait()
            run_csvs.append(run.output_csv_fname)
        self.niter, self._summary = stansummary_csv(run_csvs)

    @property
    def summary(self):
        """Retrive summary results for this RunSet.
        """
        if self._summary is None:
            self._build_summary()
        return self._summary

    def __getitem__(self, key):
        """Retrieve element from summary.
        """
        return self.summary[key]

    def flat_field(self, field, only_params=True):
        """Collect all values from summary for a given column, e.g. R_hat.
        """
        vals = []
        for key, val in self.summary.items():
            if only_params and key.endswith('__'):
                continue
            vals.append(val[field].reshape((-1, )))
        return np.hstack(vals)

    @property
    def R_hats(self):
        """Retrive vector of all R_hat values.
        """
        return self.flat_field('R_hat')

    @property
    def N_Eff(self):
        """Retrieve vector of all N_eff values.
        """
        return self.flat_field('N_Eff')

    @property
    def N_Eff_per_iter(self):
        """Retrieve vector of N_eff / niter.
        """
        return self.N_Eff / self.niter

    @property
    def csv(self):
        """Return all output samples for Runs in this Runset.
        """
        if self._csv is None:
            self._csv = io.merge_csv_data(*[r.csv for r in self.runs])
        return self._csv

    def __iter__(self):
        """Iterate over Runs in this RunSet.
        """
        return iter(self.runs)
