import json
import os
import os.path
from pprint import pformat

class Conf(object):
    _config_location = '../config.json'

    def __init__(self):
        if os.path.exists(self._config_location):
            self.__dict__ = json.load(open(self._config_location))
        else:
            self.__dict__ = {}

    def __getitem__(self, item):
        if (item in self.__dict__):
            return self.__dict__[item]
        else:
            return None

    def __repr__(self):
        return 'config file: "{}" entries:\n"{}"'.format(
            self._config_location, pformat(self.__dict__))


class Model(object):
    """Stan model."""

    def __init__(self, name=None, stan_file=None, exe_file=None):
        """Initialize object, name, stan_file args required."""
        self.name = name
        """defaults to base name of Stan program file."""
        self.stan_file = stan_file
        """full path to Stan program src."""
        self.exe_file = exe_file
        """full path to compiled c++ executible."""
        if name is None:
            raise ValueError("attribute name unspecified")
        if stan_file is None:
            raise ValueError("attribute stan_file unspecified")
        if not os.path.exists(stan_file):
            raise ValueError('no such stan_file {}'.format(self.stan_file))

    def __repr__(self):
        return 'Model(name="{}", stan_file="{}", exe_file="{}")'.format(
            self.name, self.stan_file, self.exe_file)
    
    def code(self):
        """Return Stan program as a string."""
        code = None
        try:
            with open(self.stan_file, 'r') as fd:
                code = fd.read()
        except IOError:
            print('Cannot read file: {}'.format(self.stan_file))
        return code

from utils import rdump

class StanData(object):
    """Stan model data or inits."""
    
    def __init__(self, rdump_file=None):
        """Initialize object."""
        self.rdump_file = rdump_file
        """path to rdump file."""
        if not os.path.exists(rdump_file):
            try:
                open(rdump_file,'w')
            except OSError:
                raise Exception('invalid rdump_file name {}'.format(self.rdump_file))

    def __repr__(self):
        return 'StanData(rdump_file="{}")'.format(self.rdump_file)

    def write_rdump(self, dict):
        rdump(self.rdump_file, dict)


class RunSet(object):
    """Record call to NUTS sampler for Stan model."""
    def __init__(self, num_chains, args):
        """Initialize object."""
        self.num_chains = num_chains
        """number of chains."""
        self.args = args
        """sampler args."""



class Run:
    def __init__(self,
                 id=None,
                 cmd=None,
                 start=True,
                 wait=False):

        """Create a new run of the given model, for a given method.
        """
        self.id = id
        self.cmd = cmd
        self.output_fname = os.path.join(self.tmp_dir.name, 'output.txt')
        self._output_fd = None
        self.proc = None
        self.stdout = None
        if start:
            self.start(wait=wait)

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
