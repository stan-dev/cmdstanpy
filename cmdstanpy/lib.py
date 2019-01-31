import json
import os
import os.path

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


class Model(object):
    """Stan model."""

    def __init__(self, name=None, stan_file=None, exe_file=None):
        """Initialize object, all attributes required."""
        self.name = name
        """defaults to base name of Stan program file."""
        self.stan_file = stan_file
        """full path to Stan program src."""
        self.exe_file = exe_file
        """full path to compiled c++ executible."""
        if name is None:
            raise Exception("attribute name unspecified")
        if stan_file is None:
            raise Exception("attribute stan_file unspecified")
        if not os.path.exists(stan_file):
            raise Exception('no such stan_file {}'.format(self.stan_file))

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
