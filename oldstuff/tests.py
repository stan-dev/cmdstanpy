import os
import os.path
import unittest
from lib import Conf, Model, StanData
from cmds import *


myconf = Conf()
cmdstan_path = myconf['cmdstan']


if __name__ == '__main__':
    unittest.main()
