import os
import os.path
import unittest
from cmdstanpy import config

class test1(unittest.TestCase):

    def test_path(self):
        print('compile_model, path: {}'.format(config.CMDSTAN_PATH))
        pass
    
if __name__ == '__main__':
    unittest.main()
