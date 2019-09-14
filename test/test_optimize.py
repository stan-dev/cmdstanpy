import os
import unittest

from cmdstanpy.cmdstan_args import Method, SamplerArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION
from cmdstanpy.model import Model
from cmdstanpy.stanfit import RunSet
from contextlib import contextmanager
import logging
from multiprocessing import cpu_count
import numpy as np
import sys
from testfixtures import LogCapture

here = os.path.dirname(os.path.abspath(__file__))
datafiles_path = os.path.join(here, 'data')

# class OptimizeTest(unittest.TestCase):
#     def test_optimize_works(self):
#         exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
#         stan = os.path.join(datafiles_path, 'bernoulli.stan')
#         model = Model(stan_file=stan, exe_file=exe)
#         jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
#         jinit = os.path.join(datafiles_path, 'bernoulli.init.json')
#         fit = model.optimize(
#             data=jdata,
#             seed=1239812093,
#             inits=jinit,
#             algorithm='BFGS',
#             init_alpha=0.001,
#             iter=100,
#         )
# 
#         # check if calling sample related stuff fails
#         with self.assertRaises(RuntimeError):
#             fit.summary()
#         with self.assertRaises(RuntimeError):
#             _ = fit.sample
#         with self.assertRaises(RuntimeError):
#             fit.diagnose()
# 
#         # test numpy output
#         self.assertAlmostEqual(fit.optimized_params_np[0], -5, places=2)
#         self.assertAlmostEqual(fit.optimized_params_np[1], 0.2, places=3)
# 
#         # test pandas output
#         self.assertEqual(
#             fit.optimized_params_np[0], fit.optimized_params_pd['lp__'][0]
#         )
#         self.assertEqual(
#             fit.optimized_params_np[1], fit.optimized_params_pd['theta'][0]
#         )
# 
#         # test dict output
#         self.assertEqual(
#             fit.optimized_params_np[0], fit.optimized_params_dict['lp__']
#         )
#         self.assertEqual(
#             fit.optimized_params_np[1], fit.optimized_params_dict['theta']
#         )
# 
#     def test_optimize_works_dict(self):
#         import json
# 
#         exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
#         stan = os.path.join(datafiles_path, 'bernoulli.stan')
#         model = Model(stan_file=stan, exe_file=exe)
#         with open(os.path.join(datafiles_path, 'bernoulli.data.json')) as d:
#             data = json.load(d)
#         with open(os.path.join(datafiles_path, 'bernoulli.init.json')) as d:
#             init = json.load(d)
#         fit = model.optimize(
#             data=data,
#             seed=1239812093,
#             inits=init,
#             algorithm='BFGS',
#             init_alpha=0.001,
#             iter=100,
#         )
# 
#         # test numpy output
#         self.assertAlmostEqual(fit.optimized_params_np[0], -5, places=2)
#         self.assertAlmostEqual(fit.optimized_params_np[1], 0.2, places=3)

if __name__ == '__main__':
    unittest.main()
