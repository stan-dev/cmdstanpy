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


class GenerateQuantitiesTest(unittest.TestCase):
    def test_gen_quantities_good(self):
        stan = os.path.join(datafiles_path, 'bernoulli_ppc.stan')
        model = Model(stan_file=stan)
        model.compile()

        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')

        # synthesize runset for existing sample
        goodfiles_path = os.path.join(datafiles_path, 'runset-good')
        output = os.path.join(goodfiles_path, 'bern')
        sampler_args = SamplerArgs(
            sampling_iters=100, max_treedepth=11, adapt_delta=0.95
        )
        cmdstan_args = CmdStanArgs(
            model_name=model.name,
            model_exe=model.exe_file,
            chain_ids=[1, 2, 3, 4],
            seed=12345,
            data=jdata,
            output_basename=output,
            method_args=sampler_args,
        )
        sampler_runset = RunSet(args=cmdstan_args, chains=4)
        for i in range(4):
            sampler_runset._set_retcode(i, 0)

        bern_gqs = model.run_generated_quantities(
            csv_files=sampler_runset.csv_files, data=jdata
        )
        self.assertEqual(
            bern_gqs.runset._args.method, Method.GENERATE_QUANTITIES
        )

        # check results - ouput files, quantities of interest, draws
        self.assertEqual(bern_gqs.runset.chains, 4)
        for i in range(bern_gqs.runset.chains):
            self.assertEqual(bern_gqs.runset._retcode(i), 0)
            csv_file = bern_gqs.runset.csv_files[i]
            self.assertTrue(os.path.exists(csv_file))
        column_names = [
            'y_rep.1',
            'y_rep.2',
            'y_rep.3',
            'y_rep.4',
            'y_rep.5',
            'y_rep.6',
            'y_rep.7',
            'y_rep.8',
            'y_rep.9',
            'y_rep.10',
        ]
        self.assertEqual(bern_gqs.column_names, tuple(column_names))


if __name__ == '__main__':
    unittest.main()
