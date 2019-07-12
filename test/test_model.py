import io
import os
import sys
import unittest

from cmdstanpy import TMPDIR
from cmdstanpy.utils import EXTENSION
from cmdstanpy.model import Model
import numpy as np

datafiles_path = os.path.join('test', 'data')

code = '''data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);
  for (n in 1:N)
    y[n] ~ bernoulli(theta);
}
'''


class ModelTest(unittest.TestCase):
    def test_model_good(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)

        model = Model(stan_file=stan)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(None, model.exe_file)

        model = Model(stan_file=stan, exe_file=exe)
        self.assertEqual(exe, model.exe_file)

    def test_model_bad(self):
        with self.assertRaises(Exception):
            model = Model(stan_file='xdlfkjx', exe_file='sdfndjsds')

        stan = os.path.join(datafiles_path, 'b')
        with self.assertRaises(Exception):
            model = Model(stan_file=stan)

    def test_repr(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        model = Model(stan_file=stan)
        s = repr(model)
        self.assertIn('name=bernoulli', s)

    def test_print(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        model = Model(stan_file=stan)
        self.assertEqual(code, model.code())

    def test_model_compile(self):
        stan = os.path.join(datafiles_path, 'bernoulli space.stan')
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        model = Model(stan_file=stan)
        self.assertEqual(None, model.exe_file)
        model.compile()
        self.assertTrue(model.exe_file.endswith(exe))

        model = Model(stan_file=stan)
        if os.path.exists(exe):
            os.remove(exe)
        model.compile()
        self.assertTrue(model.exe_file.endswith(exe))

        stan = os.path.join(datafiles_path, 'bernoulli_include.stan')
        exe = os.path.join(datafiles_path, 'bernoulli_include')
        here = os.path.dirname(os.path.abspath(__file__))
        datafiles_abspath = os.path.join(here, 'data')
        include_paths = [datafiles_abspath]
        if os.path.exists(exe):
            os.remove(exe)
        model = Model(stan_file=stan)
        model.compile(include_paths=include_paths)
        self.assertEqual(stan, model.stan_file)
        self.assertTrue(model.exe_file.endswith(exe))

    # TODO: test compile with existing exe - timestamp on exe unchanged
    # TODO: test overwrite with existing exe - timestamp on exe updated


class OptimizeTest(unittest.TestCase):
    def test_optimize_works(self):
        exe = os.path.join(datafiles_path, 'bernoulli')
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        model = Model(stan_file=stan, exe_file=exe)
        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        jinit = os.path.join(datafiles_path, 'bernoulli.init.json')
        fit = model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm="BFGS",
            init_alpha=0.001,
            iter=100)

        # check if calling sample related stuff fails
        with self.assertRaises(RuntimeError):
            fit.summary()
        with self.assertRaises(RuntimeError):
            _ = fit.sample
        with self.assertRaises(RuntimeError):
            fit.diagnose()

        # test numpy output
        self.assertAlmostEqual(fit.optimized_params_np[0], -5, places=2)
        self.assertAlmostEqual(fit.optimized_params_np[1], 0.2, places=3)

        # test pandas output
        self.assertEqual(
            fit.optimized_params_np[0],
            fit.optimized_params_pd["lp__"][0]
        )
        self.assertEqual(
            fit.optimized_params_np[1],
            fit.optimized_params_pd["theta"][0]
        )

        # test dict output
        self.assertEqual(
            fit.optimized_params_np[0],
            fit.optimized_params_dict["lp__"]
        )
        self.assertEqual(
            fit.optimized_params_np[1],
            fit.optimized_params_dict["theta"]
        )


class SampleTest(unittest.TestCase):
    def test_bernoulli_good(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        bern_model = Model(stan_file=stan, exe_file=exe)
        bern_model.compile()

        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        bern_fit = bern_model.sample(data=jdata,
                                     chains=4,
                                     cores=2,
                                     seed=12345,
                                     sampling_iters=100)

        for i in range(bern_fit.chains):
            csv_file = bern_fit.csv_files[i]
            txt_file = ''.join([os.path.splitext(csv_file)[0], '.txt'])
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(txt_file))

        self.assertEqual(bern_fit.chains, 4)
        self.assertEqual(bern_fit.draws, 100)
        column_names = [
            'lp__',
            'accept_stat__',
            'stepsize__',
            'treedepth__',
            'n_leapfrog__',
            'divergent__',
            'energy__',
            'theta',
        ]
        self.assertEqual(bern_fit.column_names, tuple(column_names))

        bern_sample = bern_fit.sample
        self.assertEqual(bern_sample.shape, (100, 4, len(column_names)))

        self.assertEqual(bern_fit.metric_type, 'diag_e')
        self.assertEqual(bern_fit.stepsize.shape, (4,))
        self.assertEqual(bern_fit.metric.shape, (4, 1))

        output = os.path.join(datafiles_path, 'test1-bernoulli-output')
        bern_fit = bern_model.sample(data=jdata,
                                     chains=4,
                                     cores=2,
                                     seed=12345,
                                     sampling_iters=100,
                                     csv_basename=output)
        for i in range(bern_fit.chains):
            csv_file = bern_fit.csv_files[i]
            txt_file = ''.join([os.path.splitext(csv_file)[0], '.txt'])
            self.assertTrue(os.path.exists(csv_file))
            self.assertTrue(os.path.exists(txt_file))
        bern_sample = bern_fit.sample
        self.assertEqual(bern_sample.shape, (100, 4, len(column_names)))
        for i in range(bern_fit.chains):  # cleanup datafile_path dir
            os.remove(bern_fit.csv_files[i])
            os.remove(bern_fit.console_files[i])

        rdata = os.path.join(datafiles_path, 'bernoulli.data.R')
        bern_fit = bern_model.sample(data=rdata,
                                     chains=4,
                                     cores=2,
                                     seed=12345,
                                     sampling_iters=100)
        bern_sample = bern_fit.sample
        self.assertEqual(bern_sample.shape, (100, 4, len(column_names)))

        data_dict = {'N': 10, 'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
        bern_fit = bern_model.sample(data=data_dict,
                                     chains=4,
                                     cores=2,
                                     seed=12345,
                                     sampling_iters=100)
        bern_sample = bern_fit.sample
        self.assertEqual(bern_sample.shape, (100, 4, len(column_names)))

        # check if  optimized_params_np returns first draw
        # (actually first row from csv)
        np.testing.assert_equal(
            bern_fit.get_drawset().to_numpy()[0],
            bern_fit.optimized_params_np
        )

    def test_bernoulli_bad(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        bern_model = Model(stan_file=stan, exe_file=exe)
        bern_model.compile()

        with self.assertRaisesRegex(Exception, 'Error during sampling'):
            bern_fit = bern_model.sample(chains=4,
                                         cores=2,
                                         seed=12345,
                                         sampling_iters=100)


if __name__ == '__main__':
    unittest.main()
