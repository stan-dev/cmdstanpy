import os
import unittest

from cmdstanpy.cmdstan_args import Method, SamplerArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION
from cmdstanpy.model import Model
from contextlib import contextmanager
import sys

here = os.path.dirname(os.path.abspath(__file__))
datafiles_path = os.path.join(here, 'data')

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

    def test_model_good_no_source(self):
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        model = Model(exe_file=exe)
        self.assertEqual(exe, model.exe_file)
        self.assertEqual('bernoulli', model.name)

        with self.assertRaises(RuntimeError):
            model.code()
        with self.assertRaises(RuntimeError):
            model.compile()

    def test_model_none(self):
        with self.assertRaises(ValueError):
            _ = Model(exe_file=None, stan_file=None)

    def test_model_bad(self):
        with self.assertRaises(Exception):
            model = Model(stan_file='xdlfkjx', exe_file='sdfndjsds')

        stan = os.path.join(datafiles_path, 'b')
        with self.assertRaises(Exception):
            model = Model(stan_file=stan)

        stan = os.path.join(datafiles_path, 'bad_syntax.stan')
        model = Model(stan_file=stan)
        model.compile()

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
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)

        model = Model(stan_file=stan)
        self.assertEqual(None, model.exe_file)
        model.compile()
        self.assertTrue(model.exe_file.endswith(exe.replace('\\', '/')))

        model = Model(stan_file=stan)
        if os.path.exists(exe):
            os.remove(exe)
        model.compile()
        self.assertTrue(model.exe_file.endswith(exe.replace('\\', '/')))

        # test compile with existing exe - timestamp on exe unchanged
        exe_time = os.path.getmtime(model.exe_file)
        model2 = Model(stan_file=stan)
        model2.compile()
        self.assertEqual(exe_time,os.path.getmtime(model2.exe_file))


        # test overwrite with existing exe - timestamp on exe updated
        self.assertTrue(os.path.exists(exe))
        exe_time = os.path.getmtime(exe)
        model3 = Model(stan_file=stan, exe_file=exe)
        model3.compile(overwrite=True)
        self.assertNotEqual(exe_time,os.path.getmtime(model3.exe_file))

        stan = os.path.join(datafiles_path, 'bernoulli_include.stan')
        exe = os.path.join(datafiles_path, 'bernoulli_include' + EXTENSION)
        here = os.path.dirname(os.path.abspath(__file__))
        datafiles_abspath = os.path.join(here, 'data')
        include_paths = [datafiles_abspath]
        if os.path.exists(exe):
            os.remove(exe)
        model = Model(stan_file=stan)
        model.compile(include_paths=include_paths)
        self.assertEqual(stan, model.stan_file)
        self.assertTrue(model.exe_file.endswith(exe.replace('\\', '/')))


if __name__ == '__main__':
    unittest.main()
