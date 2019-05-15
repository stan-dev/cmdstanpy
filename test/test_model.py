import os
import os.path
import unittest
from cmdstanpy.lib import Model
from cmdstanpy.cmds import compile_model

datafiles_path = os.path.join('test', 'data')

code = ('''data {
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
''')


class ModelTest(unittest.TestCase):
    def test_model_1arg(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        model = Model(stan)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(None, model.exe_file)

    def test_model_2arg(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, exe_file=exe)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(exe, model.exe_file)

    def test_repr(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        s = repr(model)
        self.assertTrue(stan in s)
        self.assertTrue(exe in s)

    def test_print(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli')
        model = Model(exe_file=exe, stan_file=stan)
        self.assertEqual(code, model.code())

    def test_error_1(self):
        with self.assertRaises(Exception):
            model = Model('xdlfkjx', 'sdfndjsds')

    def test_error_2(self):
        stan = os.path.join(datafiles_path, 'b')
        with self.assertRaises(Exception):
            model = Model(stan)


if __name__ == '__main__':
    unittest.main()
