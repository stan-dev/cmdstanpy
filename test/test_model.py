import os
import os.path
import unittest
from cmdstanpy.lib import Model
from cmdstanpy.cmds import compile_model

datafiles_path = os.path.expanduser(
    os.path.join("~", "github", "stan-dev", "cmdstanpy", "test", "files-data"))

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
    def test_model1(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        if not os.path.exists(exe):
            compile_model(stan)
        model = Model(stan, name="bernoulli", exe_file=exe)
        print('model: {}'.format(model))
        self.assertEqual("bernoulli", model.name)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(exe, model.exe_file)

    def test_model2(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        self.assertEqual("bern", model.name)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(exe, model.exe_file)

    def test_repr(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        m2 = eval(repr(model))
        self.assertEqual("bern", m2.name)
        self.assertEqual(stan, m2.stan_file)
        self.assertEqual(exe, m2.exe_file)

    def test_print(self):
        stan = os.path.join(datafiles_path, "bernoulli.stan")
        exe = os.path.join(datafiles_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        self.assertEqual(code, model.code())

    def test_error(self):
        with self.assertRaises(Exception):
            model = Model("xdlfkjx", "sdfndjsds", "sdslkjz")
            model   # silence lint checker


if __name__ == '__main__':
    unittest.main()
