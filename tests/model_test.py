import os
import os.path
import unittest
from cmdstanpy.lib import Model

examples_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                "cmdstanpy", "dev", "cmdstan", "examples", "bernoulli"))
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

class BaseTestCase(unittest.TestCase):

    def setUp(self):
        exe = os.path.join(examples_path, "bernoulli")
        # bernoulli.stan must be compiled in order for tests to pass - do this once
        if not os.path.exists(exe):
            pass

    def tearDown(self):
        pass

class ModelTest(BaseTestCase):

    def test_model1(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model("bern",stan, exe)
        self.assertEqual("bern", model.name)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(exe, model.exe_file)

    def test_model2(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        self.assertEqual("bern", model.name)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(exe, model.exe_file)

    def test_repr(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        m2 = eval(repr(model))
        self.assertEqual("bern", model.name)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(exe, model.exe_file)

    def test_print(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        self.assertEqual(code, model.code())

    def test_error(self):
        with self.assertRaises(Exception):
            model = Model("foo","bar","baz")

if __name__ == '__main__':
    unittest.main()
