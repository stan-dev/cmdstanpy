import os
import os.path
import unittest
from lib import Conf, Model, StanData
from cmds import *

examples_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                "cmdstanpy", "dev", "cmdstan", "examples", "bernoulli"))

myconf = Conf()
cmdstan_path = myconf['cmdstan']

class BaseTestCase(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

class ConfTest(BaseTestCase):

    def test_get(self):
        testconf = Conf()
        self.assertEqual(myconf['cmdstan'],
                             os.path.expanduser("~/github/stan-dev/cmdstanpy/dev/cmdstan"))
        self.assertEqual(myconf['foo'], None)
        
    def test_repr(self):
        testconf = Conf()

    
rdump = ('''y <- c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)
N <- 10
''')
class StanDataTest(BaseTestCase):

    def test_standata_existing(self):
        rdump = os.path.join(examples_path, "bernoulli.data.R")
        standata = StanData(rdump)

    def test_standata_new(self):
        json_file = os.path.join(examples_path, "bernoulli.data.json")
        dict = json.load(open(json_file))
        rdump_file = os.path.join(examples_path, "bernoulli.rdump")
        standata = StanData(rdump_file)
        standata.write_rdump(dict)
        with open(rdump_file, 'r') as myfile:
            new_data=myfile.read()
        self.assertEqual(rdump, new_data)

    def test_standata_bad(self):
        with self.assertRaises(Exception):
            standata = StanData("/no/such/path")


class CompileTest(BaseTestCase):

    def test_compile_good(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        model = compile_model(stan)
        print(model)
        exe = os.path.join(examples_path, "bernoulli")
        self.assertEqual("bernoulli", model.name)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(exe, model.exe_file)

    def test_compile_bad(self):
        stan = os.path.join(examples_path, "bbad.stan")
        with self.assertRaises(Exception):
            model = compile_model(stan)

    # test overwrite logic


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

class SampleTest(BaseTestCase):
    def test_sample_1(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        model = compile_model(stan)
        sample(model)

    def test_sample_2(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        model = compile_model(stan)
        sample(model, adapt_delta=0.95)

    def test_sample_3(self):
        rdump_file = os.path.join(examples_path, "bernoulli.rdump")
        standata = StanData(rdump_file)
        stan = os.path.join(examples_path, "bernoulli.stan")
        model = compile_model(stan)
        sample(model, data_file=rdump_file, adapt_delta=0.95)

    def test_sample_4(self):
        rdump_file = os.path.join(examples_path, "bernoulli.rdump")
        standata = StanData(rdump_file)
        stan = os.path.join(examples_path, "bernoulli.stan")
        model = compile_model(stan)
        sample(model, data_file=rdump_file, fixed_param=True)


if __name__ == '__main__':
    unittest.main()
