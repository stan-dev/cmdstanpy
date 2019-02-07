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
        rdump_file = os.path.join(examples_path, "bernoulli.data2.R")
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
        model = Model(stan, "bern", exe)
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
            model = Model("nosuchmodel","nosuchmodel.stan","nosuchmodel.exe")


class SamplerArgsTest(BaseTestCase):
    def test_samplerargs_min(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(examples_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output)
        args.validate()
        cmd = args.compose_command('*')
        self.assertIn("id=*", cmd)
        self.assertIn("bernoulli.output-*.csv", cmd)

    def test_samplerargs_good(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        rdata = os.path.join(examples_path, "bernoulli.data.R")
        output = os.path.join(examples_path, "bernoulli.output")
        args = SamplerArgs(model, seed=12345, data_file=rdata, output_file=output,
                               nuts_max_depth=15, adapt_delta=0.99)
        cmd = args.compose_command('*')
        self.assertIn("random seed=12345", cmd)
        self.assertIn("data file=", cmd)
        self.assertIn("algorithm=hmc engine=nuts max_depth=15 adapt delta=0.99", cmd)

    def test_samplerargs_num_draws(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(examples_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, post_warmup_draws=3, warmup_draws=7)
        cmd = args.compose_command('*')
        self.assertIn("num_samples=3", cmd)
        self.assertIn("num_warmup=7", cmd)

    def test_samplerargs_thin(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(examples_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, thin=3)
        cmd = args.compose_command('*')
        self.assertIn("thin=3", cmd)


    def test_samplerargs_missing_args1(self):
        with self.assertRaises(Exception):
            args = SamplerArgs()

    def test_samplerargs_missing_args2(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        args = SamplerArgs(model)
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_model1(self):
        stan = os.path.join(examples_path, "bbad.stan")
        model = Model(stan_file=stan, name="bbad")
        args = SamplerArgs(model, )
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_model2(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "no_such_exe")
        model = Model(stan_file=stan, name="bernoulli")
        args = SamplerArgs(model, )
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_output(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        args = SamplerArgs(model, output_file="/no/such/path/to.file")
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_seed1(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(examples_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, seed="badseed")
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_seed2(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(examples_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, seed=-10)
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_data(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(examples_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, data_file="/no/such/path/to.file")
        with self.assertRaises(ValueError):
            args.validate()

    def test_samplerargs_bad_init_params(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(examples_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, init_param_values="/no/such/path/to.file")
        with self.assertRaises(ValueError):
            args.validate()
            
    def test_samplerargs_bad_metric_file(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        output = os.path.join(examples_path, "bernoulli.output")
        args = SamplerArgs(model, output_file=output, hmc_metric_file="/no/such/path/to.file")
        with self.assertRaises(ValueError):
            args.validate()
            

class RunSetTest(BaseTestCase):
    def test_runset(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        exe = os.path.join(examples_path, "bernoulli")
        model = Model(exe_file=exe, stan_file=stan, name="bern")
        rdata = os.path.join(examples_path, "bernoulli.data.R")
        output = os.path.join(examples_path, "bernoulli.output")
        args = SamplerArgs(model, seed=12345, data_file=rdata, output_file=output,
                               nuts_max_depth=15, adapt_delta=0.99)
        transcript = os.path.join(examples_path, "bernoulli.run")
        runset = RunSet(chains=4, cores=2, transcript_file=transcript, args=args)
        self.assertEqual(-1, runset.get_retcode(0))
        runset.set_retcode(0,0)
        self.assertEqual(0, runset.get_retcode(0))



class SampleTest(BaseTestCase):
    def test_sample_1_good(self):
        rdata = os.path.join(examples_path, "bernoulli.data.R")
        stan = os.path.join(examples_path, "bernoulli.stan")
        output = os.path.join(examples_path, "bernoulli.output")
        model = compile_model(stan)
        runset = sample(model, data_file=rdata, csv_output_file=output)
        for i in range(runset.chains):
            self.assertEqual(0, runset.get_retcode(i))

    def test_sample_2_missing_input(self):
        stan = os.path.join(examples_path, "bernoulli.stan")
        output = os.path.join(examples_path, "bernoulli.output")
        model = compile_model(stan)
        runset = sample(model, csv_output_file=output)
        for i in range(runset.chains):
            self.assertEqual(70, runset.get_retcode(i))


if __name__ == '__main__':
    unittest.main()
