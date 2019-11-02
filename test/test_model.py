import os
import unittest

from cmdstanpy.cmdstan_args import Method, SamplerArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION
from cmdstanpy.model import CmdStanModel
from contextlib import contextmanager

from cmdstanpy.utils import cmdstan_path

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


class CmdStanModelTest(unittest.TestCase):
    def show_cmdstan_version(self):
        print('\n\nCmdStan version: {}\n\n'.format(cmdstan_path()))
        self.assertTrue(True)

    def test_model_good(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)

        # compile on instantiation
        model = CmdStanModel(stan_file=stan)
        self.assertEqual(stan, model.stan_file)
        self.assertTrue(model.exe_file.endswith(exe.replace('\\', '/')))


        # instantiate with existing exe
        model = CmdStanModel(stan_file=stan, exe_file=exe)
        self.assertEqual(stan, model.stan_file)
        self.assertTrue(model.exe_file.endswith(exe.replace('\\', '/')))
        
        # instantiate, don't compile
        os.remove(exe)
        model = CmdStanModel(stan_file=stan, compile=False)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(None, model.exe_file)

    def test_model_good_no_source(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        model = CmdStanModel(stan_file=stan)

        model2 = CmdStanModel(exe_file=exe)
        self.assertEqual(exe, model2.exe_file)
        self.assertEqual('bernoulli', model2.name)

        with self.assertRaises(RuntimeError):
            model2.code()
        with self.assertRaises(RuntimeError):
            model2.compile()

    def test_model_none(self):
        with self.assertRaises(ValueError):
            _ = CmdStanModel(exe_file=None, stan_file=None)

    def test_model_bad(self):
        with self.assertRaises(Exception):
            model = CmdStanModel(stan_file='xdlfkjx', exe_file='sdfndjsds')

        stan = os.path.join(datafiles_path, 'b')
        with self.assertRaises(Exception):
            model = CmdStanModel(stan_file=stan)

        stan = os.path.join(datafiles_path, 'bad_syntax.stan')
        with self.assertRaises(Exception):
            model = CmdStanModel(stan_file=stan)


    def test_repr(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        model = CmdStanModel(stan_file=stan)
        s = repr(model)
        self.assertIn('name=bernoulli', s)

    def test_print(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        model = CmdStanModel(stan_file=stan)
        self.assertEqual(code, model.code())

    def test_model_compile(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)

        model = CmdStanModel(stan_file=stan)
        self.assertTrue(model.exe_file.endswith(exe.replace('\\', '/')))

        model = CmdStanModel(stan_file=stan)
        self.assertTrue(model.exe_file.endswith(exe.replace('\\', '/')))
        old_exe_time = os.path.getmtime(model.exe_file)
        os.remove(exe)
        model.compile()
        new_exe_time = os.path.getmtime(model.exe_file)
        self.assertTrue(new_exe_time > old_exe_time)

        # test compile with existing exe - timestamp on exe unchanged
        exe_time = os.path.getmtime(model.exe_file)
        model2 = CmdStanModel(stan_file=stan)
        self.assertEqual(exe_time,os.path.getmtime(model2.exe_file))

    def test_model_includes(self):
        stan = os.path.join(datafiles_path, 'bernoulli_include.stan')
        exe = os.path.join(datafiles_path, 'bernoulli_include' + EXTENSION)
        here = os.path.dirname(os.path.abspath(__file__))
        datafiles_abspath = os.path.join(here, 'data')
        include_paths = [datafiles_abspath]
        if os.path.exists(exe):
            os.remove(exe)
        model = CmdStanModel(stan_file=stan, include_paths=include_paths)
        self.assertEqual(stan, model.stan_file)
        self.assertTrue(model.exe_file.endswith(exe.replace('\\', '/')))

        os.remove(os.path.join(datafiles_path, 'bernoulli_include' + '.hpp'))
        os.remove(exe)
        model2 = CmdStanModel(stan_file=stan)
        self.assertEqual(model2.include_paths, include_paths)

        model3 = CmdStanModel(stan_file=stan)
        self.assertTrue(model3.exe_file.endswith(exe.replace('\\', '/')))



if __name__ == '__main__':
    unittest.main()
