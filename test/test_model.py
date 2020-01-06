"""CmdStanModel tests"""

import os
import shutil
import unittest
import pytest
from testfixtures import LogCapture
import numpy as np

from cmdstanpy.utils import EXTENSION
from cmdstanpy.model import CmdStanModel

from cmdstanpy.utils import cmdstan_path

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')

CODE = """data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);  // uniform prior on interval 0,1
  y ~ bernoulli(theta);
}
"""


class CmdStanModelTest(unittest.TestCase):

    # pylint: disable=no-self-use
    @pytest.fixture(scope='class', autouse=True)
    def do_clean_up(self):
        for root, _, files in os.walk(DATAFILES_PATH):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext.lower() in ('.o', '.hpp', '.exe', ''):
                    filepath = os.path.join(root, filename)
                    os.remove(filepath)

    def show_cmdstan_version(self):
        print('\n\nCmdStan version: {}\n\n'.format(cmdstan_path()))
        self.assertTrue(True)

    def test_model_good(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)

        # compile on instantiation
        model = CmdStanModel(stan_file=stan)
        self.assertEqual(stan, model.stan_file)
        self.assertTrue(model.exe_file.endswith(exe.replace('\\', '/')))

        # instantiate with existing exe
        model = CmdStanModel(stan_file=stan, exe_file=exe)
        self.assertEqual(stan, model.stan_file)
        self.assertTrue(model.exe_file.endswith(exe))

        # instantiate with existing exe only - no model
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        model2 = CmdStanModel(exe_file=exe)
        self.assertEqual(exe, model2.exe_file)
        self.assertEqual('bernoulli', model2.name)
        with self.assertRaises(RuntimeError):
            model2.code()
        with self.assertRaises(RuntimeError):
            model2.compile()

        # instantiate, don't compile
        os.remove(exe)
        model = CmdStanModel(stan_file=stan, compile=False)
        self.assertEqual(stan, model.stan_file)
        self.assertEqual(None, model.exe_file)

    def test_model_paths(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
        # pylint: disable=unused-variable
        model = CmdStanModel(stan_file=stan)  # instantiates exe
        self.assertTrue(os.path.exists(exe))

        dotdot_stan = os.path.realpath(os.path.join('..', 'bernoulli.stan'))
        dotdot_exe = os.path.realpath(
            os.path.join('..', 'bernoulli' + EXTENSION)
        )
        shutil.copyfile(stan, dotdot_stan)
        shutil.copyfile(exe, dotdot_exe)
        model1 = CmdStanModel(
            stan_file=os.path.join('..', 'bernoulli.stan'),
            exe_file=os.path.join('..', 'bernoulli' + EXTENSION),
        )
        self.assertEqual(model1.stan_file, dotdot_stan)
        self.assertEqual(model1.exe_file, dotdot_exe)
        os.remove(dotdot_stan)
        os.remove(dotdot_exe)

        tilde_stan = os.path.realpath(
            os.path.join(os.path.expanduser('~'), 'bernoulli.stan')
        )
        tilde_exe = os.path.realpath(
            os.path.join(os.path.expanduser('~'), 'bernoulli' + EXTENSION)
        )
        shutil.copyfile(stan, tilde_stan)
        shutil.copyfile(exe, tilde_exe)
        model2 = CmdStanModel(
            stan_file=os.path.join('~', 'bernoulli.stan'),
            exe_file=os.path.join('~', 'bernoulli' + EXTENSION),
        )
        self.assertEqual(model2.stan_file, tilde_stan)
        self.assertEqual(model2.exe_file, tilde_exe)
        os.remove(tilde_stan)
        os.remove(tilde_exe)

    def test_model_none(self):
        with self.assertRaises(ValueError):
            _ = CmdStanModel(exe_file=None, stan_file=None)

    def test_model_file_does_not_exist(self):
        with self.assertRaises(Exception):
            CmdStanModel(stan_file='xdlfkjx', exe_file='sdfndjsds')

        stan = os.path.join(DATAFILES_PATH, 'b')
        with self.assertRaises(Exception):
            CmdStanModel(stan_file=stan)

    def test_model_syntax_error(self):
        stan = os.path.join(DATAFILES_PATH, 'bad_syntax.stan')

        with LogCapture() as log:
            with self.assertRaises(Exception):
                CmdStanModel(stan_file=stan)

            # Join all the log messages into one string
            error_message = "@( * O * )@".join(np.array(log.actual())[:, -1])

            # Ensure the new line character in error message is not escaped
            # so the error message is readable
            self.assertRegex(error_message, r'PARSER EXPECTED: ";"\n')

    def test_repr(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        model = CmdStanModel(stan_file=stan)
        model_repr = repr(model)
        self.assertIn('name=bernoulli', model_repr)

    def test_print(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        model = CmdStanModel(stan_file=stan)
        self.assertEqual(CODE, model.code())

    def test_model_compile(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)

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
        self.assertEqual(exe_time, os.path.getmtime(model2.exe_file))

    def test_model_compile_includes(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_include.stan')
        exe = os.path.join(DATAFILES_PATH, 'bernoulli_include' + EXTENSION)
        if os.path.exists(exe):
            os.remove(exe)

        datafiles_abspath = os.path.join(HERE, 'data')
        include_paths = [datafiles_abspath]

        # test compile with explicit include paths
        model = CmdStanModel(stan_file=stan, include_paths=include_paths)
        self.assertEqual(stan, model.stan_file)
        self.assertTrue(model.exe_file.endswith(exe.replace('\\', '/')))

        # test compile - implicit include path is current dir
        os.remove(os.path.join(DATAFILES_PATH, 'bernoulli_include' + '.hpp'))
        os.remove(os.path.join(DATAFILES_PATH, 'bernoulli_include' + '.o'))
        os.remove(exe)
        model2 = CmdStanModel(stan_file=stan)
        self.assertEqual(model2.include_paths, include_paths)

        # already compiled
        model3 = CmdStanModel(stan_file=stan)
        self.assertTrue(model3.exe_file.endswith(exe.replace('\\', '/')))


if __name__ == '__main__':
    unittest.main()
