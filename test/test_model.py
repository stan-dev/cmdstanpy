"""CmdStanModel tests"""

import os
import shutil
import unittest
from unittest.mock import Mock
import pytest
from testfixtures import LogCapture
import numpy as np
import tqdm

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

BERN_STAN = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
BERN_EXE = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)


class CmdStanModelTest(unittest.TestCase):

    # pylint: disable=no-self-use
    @pytest.fixture(scope='class', autouse=True)
    def do_clean_up(self):
        for root, _, files in os.walk(DATAFILES_PATH):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext.lower() in ('.o', '.d', '.hpp', '.exe', ''):
                    filepath = os.path.join(root, filename)
                    os.remove(filepath)

    def show_cmdstan_version(self):
        print('\n\nCmdStan version: {}\n\n'.format(cmdstan_path()))
        self.assertTrue(True)

    def test_model_good(self):
        # compile on instantiation
        model = CmdStanModel(model_name='bern', stan_file=BERN_STAN)
        self.assertEqual(BERN_STAN, model.stan_file)
        self.assertTrue(model.exe_file.endswith(BERN_EXE.replace('\\', '/')))
        self.assertEqual('bern', model.name)

        # instantiate with existing exe
        model = CmdStanModel(stan_file=BERN_STAN, exe_file=BERN_EXE)
        self.assertEqual(BERN_STAN, model.stan_file)
        self.assertTrue(model.exe_file.endswith(BERN_EXE))
        self.assertEqual('bernoulli', model.name)

        # instantiate with existing exe only - no model
        model2 = CmdStanModel(exe_file=BERN_EXE)
        self.assertEqual(BERN_EXE, model2.exe_file)
        self.assertEqual('bernoulli', model2.name)
        with self.assertRaises(RuntimeError):
            model2.code()
        with self.assertRaises(RuntimeError):
            model2.compile()

        # instantiate, don't compile
        os.remove(BERN_EXE)
        model = CmdStanModel(stan_file=BERN_STAN, compile=False)
        self.assertEqual(BERN_STAN, model.stan_file)
        self.assertEqual(None, model.exe_file)

    def test_model_bad(self):
        with self.assertRaises(ValueError):
            CmdStanModel(stan_file=None, exe_file=None)
        with self.assertRaises(ValueError):
            CmdStanModel(model_name='bad')

    def test_stanc_options(self):
        opts = {
            'O': True,
            'allow_undefined': True,
            'use-opencl': True,
            'name': 'foo',
        }
        model = CmdStanModel(
            stan_file=BERN_STAN, compile=False, stanc_options=opts
        )
        stanc_opts = model.stanc_options
        self.assertTrue(stanc_opts['O'])
        self.assertTrue(stanc_opts['allow_undefined'])
        self.assertTrue(stanc_opts['use-opencl'])
        self.assertTrue(stanc_opts['name'] == 'foo')

        cpp_opts = model.cpp_options
        self.assertEqual(cpp_opts['STAN_OPENCL'], 'TRUE')

        with self.assertRaises(ValueError):
            bad_opts = {'X': True}
            model = CmdStanModel(
                stan_file=BERN_STAN, compile=False, stanc_options=bad_opts
            )
        with self.assertRaises(ValueError):
            bad_opts = {'include_paths': True}
            model = CmdStanModel(
                stan_file=BERN_STAN, compile=False, stanc_options=bad_opts
            )
        with self.assertRaises(ValueError):
            bad_opts = {'include_paths': 'lkjdf'}
            model = CmdStanModel(
                stan_file=BERN_STAN, compile=False, stanc_options=bad_opts
            )

    def test_cpp_options(self):
        opts = {
            'STAN_OPENCL': 'TRUE',
            'STAN_MPI': 'TRUE',
            'STAN_THREADS': 'TRUE',
        }
        model = CmdStanModel(
            stan_file=BERN_STAN, compile=False, cpp_options=opts
        )
        cpp_opts = model.cpp_options
        self.assertEqual(cpp_opts['STAN_OPENCL'], 'TRUE')
        self.assertEqual(cpp_opts['STAN_MPI'], 'TRUE')
        self.assertEqual(cpp_opts['STAN_THREADS'], 'TRUE')

    def test_model_paths(self):
        # pylint: disable=unused-variable
        model = CmdStanModel(stan_file=BERN_STAN)  # instantiates exe
        self.assertTrue(os.path.exists(BERN_EXE))

        dotdot_stan = os.path.realpath(os.path.join('..', 'bernoulli.stan'))
        dotdot_exe = os.path.realpath(
            os.path.join('..', 'bernoulli' + EXTENSION)
        )
        shutil.copyfile(BERN_STAN, dotdot_stan)
        shutil.copyfile(BERN_EXE, dotdot_exe)
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
        shutil.copyfile(BERN_STAN, tilde_stan)
        shutil.copyfile(BERN_EXE, tilde_exe)
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
        with self.assertRaises(ValueError):
            CmdStanModel(stan_file='xdlfkjx', exe_file='sdfndjsds')

        stan = os.path.join(DATAFILES_PATH, 'b')
        with self.assertRaises(ValueError):
            CmdStanModel(stan_file=stan)

    def test_model_syntax_error(self):
        stan = os.path.join(DATAFILES_PATH, 'bad_syntax.stan')
        with LogCapture() as log:
            with self.assertRaises(Exception):
                CmdStanModel(stan_file=stan)

            # Join all the log messages into one string
            error_message = '@( * O * )@'.join(np.array(log.actual())[:, -1])

            # Ensure the new line character in error message is not escaped
            # so the error message is readable
            self.assertRegex(error_message, r'parsing error:(\r\n|\r|\n)')

    def test_repr(self):
        model = CmdStanModel(stan_file=BERN_STAN)
        model_repr = model.__repr__()
        self.assertIn('name=bernoulli', model_repr)

    def test_print(self):
        model = CmdStanModel(stan_file=BERN_STAN)
        self.assertEqual(CODE, model.code())

    def test_model_compile(self):
        model = CmdStanModel(stan_file=BERN_STAN)
        self.assertTrue(model.exe_file.endswith(BERN_EXE.replace('\\', '/')))

        model = CmdStanModel(stan_file=BERN_STAN)
        self.assertTrue(model.exe_file.endswith(BERN_EXE.replace('\\', '/')))
        old_exe_time = os.path.getmtime(model.exe_file)
        os.remove(BERN_EXE)
        model.compile()
        new_exe_time = os.path.getmtime(model.exe_file)
        self.assertTrue(new_exe_time > old_exe_time)

        # test compile with existing exe - timestamp on exe unchanged
        exe_time = os.path.getmtime(model.exe_file)
        model2 = CmdStanModel(stan_file=BERN_STAN)
        self.assertEqual(exe_time, os.path.getmtime(model2.exe_file))

    def test_model_includes_explicit(self):
        if os.path.exists(BERN_EXE):
            os.remove(BERN_EXE)
        model = CmdStanModel(
            stan_file=BERN_STAN, stanc_options={'include_paths': DATAFILES_PATH}
        )
        self.assertEqual(BERN_STAN, model.stan_file)
        self.assertTrue(model.exe_file.endswith(BERN_EXE.replace('\\', '/')))

    def test_model_includes_implicit(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_include.stan')
        exe = os.path.join(DATAFILES_PATH, 'bernoulli_include' + EXTENSION)
        if os.path.exists(exe):
            os.remove(exe)
        model2 = CmdStanModel(stan_file=stan)
        self.assertTrue(model2.exe_file.endswith(exe.replace('\\', '/')))

    def test_read_progress(self):
        model = CmdStanModel(stan_file=BERN_STAN, compile=False)

        proc_mock = Mock()
        proc_mock.poll.side_effect = [None, None, 'finish']
        stan_output1 = 'Iteration: 12100 / 31000 [ 39%]  (Warmup)'
        stan_output2 = 'Iteration: 14000 / 31000 [ 45%]  (Warmup)'
        pbar = tqdm.tqdm(desc='Chain 1 - warmup', position=1, total=1)

        proc_mock.stdout.readline.side_effect = [
            stan_output1.encode('utf-8'),
            stan_output2.encode('utf-8'),
        ]

        with LogCapture() as log:
            result = model._read_progress(proc=proc_mock, pbar=pbar, idx=0)
            self.assertEqual([], log.actual())
            self.assertEqual(31000, pbar.total)

            # Expect progress bar output to be something like this:
            # 'Chain 1 -   done:  45%|████▌     | 14000/31000'
            # --------

            self.assertIn('Chain 1 -   done:  45%', str(pbar))
            self.assertIn('14000/31000', str(pbar))

            # Check Stan's output is returned
            output = result.decode('utf-8')
            self.assertIn(stan_output1, output)
            self.assertIn(stan_output2, output)


if __name__ == '__main__':
    unittest.main()
