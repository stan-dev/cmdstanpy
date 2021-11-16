"""CmdStanModel tests"""

import logging
import os
import shutil
import tempfile
import unittest
from test import CustomTestCase

from testfixtures import LogCapture, StringComparison

from cmdstanpy.model import CmdStanModel
from cmdstanpy.utils import EXTENSION

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
BERN_DATA = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
BERN_EXE = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
BERN_BASENAME = 'bernoulli'


class CmdStanModelTest(CustomTestCase):
    def test_model_good(self):
        # compile on instantiation, override model name
        model = CmdStanModel(model_name='bern', stan_file=BERN_STAN)
        self.assertEqual(BERN_STAN, model.stan_file)
        self.assertPathsEqual(model.exe_file, BERN_EXE)
        self.assertEqual('bern', model.name)

        # compile with external header
        model = CmdStanModel(
            stan_file=os.path.join(DATAFILES_PATH, "external.stan"),
            user_header=os.path.join(DATAFILES_PATH, 'return_one.hpp'),
        )

        # default model name
        model = CmdStanModel(stan_file=BERN_STAN)
        self.assertEqual(BERN_BASENAME, model.name)

        # instantiate with existing exe
        model = CmdStanModel(stan_file=BERN_STAN, exe_file=BERN_EXE)
        self.assertEqual(BERN_STAN, model.stan_file)
        self.assertPathsEqual(model.exe_file, BERN_EXE)

    def test_ctor_compile_arg(self):
        # instantiate, don't compile
        if os.path.exists(BERN_EXE):
            os.remove(BERN_EXE)
        model = CmdStanModel(stan_file=BERN_STAN, compile=False)
        self.assertEqual(BERN_STAN, model.stan_file)
        self.assertEqual(None, model.exe_file)

        model = CmdStanModel(stan_file=BERN_STAN, compile=True)
        self.assertPathsEqual(model.exe_file, BERN_EXE)
        exe_time = os.path.getmtime(model.exe_file)

        model = CmdStanModel(stan_file=BERN_STAN)
        self.assertTrue(exe_time == os.path.getmtime(model.exe_file))

        model = CmdStanModel(stan_file=BERN_STAN, compile='force')
        self.assertTrue(exe_time < os.path.getmtime(model.exe_file))

    def test_exe_only(self):
        model = CmdStanModel(stan_file=BERN_STAN)
        self.assertEqual(BERN_EXE, model.exe_file)
        exe_only = os.path.join(DATAFILES_PATH, 'exe_only')
        shutil.copyfile(model.exe_file, exe_only)

        model2 = CmdStanModel(exe_file=exe_only)
        with self.assertRaises(RuntimeError):
            model2.code()
        with self.assertRaises(RuntimeError):
            model2.compile()
        self.assertFalse(model2._fixed_param)

    def test_fixed_param(self):
        stan = os.path.join(DATAFILES_PATH, 'datagen_poisson_glm.stan')
        model = CmdStanModel(stan_file=stan)
        self.assertTrue(model._fixed_param)

    # pylint: disable=no-self-use
    def test_model_pedantic(self):
        stan_file = os.path.join(DATAFILES_PATH, 'bernoulli_pedantic.stan')
        with LogCapture(level=logging.WARNING) as log:
            logging.getLogger()
            model = CmdStanModel(model_name='bern', stan_file=stan_file)
            model.compile(force=True, stanc_options={'warn-pedantic': True})
        log.check_present(
            (
                'cmdstanpy',
                'WARNING',
                StringComparison(r'(?s).*The parameter theta has no priors.*'),
            )
        )

    def test_model_bad(self):
        with self.assertRaises(ValueError):
            CmdStanModel(stan_file=None, exe_file=None)
        with self.assertRaises(ValueError):
            CmdStanModel(model_name='bad')
        with self.assertRaises(ValueError):
            CmdStanModel(model_name='', stan_file=BERN_STAN)
        with self.assertRaises(ValueError):
            CmdStanModel(model_name='   ', stan_file=BERN_STAN)
        with self.assertRaises(ValueError):
            CmdStanModel(
                stan_file=os.path.join(DATAFILES_PATH, "external.stan")
            )
        CmdStanModel(stan_file=BERN_STAN)
        os.remove(BERN_EXE)
        with self.assertRaises(ValueError):
            CmdStanModel(stan_file=BERN_STAN, exe_file=BERN_EXE)

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

    def test_model_info(self):
        model = CmdStanModel(stan_file=BERN_STAN, compile=False)
        model.compile(force=True)
        info_dict = model.exe_info()
        self.assertEqual(info_dict['STAN_THREADS'].lower(), 'false')

        if model.exe_file is not None and os.path.exists(model.exe_file):
            os.remove(model.exe_file)
        empty_dict = model.exe_info()
        self.assertEqual(len(empty_dict), 0)

        model_info = model.src_info()
        self.assertNotEqual(model_info, {})
        self.assertIn('theta', model_info['parameters'])

    def test_compile_force(self):
        if os.path.exists(BERN_EXE):
            os.remove(BERN_EXE)
        model = CmdStanModel(stan_file=BERN_STAN, compile=False, cpp_options={})
        self.assertIsNone(model.exe_file)

        model.compile(force=True)
        self.assertIsNotNone(model.exe_file)
        self.assertTrue(os.path.exists(model.exe_file))

        info_dict = model.exe_info()
        print(f'info={info_dict}')
        self.assertEqual(info_dict['STAN_THREADS'].lower(), 'false')

        more_opts = {'STAN_THREADS': 'TRUE'}

        model.compile(force=True, cpp_options=more_opts)
        self.assertIsNotNone(model.exe_file)
        self.assertTrue(os.path.exists(model.exe_file))

        info_dict2 = model.exe_info()
        print(f'info2={info_dict2}')
        self.assertEqual(info_dict2['STAN_THREADS'].lower(), 'true')

        override_opts = {'STAN_NO_RANGE_CHECKS': 'TRUE'}

        model.compile(
            force=True, cpp_options=override_opts, override_options=True
        )
        info_dict3 = model.exe_info()
        print(f'info3={info_dict3}')
        self.assertEqual(info_dict3['STAN_THREADS'].lower(), 'false')
        # cmdstan#1056
        # self.assertEqual(info_dict3['STAN_NO_RANGE_CHECKS'].lower(), 'true')

        model.compile(force=True, cpp_options=more_opts)
        info_dict4 = model.exe_info()
        self.assertEqual(info_dict4['STAN_THREADS'].lower(), 'true')

        # test compile='force' in constructor
        model2 = CmdStanModel(stan_file=BERN_STAN, compile='force')
        info_dict5 = model2.exe_info()
        self.assertEqual(info_dict5['STAN_THREADS'].lower(), 'false')

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
        with LogCapture(level=logging.WARNING) as log:
            logging.getLogger()
            with self.assertRaises(ValueError):
                CmdStanModel(stan_file=stan)
        log.check_present(
            ('cmdstanpy', 'WARNING', StringComparison(r'(?s).*Syntax error.*'))
        )

    def test_repr(self):
        model = CmdStanModel(stan_file=BERN_STAN)
        model_repr = model.__repr__()
        self.assertIn('name=bernoulli', model_repr)

    def test_print(self):
        model = CmdStanModel(stan_file=BERN_STAN)
        self.assertEqual(CODE, model.code())

    def test_model_compile(self):
        model = CmdStanModel(stan_file=BERN_STAN)
        self.assertPathsEqual(model.exe_file, BERN_EXE)

        model = CmdStanModel(stan_file=BERN_STAN)
        self.assertPathsEqual(model.exe_file, BERN_EXE)
        old_exe_time = os.path.getmtime(model.exe_file)
        os.remove(BERN_EXE)
        model.compile()
        new_exe_time = os.path.getmtime(model.exe_file)
        self.assertTrue(new_exe_time > old_exe_time)

        # test compile with existing exe - timestamp on exe unchanged
        exe_time = os.path.getmtime(model.exe_file)
        model2 = CmdStanModel(stan_file=BERN_STAN)
        self.assertEqual(exe_time, os.path.getmtime(model2.exe_file))

    def test_model_compile_space(self):
        with tempfile.TemporaryDirectory(
            prefix="cmdstanpy_testfolder_"
        ) as tmp_path:
            path_with_space = os.path.join(tmp_path, "space in path")
            os.makedirs(path_with_space, exist_ok=True)
            bern_stan_new = os.path.join(
                path_with_space, os.path.split(BERN_STAN)[1]
            )
            bern_exe_new = os.path.join(
                path_with_space, os.path.split(BERN_EXE)[1]
            )
            shutil.copyfile(BERN_STAN, bern_stan_new)
            model = CmdStanModel(stan_file=bern_stan_new)

            old_exe_time = os.path.getmtime(model.exe_file)
            os.remove(bern_exe_new)
            model.compile()
            new_exe_time = os.path.getmtime(model.exe_file)
            self.assertTrue(new_exe_time > old_exe_time)

            # test compile with existing exe - timestamp on exe unchanged
            exe_time = os.path.getmtime(model.exe_file)
            model2 = CmdStanModel(stan_file=bern_stan_new)
            self.assertEqual(exe_time, os.path.getmtime(model2.exe_file))

    def test_model_includes_explicit(self):
        if os.path.exists(BERN_EXE):
            os.remove(BERN_EXE)
        model = CmdStanModel(
            stan_file=BERN_STAN, stanc_options={'include_paths': DATAFILES_PATH}
        )
        self.assertEqual(BERN_STAN, model.stan_file)
        self.assertPathsEqual(model.exe_file, BERN_EXE)

    def test_model_includes_implicit(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_include.stan')
        exe = os.path.join(DATAFILES_PATH, 'bernoulli_include' + EXTENSION)
        if os.path.exists(exe):
            os.remove(exe)
        model2 = CmdStanModel(stan_file=stan)
        self.assertPathsEqual(model2.exe_file, exe)


if __name__ == '__main__':
    unittest.main()
