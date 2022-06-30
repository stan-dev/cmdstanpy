"""utils test"""

import collections.abc
import contextlib
import io
import json
import logging
import os
import pathlib
import platform
import random
import shutil
import stat
import string
import tempfile
import unittest
from test import CustomTestCase

import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture, StringComparison

from cmdstanpy import _DOT_CMDSTAN, _TMPDIR
from cmdstanpy.model import CmdStanModel
from cmdstanpy.progress import _disable_progress, allow_show_progress
from cmdstanpy.utils import (
    EXTENSION,
    BaseType,
    MaybeDictToFilePath,
    SanitizedOrTmpFilePath,
    check_sampler_csv,
    cmdstan_path,
    cmdstan_version,
    cmdstan_version_before,
    do_command,
    flatten_chains,
    get_latest_cmdstan,
    install_cmdstan,
    parse_method_vars,
    parse_rdump_value,
    parse_stan_vars,
    pushd,
    read_metric,
    rload,
    set_cmdstan_path,
    validate_cmdstan_path,
    validate_dir,
    windows_short_path,
    write_stan_json,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')
BERN_STAN = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
BERN_DATA = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
BERN_EXE = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)


class CmdStanPathTest(CustomTestCase):
    def test_default_path(self):
        if 'CMDSTAN' in os.environ:
            self.assertPathsEqual(cmdstan_path(), os.environ['CMDSTAN'])
            path = os.environ['CMDSTAN']
            with self.modified_environ('CMDSTAN'):
                self.assertNotIn('CMDSTAN', os.environ)
                set_cmdstan_path(path)
                self.assertPathsEqual(cmdstan_path(), path)
                self.assertIn('CMDSTAN', os.environ)
        else:
            cmdstan_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))
            install_version = os.path.join(
                cmdstan_dir, get_latest_cmdstan(cmdstan_dir)
            )
            self.assertTrue(os.path.samefile(cmdstan_path(), install_version))
            self.assertIn('CMDSTAN', os.environ)

    def test_non_spaces_location(self):
        with tempfile.TemporaryDirectory(
            prefix="cmdstan_tests", dir=_TMPDIR
        ) as tmpdir:
            good_path = os.path.join(tmpdir, 'good_dir')
            os.mkdir(good_path)
            with SanitizedOrTmpFilePath(good_path) as (pth, is_changed):
                self.assertPathsEqual(pth, good_path)
                self.assertFalse(is_changed)

            # prepare files for test
            bad_path = os.path.join(tmpdir, 'bad dir')
            os.makedirs(bad_path, exist_ok=True)
            stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
            stan_bad = os.path.join(bad_path, 'bad name.stan')
            shutil.copy(stan, stan_bad)

            stan_copied = None
            try:
                with SanitizedOrTmpFilePath(stan_bad) as (pth, is_changed):
                    stan_copied = pth
                    self.assertTrue(os.path.exists(stan_copied))
                    self.assertNotIn(' ', stan_copied)
                    self.assertTrue(is_changed)
                    raise RuntimeError
            except RuntimeError:
                pass

            if platform.system() != 'Windows':
                self.assertFalse(os.path.exists(stan_copied))

            # cleanup after test
            shutil.rmtree(good_path, ignore_errors=True)
            shutil.rmtree(bad_path, ignore_errors=True)

    def test_set_path(self):
        if 'CMDSTAN' in os.environ:
            self.assertPathsEqual(cmdstan_path(), os.environ['CMDSTAN'])
        else:
            cmdstan_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))
            install_version = os.path.join(
                cmdstan_dir, get_latest_cmdstan(cmdstan_dir)
            )
            set_cmdstan_path(install_version)
            self.assertPathsEqual(install_version, cmdstan_path())
            self.assertPathsEqual(install_version, os.environ['CMDSTAN'])

    def test_validate_path(self):
        if 'CMDSTAN' in os.environ:
            install_version = os.environ.get('CMDSTAN')
        else:
            cmdstan_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))

            install_version = os.path.join(
                cmdstan_dir, get_latest_cmdstan(cmdstan_dir)
            )

        set_cmdstan_path(install_version)
        validate_cmdstan_path(install_version)
        path_foo = os.path.abspath(os.path.join('releases', 'foo'))
        with self.assertRaisesRegex(ValueError, 'No CmdStan directory'):
            validate_cmdstan_path(path_foo)

        folder_name = ''.join(
            random.choice(string.ascii_letters) for _ in range(10)
        )
        while os.path.exists(folder_name):
            folder_name = ''.join(
                random.choice(string.ascii_letters) for _ in range(10)
            )
        folder = pathlib.Path(folder_name)
        folder.mkdir(parents=True)
        (folder / "makefile").touch()

        with self.assertRaisesRegex(ValueError, 'missing binaries'):
            validate_cmdstan_path(str(folder.absolute()))
        shutil.rmtree(folder)

    def test_validate_dir(self):
        with tempfile.TemporaryDirectory(
            prefix="cmdstan_tests", dir=_TMPDIR
        ) as tmpdir:
            path = os.path.join(tmpdir, 'cmdstan-M.m.p')
            self.assertFalse(os.path.exists(path))
            validate_dir(path)
            self.assertTrue(os.path.exists(path))

            _, file = tempfile.mkstemp(dir=_TMPDIR)
            with self.assertRaisesRegex(Exception, 'File exists'):
                validate_dir(file)

            if platform.system() != 'Windows':
                with self.assertRaisesRegex(
                    Exception, 'Cannot create directory'
                ):
                    dir = tempfile.mkdtemp(dir=_TMPDIR)
                    os.chmod(dir, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                    validate_dir(os.path.join(dir, 'cmdstan-M.m.p'))

    def test_munge_cmdstan_versions(self):
        with tempfile.TemporaryDirectory(
            prefix="cmdstan_tests", dir=_TMPDIR
        ) as tmpdir:
            tdir = os.path.join(tmpdir, 'tmpdir_xxx')
            os.makedirs(tdir)
            os.makedirs(os.path.join(tdir, 'cmdstan-2.22.0-rc1'))
            os.makedirs(os.path.join(tdir, 'cmdstan-2.22.0-rc2'))
            self.assertEqual(get_latest_cmdstan(tdir), 'cmdstan-2.22.0-rc2')

            os.makedirs(os.path.join(tdir, 'cmdstan-2.22.0'))
            self.assertEqual(get_latest_cmdstan(tdir), 'cmdstan-2.22.0')

    def test_cmdstan_version_before(self):
        cmdstan_path()  # sets os.environ['CMDSTAN']
        self.assertTrue(cmdstan_version_before(99, 99))
        self.assertFalse(cmdstan_version_before(1, 1))

    def test_cmdstan_version(self):
        with tempfile.TemporaryDirectory(
            prefix="cmdstan_tests", dir=_TMPDIR
        ) as tmpdir:
            tdir = pathlib.Path(tmpdir) / 'tmpdir_xxx'
            fake_path = tdir / 'cmdstan-2.22.0'
            fake_bin = fake_path / 'bin'
            fake_bin.mkdir(parents=True)
            fake_makefile = fake_path / 'makefile'
            fake_makefile.touch()
            (fake_bin / f'stanc{EXTENSION}').touch()
            with self.modified_environ(CMDSTAN=str(fake_path)):
                self.assertTrue(str(fake_path) == cmdstan_path())
                with open(fake_makefile, 'w') as fd:
                    fd.write('...  CMDSTAN_VERSION := dont_need_no_mmp\n\n')
                expect = (
                    'Cannot parse version, expected "<major>.<minor>.<patch>", '
                    'found: "dont_need_no_mmp".'
                )
                with LogCapture() as log:
                    cmdstan_version()
                log.check_present(('cmdstanpy', 'INFO', expect))

                fake_makefile.unlink()
                expect = (
                    'CmdStan installation {} missing makefile, '
                    'cannot get version.'.format(fake_path)
                )
                with LogCapture() as log:
                    cmdstan_version()
                log.check_present(('cmdstanpy', 'INFO', expect))
        cmdstan_path()


class DataFilesTest(unittest.TestCase):
    def test_dict_to_file(self):
        file_good = os.path.join(DATAFILES_PATH, 'bernoulli_output_1.csv')
        dict_good = {'a': 0.5}
        created_tmp = None
        with MaybeDictToFilePath(file_good, dict_good) as (fg1, fg2):
            self.assertTrue(os.path.exists(fg1))
            self.assertTrue(os.path.exists(fg2))
            with open(fg2) as fg2_d:
                self.assertEqual(json.load(fg2_d), dict_good)
            created_tmp = fg2
        self.assertTrue(os.path.exists(file_good))
        self.assertFalse(os.path.exists(created_tmp))

        with self.assertRaises(ValueError):
            with MaybeDictToFilePath(123, dict_good) as (fg1, fg2):
                pass

    def test_write_stan_json(self):
        def cmp(d1, d2):
            self.assertEqual(d1.keys(), d2.keys())
            for k in d1:
                data_1 = d1[k]
                data_2 = d2[k]
                if isinstance(data_2, collections.abc.Collection):
                    data_2 = np.asarray(data_2).tolist()

                self.assertEqual(data_1, data_2)

        dict_list = {'a': [1.0, 2.0, 3.0]}
        file_list = os.path.join(_TMPDIR, 'list.json')
        write_stan_json(file_list, dict_list)
        with open(file_list) as fd:
            cmp(json.load(fd), dict_list)

        arr = np.repeat(3, 4)
        dict_vec = {'a': arr}
        file_vec = os.path.join(_TMPDIR, 'vec.json')
        write_stan_json(file_vec, dict_vec)
        with open(file_vec) as fd:
            cmp(json.load(fd), dict_vec)

        dict_bool = {'a': False}
        file_bool = os.path.join(_TMPDIR, 'bool.json')
        write_stan_json(file_bool, dict_bool)
        with open(file_bool) as fd:
            cmp(json.load(fd), {'a': 0})

        dict_none = {'a': None}
        file_none = os.path.join(_TMPDIR, 'none.json')
        write_stan_json(file_none, dict_none)
        with open(file_none) as fd:
            cmp(json.load(fd), dict_none)

        series = pd.Series(arr)
        dict_vec_pd = {'a': series}
        file_vec_pd = os.path.join(_TMPDIR, 'vec_pd.json')
        write_stan_json(file_vec_pd, dict_vec_pd)
        with open(file_vec_pd) as fd:
            cmp(json.load(fd), dict_vec_pd)

        df_vec = pd.DataFrame(dict_list)
        file_pd = os.path.join(_TMPDIR, 'pd.json')
        write_stan_json(file_pd, df_vec)
        with open(file_pd) as fd:
            cmp(json.load(fd), dict_list)

        dict_zero_vec = {'a': []}
        file_zero_vec = os.path.join(_TMPDIR, 'empty_vec.json')
        write_stan_json(file_zero_vec, dict_zero_vec)
        with open(file_zero_vec) as fd:
            cmp(json.load(fd), dict_zero_vec)

        dict_zero_matrix = {'a': [[], [], []]}
        file_zero_matrix = os.path.join(_TMPDIR, 'empty_matrix.json')
        write_stan_json(file_zero_matrix, dict_zero_matrix)
        with open(file_zero_matrix) as fd:
            cmp(json.load(fd), dict_zero_matrix)

        arr = np.zeros(shape=(5, 0))
        dict_zero_matrix = {'a': arr}
        file_zero_matrix = os.path.join(_TMPDIR, 'empty_matrix.json')
        write_stan_json(file_zero_matrix, dict_zero_matrix)
        with open(file_zero_matrix) as fd:
            cmp(json.load(fd), dict_zero_matrix)

        arr = np.zeros(shape=(2, 3, 4))
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertEqual(arr.shape, (2, 3, 4))

        dict_3d_matrix = {'a': arr}
        file_3d_matrix = os.path.join(_TMPDIR, '3d_matrix.json')
        write_stan_json(file_3d_matrix, dict_3d_matrix)
        with open(file_3d_matrix) as fd:
            cmp(json.load(fd), dict_3d_matrix)

        scalr = np.int32(1)
        self.assertTrue(type(scalr).__module__ == 'numpy')
        dict_scalr = {'a': scalr}
        file_scalr = os.path.join(_TMPDIR, 'scalr.json')
        write_stan_json(file_scalr, dict_scalr)
        with open(file_scalr) as fd:
            cmp(json.load(fd), dict_scalr)

        # custom Stan serialization
        dict_inf_nan = {
            'a': np.array(
                [
                    [-np.inf, np.inf, np.NaN],
                    [-float('inf'), float('inf'), float('NaN')],
                    [
                        np.float32(-np.inf),
                        np.float32(np.inf),
                        np.float32(np.NaN),
                    ],
                    [1e200 * -1e200, 1e220 * 1e200, -np.nan],
                ]
            )
        }
        dict_inf_nan_exp = {'a': [["-inf", "+inf", "NaN"]] * 4}
        file_fin = os.path.join(_TMPDIR, 'inf.json')
        write_stan_json(file_fin, dict_inf_nan)
        with open(file_fin) as fd:
            cmp(json.load(fd), dict_inf_nan_exp)

        dict_complex = {'a': np.array([np.complex64(3), 3 + 4j])}
        dict_complex_exp = {'a': [[3, 0], [3, 4]]}
        file_complex = os.path.join(_TMPDIR, 'complex.json')
        write_stan_json(file_complex, dict_complex)
        with open(file_complex) as fd:
            cmp(json.load(fd), dict_complex_exp)

    def test_write_stan_json_bad(self):
        file_bad = os.path.join(_TMPDIR, 'bad.json')

        dict_badtype = {'a': 'a string'}
        with self.assertRaises(TypeError):
            write_stan_json(file_bad, dict_badtype)

        dict_badtype_nested = {'a': ['a string']}
        with self.assertRaises(ValueError):
            write_stan_json(file_bad, dict_badtype_nested)


class ReadStanCsvTest(CustomTestCase):
    def test_check_sampler_csv_1(self):
        csv_good = os.path.join(DATAFILES_PATH, 'bernoulli_output_1.csv')
        dict = check_sampler_csv(
            path=csv_good,
            is_fixed_param=False,
            iter_warmup=100,
            iter_sampling=10,
            thin=1,
        )
        self.assertEqual('bernoulli_model', dict['model'])
        self.assertEqual(10, dict['num_samples'])
        self.assertEqual(10, dict['draws_sampling'])
        self.assertEqual(8, len(dict['column_names']))

        with self.assertRaisesRegexNested(
            ValueError, 'config error, expected thin = 2'
        ):
            check_sampler_csv(
                path=csv_good, iter_warmup=100, iter_sampling=20, thin=2
            )
        with self.assertRaisesRegexNested(
            ValueError, 'config error, expected save_warmup'
        ):
            check_sampler_csv(
                path=csv_good,
                iter_warmup=100,
                iter_sampling=10,
                save_warmup=True,
            )
        with self.assertRaisesRegexNested(ValueError, 'expected 1000 draws'):
            check_sampler_csv(path=csv_good, iter_warmup=100)

    def test_check_sampler_csv_2(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'no_such_file.csv')
        with self.assertRaises(Exception):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_3(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_cols.csv')
        with self.assertRaisesRegexNested(Exception, '8 items'):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_4(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_rows.csv')
        with self.assertRaisesRegexNested(Exception, 'found 9'):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_1(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_metric_1.csv')
        with self.assertRaisesRegexNested(Exception, 'expecting metric'):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_2(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_metric_2.csv')
        with self.assertRaisesRegexNested(Exception, 'invalid step size'):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_3(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_metric_3.csv')
        with self.assertRaisesRegexNested(
            Exception, 'invalid or missing mass matrix specification'
        ):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_4(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_metric_4.csv')
        with self.assertRaisesRegexNested(
            Exception, 'invalid or missing mass matrix specification'
        ):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_thin(self):
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        bern_model.compile()
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_fit = bern_model.sample(
            data=jdata,
            chains=1,
            parallel_chains=1,
            seed=12345,
            iter_sampling=490,
            iter_warmup=490,
            thin=7,
            max_treedepth=11,
            adapt_delta=0.98,
        )
        csv_file = bern_fit.runset.csv_files[0]
        dict = check_sampler_csv(
            path=csv_file,
            is_fixed_param=False,
            iter_sampling=490,
            iter_warmup=490,
            thin=7,
        )
        self.assertEqual(dict['num_samples'], 490)
        self.assertEqual(dict['thin'], 7)
        self.assertEqual(dict['draws_sampling'], 70)
        self.assertEqual(dict['seed'], 12345)
        self.assertEqual(dict['max_depth'], 11)
        self.assertEqual(dict['delta'], 0.98)

        with self.assertRaisesRegexNested(ValueError, 'config error'):
            check_sampler_csv(
                path=csv_file,
                is_fixed_param=False,
                iter_sampling=490,
                iter_warmup=490,
                thin=9,
            )
        with self.assertRaisesRegexNested(
            ValueError, 'expected 490 draws, found 70'
        ):
            check_sampler_csv(
                path=csv_file,
                is_fixed_param=False,
                iter_sampling=490,
                iter_warmup=490,
            )


class ReadMetricTest(unittest.TestCase):
    def test_metric_json_vec(self):
        metric_file = os.path.join(DATAFILES_PATH, 'metric_diag.data.json')
        dims = read_metric(metric_file)
        self.assertEqual(1, len(dims))
        self.assertEqual(3, dims[0])

    def test_metric_json_matrix(self):
        metric_file = os.path.join(DATAFILES_PATH, 'metric_dense.data.json')
        dims = read_metric(metric_file)
        self.assertEqual(2, len(dims))
        self.assertEqual(dims[0], dims[1])

    def test_metric_rdump_vec(self):
        metric_file = os.path.join(DATAFILES_PATH, 'metric_diag.data.R')
        dims = read_metric(metric_file)
        self.assertEqual(1, len(dims))
        self.assertEqual(3, dims[0])

    def test_metric_rdump_matrix(self):
        metric_file = os.path.join(DATAFILES_PATH, 'metric_dense.data.R')
        dims = read_metric(metric_file)
        self.assertEqual(2, len(dims))
        self.assertEqual(dims[0], dims[1])

    def test_metric_json_bad(self):
        metric_file = os.path.join(DATAFILES_PATH, 'metric_bad.data.json')
        with self.assertRaisesRegex(
            Exception, 'bad or missing entry "inv_metric"'
        ):
            read_metric(metric_file)

    def test_metric_rdump_bad_1(self):
        metric_file = os.path.join(DATAFILES_PATH, 'metric_bad_1.data.R')
        with self.assertRaisesRegex(
            Exception, 'bad or missing entry "inv_metric"'
        ):
            read_metric(metric_file)

    def test_metric_rdump_bad_2(self):
        metric_file = os.path.join(DATAFILES_PATH, 'metric_bad_2.data.R')
        with self.assertRaisesRegex(
            Exception, 'bad or missing entry "inv_metric"'
        ):
            read_metric(metric_file)

    def test_metric_missing(self):
        metric_file = os.path.join(DATAFILES_PATH, 'no_such_file.json')
        with self.assertRaisesRegex(Exception, 'No such file or directory'):
            read_metric(metric_file)


@pytest.mark.skipif(platform.system() != 'Windows', reason='Windows only tests')
class WindowsShortPath(unittest.TestCase):
    def test_windows_short_path_directory(self):
        with tempfile.TemporaryDirectory(
            prefix="cmdstan_tests", dir=_TMPDIR
        ) as tmpdir:
            original_path = os.path.join(tmpdir, 'new path')
            os.makedirs(original_path, exist_ok=True)
            self.assertTrue(os.path.exists(original_path))
            self.assertIn(' ', original_path)
            short_path = windows_short_path(original_path)
            self.assertTrue(os.path.exists(short_path))
            self.assertNotEqual(original_path, short_path)
            self.assertNotIn(' ', short_path)

    def test_windows_short_path_file(self):
        with tempfile.TemporaryDirectory(
            prefix="cmdstan_tests", dir=_TMPDIR
        ) as tmpdir:
            original_path = os.path.join(tmpdir, 'new path', 'my_file.csv')
            os.makedirs(os.path.split(original_path)[0], exist_ok=True)
            self.assertTrue(os.path.exists(os.path.split(original_path)[0]))
            self.assertIn(' ', original_path)
            self.assertEqual(os.path.splitext(original_path)[1], '.csv')
            short_path = windows_short_path(original_path)
            self.assertTrue(os.path.exists(os.path.split(short_path)[0]))
            self.assertNotEqual(original_path, short_path)
            self.assertNotIn(' ', short_path)
            self.assertEqual(os.path.splitext(short_path)[1], '.csv')

    def test_windows_short_path_file_with_space(self):
        """Test that the function doesn't touch filename."""
        with tempfile.TemporaryDirectory(
            prefix="cmdstan_tests", dir=_TMPDIR
        ) as tmpdir:
            original_path = os.path.join(tmpdir, 'new path', 'my file.csv')
            os.makedirs(os.path.split(original_path)[0], exist_ok=True)
            self.assertTrue(os.path.exists(os.path.split(original_path)[0]))
            self.assertIn(' ', original_path)
            short_path = windows_short_path(original_path)
            self.assertTrue(os.path.exists(os.path.split(short_path)[0]))
            self.assertNotEqual(original_path, short_path)
            self.assertIn(' ', short_path)
            self.assertEqual(os.path.splitext(short_path)[1], '.csv')


class RloadTest(unittest.TestCase):
    def test_rload_metric(self):
        dfile = os.path.join(DATAFILES_PATH, 'metric_diag.data.R')
        data_dict = rload(dfile)
        self.assertEqual(data_dict['inv_metric'].shape, (3,))

        dfile = os.path.join(DATAFILES_PATH, 'metric_dense.data.R')
        data_dict = rload(dfile)
        self.assertEqual(data_dict['inv_metric'].shape, (3, 3))

    def test_rload_data(self):
        dfile = os.path.join(DATAFILES_PATH, 'rdump_test.data.R')
        data_dict = rload(dfile)
        self.assertEqual(data_dict['N'], 128)
        self.assertEqual(data_dict['M'], 2)
        self.assertEqual(data_dict['x'].shape, (128, 2))

    def test_rload_jags_data(self):
        dfile = os.path.join(DATAFILES_PATH, 'rdump_jags.data.R')
        data_dict = rload(dfile)
        self.assertEqual(data_dict['N'], 128)
        self.assertEqual(data_dict['M'], 2)
        self.assertEqual(data_dict['y'].shape, (128,))

    def test_rload_wrong_data(self):
        dfile = os.path.join(DATAFILES_PATH, 'metric_diag.data.json')
        data_dict = rload(dfile)
        self.assertEqual(data_dict, None)

    def test_rload_bad_data_1(self):
        dfile = os.path.join(DATAFILES_PATH, 'rdump_bad_1.data.R')
        with self.assertRaises(ValueError):
            rload(dfile)

    def test_rload_bad_data_2(self):
        dfile = os.path.join(DATAFILES_PATH, 'rdump_bad_2.data.R')
        with self.assertRaises(ValueError):
            rload(dfile)

    def test_rload_bad_data_3(self):
        dfile = os.path.join(DATAFILES_PATH, 'rdump_bad_3.data.R')
        with self.assertRaises(ValueError):
            rload(dfile)

    def test_parse_rdump_value(self):
        struct1 = (
            'structure(c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16),.Dim=c(2,8))'
        )
        v_struct1 = parse_rdump_value(struct1)
        self.assertEqual(v_struct1.shape, (2, 8))
        self.assertEqual(v_struct1[1, 0], 2)
        self.assertEqual(v_struct1[0, 7], 15)

        struct2 = (
            'structure(c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16),.Dim=c(1,16))'
        )
        v_struct2 = parse_rdump_value(struct2)
        self.assertEqual(v_struct2.shape, (1, 16))

        struct3 = (
            'structure(c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16),.Dim=c(8,2))'
        )
        v_struct3 = parse_rdump_value(struct3)
        self.assertEqual(v_struct3.shape, (8, 2))
        self.assertEqual(v_struct3[1, 0], 2)
        self.assertEqual(v_struct3[7, 0], 8)
        self.assertEqual(v_struct3[0, 1], 9)
        self.assertEqual(v_struct3[6, 1], 15)


class ParseVarsTest(unittest.TestCase):
    def test_parse_empty(self):
        x = []
        sampler_vars = parse_method_vars(x)
        self.assertEqual(len(sampler_vars), 0)
        stan_vars_dims, stan_vars_cols, stan_var_types = parse_stan_vars(x)
        self.assertEqual(len(stan_vars_dims), 0)
        self.assertEqual(len(stan_vars_cols), 0)
        self.assertEqual(len(stan_var_types), 0)

    def test_parse_missing(self):
        with self.assertRaises(ValueError):
            parse_method_vars(None)
        with self.assertRaises(ValueError):
            parse_stan_vars(None)

    def test_parse_method_vars(self):
        x = [
            'lp__',
            'accept_stat__',
            'stepsize__',
            'treedepth__',
            'n_leapfrog__',
            'divergent__',
            'energy__',
            'theta[1]',
            'theta[2]',
            'theta[3]',
            'theta[4]',
            'z_init[1]',
            'z_init[2]',
        ]
        vars_dict = parse_method_vars(x)
        self.assertEqual(len(vars_dict), 7)
        self.assertEqual(vars_dict['lp__'], (0,))
        self.assertEqual(vars_dict['stepsize__'], (2,))

    def test_parse_scalars(self):
        x = ['lp__', 'foo']
        dims_map, cols_map, _ = parse_stan_vars(x)
        self.assertEqual(len(dims_map), 1)
        self.assertEqual(dims_map['foo'], ())
        self.assertEqual(len(cols_map), 1)
        self.assertEqual(cols_map['foo'], (1,))

        dims_map = {}
        cols_map = {}
        x = ['lp__', 'foo1', 'foo2']
        dims_map, cols_map, _ = parse_stan_vars(x)
        self.assertEqual(len(dims_map), 2)
        self.assertEqual(dims_map['foo1'], ())
        self.assertEqual(dims_map['foo2'], ())
        self.assertEqual(len(cols_map), 2)
        self.assertEqual(cols_map['foo1'], (1,))
        self.assertEqual(cols_map['foo2'], (2,))

        dims_map = {}
        cols_map = {}
        x = ['lp__', 'z[real]', 'z[imag]']
        dims_map, cols_map, types_map = parse_stan_vars(x)
        self.assertEqual(len(dims_map), 1)
        self.assertEqual(dims_map['z'], (2,))
        self.assertEqual(types_map['z'], BaseType.COMPLEX)

    def test_parse_containers(self):
        # demonstrates flaw in shortcut to get container dims
        x = [
            'lp__',
            'accept_stat__',
            'foo',
            'phi[1]',
            'phi[2]',
            'phi[3]',
            'phi[10]',
            'bar',
        ]
        dims_map, cols_map, _ = parse_stan_vars(x)
        self.assertEqual(len(dims_map), 3)
        self.assertEqual(dims_map['foo'], ())
        self.assertEqual(dims_map['phi'], (10,))  # sic
        self.assertEqual(dims_map['bar'], ())
        self.assertEqual(len(cols_map), 3)
        self.assertEqual(cols_map['foo'], (2,))
        self.assertEqual(
            cols_map['phi'],
            (
                3,
                4,
                5,
                6,
            ),
        )
        self.assertEqual(cols_map['bar'], (7,))

        x = [
            'lp__',
            'accept_stat__',
            'foo',
            'phi[1]',
            'phi[2]',
            'phi[3]',
            'phi[10,10]',
            'bar',
        ]
        dims_map = {}
        cols_map = {}
        dims_map, cols_map, _ = parse_stan_vars(x)
        self.assertEqual(len(dims_map), 3)
        self.assertEqual(
            dims_map['phi'],
            (
                10,
                10,
            ),
        )
        self.assertEqual(len(cols_map), 3)
        self.assertEqual(
            cols_map['phi'],
            (
                3,
                4,
                5,
                6,
            ),
        )

        x = [
            'lp__',
            'accept_stat__',
            'foo',
            'phi[10,10,10]',
        ]
        dims_map = {}
        cols_map = {}
        dims_map, cols_map, _ = parse_stan_vars(x)
        self.assertEqual(len(dims_map), 2)
        self.assertEqual(
            dims_map['phi'],
            (
                10,
                10,
                10,
            ),
        )
        self.assertEqual(len(cols_map), 2)
        self.assertEqual(cols_map['phi'], (3,))


class DoCommandTest(unittest.TestCase):
    def test_capture_console(self):
        tmp = io.StringIO()
        do_command(cmd=['ls'], cwd=HERE, fd_out=tmp)
        self.assertIn('test_utils.py', tmp.getvalue())

    def test_exit(self):
        sys_stdout = io.StringIO()
        with contextlib.redirect_stdout(sys_stdout):
            args = ['bash', '/bin/junk']
            with self.assertRaises(RuntimeError):
                do_command(args, HERE)


class PushdTest(unittest.TestCase):
    def test_restore_cwd(self):
        "Ensure do_command in a different cwd restores cwd after error."
        cwd = os.getcwd()
        with self.assertRaises(RuntimeError):
            with pushd(os.path.dirname(cwd)):
                raise RuntimeError('error')
        self.assertEqual(cwd, os.getcwd())


class FlattenTest(unittest.TestCase):
    def test_good(self):
        array_3d = np.empty((200, 4, 4))
        vals = [1.0, 2.0, 3.0, 4.0]
        pos = [(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3)]
        draws, chains, cols = zip(*pos)
        array_3d[draws, chains, cols] = vals
        flattened = flatten_chains(array_3d)

        self.assertEqual(flattened.shape, (800, 4))
        self.assertEqual(flattened[0, 0], 1.0)
        self.assertEqual(flattened[200, 1], 2.0)
        self.assertEqual(flattened[400, 2], 3.0)
        self.assertEqual(flattened[600, 3], 4.0)

    def test_bad(self):
        array_2d = np.empty((200, 4))
        with self.assertRaisesRegex(ValueError, 'Expecting 3D array'):
            flatten_chains(array_2d)


class InstallCmdstanFunctionTest(CustomTestCase):
    def test_bad_version(self):
        with LogCapture() as log:
            res = install_cmdstan(version="0.00.0")
        log.check_present(
            (
                "cmdstanpy",
                "WARNING",
                StringComparison("CmdStan installation failed.\nVersion*"),
            )
        )
        self.assertFalse(res)

    def test_interactive_extra_args(self):
        with LogCapture() as log:
            with self.replace_stdin("9.99.9\n"):
                res = install_cmdstan(version="2.29.2", interactive=True)
        log.check_present(
            (
                "cmdstanpy",
                "WARNING",
                "Interactive installation requested but other arguments"
                " were used.\n\tThese values will be ignored!",
            )
        )
        self.assertFalse(res)


@pytest.mark.order(-1)
class ShowProgressTest(unittest.TestCase):
    # this test must run after any tests that check tqdm progress bars
    def test_show_progress_fns(self):
        self.assertTrue(allow_show_progress())
        with LogCapture() as log:
            logging.getLogger()
            try:
                raise ValueError("error")
            except ValueError as e:
                _disable_progress(e)
        log.check_present(
            (
                'cmdstanpy',
                'ERROR',
                'Error in progress bar initialization:\n'
                '\terror\n'
                'Disabling progress bars for this session',
            )
        )
        self.assertFalse(allow_show_progress())
        try:
            raise ValueError("error")
        except ValueError as e:
            with LogCapture() as log:
                logging.getLogger()
                _disable_progress(e)
        msgs = ' '.join(log.actual())
        # msg should only be printed once per session - check not found
        self.assertEqual(
            -1, msgs.find('Disabling progress bars for this session')
        )
        self.assertFalse(allow_show_progress())


if __name__ == '__main__':
    unittest.main()
