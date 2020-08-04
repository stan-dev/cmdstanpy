"""utils test"""

import json
import os
import unittest
import platform
import shutil
import string
import random

import numpy as np

from cmdstanpy import _TMPDIR
from cmdstanpy.utils import (
    cmdstan_path,
    set_cmdstan_path,
    validate_cmdstan_path,
    get_latest_cmdstan,
    check_sampler_csv,
    MaybeDictToFilePath,
    read_metric,
    TemporaryCopiedFile,
    windows_short_path,
    rdump,
    rload,
    parse_rdump_value,
    jsondump,
    parse_var_dims,
)
from cmdstanpy.model import CmdStanModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


class CmdStanPathTest(unittest.TestCase):
    def test_default_path(self):
        cur_value = None
        if 'CMDSTAN' in os.environ:
            cur_value = os.environ['CMDSTAN']
        try:
            if 'CMDSTAN' in os.environ:
                self.assertEqual(cmdstan_path(), os.environ['CMDSTAN'])
                path = os.environ['CMDSTAN']
                del os.environ['CMDSTAN']
                self.assertFalse('CMDSTAN' in os.environ)
                set_cmdstan_path(path)
                self.assertEqual(cmdstan_path(), path)
                self.assertTrue('CMDSTAN' in os.environ)
            else:
                install_dir = os.path.expanduser(
                    os.path.join('~', '.cmdstanpy')
                )
                install_version = os.path.expanduser(
                    os.path.join(install_dir, get_latest_cmdstan(install_dir))
                )
                self.assertTrue(
                    os.path.samefile(cmdstan_path(), install_version)
                )
                self.assertTrue('CMDSTAN' in os.environ)
        finally:
            if cur_value is not None:
                os.environ['CMDSTAN'] = cur_value
            else:
                if 'CMDSTAN' in os.environ:
                    del os.environ['CMDSTAN']

    def test_non_spaces_location(self):
        good_path = os.path.join(_TMPDIR, 'good_dir')
        with TemporaryCopiedFile(good_path) as (pth, is_changed):
            self.assertEqual(pth, good_path)
            self.assertFalse(is_changed)

        # prepare files for test
        bad_path = os.path.join(_TMPDIR, 'bad dir')
        os.makedirs(bad_path, exist_ok=True)
        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        stan_bad = os.path.join(bad_path, 'bad name.stan')
        shutil.copy(stan, stan_bad)

        stan_copied = None
        try:
            with TemporaryCopiedFile(stan_bad) as (pth, is_changed):
                stan_copied = pth
                self.assertTrue(os.path.exists(stan_copied))
                self.assertTrue(' ' not in stan_copied)
                self.assertTrue(is_changed)
                raise RuntimeError
        except RuntimeError:
            pass

        if platform.system() != 'Windows':
            self.assertFalse(os.path.exists(stan_copied))

        # cleanup after test
        shutil.rmtree(bad_path, ignore_errors=True)

    def test_set_path(self):
        if 'CMDSTAN' in os.environ:
            self.assertEqual(cmdstan_path(), os.environ['CMDSTAN'])
        else:
            install_dir = os.path.expanduser(os.path.join('~', '.cmdstanpy'))
            install_version = os.path.expanduser(
                os.path.join(install_dir, get_latest_cmdstan(install_dir))
            )
            set_cmdstan_path(install_version)
            self.assertEqual(install_version, cmdstan_path())
            self.assertEqual(install_version, os.environ['CMDSTAN'])

    def test_validate_path(self):
        install_dir = os.path.expanduser(os.path.join('~', '.cmdstanpy'))
        install_version = os.path.expanduser(
            os.path.join(install_dir, get_latest_cmdstan(install_dir))
        )
        set_cmdstan_path(install_version)
        validate_cmdstan_path(install_version)
        path_foo = os.path.abspath(os.path.join('releases', 'foo'))
        with self.assertRaisesRegex(ValueError, 'no such CmdStan directory'):
            validate_cmdstan_path(path_foo)
        folder_name = ''.join(
            random.choice(string.ascii_letters) for _ in range(10)
        )
        while os.path.exists(folder_name):
            folder_name = ''.join(
                random.choice(string.ascii_letters) for _ in range(10)
            )
        os.makedirs(folder_name)
        path_test = os.path.abspath(folder_name)
        with self.assertRaisesRegex(ValueError, 'no CmdStan binaries'):
            validate_cmdstan_path(path_test)
        shutil.rmtree(folder_name)

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

    def test_jsondump(self):
        def cmp(d1, d2):
            self.assertEqual(d1.keys(), d2.keys())
            for k in d1:
                data_1 = d1[k]
                data_2 = d2[k]
                if isinstance(data_2, np.ndarray):
                    data_2 = data_2.tolist()
                self.assertEqual(data_1, data_2)

        dict_list = {'a': [1.0, 2.0, 3.0]}
        file_list = os.path.join(_TMPDIR, 'list.json')
        jsondump(file_list, dict_list)
        with open(file_list) as fd:
            cmp(json.load(fd), dict_list)

        dict_vec = {'a': np.repeat(3, 4)}
        file_vec = os.path.join(_TMPDIR, 'vec.json')
        jsondump(file_vec, dict_vec)
        with open(file_vec) as fd:
            cmp(json.load(fd), dict_vec)

        dict_zero_vec = {'a': []}
        file_zero_vec = os.path.join(_TMPDIR, 'empty_vec.json')
        jsondump(file_zero_vec, dict_zero_vec)
        with open(file_zero_vec) as fd:
            cmp(json.load(fd), dict_zero_vec)

        dict_zero_matrix = {'a': [[], [], []]}
        file_zero_matrix = os.path.join(_TMPDIR, 'empty_matrix.json')
        jsondump(file_zero_matrix, dict_zero_matrix)
        with open(file_zero_matrix) as fd:
            cmp(json.load(fd), dict_zero_matrix)

        arr = np.zeros(shape=(5, 0))
        dict_zero_matrix = {'a': arr}
        file_zero_matrix = os.path.join(_TMPDIR, 'empty_matrix.json')
        jsondump(file_zero_matrix, dict_zero_matrix)
        with open(file_zero_matrix) as fd:
            cmp(json.load(fd), dict_zero_matrix)


class ReadStanCsvTest(unittest.TestCase):
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
        self.assertFalse('save_warmup' in dict)
        self.assertEqual(10, dict['draws_sampling'])
        self.assertEqual(8, len(dict['column_names']))

        with self.assertRaisesRegex(
            ValueError, 'config error, expected thin = 2'
        ):
            check_sampler_csv(
                path=csv_good, iter_warmup=100, iter_sampling=20, thin=2
            )
        with self.assertRaisesRegex(
            ValueError, 'config error, expected save_warmup'
        ):
            check_sampler_csv(
                path=csv_good,
                iter_warmup=100,
                iter_sampling=10,
                save_warmup=True,
            )
        with self.assertRaisesRegex(ValueError, 'expected 1000 draws'):
            check_sampler_csv(path=csv_good, iter_warmup=100)

    def test_check_sampler_csv_2(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'no_such_file.csv')
        with self.assertRaises(Exception):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_3(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_cols.csv')
        with self.assertRaisesRegex(Exception, '8 items'):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_4(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_rows.csv')
        with self.assertRaisesRegex(Exception, 'found 9'):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_1(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_metric_1.csv')
        with self.assertRaisesRegex(Exception, 'expecting metric'):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_2(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_metric_2.csv')
        with self.assertRaisesRegex(Exception, 'invalid stepsize'):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_3(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_metric_3.csv')
        with self.assertRaisesRegex(
            Exception, 'invalid or missing mass matrix specification'
        ):
            check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_4(self):
        csv_bad = os.path.join(DATAFILES_PATH, 'output_bad_metric_4.csv')
        with self.assertRaisesRegex(
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

        with self.assertRaisesRegex(ValueError, 'config error'):
            check_sampler_csv(
                path=csv_file,
                is_fixed_param=False,
                iter_sampling=490,
                iter_warmup=490,
                thin=9,
            )
        with self.assertRaisesRegex(ValueError, 'expected 490 draws, found 70'):
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


# pylint: disable=no-self-use
class WindowsShortPath(unittest.TestCase):
    def test_windows_short_path_directory(self):
        if platform.system() != 'Windows':
            return
        original_path = os.path.join(_TMPDIR, 'new path')
        os.makedirs(original_path, exist_ok=True)
        assert os.path.exists(original_path)
        assert ' ' in original_path
        short_path = windows_short_path(original_path)
        assert os.path.exists(short_path)
        assert original_path != short_path
        assert ' ' not in short_path

    def test_windows_short_path_file(self):
        if platform.system() != 'Windows':
            return
        original_path = os.path.join(_TMPDIR, 'new path', 'my_file.csv')
        os.makedirs(os.path.split(original_path)[0], exist_ok=True)
        assert os.path.exists(os.path.split(original_path)[0])
        assert ' ' in original_path
        assert os.path.splitext(original_path)[1] == '.csv'
        short_path = windows_short_path(original_path)
        assert os.path.exists(os.path.split(short_path)[0])
        assert original_path != short_path
        assert ' ' not in short_path
        assert os.path.splitext(short_path)[1] == '.csv'

    def test_windows_short_path_file_with_space(self):
        """Test that the function doesn't touch filename."""
        if platform.system() != 'Windows':
            return
        original_path = os.path.join(_TMPDIR, 'new path', 'my file.csv')
        os.makedirs(os.path.split(original_path)[0], exist_ok=True)
        assert os.path.exists(os.path.split(original_path)[0])
        assert ' ' in original_path
        short_path = windows_short_path(original_path)
        assert os.path.exists(os.path.split(short_path)[0])
        assert original_path != short_path
        assert ' ' in short_path
        assert os.path.splitext(short_path)[1] == '.csv'


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

    def test_roundtrip_metric(self):
        dfile = os.path.join(DATAFILES_PATH, 'metric_diag.data.R')
        data_dict_1 = rload(dfile)
        self.assertEqual(data_dict_1['inv_metric'].shape, (3,))

        dfile_tmp = os.path.join(DATAFILES_PATH, 'tmp.data.R')
        rdump(dfile_tmp, data_dict_1)
        data_dict_2 = rload(dfile_tmp)

        self.assertTrue('inv_metric' in data_dict_2)
        for i, x in enumerate(data_dict_2['inv_metric']):
            self.assertEqual(x, data_dict_2['inv_metric'][i])

        os.remove(dfile_tmp)

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


class ParseVarDimsTest(unittest.TestCase):
    def test_parse_empty(self):
        x = []
        vars_dict = parse_var_dims(x)
        self.assertEqual(len(vars_dict), 0)

    def test_parse_missing(self):
        with self.assertRaises(ValueError):
            parse_var_dims(None)

    def test_parse_scalar(self):
        x = ['foo']
        vars_dict = parse_var_dims(x)
        self.assertEqual(len(vars_dict), 1)
        self.assertEqual(vars_dict['foo'], 1)

    def test_parse_scalars(self):
        x = ['foo', 'foo1']
        vars_dict = parse_var_dims(x)
        self.assertEqual(len(vars_dict), 2)
        self.assertEqual(vars_dict['foo'], 1)
        self.assertEqual(vars_dict['foo1'], 1)

    def test_parse_scalar_vec_scalar(self):
        x = [
            'foo',
            'phi.1',
            'phi.2',
            'phi.3',
            'phi.4',
            'phi.5',
            'phi.6',
            'phi.7',
            'phi.8',
            'phi.9',
            'phi.10',
            'bar',
        ]
        vars_dict = parse_var_dims(x)
        self.assertEqual(len(vars_dict), 3)
        self.assertEqual(vars_dict['foo'], 1)
        self.assertEqual(vars_dict['phi'], (10,))
        self.assertEqual(vars_dict['bar'], 1)

    def test_parse_scalar_matrix_vec(self):
        x = [
            'foo',
            'phi.1.1',
            'phi.1.2',
            'phi.1.3',
            'phi.1.4',
            'phi.1.5',
            'phi.2.1',
            'phi.2.2',
            'phi.2.3',
            'phi.2.4',
            'phi.2.5',
            'bar.1',
            'bar.2',
        ]
        vars_dict = parse_var_dims(x)
        self.assertEqual(len(vars_dict), 3)
        self.assertEqual(vars_dict['foo'], 1)
        self.assertEqual(vars_dict['phi'], (2, 5))
        self.assertEqual(vars_dict['bar'], (2,))


if __name__ == '__main__':
    unittest.main()
