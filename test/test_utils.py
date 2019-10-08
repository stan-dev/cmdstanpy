import json
import os
import unittest
import platform
import tempfile
import shutil
import string
import random

from cmdstanpy import TMPDIR
from cmdstanpy.utils import (
    EXTENSION,
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
)
from cmdstanpy.model import Model

here = os.path.dirname(os.path.abspath(__file__))
datafiles_path = os.path.join(here, 'data')


class CmdStanPathTest(unittest.TestCase):
    def test_default_path(self):
        abs_rel_path = os.path.expanduser(
            os.path.join('~', '.cmdstanpy', 'cmdstan')
        )
        self.assertTrue(cmdstan_path().startswith(abs_rel_path))

    def test_non_spaces_location(self):
        good_path = os.path.join(TMPDIR, 'good_dir')
        with TemporaryCopiedFile(good_path) as (p, is_changed):
            self.assertEqual(p, good_path)
            self.assertFalse(is_changed)

        # prepare files for test
        bad_path = os.path.join(TMPDIR, 'bad dir')
        os.makedirs(bad_path, exist_ok=True)
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        stan_bad = os.path.join(bad_path, 'bad name.stan')
        shutil.copy(stan, stan_bad)

        stan_copied = None
        try:
            with TemporaryCopiedFile(stan_bad) as (p, is_changed):
                stan_copied = p
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
        install_dir = os.path.expanduser(os.path.join('~', '.cmdstanpy'))
        install_version = os.path.expanduser(
            os.path.join(install_dir, get_latest_cmdstan(install_dir))
        )
        set_cmdstan_path(install_version)
        self.assertEqual(install_version, cmdstan_path())

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
        folder_name = "".join(
            random.choice(string.ascii_letters) for _ in range(10)
        )
        while os.path.exists(folder_name):
            folder_name = "".join(
                random.choice(string.ascii_letters) for _ in range(10)
            )
        os.makedirs(folder_name)
        path_test = os.path.abspath(folder_name)
        with self.assertRaisesRegex(ValueError, 'no CmdStan binaries'):
            validate_cmdstan_path(path_test)
        shutil.rmtree(folder_name)

    def test_dict_to_file(self):
        file_good = os.path.join(datafiles_path, 'bernoulli_output_1.csv')
        dict_good = {'a': 'A'}
        created_tmp = None
        with MaybeDictToFilePath(file_good, dict_good) as (f1, f2):
            self.assertTrue(os.path.exists(f1))
            self.assertTrue(os.path.exists(f2))
            with open(f2) as f2_d:
                self.assertEqual(json.load(f2_d), dict_good)
            created_tmp = f2
        self.assertTrue(os.path.exists(file_good))
        self.assertFalse(os.path.exists(created_tmp))

        with self.assertRaises(ValueError):
            with MaybeDictToFilePath(123, dict_good) as (f1, f2):
                pass


class ReadStanCsvTest(unittest.TestCase):
    def test_check_sampler_csv_1(self):
        csv_good = os.path.join(datafiles_path, 'bernoulli_output_1.csv')
        dict = check_sampler_csv(csv_good)
        self.assertEqual('bernoulli_model', dict['model'])
        self.assertEqual(10, dict['num_samples'])
        self.assertFalse('save_warmup' in dict)
        self.assertEqual(10, dict['draws'])
        self.assertEqual(8, len(dict['column_names']))

    def test_check_sampler_csv_2(self):
        csv_bad = os.path.join(datafiles_path, 'no_such_file.csv')
        with self.assertRaises(Exception):
            dict = check_sampler_csv(csv_bad)

    def test_check_sampler_csv_3(self):
        csv_bad = os.path.join(datafiles_path, 'output_bad_cols.csv')
        with self.assertRaisesRegex(Exception, '8 items'):
            dict = check_sampler_csv(csv_bad)

    def test_check_sampler_csv_4(self):
        csv_bad = os.path.join(datafiles_path, 'output_bad_rows.csv')
        with self.assertRaisesRegex(Exception, 'found 9'):
            dict = check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_1(self):
        csv_bad = os.path.join(datafiles_path, 'output_bad_metric_1.csv')
        with self.assertRaisesRegex(Exception, 'expecting metric'):
            dict = check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_2(self):
        csv_bad = os.path.join(datafiles_path, 'output_bad_metric_2.csv')
        with self.assertRaisesRegex(Exception, 'invalid stepsize'):
            dict = check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_3(self):
        csv_bad = os.path.join(datafiles_path, 'output_bad_metric_3.csv')
        with self.assertRaisesRegex(
            Exception, 'invalid or missing mass matrix specification'
        ):
            dict = check_sampler_csv(csv_bad)

    def test_check_sampler_csv_metric_4(self):
        csv_bad = os.path.join(datafiles_path, 'output_bad_metric_4.csv')
        with self.assertRaisesRegex(
            Exception, 'invalid or missing mass matrix specification'
        ):
            dict = check_sampler_csv(csv_bad)

    def test_check_sampler_csv_thin(self):
        stan = os.path.join(datafiles_path, 'bernoulli.stan')
        exe = os.path.join(datafiles_path, 'bernoulli' + EXTENSION)
        bern_model = Model(stan_file=stan, exe_file=exe)
        bern_model.compile()

        jdata = os.path.join(datafiles_path, 'bernoulli.data.json')
        bern_fit = bern_model.sample(
            data=jdata,
            chains=1,
            cores=1,
            seed=12345,
            sampling_iters=490,
            warmup_iters=490,
            thin=7,
            max_treedepth=11,
            adapt_delta=0.98,
        )
        csv_file = bern_fit.runset.csv_files[0]
        dict = check_sampler_csv(csv_file)
        self.assertEqual(dict['num_samples'], 490)
        self.assertEqual(dict['thin'], 7)
        self.assertEqual(dict['draws'], 70)
        self.assertEqual(dict['seed'], 12345)
        self.assertEqual(dict['max_depth'], 11)
        self.assertEqual(dict['delta'], 0.98)


class ReadMetricTest(unittest.TestCase):
    def test_metric_json_vec(self):
        metric_file = os.path.join(datafiles_path, 'metric_diag.data.json')
        dims = read_metric(metric_file)
        self.assertEqual(1, len(dims))
        self.assertEqual(3, dims[0])

    def test_metric_json_matrix(self):
        metric_file = os.path.join(datafiles_path, 'metric_dense.data.json')
        dims = read_metric(metric_file)
        self.assertEqual(2, len(dims))
        self.assertEqual(dims[0], dims[1])

    def test_metric_rdump_vec(self):
        metric_file = os.path.join(datafiles_path, 'metric_diag.data.R')
        dims = read_metric(metric_file)
        self.assertEqual(1, len(dims))
        self.assertEqual(3, dims[0])

    def test_metric_rdump_matrix(self):
        metric_file = os.path.join(datafiles_path, 'metric_dense.data.R')
        dims = read_metric(metric_file)
        self.assertEqual(2, len(dims))
        self.assertEqual(dims[0], dims[1])

    def test_metric_json_bad(self):
        metric_file = os.path.join(datafiles_path, 'metric_bad.data.json')
        with self.assertRaisesRegex(
            Exception, 'bad or missing entry "inv_metric"'
        ):
            dims = read_metric(metric_file)

    def test_metric_rdump_bad_1(self):
        metric_file = os.path.join(datafiles_path, 'metric_bad_1.data.R')
        with self.assertRaisesRegex(
            Exception, 'bad or missing entry "inv_metric"'
        ):
            dims = read_metric(metric_file)

    def test_metric_rdump_bad_2(self):
        metric_file = os.path.join(datafiles_path, 'metric_bad_2.data.R')
        with self.assertRaisesRegex(
            Exception, 'bad or missing entry "inv_metric"'
        ):
            dims = read_metric(metric_file)

    def test_metric_missing(self):
        metric_file = os.path.join(datafiles_path, 'no_such_file.json')
        with self.assertRaisesRegex(Exception, 'No such file or directory'):
            dims = read_metric(metric_file)


class WindowsShortPath(unittest.TestCase):
    def test_windows_short_path_directory(self):
        if platform.system() != 'Windows':
            return
        original_path = os.path.join(TMPDIR, 'new path')
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
        original_path = os.path.join(TMPDIR, 'new path', 'my_file.csv')
        os.makedirs(os.path.split(original_path)[0], exist_ok=True)
        assert os.path.exists(os.path.split(original_path)[0])
        assert ' ' in original_path
        assert '.csv' == os.path.splitext(original_path)[1]
        short_path = windows_short_path(original_path)
        assert os.path.exists(os.path.split(short_path)[0])
        assert original_path != short_path
        assert ' ' not in short_path
        assert '.csv' == os.path.splitext(short_path)[1]

    def test_windows_short_path_file_with_space(self):
        """Test that the function doesn't touch filename."""
        if platform.system() != 'Windows':
            return
        original_path = os.path.join(TMPDIR, 'new path', 'my file.csv')
        os.makedirs(os.path.split(original_path)[0], exist_ok=True)
        assert os.path.exists(os.path.split(original_path)[0])
        assert ' ' in original_path
        short_path = windows_short_path(original_path)
        assert os.path.exists(os.path.split(short_path)[0])
        assert original_path != short_path
        assert ' ' in short_path
        assert '.csv' == os.path.splitext(short_path)[1]


class RloadTest(unittest.TestCase):
    def test_rload_metric(self):
        dfile = os.path.join(datafiles_path, 'metric_diag.data.R')
        data_dict = rload(dfile)
        self.assertEqual(data_dict['inv_metric'].shape, (3,))

        dfile = os.path.join(datafiles_path, 'metric_dense.data.R')
        data_dict = rload(dfile)
        self.assertEqual(data_dict['inv_metric'].shape, (3, 3))

    def test_rload_data(self):
        dfile = os.path.join(datafiles_path, 'rdump_test.data.R')
        data_dict = rload(dfile)
        self.assertEqual(data_dict['N'], 128)
        self.assertEqual(data_dict['M'], 2)
        self.assertEqual(data_dict['x'].shape, (128, 2))

    def test_rload_jags_data(self):
        dfile = os.path.join(datafiles_path, 'rdump_jags.data.R')
        data_dict = rload(dfile)
        self.assertEqual(data_dict['N'], 128)
        self.assertEqual(data_dict['M'], 2)
        self.assertEqual(data_dict['y'].shape, (128,))

    def test_rload_wrong_data(self):
        dfile = os.path.join(datafiles_path, 'metric_diag.data.json')
        data_dict = rload(dfile)
        self.assertEqual(data_dict, None)

    def test_rload_bad_data_1(self):
        dfile = os.path.join(datafiles_path, 'rdump_bad_1.data.R')
        with self.assertRaises(ValueError):
            data_dict = rload(dfile)

    def test_rload_bad_data_2(self):
        dfile = os.path.join(datafiles_path, 'rdump_bad_2.data.R')
        with self.assertRaises(ValueError):
            data_dict = rload(dfile)

    def test_rload_bad_data_3(self):
        dfile = os.path.join(datafiles_path, 'rdump_bad_3.data.R')
        with self.assertRaises(ValueError):
            data_dict = rload(dfile)

    def test_roundtrip_metric(self):
        dfile = os.path.join(datafiles_path, 'metric_diag.data.R')
        data_dict_1 = rload(dfile)
        self.assertEqual(data_dict_1['inv_metric'].shape, (3,))

        dfile_tmp = os.path.join(datafiles_path, 'tmp.data.R')
        rdump(dfile_tmp, data_dict_1)
        data_dict_2 = rload(dfile_tmp)

        self.assertTrue('inv_metric' in data_dict_2)
        for i, x in enumerate(data_dict_2['inv_metric']):
            self.assertEqual(x, data_dict_2['inv_metric'][i])

        os.remove(dfile_tmp)

    def test_parse_rdump_value(self):
        s1 = 'structure(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),.Dim=c(2,8))'
        v_s1 = parse_rdump_value(s1)
        self.assertEqual(v_s1.shape, (2, 8))
        self.assertEqual(v_s1[1, 0], 2)
        self.assertEqual(v_s1[0, 7], 15)

        s2 = 'structure(c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16),.Dim=c(1,16))'
        v_s2 = parse_rdump_value(s2)
        self.assertEqual(v_s2.shape, (1, 16))

        s3 = 'structure(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),.Dim = c(8, 2))'
        v_s3 = parse_rdump_value(s3)
        self.assertEqual(v_s3.shape, (8, 2))
        self.assertEqual(v_s3[1, 0], 2)
        self.assertEqual(v_s3[7, 0], 8)
        self.assertEqual(v_s3[0, 1], 9)
        self.assertEqual(v_s3[6, 1], 15)


if __name__ == '__main__':
    unittest.main()
