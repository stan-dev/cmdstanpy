import os
import os.path
import unittest
from cmdstanpy.utils import *

class ReadStanCsvTest(unittest.TestCase):

    def test_scan_csv_1(self):
        csv_good = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                    "cmdstanpy", "test", "files", "bernoulli_output_1.csv"))
        dict = scan_stan_csv(csv_good)
        self.assertEqual("bernoulli_model", dict['model'])
        self.assertEqual('10',dict['num_samples'])
        self.assertFalse('save_warmup' in dict)
        self.assertEqual(10,dict['draws'])
        self.assertEqual(8,len(dict['col_headers']))

    def test_scan_csv_2(self):
        csv_bad = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                        "cmdstanpy", "test", "files", "no_such_file.csv"))
        with self.assertRaises(Exception):
            dict = scan_stan_csv(csv_bad)

    def test_scan_csv_3(self):
        csv_bad = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                        "cmdstanpy", "test", "files", "output_bad_cols.csv"))
        with self.assertRaisesRegexp(Exception, "8 columns"):
            dict = scan_stan_csv(csv_bad)

    def test_scan_csv_4(self):
        csv_bad = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                        "cmdstanpy", "test", "files", "output_bad_rows.csv"))
        with self.assertRaisesRegexp(Exception, "10 draws"):
            dict = scan_stan_csv(csv_bad)

if __name__ == '__main__':
    unittest.main()
