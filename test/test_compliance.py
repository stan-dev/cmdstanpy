"""Testing for things like pickleability, deep copying"""

import copy
import pathlib
import pickle
import unittest

import cmdstanpy

DATAFILES_PATH = pathlib.Path(__file__).parent.resolve() / 'data'


class SampleCompliance(unittest.TestCase):
    def test_sample_pickle_ability(self):
        csvfiles_path = DATAFILES_PATH / 'lotka-volterra.csv'
        fit = cmdstanpy.from_csv(path=csvfiles_path)
        keys = fit.stan_variables().keys()
        pickled = pickle.dumps(fit)
        del fit
        unpickled = pickle.loads(pickled)
        self.assertSequenceEqual(keys, unpickled.stan_variables().keys())

    def test_sample_copy_ability(self):
        csvfiles_path = DATAFILES_PATH / 'lotka-volterra.csv'
        fit = cmdstanpy.from_csv(path=csvfiles_path)
        fit2 = copy.deepcopy(fit)
        self.assertSequenceEqual(
            fit.stan_variables().keys(), fit2.stan_variables().keys()
        )


class OptimizeCompliance(unittest.TestCase):
    def test_optimize_pickle_ability(self):
        csvfiles_path = DATAFILES_PATH / 'optimize' / 'rosenbrock_mle.csv'
        fit = cmdstanpy.from_csv(path=csvfiles_path)
        keys = fit.stan_variables().keys()
        pickled = pickle.dumps(fit)
        del fit
        unpickled = pickle.loads(pickled)
        self.assertSequenceEqual(keys, unpickled.stan_variables().keys())

    def test_optimize_copy_ability(self):
        csvfiles_path = DATAFILES_PATH / 'optimize' / 'rosenbrock_mle.csv'
        fit = cmdstanpy.from_csv(path=csvfiles_path)
        fit2 = copy.deepcopy(fit)
        self.assertSequenceEqual(
            fit.stan_variables().keys(), fit2.stan_variables().keys()
        )


class VariationalCompliance(unittest.TestCase):
    def test_variational_pickle_ability(self):
        csvfiles_path = DATAFILES_PATH / 'variational'
        fit = cmdstanpy.from_csv(path=csvfiles_path)
        keys = fit.stan_variables().keys()
        pickled = pickle.dumps(fit)
        del fit
        unpickled = pickle.loads(pickled)
        self.assertSequenceEqual(keys, unpickled.stan_variables().keys())

    def test_variational_copy_ability(self):
        csvfiles_path = DATAFILES_PATH / 'variational'
        fit = cmdstanpy.from_csv(path=csvfiles_path)
        fit2 = copy.deepcopy(fit)
        self.assertSequenceEqual(
            fit.stan_variables().keys(), fit2.stan_variables().keys()
        )
