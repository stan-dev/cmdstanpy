"""Tests for the Laplace sampling method."""

import os

import cmdstanpy

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_laplace_from_csv():
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    fit = model.laplace_sample(
        data={},
        mode=os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv'),
    )
    assert 'x' in fit.stan_variables()
    assert 'y' in fit.stan_variables()
    assert isinstance(fit.mode, cmdstanpy.CmdStanMLE)


def test_laplace_runs_opt():
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    fit1 = model.laplace_sample(data={}, seed=1234)
    assert isinstance(fit1.mode, cmdstanpy.CmdStanMLE)

    assert fit1.mode.metadata.cmdstan_config['seed'] == 1234
    assert fit1._metadata.cmdstan_config['seed'] == 1234
