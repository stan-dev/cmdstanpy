"""
    Tests for the Pathfinder method.
"""

import contextlib
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

import cmdstanpy
from cmdstanpy.stanfit import from_csv

HERE = Path(__file__).parent
DATAFILES_PATH = HERE / 'data'


def test_pathfinder_outputs():
    stan = DATAFILES_PATH / 'bernoulli.stan'
    bern_model = cmdstanpy.CmdStanModel(stan_file=stan)
    jdata = str(DATAFILES_PATH / 'bernoulli.data.json')

    draws = 1232
    pathfinder = bern_model.pathfinder(
        data=jdata,
        draws=draws,
    )
    assert 'lp__' in pathfinder.column_names
    assert 'lp_approx__' in pathfinder.column_names
    assert 'theta' in pathfinder.column_names

    theta = pathfinder.theta
    assert theta.shape == (draws,)
    assert 0.23 < theta.mean() < 0.27

    assert pathfinder.is_resampled

    assert pathfinder.draws().shape == (draws, 3)


def test_pathfinder_from_csv():
    pathfinder_outputs = DATAFILES_PATH / 'pathfinder'
    pathfinder = from_csv(pathfinder_outputs)
    assert isinstance(pathfinder, cmdstanpy.CmdStanPathfinder)
    assert 'lp__' in pathfinder.column_names
    assert 'lp_approx__' in pathfinder.column_names
    assert 'theta' in pathfinder.column_names
    theta = pathfinder.theta
    assert theta.shape == (1000,)
    assert 0.23 < theta.mean() < 0.27


def test_single_pathfinder():
    stan = DATAFILES_PATH / 'bernoulli.stan'
    bern_model = cmdstanpy.CmdStanModel(stan_file=stan)
    jdata = str(DATAFILES_PATH / 'bernoulli.data.json')

    draws = 234
    pathfinder = bern_model.pathfinder(
        data=jdata,
        num_paths=1,
        draws=draws,
    )

    assert not pathfinder.is_resampled

    theta = pathfinder.theta
    assert theta.shape == (draws,)

    pathfinder2 = bern_model.pathfinder(
        data=jdata,
        num_paths=1,
        num_single_draws=draws,
    )

    theta2 = pathfinder2.theta
    assert theta2.shape == (draws,)

    with pytest.raises(ValueError, match="Cannot specify both"):
        bern_model.pathfinder(
            data=jdata,
            num_paths=1,
            draws=draws,
            num_single_draws=draws // 2,
        )


def test_pathfinder_create_inits():
    stan = DATAFILES_PATH / 'bernoulli.stan'
    bern_model = cmdstanpy.CmdStanModel(stan_file=stan)
    jdata = str(DATAFILES_PATH / 'bernoulli.data.json')

    pathfinder = bern_model.pathfinder(data=jdata)

    inits = pathfinder.create_inits()
    assert isinstance(inits, list)
    assert len(inits) == 4
    assert isinstance(inits[0], dict)
    assert 'theta' in inits[0]

    inits_10 = pathfinder.create_inits(chains=10)
    assert isinstance(inits_10, list)
    assert len(inits_10) == 10

    inits_1 = pathfinder.create_inits(chains=1)
    assert isinstance(inits_1, dict)
    assert 'theta' in inits_1
    assert len(inits_1) == 1

    seeded = pathfinder.create_inits(seed=1234)
    seeded2 = pathfinder.create_inits(seed=1234)
    assert all(
        init1['theta'] == init2['theta']
        for init1, init2 in zip(seeded, seeded2)
    )


def test_pathfinder_init_sampling():
    logistic_stan = DATAFILES_PATH / 'logistic.stan'
    logistic_model = cmdstanpy.CmdStanModel(stan_file=logistic_stan)
    logistic_data = str(DATAFILES_PATH / 'logistic.data.R')
    pathfinder = logistic_model.pathfinder(data=logistic_data)

    fit = logistic_model.sample(
        data=logistic_data,
        inits=pathfinder.create_inits(),
    )

    assert fit.chains == 4
    assert fit.draws().shape == (1000, 4, 9)


def test_inits_for_pathfinder():
    stan = DATAFILES_PATH / 'bernoulli.stan'
    bern_model = cmdstanpy.CmdStanModel(stan_file=stan)
    jdata = str(DATAFILES_PATH / 'bernoulli.data.json')
    bern_model.pathfinder(
        jdata, inits=[{"theta": 0.1}, {"theta": 0.9}], num_paths=2
    )

    # second path is initialized too large!
    with contextlib.redirect_stdout(StringIO()) as captured:
        bern_model.pathfinder(
            jdata,
            inits=[{"theta": 0.1}, {"theta": 1.1}],
            num_paths=2,
            show_console=True,
        )

    assert "Bounded variable is 1.1" in captured.getvalue()


def test_pathfinder_no_psis():
    stan = DATAFILES_PATH / 'bernoulli.stan'
    bern_model = cmdstanpy.CmdStanModel(stan_file=stan)
    jdata = str(DATAFILES_PATH / 'bernoulli.data.json')

    pathfinder = bern_model.pathfinder(data=jdata, psis_resample=False)

    assert not pathfinder.is_resampled
    assert pathfinder.draws().shape == (4000, 3)


def test_pathfinder_no_lp_calc():
    stan = DATAFILES_PATH / 'bernoulli.stan'
    bern_model = cmdstanpy.CmdStanModel(stan_file=stan)
    jdata = str(DATAFILES_PATH / 'bernoulli.data.json')

    pathfinder = bern_model.pathfinder(data=jdata, calculate_lp=False)

    assert not pathfinder.is_resampled
    assert pathfinder.draws().shape == (4000, 3)
    n_lp_nan = np.sum(np.isnan(pathfinder.method_variables()['lp__']))
    assert n_lp_nan < 4000  # some lp still calculated during pathfinder
    assert n_lp_nan > 3000  # but most are not


def test_pathfinder_threads():
    stan = DATAFILES_PATH / 'bernoulli.stan'
    bern_model = cmdstanpy.CmdStanModel(stan_file=stan)
    jdata = str(DATAFILES_PATH / 'bernoulli.data.json')

    bern_model.pathfinder(data=jdata, num_threads=1)

    with pytest.raises(ValueError, match="STAN_THREADS"):
        bern_model.pathfinder(data=jdata, num_threads=4)

    bern_model = cmdstanpy.CmdStanModel(
        stan_file=stan, cpp_options={'STAN_THREADS': True}, force_compile=True
    )
    pathfinder = bern_model.pathfinder(data=jdata, num_threads=4)
    assert pathfinder.draws().shape == (1000, 3)
