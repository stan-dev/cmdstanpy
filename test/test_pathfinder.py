"""
    Tests for the Pathfinder method.
"""

from pathlib import Path

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
