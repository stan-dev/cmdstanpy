"""Tests for the Laplace sampling method."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import cmdstanpy
from cmdstanpy.stanfit import from_csv, from_output_files

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_laplace_from_opt_csv() -> None:
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    fit = model.laplace_sample(
        data={},
        mode=os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv'),
        jacobian=False,
    )
    assert 'x' in fit.stan_variables()
    assert 'y' in fit.stan_variables()
    assert isinstance(fit.mode, cmdstanpy.CmdStanMLE)


def test_laplace_from_output_files() -> None:
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    with TemporaryDirectory() as directory:
        fit = model.laplace_sample(
            data={},
            seed=1234,
            output_dir=directory,
        )
        fit2 = from_output_files(fit.csv_file)
        assert isinstance(fit2, cmdstanpy.CmdStanLaplace)
        assert 'x' in fit2.stan_variables()
        assert 'y' in fit2.stan_variables()
        assert isinstance(fit2.mode, cmdstanpy.CmdStanMLE)

        with pytest.deprecated_call(match='use from_output_files instead'):
            fit_from_csv = from_csv([fit.csv_file])
        assert isinstance(fit_from_csv, cmdstanpy.CmdStanLaplace)
        assert isinstance(fit_from_csv.mode, cmdstanpy.CmdStanMLE)

        # An explicit Laplace manifest includes the Laplace and mode CSV/config.
        fit3 = from_output_files(
            [
                os.path.join(directory, filename)
                for filename in os.listdir(directory)
                if filename.endswith((".csv", "_config.json"))
                and "profile" not in filename
            ]
        )
        assert isinstance(fit3, cmdstanpy.CmdStanLaplace)
        assert 'x' in fit3.stan_variables()
        assert 'y' in fit3.stan_variables()
        assert isinstance(fit3.mode, cmdstanpy.CmdStanMLE)


def test_laplace_save_output_files(tmp_path: Path) -> None:
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    fit = model.laplace_sample(data={}, seed=1234)

    destination = tmp_path / 'saved'
    fit.save_output_files(os.fspath(destination))

    rebuilt = from_output_files(destination)
    assert isinstance(rebuilt, cmdstanpy.CmdStanLaplace)
    assert isinstance(rebuilt.mode, cmdstanpy.CmdStanMLE)
    assert Path(rebuilt.mode.csv_file).parent == destination


def test_laplace_missing_mode_files(tmp_path: Path) -> None:
    # the laplace fixture records a stale mode path; without the mode's
    # output files next to the laplace CSV either, loading must fail
    import shutil

    for name in ('rosenbrock_laplace.csv', 'rosenbrock_laplace_config.json'):
        shutil.copy(
            os.path.join(DATAFILES_PATH, 'laplace', name),
            os.path.join(tmp_path, name),
        )
    with pytest.raises(ValueError, match=r'optimization mode'):
        from_output_files(os.path.join(tmp_path, 'rosenbrock_laplace.csv'))


def test_laplace_runs_opt() -> None:
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    fit1 = model.laplace_sample(data={}, seed=1234, opt_args={'iter': 1003})
    assert isinstance(fit1.mode, cmdstanpy.CmdStanMLE)


def test_laplace_bad_jacobian_mismatch() -> None:
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    with pytest.raises(ValueError):
        model.laplace_sample(
            data={},
            mode=os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv'),
            jacobian=True,
        )


def test_laplace_bad_two_modes() -> None:
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    with pytest.raises(ValueError):
        model.laplace_sample(
            data={},
            mode=os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv'),
            opt_args={'iter': 1003},
            jacobian=False,
        )


def test_laplace_outputs() -> None:
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    fit = model.laplace_sample(data={}, seed=1234, draws=123)

    variables = fit.stan_variables()
    assert 'x' in variables
    assert 'y' in variables
    assert variables['x'].shape == (123,)

    np.testing.assert_array_equal(variables['x'], fit.x)

    fit_pd = fit.draws_pd()
    assert 'x' in fit_pd.columns
    assert 'y' in fit_pd.columns
    assert fit_pd['x'].shape == (123,)


def test_laplace_create_inits() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = cmdstanpy.CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    laplace = bern_model.laplace_sample(data=jdata)

    inits = laplace.create_inits()
    assert isinstance(inits, list)
    assert len(inits) == 4
    assert isinstance(inits[0], dict)
    assert 'theta' in inits[0]

    inits_10 = laplace.create_inits(chains=10)
    assert isinstance(inits_10, list)
    assert len(inits_10) == 10

    inits_1 = laplace.create_inits(chains=1)
    assert isinstance(inits_1, dict)
    assert 'theta' in inits_1
    assert len(inits_1) == 1

    seeded = laplace.create_inits(seed=1234)
    seeded2 = laplace.create_inits(seed=1234)
    assert isinstance(seeded, list)
    assert isinstance(seeded2, list)
    assert all(
        init1['theta'] == init2['theta']
        for init1, init2 in zip(seeded, seeded2)
    )


def test_laplace_init_sampling() -> None:
    stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
    logistic_model = cmdstanpy.CmdStanModel(stan_file=stan)
    logistic_data = os.path.join(DATAFILES_PATH, 'logistic.data.R')

    laplace = logistic_model.laplace_sample(data=logistic_data)
    inits = laplace.create_inits()

    fit = logistic_model.sample(data=logistic_data, inits=inits)

    assert fit.chains == 4
    assert fit.draws().shape == (1000, 4, 9)
