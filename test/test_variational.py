"""CmdStan method variational tests"""

import contextlib
import io
import logging
import os
import pickle
import shutil
from math import fabs
from test import check_present

import numpy as np
import pytest

from cmdstanpy.model import CmdStanModel
from cmdstanpy.stanfit import CmdStanVB, from_output_files

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_instantiate_from_output_files() -> None:
    csv_path = os.path.join(
        DATAFILES_PATH, 'variational', 'eta_should_be_big_vb.csv'
    )
    variational = from_output_files(csv_path)
    assert isinstance(variational, CmdStanVB)
    assert 'CmdStanVB: model=eta_should_be_big' in repr(variational)
    assert 'method=variational' in repr(variational)
    assert variational.column_names == (
        'lp__',
        'log_p__',
        'log_g__',
        'mu[1]',
        'mu[2]',
    )
    assert variational.variational_sample.shape == (1000, 5)


def test_variables() -> None:
    # pylint: disable=C0103
    stan = os.path.join(DATAFILES_PATH, 'variational', 'eta_should_be_big.stan')
    model = CmdStanModel(stan_file=stan)
    variational = model.variational(algorithm='meanfield', seed=999999)
    assert variational.column_names == (
        'lp__',
        'log_p__',
        'log_g__',
        'mu[1]',
        'mu[2]',
    )
    assert len(variational.metadata.stan_vars) == 1
    assert 'mu' in variational.metadata.stan_vars
    assert variational.metadata.stan_vars['mu'].dimensions == (2,)
    mu = variational.stan_variable(var='mu', mean=True)
    assert mu.shape == (2,)
    mu = variational.stan_variable(var='mu', mean=False)
    assert mu.shape == (1000, 2)
    with pytest.raises(ValueError):
        variational.stan_variable(var='eta')
    with pytest.raises(ValueError):
        variational.stan_variable(var='lp__')


def test_variables_3d() -> None:
    # construct fit using existing sampler output
    stan = os.path.join(DATAFILES_PATH, 'multidim_vars.stan')
    jdata = os.path.join(DATAFILES_PATH, 'logistic.data.R')
    multidim_model = CmdStanModel(stan_file=stan)
    multidim_variational = multidim_model.variational(
        data=jdata,
        seed=1239812093,
        algorithm='meanfield',
    )
    assert len(multidim_variational.metadata.stan_vars) == 3
    assert 'y_rep' in multidim_variational.metadata.stan_vars
    assert multidim_variational.metadata.stan_vars['y_rep'].dimensions == (
        5,
        4,
        3,
    )
    var_y_rep = multidim_variational.stan_variable(var='y_rep', mean=False)
    assert var_y_rep.shape == (1000, 5, 4, 3)
    var_beta = multidim_variational.stan_variable(var='beta', mean=False)
    assert var_beta.shape == (1000, 2)
    var_frac_60 = multidim_variational.stan_variable(var='frac_60', mean=False)
    assert var_frac_60.shape == (1000,)
    vars = multidim_variational.stan_variables(mean=False)
    assert len(vars) == len(multidim_variational.metadata.stan_vars)
    assert 'y_rep' in vars
    assert vars['y_rep'].shape == (1000, 5, 4, 3)
    assert 'beta' in vars
    assert vars['beta'].shape == (1000, 2)
    assert 'frac_60' in vars
    assert vars['frac_60'].shape == (1000,)


def test_variational_good() -> None:
    stan = os.path.join(DATAFILES_PATH, 'variational', 'eta_should_be_big.stan')
    model = CmdStanModel(stan_file=stan)
    variational = model.variational(algorithm='meanfield', seed=999999)
    assert variational.column_names == (
        'lp__',
        'log_p__',
        'log_g__',
        'mu[1]',
        'mu[2]',
    )
    # fixed seed, id=1 by default will give known output values
    assert variational.eta == 10
    np.testing.assert_almost_equal(
        variational.variational_params_dict['mu[1]'], 302.142, decimal=2
    )
    np.testing.assert_almost_equal(
        variational.variational_params_dict['mu[2]'], 361.005, decimal=2
    )
    np.testing.assert_almost_equal(
        variational.variational_params_np[0],
        variational.variational_params_pd['lp__'][0],
    )
    assert (
        variational.variational_params_np[3]
        == variational.variational_params_dict['mu[1]']
    )
    np.testing.assert_almost_equal(
        variational.variational_params_np[4],
        variational.variational_params_dict['mu[2]'],
    )
    assert variational.variational_sample.shape == (1000, 5)


def test_variational_eta_small() -> None:
    stan = os.path.join(
        DATAFILES_PATH, 'variational', 'eta_should_be_small.stan'
    )
    model = CmdStanModel(stan_file=stan)
    variational = model.variational(algorithm='meanfield', seed=1234)
    assert variational.column_names == (
        'lp__',
        'log_p__',
        'log_g__',
        'mu[1]',
        'mu[2]',
    )
    np.testing.assert_almost_equal(
        fabs(variational.variational_params_dict['mu[1]']), 0.08, decimal=1
    )
    np.testing.assert_almost_equal(
        fabs(variational.variational_params_dict['mu[2]']), 0.09, decimal=1
    )


def test_variational_eta_fail(caplog: pytest.LogCaptureFixture) -> None:
    stan = os.path.join(DATAFILES_PATH, 'variational', 'eta_should_fail.stan')
    model = CmdStanModel(stan_file=stan)
    with pytest.raises(
        RuntimeError,
        match=r'algorithm may not have converged\.\n.*require_converged',
    ):
        model.variational(algorithm='meanfield', seed=12345)

    with caplog.at_level(logging.WARNING):
        model.variational(
            algorithm='meanfield', seed=12345, require_converged=False
        )
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            'The algorithm may not have converged.\n'
            'Proceeding because require_converged is set to False',
        ),
    )


def test_single_row_csv() -> None:
    stan = os.path.join(DATAFILES_PATH, 'matrix_var.stan')
    model = CmdStanModel(stan_file=stan)
    # testing data parsing, allow non-convergence
    vb_fit = model.variational(require_converged=False, seed=12345)

    assert vb_fit.stan_variable('theta', mean=False).shape == (1000,)
    z_as_ndarray = vb_fit.stan_variable(var="z", mean=False)
    assert z_as_ndarray.shape == (1000, 4, 3)
    for i in range(4):
        for j in range(3):
            assert int(z_as_ndarray[0, i, j]) == i + 1


def test_show_console() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    sys_stdout = io.StringIO()
    with contextlib.redirect_stdout(sys_stdout):
        # testing data parsing, allow non-convergence
        bern_model.variational(
            data=jdata,
            show_console=True,
            require_converged=False,
            seed=12345,
        )
    console = sys_stdout.getvalue()
    assert 'Chain [1] method = variational' in console


def test_exe_only() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    exe_only = os.path.join(DATAFILES_PATH, 'exe_only')
    shutil.copyfile(bern_model.exe_file, exe_only)
    os.chmod(exe_only, 0o755)

    bern2_model = CmdStanModel(exe_file=exe_only)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    variational = bern2_model.variational(
        data=jdata,
        require_converged=False,
        seed=12345,
        algorithm='meanfield',
    )
    assert variational.variational_sample.shape == (1000, 4)


def test_complex_output() -> None:
    stan = os.path.join(DATAFILES_PATH, 'complex_var.stan')
    model = CmdStanModel(stan_file=stan)
    fit = model.variational(
        require_converged=False,
        seed=12345,
        algorithm='meanfield',
    )

    assert fit.stan_variable('zs', mean=False).shape == (1000, 2, 3)
    np.testing.assert_equal(fit.stan_variable("z", mean=True), 3 + 4j)
    np.testing.assert_equal(fit.z, np.repeat(3 + 4j, 1000))

    np.testing.assert_allclose(
        fit.stan_variable('zs', mean=False)[0],
        np.array([[3, 4j, 5], [1j, 2j, 3j]]),
    )

    # make sure the name 'imag' isn't magic
    assert fit.stan_variable('imag', mean=False).shape == (
        1000,
        2,
    )


def test_attrs() -> None:
    stan = os.path.join(DATAFILES_PATH, 'named_output.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    fit = model.variational(
        data=jdata,
        require_converged=False,
        seed=12345,
        algorithm='meanfield',
    )

    np.testing.assert_equal(fit.a, np.repeat(4.5, 1000))
    assert fit.b.shape == (1000, 3)
    assert fit.theta.shape == (1000,)

    assert fit.stan_variable('thin', mean=True) == 3.5

    assert isinstance(fit.variational_params_np, np.ndarray)
    assert fit.stan_variable('variational_params_np', mean=True) == 0

    with pytest.raises(AttributeError, match='Unknown variable name:'):
        dummy = fit.c


def test_timeout() -> None:
    stan = os.path.join(DATAFILES_PATH, 'timeout.stan')
    timeout_model = CmdStanModel(stan_file=stan)
    with pytest.raises(TimeoutError):
        timeout_model.variational(timeout=0.1, data={'loop': 1})


def test_serialization() -> None:
    stan = os.path.join(DATAFILES_PATH, 'variational', 'eta_should_be_big.stan')
    model = CmdStanModel(stan_file=stan)
    variational1 = model.variational(algorithm='meanfield', seed=999999)
    dumped = pickle.dumps(variational1)
    shutil.rmtree(os.path.dirname(variational1.csv_file))
    variational2: CmdStanVB = pickle.loads(dumped)
    np.testing.assert_array_equal(
        variational1.variational_sample, variational2.variational_sample
    )
    assert (
        variational1.variational_params_dict
        == variational2.variational_params_dict
    )


def test_variational_create_inits() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    vb = bern_model.variational(data=jdata, seed=11235, draws=10)

    inits = vb.create_inits()
    assert isinstance(inits, list)
    assert len(inits) == 4
    assert isinstance(inits[0], dict)
    assert 'theta' in inits[0]

    inits_10 = vb.create_inits(seed=0, chains=10)
    assert isinstance(inits_10, list)
    assert len(inits_10) == 10
    np.testing.assert_array_equal(
        np.sort(np.asarray([init['theta'] for init in inits_10])),
        np.sort(vb.theta),
    )

    inits_1 = vb.create_inits(chains=1)
    assert isinstance(inits_1, dict)
    assert 'theta' in inits_1
    assert len(inits_1) == 1

    seeded = vb.create_inits(seed=1234)
    seeded2 = vb.create_inits(seed=1234)
    assert isinstance(seeded, list)
    assert isinstance(seeded2, list)
    assert len(seeded) == len(seeded2) == 4
    assert all(
        init1['theta'] == init2['theta']
        for init1, init2 in zip(seeded, seeded2)
    )


def test_variational_init_sampling() -> None:
    stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
    logistic_model = CmdStanModel(stan_file=stan)
    logistic_data = os.path.join(DATAFILES_PATH, 'logistic.data.R')

    vb = logistic_model.variational(data=logistic_data, seed=11235)
    inits = vb.create_inits()

    fit = logistic_model.sample(data=logistic_data, inits=inits)

    assert fit.chains == 4
    assert fit.draws().shape == (1000, 4, 9)
