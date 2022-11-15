"""CmdStan method generate_quantities tests"""

import contextlib
import io
import json
import logging
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest
from test import without_import, check_present

import cmdstanpy.stanfit
from cmdstanpy.stanfit import CmdStanGQ
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.model import CmdStanModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_from_csv_files(caplog: pytest.LogCaptureFixture) -> None:
    # fitted_params sample - list of filenames
    goodfiles_path = os.path.join(DATAFILES_PATH, 'runset-good', 'bern')
    csv_files = []
    for i in range(4):
        csv_files.append('{}-{}.csv'.format(goodfiles_path, i + 1))

    # gq_model
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=csv_files)

    assert bern_gqs.runset._args.method == Method.GENERATE_QUANTITIES
    assert 'CmdStanGQ: model=bernoulli_ppc' in repr(bern_gqs)
    assert 'method=generate_quantities' in repr(bern_gqs)

    assert bern_gqs.runset.chains == 4
    for i in range(bern_gqs.runset.chains):
        assert bern_gqs.runset._retcode(i) == 0
        csv_file = bern_gqs.runset.csv_files[i]
        assert os.path.exists(csv_file)

    column_names = [
        'y_rep[1]',
        'y_rep[2]',
        'y_rep[3]',
        'y_rep[4]',
        'y_rep[5]',
        'y_rep[6]',
        'y_rep[7]',
        'y_rep[8]',
        'y_rep[9]',
        'y_rep[10]',
    ]
    assert bern_gqs.column_names == tuple(column_names)

    # draws()
    assert bern_gqs.draws().shape == (100, 4, 10)
    with caplog.at_level("WARNING"):
        logging.getLogger()
        bern_gqs.draws(inc_warmup=True)
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            "Sample doesn't contain draws from warmup iterations, "
            'rerun sampler with "save_warmup=True".',
        ),
    )

    # draws_pd()
    assert bern_gqs.draws_pd().shape == (400, 10)
    assert (
        bern_gqs.draws_pd(inc_sample=True).shape[1]
        == bern_gqs.mcmc_sample.draws_pd().shape[1]
        + bern_gqs.draws_pd().shape[1]
    )

    assert list(bern_gqs.draws_pd(vars=['y_rep']).columns) == column_names


def test_from_csv_files_bad() -> None:
    # gq model
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    # no filename
    with pytest.raises(ValueError):
        model.generate_quantities(data=jdata, mcmc_sample=[])

    # Stan CSV flles corrupted
    goodfiles_path = os.path.join(
        DATAFILES_PATH, 'runset-bad', 'bad-draws-bern'
    )
    csv_files = []
    for i in range(4):
        csv_files.append('{}-{}.csv'.format(goodfiles_path, i + 1))

    with pytest.raises(Exception, match='Invalid sample from Stan CSV files'):
        model.generate_quantities(data=jdata, mcmc_sample=csv_files)


def test_from_mcmc_sample() -> None:
    # fitted_params sample
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit = bern_model.sample(
        data=jdata,
        chains=4,
        parallel_chains=2,
        seed=12345,
        iter_sampling=100,
    )
    # gq_model
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
    model = CmdStanModel(stan_file=stan)

    bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)

    assert bern_gqs.runset._args.method == Method.GENERATE_QUANTITIES
    assert 'CmdStanGQ: model=bernoulli_ppc' in repr(bern_gqs)
    assert 'method=generate_quantities' in repr(bern_gqs)
    assert bern_gqs.runset.chains == 4
    for i in range(bern_gqs.runset.chains):
        assert bern_gqs.runset._retcode(i) == 0
        csv_file = bern_gqs.runset.csv_files[i]
        assert os.path.exists(csv_file)


def test_from_mcmc_sample_draws() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit = bern_model.sample(
        data=jdata,
        chains=4,
        parallel_chains=2,
        seed=12345,
        iter_sampling=100,
    )
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
    model = CmdStanModel(stan_file=stan)

    bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)

    assert bern_gqs.draws_pd().shape == (400, 10)
    assert (
        bern_gqs.draws_pd(inc_sample=True).shape[1]
        == bern_gqs.mcmc_sample.draws_pd().shape[1]
        + bern_gqs.draws_pd().shape[1]
    )
    row1_sample_pd = bern_fit.draws_pd().iloc[0]
    row1_gqs_pd = bern_gqs.draws_pd().iloc[0]
    np.testing.assert_array_equal(
        pd.concat((row1_sample_pd, row1_gqs_pd), axis=0).values,
        bern_gqs.draws_pd(inc_sample=True).iloc[0].values,
    )
    # draws_xr
    xr_data = bern_gqs.draws_xr()
    assert xr_data.y_rep.dims == ('chain', 'draw', 'y_rep_dim_0')
    assert xr_data.y_rep.values.shape == (4, 100, 10)

    xr_var = bern_gqs.draws_xr(vars='y_rep')
    assert xr_var.y_rep.dims == ('chain', 'draw', 'y_rep_dim_0')
    assert xr_var.y_rep.values.shape == (4, 100, 10)

    xr_var = bern_gqs.draws_xr(vars=['y_rep'])
    assert xr_var.y_rep.dims == ('chain', 'draw', 'y_rep_dim_0')
    assert xr_var.y_rep.values.shape == (4, 100, 10)

    xr_data_plus = bern_gqs.draws_xr(inc_sample=True)
    assert xr_data_plus.y_rep.dims == ('chain', 'draw', 'y_rep_dim_0')
    assert xr_data_plus.y_rep.values.shape == (4, 100, 10)
    assert xr_data_plus.theta.dims == ('chain', 'draw')
    assert xr_data_plus.theta.values.shape == (4, 100)


def test_from_mcmc_sample_variables() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit = bern_model.sample(
        data=jdata,
        chains=4,
        parallel_chains=2,
        seed=12345,
        iter_sampling=100,
    )
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
    model = CmdStanModel(stan_file=stan)

    bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)

    theta = bern_gqs.stan_variable(var='theta')
    assert theta.shape == (400,)
    y_rep = bern_gqs.stan_variable(var='y_rep')
    assert y_rep.shape == (400, 10)
    with pytest.raises(ValueError):
        bern_gqs.stan_variable(var='eta')
    with pytest.raises(ValueError):
        bern_gqs.stan_variable(var='lp__')

    vars_dict = bern_gqs.stan_variables()
    var_names = list(
        bern_gqs.mcmc_sample.metadata.stan_vars_cols.keys()
    ) + list(bern_gqs.metadata.stan_vars_cols.keys())
    assert set(var_names) == set(list(vars_dict.keys()))


def test_save_warmup(caplog: pytest.LogCaptureFixture) -> None:
    # fitted_params sample
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit = bern_model.sample(
        data=jdata,
        chains=4,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        save_warmup=True,
    )
    # gq_model
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
    model = CmdStanModel(stan_file=stan)

    with caplog.at_level(level=logging.WARNING):
        logging.getLogger()
        bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            'Sample contains saved warmup draws which will be used to '
            'generate additional quantities of interest.',
        ),
    )
    assert bern_gqs.draws().shape == (100, 4, 10)
    assert bern_gqs.draws(concat_chains=False, inc_warmup=False).shape == (
        100,
        4,
        10,
    )
    assert bern_gqs.draws(concat_chains=False, inc_warmup=True).shape == (
        200,
        4,
        10,
    )
    assert bern_gqs.draws(concat_chains=True, inc_warmup=False).shape == (
        400,
        10,
    )
    assert bern_gqs.draws(concat_chains=True, inc_warmup=True).shape == (
        800,
        10,
    )

    assert bern_gqs.draws_pd().shape == (400, 10)
    assert bern_gqs.draws_pd(inc_warmup=False).shape == (400, 10)
    assert bern_gqs.draws_pd(inc_warmup=True).shape == (800, 10)
    assert bern_gqs.draws_pd(vars=['y_rep'], inc_warmup=False).shape == (
        400,
        10,
    )
    assert bern_gqs.draws_pd(vars='y_rep', inc_warmup=False).shape == (400, 10)

    theta = bern_gqs.stan_variable(var='theta')
    assert theta.shape == (400,)
    y_rep = bern_gqs.stan_variable(var='y_rep')
    assert y_rep.shape == (400, 10)
    with pytest.raises(ValueError):
        bern_gqs.stan_variable(var='eta')
    with pytest.raises(ValueError):
        bern_gqs.stan_variable(var='lp__')

    vars_dict = bern_gqs.stan_variables()
    var_names = list(
        bern_gqs.mcmc_sample.metadata.stan_vars_cols.keys()
    ) + list(bern_gqs.metadata.stan_vars_cols.keys())
    assert set(var_names) == set(list(vars_dict.keys()))

    xr_data = bern_gqs.draws_xr()
    assert xr_data.y_rep.dims == ('chain', 'draw', 'y_rep_dim_0')
    assert xr_data.y_rep.values.shape == (4, 100, 10)

    xr_data_plus = bern_gqs.draws_xr(inc_sample=True)
    assert xr_data_plus.y_rep.dims == ('chain', 'draw', 'y_rep_dim_0')
    assert xr_data_plus.y_rep.values.shape == (4, 100, 10)
    assert xr_data_plus.theta.dims == ('chain', 'draw')
    assert xr_data_plus.theta.values.shape == (4, 100)

    xr_data_plus = bern_gqs.draws_xr(vars='theta', inc_sample=True)
    assert xr_data_plus.theta.dims == ('chain', 'draw')
    assert xr_data_plus.theta.values.shape == (4, 100)

    xr_data_plus = bern_gqs.draws_xr(inc_sample=True, inc_warmup=True)
    assert xr_data_plus.y_rep.dims == ('chain', 'draw', 'y_rep_dim_0')
    assert xr_data_plus.y_rep.values.shape == (4, 200, 10)
    assert xr_data_plus.theta.dims == ('chain', 'draw')
    assert xr_data_plus.theta.values.shape == (4, 200)


def test_sample_plus_quantities_dedup() -> None:
    # fitted_params - model GQ block: y_rep is PPC of theta
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit = model.sample(
        data=jdata,
        chains=4,
        parallel_chains=2,
        seed=12345,
        iter_sampling=100,
    )
    # gq_model - y_rep[n] == y[n]
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc_dup.stan')
    model = CmdStanModel(stan_file=stan)
    bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)
    # check that models have different y_rep values
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            bern_fit.stan_variable(var='y_rep'),
            bern_gqs.stan_variable(var='y_rep'),
        )
    # check that stan_variable returns values from gq model
    with open(jdata) as fd:
        bern_data = json.load(fd)
    y_rep = bern_gqs.stan_variable(var='y_rep')
    for i in range(10):
        assert y_rep[0, i] == bern_data['y'][i]


def test_no_xarray() -> None:
    with without_import('xarray', cmdstanpy.stanfit.gq):
        with pytest.raises(ImportError):
            # if this fails the testing framework is the problem
            import xarray as _  # noqa

        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        bern_model = CmdStanModel(stan_file=stan)
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_fit = bern_model.sample(
            data=jdata,
            chains=4,
            parallel_chains=2,
            seed=12345,
            iter_sampling=100,
        )
        stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
        model = CmdStanModel(stan_file=stan)

        bern_gqs = model.generate_quantities(data=jdata, mcmc_sample=bern_fit)

        with pytest.raises(RuntimeError):
            bern_gqs.draws_xr()


def test_single_row_csv() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit = bern_model.sample(
        data=jdata,
        chains=1,
        seed=12345,
        iter_sampling=1,
    )
    stan = os.path.join(DATAFILES_PATH, 'matrix_var.stan')
    model = CmdStanModel(stan_file=stan)
    gqs = model.generate_quantities(mcmc_sample=bern_fit)
    z_as_ndarray = gqs.stan_variable(var="z")
    assert z_as_ndarray.shape == (1, 4, 3)  # flattens chains
    z_as_xr = gqs.draws_xr(vars="z")
    assert z_as_xr.z.data.shape == (1, 1, 4, 3)  # keeps chains
    for i in range(4):
        for j in range(3):
            assert int(z_as_ndarray[0, i, j]) == i + 1
            assert int(z_as_xr.z.data[0, 0, i, j]) == i + 1


def test_show_console() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit = bern_model.sample(
        data=jdata,
        chains=4,
        parallel_chains=2,
        seed=12345,
        iter_sampling=100,
    )
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_ppc.stan')
    model = CmdStanModel(stan_file=stan)

    sys_stdout = io.StringIO()
    with contextlib.redirect_stdout(sys_stdout):
        model.generate_quantities(
            data=jdata,
            mcmc_sample=bern_fit,
            show_console=True,
        )
    console = sys_stdout.getvalue()
    assert 'Chain [1] method = generate' in console
    assert 'Chain [2] method = generate' in console
    assert 'Chain [3] method = generate' in console
    assert 'Chain [4] method = generate' in console


def test_complex_output() -> None:
    stan_bern = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model_bern = CmdStanModel(stan_file=stan_bern)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    fit_sampling = model_bern.sample(chains=1, iter_sampling=10, data=jdata)

    stan = os.path.join(DATAFILES_PATH, 'complex_var.stan')
    model = CmdStanModel(stan_file=stan)
    fit = model.generate_quantities(mcmc_sample=fit_sampling)

    assert fit.stan_variable('zs').shape == (10, 2, 3)
    assert fit.stan_variable('z')[0] == 3 + 4j

    np.testing.assert_allclose(
        fit.stan_variable('zs')[0], np.array([[3, 4j, 5], [1j, 2j, 3j]])
    )

    # make sure the name 'imag' isn't magic
    assert fit.stan_variable('imag').shape == (10, 2)

    assert "zs_dim_2" not in fit.draws_xr()
    # getting a raw scalar out of xarray is heavy
    assert fit.draws_xr().z.isel(chain=0, draw=1).data[()] == 3 + 4j


def test_attrs() -> None:
    stan_bern = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model_bern = CmdStanModel(stan_file=stan_bern)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    fit_sampling = model_bern.sample(chains=1, iter_sampling=10, data=jdata)

    stan = os.path.join(DATAFILES_PATH, 'named_output.stan')
    model = CmdStanModel(stan_file=stan)
    fit = model.generate_quantities(data=jdata, mcmc_sample=fit_sampling)

    assert fit.a[0] == 4.5
    assert fit.b.shape == (10, 3)
    assert fit.theta.shape == (10,)

    fit.draws()
    assert fit.stan_variable('draws')[0] == 0

    with pytest.raises(AttributeError, match='Unknown variable name:'):
        dummy = fit.c


def test_timeout() -> None:
    stan = os.path.join(DATAFILES_PATH, 'timeout.stan')
    timeout_model = CmdStanModel(stan_file=stan)
    fit = timeout_model.sample(data={'loop': 0}, chains=1, iter_sampling=10)
    with pytest.raises(TimeoutError):
        timeout_model.generate_quantities(
            timeout=0.1, mcmc_sample=fit, data={'loop': 1}
        )


@pytest.mark.order(before="test_no_xarray")
def test_serialization() -> None:
    stan_bern = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model_bern = CmdStanModel(stan_file=stan_bern)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    fit_sampling = model_bern.sample(chains=1, iter_sampling=10, data=jdata)

    stan = os.path.join(DATAFILES_PATH, 'named_output.stan')
    model = CmdStanModel(stan_file=stan)
    fit1 = model.generate_quantities(data=jdata, mcmc_sample=fit_sampling)

    dumped = pickle.dumps(fit1)
    shutil.rmtree(fit1.runset._output_dir)
    fit2: CmdStanGQ = pickle.loads(dumped)
    variables1 = fit1.stan_variables()
    variables2 = fit2.stan_variables()
    assert set(variables1) == set(variables2)
    for key, value1 in variables1.items():
        np.testing.assert_array_equal(value1, variables2[key])
