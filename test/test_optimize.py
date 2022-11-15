"""CmdStan method optimize tests"""

import contextlib
import io
import json
import logging
import os
import pickle
import shutil

import numpy as np

from cmdstanpy.cmdstan_args import CmdStanArgs, OptimizeArgs
from cmdstanpy.model import CmdStanModel
from cmdstanpy.stanfit import CmdStanMLE, RunSet, from_csv
import pytest
from test import check_present

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_instantiate() -> None:
    stan = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = CmdStanModel(stan_file=stan)
    no_data = {}
    args = OptimizeArgs(algorithm='Newton')
    cmdstan_args = CmdStanArgs(
        model_name=model.name,
        model_exe=model.exe_file,
        chain_ids=None,
        data=no_data,
        method_args=args,
    )
    runset = RunSet(args=cmdstan_args, chains=1)
    runset._csv_files = [
        os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv')
    ]
    mle = CmdStanMLE(runset)
    assert 'CmdStanMLE: model=rosenbrock' in repr(mle)
    assert 'method=optimize' in repr(mle)
    assert mle.column_names == ('lp__', 'x', 'y')
    np.testing.assert_almost_equal(mle.optimized_params_dict['x'], 1, decimal=3)
    np.testing.assert_almost_equal(mle.optimized_params_dict['y'], 1, decimal=3)


def test_instantiate_from_csvfiles() -> None:
    csvfiles_path = os.path.join(
        DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv'
    )
    mle = from_csv(path=csvfiles_path)
    assert 'CmdStanMLE: model=rosenbrock' in repr(mle)
    assert 'method=optimize' in repr(mle)
    mle.column_names == ('lp__', 'x', 'y')
    np.testing.assert_almost_equal(mle.optimized_params_dict['x'], 1, decimal=3)
    np.testing.assert_almost_equal(mle.optimized_params_dict['y'], 1, decimal=3)


def test_instantiate_from_csvfiles_save_iterations() -> None:
    csvfiles_path = os.path.join(
        DATAFILES_PATH, 'optimize', 'eight_schools_mle_iters.csv'
    )
    mle = from_csv(path=csvfiles_path)
    assert 'CmdStanMLE: model=eight_schools' in repr(mle)
    assert 'method=optimize' in repr(mle)
    assert mle.column_names == (
        'lp__',
        'mu',
        'theta[1]',
        'theta[2]',
        'theta[3]',
        'theta[4]',
        'theta[5]',
        'theta[6]',
        'theta[7]',
        'theta[8]',
        'tau',
    )
    np.testing.assert_almost_equal(
        mle.optimized_params_dict['mu'], 1.06401, decimal=3
    )
    np.testing.assert_almost_equal(
        mle.optimized_params_dict['theta[1]'], 1.06401, decimal=3
    )
    assert mle.optimized_iterations_np.shape == (173, 11)


def test_rosenbrock(caplog: pytest.LogCaptureFixture) -> None:
    stan = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = CmdStanModel(stan_file=stan)
    mle = model.optimize(algorithm='LBFGS')
    assert 'CmdStanMLE: model=rosenbrock' in repr(mle)
    assert 'method=optimize' in repr(mle)
    assert mle.converged
    assert mle.column_names == ('lp__', 'x', 'y')
    np.testing.assert_almost_equal(mle.stan_variable('x'), 1, decimal=3)
    np.testing.assert_almost_equal(mle.stan_variable('y'), 1, decimal=3)
    np.testing.assert_almost_equal(
        mle.optimized_params_pd['x'][0], 1, decimal=3
    )
    np.testing.assert_almost_equal(mle.optimized_params_np[1], 1, decimal=3)
    np.testing.assert_almost_equal(mle.optimized_params_dict['x'], 1, decimal=3)
    with caplog.at_level(logging.WARNING):
        assert mle.optimized_iterations_np is None
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            'Intermediate iterations not saved to CSV output file. '
            'Rerun the optimize method with "save_iterations=True".',
        ),
    )
    with caplog.at_level(logging.WARNING):
        assert mle.optimized_iterations_pd is None
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            'Intermediate iterations not saved to CSV output file. '
            'Rerun the optimize method with "save_iterations=True".',
        ),
    )

    mle = model.optimize(algorithm='LBFGS', save_iterations=True, seed=12345)
    assert mle.converged
    np.testing.assert_almost_equal(mle.stan_variable('x'), 1, decimal=3)
    np.testing.assert_almost_equal(mle.stan_variable('y'), 1, decimal=3)

    assert mle.optimized_params_np.shape == (3,)
    np.testing.assert_almost_equal(mle.optimized_params_np[1], 1, decimal=3)
    np.testing.assert_almost_equal(
        mle.optimized_params_pd['x'][0], 1, decimal=3
    )
    np.testing.assert_almost_equal(mle.optimized_params_dict['x'], 1, decimal=3)

    last_iter = mle.optimized_iterations_np.shape[0] - 1
    assert (
        mle.optimized_iterations_np[0, 1]
        != mle.optimized_iterations_np[last_iter, 1]
    )
    for i in range(3):
        assert (
            mle.optimized_params_np[i]
            == mle.optimized_iterations_np[last_iter, i]
        )


def test_eight_schools(caplog: pytest.LogCaptureFixture) -> None:
    stan = os.path.join(DATAFILES_PATH, 'eight_schools.stan')
    rdata = os.path.join(DATAFILES_PATH, 'eight_schools.data.R')
    model = CmdStanModel(stan_file=stan)
    with pytest.raises(RuntimeError):
        model.optimize(data=rdata, algorithm='LBFGS')

    mle = model.optimize(data=rdata, algorithm='LBFGS', require_converged=False)
    assert 'CmdStanMLE: model=eight_schools' in repr(mle)
    assert 'method=optimize' in repr(mle)
    assert not mle.converged
    with caplog.at_level(logging.WARNING):
        assert mle.optimized_params_pd.shape == (1, 11)
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            'Invalid estimate, optimization failed to converge.',
        ),
    )
    with caplog.at_level(logging.WARNING):
        mle.stan_variable('tau')
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            'Invalid estimate, optimization failed to converge.',
        ),
    )


def test_variable_bern() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    bern_mle = bern_model.optimize(
        data=jdata,
        seed=1239812093,
        algorithm='LBFGS',
        init_alpha=0.001,
        iter=100,
        tol_obj=1e-12,
        tol_rel_obj=1e4,
        tol_grad=1e-8,
        tol_rel_grad=1e7,
        tol_param=1e-8,
        history_size=5,
    )
    assert 1 == len(bern_mle.metadata.stan_vars_dims)
    assert 'theta' in bern_mle.metadata.stan_vars_dims
    assert bern_mle.metadata.stan_vars_dims['theta'] == ()
    theta = bern_mle.stan_variable(var='theta')
    assert isinstance(theta, float)
    with pytest.raises(ValueError):
        bern_mle.stan_variable(var='eta')
    with pytest.raises(ValueError):
        bern_mle.stan_variable(var='lp__')


def test_variables_3d() -> None:
    stan = os.path.join(DATAFILES_PATH, 'multidim_vars.stan')
    jdata = os.path.join(DATAFILES_PATH, 'logistic.data.R')
    multidim_model = CmdStanModel(stan_file=stan)
    multidim_mle = multidim_model.optimize(
        data=jdata,
        seed=1239812093,
        algorithm='LBFGS',
        init_alpha=0.001,
        iter=100,
        tol_obj=1e-12,
        tol_rel_obj=1e4,
        tol_grad=1e-8,
        tol_rel_grad=1e7,
        tol_param=1e-8,
        history_size=5,
    )
    assert 3 == len(multidim_mle.metadata.stan_vars_dims)
    assert 'y_rep' in multidim_mle.metadata.stan_vars_dims
    assert multidim_mle.metadata.stan_vars_dims['y_rep'] == (5, 4, 3)
    var_y_rep = multidim_mle.stan_variable(var='y_rep')
    assert var_y_rep.shape == (5, 4, 3)
    var_beta = multidim_mle.stan_variable(var='beta')
    assert var_beta.shape == (2,)  # 1-element tuple
    var_frac_60 = multidim_mle.stan_variable(var='frac_60')
    assert isinstance(var_frac_60, float)
    vars = multidim_mle.stan_variables()
    assert len(vars) == len(multidim_mle.metadata.stan_vars_dims)
    assert 'y_rep' in vars
    assert vars['y_rep'].shape == (5, 4, 3)
    assert 'beta' in vars
    assert vars['beta'].shape == (2,)
    assert 'frac_60' in vars
    assert isinstance(vars['frac_60'], float)

    multidim_mle_iters = multidim_model.optimize(
        data=jdata,
        seed=1239812093,
        algorithm='LBFGS',
        init_alpha=0.001,
        iter=100,
        tol_obj=1e-12,
        tol_rel_obj=1e4,
        tol_grad=1e-8,
        tol_rel_grad=1e7,
        tol_param=1e-8,
        history_size=5,
        save_iterations=True,
    )
    vars_iters = multidim_mle_iters.stan_variables(inc_iterations=True)
    assert len(vars_iters) == len(multidim_mle_iters.metadata.stan_vars_dims)
    assert 'y_rep' in vars_iters
    assert vars_iters['y_rep'].shape == (8, 5, 4, 3)
    assert 'beta' in vars_iters
    assert vars_iters['beta'].shape == (8, 2)
    assert 'frac_60' in vars_iters
    assert vars_iters['frac_60'].shape == (8,)


def test_optimize_good() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    jinit = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')
    mle = model.optimize(
        data=jdata,
        seed=1239812093,
        inits=jinit,
        algorithm='LBFGS',
        init_alpha=0.001,
        iter=100,
        tol_obj=1e-12,
        tol_rel_obj=1e4,
        tol_grad=1e-8,
        tol_rel_grad=1e7,
        tol_param=1e-8,
        history_size=5,
    )

    # test numpy output
    assert isinstance(mle.optimized_params_np, np.ndarray)
    np.testing.assert_almost_equal(mle.optimized_params_np[0], -5, decimal=2)
    np.testing.assert_almost_equal(mle.optimized_params_np[1], 0.2, decimal=3)

    # test pandas output
    assert mle.optimized_params_np[0] == mle.optimized_params_pd['lp__'][0]
    assert mle.optimized_params_np[1] == mle.optimized_params_pd['theta'][0]

    # test dict output
    assert mle.optimized_params_np[0] == mle.optimized_params_dict['lp__']
    assert mle.optimized_params_np[1] == mle.optimized_params_dict['theta']


def test_negative_parameter_values() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    jinit = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')

    with pytest.raises(ValueError, match='must be greater than'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            tol_obj=-1.0,
        )

    with pytest.raises(ValueError, match='must be greater than'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            tol_rel_obj=-1.0,
        )

    with pytest.raises(ValueError, match='must be greater than'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            tol_grad=-1.0,
        )

    with pytest.raises(ValueError, match='must be greater than'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            tol_rel_grad=-1.0,
        )

    with pytest.raises(ValueError, match='must be greater than'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            tol_param=-1.0,
        )

    with pytest.raises(ValueError, match='must be greater than'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            history_size=-1,
        )


def test_parameters_are_floats() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    jinit = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')

    with pytest.raises(ValueError, match='must be type of float'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            tol_obj="rabbit",
        )

    with pytest.raises(ValueError, match='must be type of float'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            tol_rel_obj="rabbit",
        )

    with pytest.raises(ValueError, match='must be type of float'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            tol_grad="rabbit",
        )

    with pytest.raises(ValueError, match='must be type of float'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            tol_rel_grad="rabbit",
        )

    with pytest.raises(ValueError, match='must be type of float'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            tol_param="rabbit",
        )

    with pytest.raises(ValueError, match='must be type of int'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='LBFGS',
            history_size="rabbit",
        )


def test_parameters_and_optimizer_compatible() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    jinit = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')

    with pytest.raises(ValueError, match='bfgs or lbfgs'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='Newton',
            tol_obj=1,
        )

    with pytest.raises(ValueError, match='bfgs or lbfgs'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='Newton',
            tol_rel_obj=1,
        )

    with pytest.raises(ValueError, match='bfgs or lbfgs'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='Newton',
            tol_grad=1,
        )

    with pytest.raises(ValueError, match='bfgs or lbfgs'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='Newton',
            tol_rel_grad=1,
        )

    with pytest.raises(ValueError, match='bfgs or lbfgs'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            tol_rel_grad=1,
        )

    with pytest.raises(ValueError, match='bfgs or lbfgs'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='Newton',
            tol_param=1,
        )

    with pytest.raises(ValueError, match='lbfgs'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='Newton',
            history_size=1,
        )

    with pytest.raises(ValueError, match='lbfgs'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            algorithm='BFGS',
            history_size=1,
        )

    with pytest.raises(ValueError, match='lbfgs'):
        model.optimize(
            data=jdata,
            seed=1239812093,
            inits=jinit,
            history_size=1,
        )


def test_optimize_good_dict() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model = CmdStanModel(stan_file=stan)
    with open(os.path.join(DATAFILES_PATH, 'bernoulli.data.json')) as fd:
        data = json.load(fd)
    with open(os.path.join(DATAFILES_PATH, 'bernoulli.init.json')) as fd:
        init = json.load(fd)
    mle = model.optimize(
        data=data,
        seed=1239812093,
        inits=init,
        algorithm='BFGS',
        init_alpha=0.001,
        iter=100,
    )
    # test numpy output
    np.testing.assert_almost_equal(mle.optimized_params_np[0], -5, decimal=2)
    np.testing.assert_almost_equal(mle.optimized_params_np[1], 0.2, decimal=3)


def test_optimize_rosenbrock() -> None:
    stan = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    rose_model = CmdStanModel(stan_file=stan)
    mle = rose_model.optimize(seed=1239812093, inits=None, algorithm='BFGS')
    assert mle.column_names == ('lp__', 'x', 'y')
    np.testing.assert_almost_equal(mle.optimized_params_dict['x'], 1, decimal=3)
    np.testing.assert_almost_equal(mle.optimized_params_dict['y'], 1, decimal=3)


def test_optimize_no_data() -> None:
    stan = os.path.join(DATAFILES_PATH, 'optimize', 'no_data.stan')
    rose_model = CmdStanModel(stan_file=stan)
    mle = rose_model.optimize(seed=1239812093)
    assert mle.column_names == ('lp__', 'a')
    np.testing.assert_almost_equal(mle.optimized_params_dict['a'], 0, decimal=3)


def test_optimize_bad() -> None:
    stan = os.path.join(DATAFILES_PATH, 'optimize', 'exponential_boundary.stan')
    exp_bound_model = CmdStanModel(stan_file=stan)
    no_data = {}
    with pytest.raises(RuntimeError, match='Error during optimization'):
        exp_bound_model.optimize(
            data=no_data, seed=1239812093, inits=None, algorithm='BFGS'
        )


def test_single_row_csv() -> None:
    stan = os.path.join(DATAFILES_PATH, 'matrix_var.stan')
    model = CmdStanModel(stan_file=stan)
    mle = model.optimize()
    assert isinstance(mle.stan_variable('theta'), float)
    z_as_ndarray = mle.stan_variable(var="z")
    assert z_as_ndarray.shape == (4, 3)
    for i in range(4):
        for j in range(3):
            assert int(z_as_ndarray[i, j]) == i + 1


def test_show_console() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    sys_stdout = io.StringIO()
    with contextlib.redirect_stdout(sys_stdout):
        bern_model.optimize(
            data=jdata,
            show_console=True,
        )
    console = sys_stdout.getvalue()
    assert 'Chain [1] method = optimize' in console


def test_exe_only() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    exe_only = os.path.join(DATAFILES_PATH, 'exe_only')
    shutil.copyfile(bern_model.exe_file, exe_only)
    os.chmod(exe_only, 0o755)

    bern2_model = CmdStanModel(exe_file=exe_only)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    mle = bern2_model.optimize(data=jdata)
    assert mle.optimized_params_np[0] == mle.optimized_params_dict['lp__']
    assert mle.optimized_params_np[1] == mle.optimized_params_dict['theta']


def test_complex_output() -> None:
    stan = os.path.join(DATAFILES_PATH, 'complex_var.stan')
    model = CmdStanModel(stan_file=stan)
    fit = model.optimize()

    assert fit.stan_variable('zs').shape == (2, 3)
    assert fit.stan_variable('z') == 3 + 4j

    np.testing.assert_allclose(
        fit.stan_variable('zs'), np.array([[3, 4j, 5], [1j, 2j, 3j]])
    )

    # make sure the name 'imag' isn't magic
    assert fit.stan_variable('imag').shape == (2,)


def test_attrs() -> None:
    stan = os.path.join(DATAFILES_PATH, 'named_output.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    fit = model.optimize(data=jdata)

    assert fit.a == 4.5
    assert fit.b.shape == (3,)
    assert isinstance(fit.theta, float)

    assert fit.stan_variable('thin') == 3.5

    assert isinstance(fit.optimized_params_np, np.ndarray)
    assert fit.stan_variable('optimized_params_np') == 0

    with pytest.raises(AttributeError, match='Unknown variable name:'):
        dummy = fit.c


def test_timeout() -> None:
    stan = os.path.join(DATAFILES_PATH, 'timeout.stan')
    timeout_model = CmdStanModel(stan_file=stan)
    with pytest.raises(TimeoutError):
        timeout_model.optimize(data={'loop': 1}, timeout=0.1)


def test_serialization() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    jinit = os.path.join(DATAFILES_PATH, 'bernoulli.init.json')
    mle1 = model.optimize(
        data=jdata,
        seed=1239812093,
        inits=jinit,
        algorithm='LBFGS',
        init_alpha=0.001,
        iter=100,
        tol_obj=1e-12,
        tol_rel_obj=1e4,
        tol_grad=1e-8,
        tol_rel_grad=1e7,
        tol_param=1e-8,
        history_size=5,
    )
    dumped = pickle.dumps(mle1)
    shutil.rmtree(mle1.runset._output_dir)
    mle2: CmdStanMLE = pickle.loads(dumped)
    np.testing.assert_array_equal(
        mle1.optimized_params_np, mle2.optimized_params_np
    )
