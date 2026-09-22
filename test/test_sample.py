"""CmdStan method sample tests"""

import contextlib
import io
import json
import logging
import os
import pickle
import platform
import re
import shutil
import stat
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from test import check_present, raises_nested, without_import
from time import time

import numpy as np
import pytest

import cmdstanpy.stanfit
from cmdstanpy import _TMPDIR
from cmdstanpy.model import CmdStanModel
from cmdstanpy.stanfit import CmdStanMCMC, from_output_files

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')
GOODFILES_PATH = os.path.join(DATAFILES_PATH, 'runset-good')
BADFILES_PATH = os.path.join(DATAFILES_PATH, 'runset-bad')

# metadata should make this unnecessary
SAMPLER_STATE = [
    'lp__',
    'accept_stat__',
    'stepsize__',
    'treedepth__',
    'n_leapfrog__',
    'divergent__',
    'energy__',
]
# metadata should make this unnecessary
BERNOULLI_COLS = SAMPLER_STATE + ['theta']


@pytest.mark.parametrize(
    'stanfile',
    [
        'bernoulli.stan',
        'bernoulli with space in name.stan',
        'path with space/bernoulli_path_with_space.stan',
        'path~with~tilde/bernoulli_path_with_tilde.stan',
    ],
)
def test_bernoulli_good(stanfile: str) -> None:
    stan = os.path.join(DATAFILES_PATH, stanfile)
    bern_model = CmdStanModel(stan_file=stan, force_compile=True)

    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=200,
        iter_sampling=100,
        show_progress=False,
    )
    assert 'CmdStanMCMC: model=' in repr(bern_fit)
    assert 'method=sample' in repr(bern_fit)

    assert bern_fit.config.method_config.method == 'sample'

    for i in range(bern_fit.chains):
        csv_file = bern_fit.csv_files[i]
        # NB: This will fail if STAN_THREADS is enabled
        # due to sampling only producing 1 stdout file in that case
        stdout_file = bern_fit.stdout_files[i]  # type: ignore
        assert os.path.exists(csv_file)
        assert os.path.exists(stdout_file)

    assert bern_fit.chains == 2
    assert bern_fit.thin == 1
    assert bern_fit.num_draws_warmup == 200
    assert bern_fit.num_draws_sampling == 100
    assert bern_fit.column_names == tuple(BERNOULLI_COLS)

    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))
    assert bern_fit.metric_type == 'diag_e'
    assert bern_fit.step_size is not None
    assert bern_fit.step_size.shape == (2,)
    assert bern_fit.inv_metric is not None
    assert bern_fit.inv_metric.shape == (2, 1)

    assert bern_fit.draws(concat_chains=True).shape == (
        200,
        len(BERNOULLI_COLS),
    )

    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=200,
        iter_sampling=100,
        metric='dense_e',
        show_progress=False,
    )
    assert 'CmdStanMCMC: model=' in repr(bern_fit)
    assert 'method=sample' in repr(bern_fit)

    assert bern_fit.config.method_config.method == 'sample'

    for i in range(bern_fit.chains):
        csv_file = bern_fit.csv_files[i]
        stdout_file = bern_fit.stdout_files[i]  # type: ignore
        assert os.path.exists(csv_file)
        assert os.path.exists(stdout_file)

    assert bern_fit.chains == 2
    assert bern_fit.num_draws_sampling == 100
    assert bern_fit.column_names == tuple(BERNOULLI_COLS)

    bern_sample = bern_fit.draws()
    assert bern_sample.shape == (100, 2, len(BERNOULLI_COLS))
    assert bern_fit.metric_type == 'dense_e'
    assert bern_fit.step_size is not None
    assert bern_fit.step_size.shape == (2,)
    assert bern_fit.inv_metric is not None
    assert bern_fit.inv_metric.shape == (2, 1, 1)

    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        output_dir=DATAFILES_PATH,
        show_progress=False,
    )
    for i in range(bern_fit.chains):
        csv_file = bern_fit.csv_files[i]
        stdout_file = bern_fit.stdout_files[i]  # type: ignore
        assert os.path.exists(csv_file)
        assert os.path.exists(stdout_file)
    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))
    for attr in (  # cleanup datafile_path dir
        'csv_files',
        'stdout_files',
        'config_files',
        'metric_files',
    ):
        files = getattr(bern_fit, attr)
        if files is None:
            continue
        for f in files:
            if os.path.exists(f):
                os.remove(f)
    rdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.R')
    bern_fit = bern_model.sample(
        data=rdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        show_progress=False,
    )
    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))

    data_dict = {'N': 10, 'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    bern_fit = bern_model.sample(
        data=data_dict,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        show_progress=False,
    )
    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))

    np_scalr_10 = np.int32(10)
    data_dict = {'N': np_scalr_10, 'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    bern_fit = bern_model.sample(
        data=data_dict,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        show_progress=False,
    )
    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))


@pytest.mark.parametrize("stanfile", ["bernoulli.stan"])
def test_bernoulli_unit_e(stanfile: str) -> None:
    stan = os.path.join(DATAFILES_PATH, stanfile)
    bern_model = CmdStanModel(stan_file=stan)

    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        metric='unit_e',
        show_progress=False,
    )
    assert bern_fit.metric_type == 'unit_e'
    assert bern_fit.inv_metric is None
    assert bern_fit.step_size is not None
    assert bern_fit.step_size.shape == (2,)

    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))


def test_init_types() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        inits=1.1,
        show_progress=False,
    )

    bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        inits=1,
        show_progress=False,
    )

    # Save init to json
    inits_path1 = os.path.join(_TMPDIR, 'inits_test_1.json')
    with open(inits_path1, 'w') as fd:
        json.dump({'theta': 0.1}, fd)
    inits_path2 = os.path.join(_TMPDIR, 'inits_test_2.json')
    with open(inits_path2, 'w') as fd:
        json.dump({'theta': 0.9}, fd)

    bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        inits=inits_path1,
        show_progress=False,
    )

    bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        inits=[inits_path1, inits_path2],
        show_progress=False,
        force_one_process_per_chain=False,
    )

    bern_model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        inits=[inits_path1, inits_path2],
        show_progress=False,
        force_one_process_per_chain=True,
    )

    with pytest.raises(ValueError):
        bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_warmup=100,
            iter_sampling=100,
            inits=-1,
        )

    # test that inits are actually used by having a bad one
    init_1 = {"theta": 0.2}
    init_2 = {"theta": 4.0}
    with pytest.raises(RuntimeError):
        bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            inits=[init_1, init_2],
            iter_warmup=100,
            iter_sampling=100,
            force_one_process_per_chain=True,
            show_progress=False,
        )
    # https://github.com/stan-dev/cmdstan/pull/1191
    with pytest.raises(RuntimeError):
        bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            inits=[init_1, init_2],
            iter_warmup=100,
            iter_sampling=100,
            force_one_process_per_chain=False,
            show_progress=False,
        )


def test_bernoulli_bad() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)

    with pytest.raises(RuntimeError, match='variable does not exist'):
        bern_model.sample()

    with pytest.raises(RuntimeError, match='variable does not exist'):
        bern_model.sample(data={'foo': 1})

    if platform.system() != 'Windows':
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        dirname1 = 'tmp1' + str(time())
        os.mkdir(dirname1, mode=644)
        dirname2 = 'tmp2' + str(time())
        path = os.path.join(dirname1, dirname2)
        with pytest.raises(ValueError, match='Invalid path for output files'):
            bern_model.sample(data=jdata, chains=1, output_dir=path)
        os.rmdir(dirname1)


def test_multi_proc_1(caplog: pytest.LogCaptureFixture) -> None:
    logistic_stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
    logistic_model = CmdStanModel(stan_file=logistic_stan)
    logistic_data = os.path.join(DATAFILES_PATH, 'logistic.data.R')

    with caplog.at_level(logging.INFO):
        logging.getLogger()
        logistic_model.sample(
            data=logistic_data,
            chains=2,
            parallel_chains=1,
            iter_sampling=200,
            iter_warmup=200,
            show_console=True,
        )
    check_present(
        caplog,
        ('cmdstanpy', 'INFO', 'Chain [1] done processing'),
        ('cmdstanpy', 'INFO', 'Chain [2] start processing'),
    )


def test_multi_proc_2(caplog: pytest.LogCaptureFixture) -> None:
    logistic_stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
    logistic_model = CmdStanModel(stan_file=logistic_stan)
    logistic_data = os.path.join(DATAFILES_PATH, 'logistic.data.R')

    with caplog.at_level(logging.INFO):
        logging.getLogger()
        logistic_model.sample(
            data=logistic_data,
            chains=4,
            parallel_chains=2,
            iter_sampling=200,
            iter_warmup=200,
            show_console=True,
        )
    if cpu_count() >= 4:
        # finish chains 1, 2 before starting chains 3, 4
        check_present(
            caplog,
            ('cmdstanpy', 'INFO', 'Chain [1] done processing'),
            ('cmdstanpy', 'INFO', 'Chain [4] start processing'),
        )
    if cpu_count() >= 4:
        with caplog.at_level(logging.INFO):
            logging.getLogger()
            logistic_model.sample(
                data=logistic_data,
                chains=4,
                parallel_chains=4,
                iter_sampling=200,
                iter_warmup=200,
                show_console=True,
            )
            check_present(
                caplog,
                ('cmdstanpy', 'INFO', 'Chain [4] start processing'),
                ('cmdstanpy', 'INFO', 'Chain [1] done processing'),
            )


def test_num_threads_msgs(caplog: pytest.LogCaptureFixture) -> None:
    logistic_stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
    logistic_model = CmdStanModel(stan_file=logistic_stan)
    logistic_data = os.path.join(DATAFILES_PATH, 'logistic.data.R')

    with caplog.at_level(logging.DEBUG):
        logging.getLogger()
        logistic_model.sample(
            data=logistic_data,
            chains=1,
            parallel_chains=1,
            threads_per_chain=7,
            iter_sampling=200,
            iter_warmup=200,
            show_progress=False,
        )
    check_present(
        caplog, ('cmdstanpy', 'DEBUG', 'running CmdStan, num_threads: 7')
    )
    with caplog.at_level(logging.DEBUG):
        logging.getLogger()
        logistic_model.sample(
            data=logistic_data,
            chains=7,
            parallel_chains=1,
            threads_per_chain=5,
            iter_sampling=200,
            iter_warmup=200,
            show_progress=False,
        )
    check_present(
        caplog, ('cmdstanpy', 'DEBUG', 'running CmdStan, num_threads: 5')
    )
    with caplog.at_level(logging.INFO):
        logging.getLogger()
        logistic_model.sample(
            data=logistic_data,
            chains=1,
            parallel_chains=7,
            threads_per_chain=5,
            iter_sampling=200,
            iter_warmup=200,
            show_progress=False,
        )
    check_present(
        caplog,
        (
            'cmdstanpy',
            'INFO',
            'Requested 7 parallel_chains but only 1 required, '
            'will run all chains in parallel.',
        ),
    )


def test_multi_proc_threads(caplog: pytest.LogCaptureFixture) -> None:

    logistic_stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
    logistic_model = CmdStanModel(
        stan_file=logistic_stan,
        cpp_options={'STAN_THREADS': 'TRUE'},
        force_compile=True,
    )
    info_dict = logistic_model.exe_info()
    assert info_dict is not None
    assert 'STAN_THREADS' in info_dict
    assert info_dict['STAN_THREADS'] == 'true'

    logistic_data = os.path.join(DATAFILES_PATH, 'logistic.data.R')
    with caplog.at_level(logging.DEBUG):
        logging.getLogger()
        logistic_model.sample(
            data=logistic_data,
            chains=4,
            parallel_chains=4,
            threads_per_chain=5,
            iter_sampling=200,
            iter_warmup=200,
            show_progress=False,
        )
    check_present(
        caplog, ('cmdstanpy', 'DEBUG', 'running CmdStan, num_threads: 20')
    )


def test_multi_proc_err_msgs() -> None:
    logistic_stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
    logistic_model = CmdStanModel(stan_file=logistic_stan)
    logistic_data = os.path.join(DATAFILES_PATH, 'logistic.data.R')

    with pytest.raises(
        ValueError, match='parallel_chains must be a positive integer'
    ):
        logistic_model.sample(data=logistic_data, chains=4, parallel_chains=-4)
    with pytest.raises(
        ValueError, match='threads_per_chain must be a positive integer'
    ):
        logistic_model.sample(
            data=logistic_data, chains=4, threads_per_chain=-4
        )


def test_fixed_param_good() -> None:
    stan = os.path.join(DATAFILES_PATH, 'datagen_poisson_glm.stan')
    datagen_model = CmdStanModel(stan_file=stan)
    datagen_fit = datagen_model.sample(
        seed=12345, chains=1, iter_sampling=100, fixed_param=True
    )
    assert datagen_fit.config.method_config.method == 'sample'
    assert datagen_fit.metric_type is None
    assert datagen_fit.inv_metric is None
    assert datagen_fit.step_size is None
    assert datagen_fit.divergences is None
    assert datagen_fit.max_treedepths is None

    for i in range(datagen_fit.chains):
        csv_file = datagen_fit.csv_files[i]
        stdout_file = datagen_fit.stdout_files[i]  # type: ignore
        assert os.path.exists(csv_file)
        assert os.path.exists(stdout_file)

    assert datagen_fit.chains == 1

    column_names = [
        'lp__',
        'accept_stat__',
        'N',
        'y_sim[1]',
        'y_sim[2]',
        'y_sim[3]',
        'y_sim[4]',
        'y_sim[5]',
        'y_sim[6]',
        'y_sim[7]',
        'y_sim[8]',
        'y_sim[9]',
        'y_sim[10]',
        'y_sim[11]',
        'y_sim[12]',
        'y_sim[13]',
        'y_sim[14]',
        'y_sim[15]',
        'y_sim[16]',
        'y_sim[17]',
        'y_sim[18]',
        'y_sim[19]',
        'y_sim[20]',
        'x_sim[1]',
        'x_sim[2]',
        'x_sim[3]',
        'x_sim[4]',
        'x_sim[5]',
        'x_sim[6]',
        'x_sim[7]',
        'x_sim[8]',
        'x_sim[9]',
        'x_sim[10]',
        'x_sim[11]',
        'x_sim[12]',
        'x_sim[13]',
        'x_sim[14]',
        'x_sim[15]',
        'x_sim[16]',
        'x_sim[17]',
        'x_sim[18]',
        'x_sim[19]',
        'x_sim[20]',
        'pop_sim[1]',
        'pop_sim[2]',
        'pop_sim[3]',
        'pop_sim[4]',
        'pop_sim[5]',
        'pop_sim[6]',
        'pop_sim[7]',
        'pop_sim[8]',
        'pop_sim[9]',
        'pop_sim[10]',
        'pop_sim[11]',
        'pop_sim[12]',
        'pop_sim[13]',
        'pop_sim[14]',
        'pop_sim[15]',
        'pop_sim[16]',
        'pop_sim[17]',
        'pop_sim[18]',
        'pop_sim[19]',
        'pop_sim[20]',
        'alpha_sim',
        'beta_sim',
        'eta[1]',
        'eta[2]',
        'eta[3]',
        'eta[4]',
        'eta[5]',
        'eta[6]',
        'eta[7]',
        'eta[8]',
        'eta[9]',
        'eta[10]',
        'eta[11]',
        'eta[12]',
        'eta[13]',
        'eta[14]',
        'eta[15]',
        'eta[16]',
        'eta[17]',
        'eta[18]',
        'eta[19]',
        'eta[20]',
    ]
    assert datagen_fit.column_names == tuple(column_names)
    assert datagen_fit.num_draws_sampling == 100
    assert datagen_fit.draws().shape == (100, 1, len(column_names))
    assert datagen_fit.inv_metric is None
    assert datagen_fit.metric_type is None
    assert datagen_fit.step_size is None


def test_sample_no_params() -> None:
    stan = os.path.join(DATAFILES_PATH, 'datagen_poisson_glm.stan')
    datagen_model = CmdStanModel(stan_file=stan)
    datagen_fit = datagen_model.sample(iter_sampling=100, show_progress=False)
    summary = datagen_fit.summary()

    assert 'lp__' in list(summary.index)
    assert datagen_fit.step_size is not None
    assert np.isnan(datagen_fit.step_size).all()

    exe_only = os.path.join(DATAFILES_PATH, 'exe_only')
    shutil.copyfile(datagen_model.exe_file, exe_only)
    os.chmod(exe_only, 0o755)
    datagen2_model = CmdStanModel(exe_file=exe_only)
    datagen2_fit = datagen2_model.sample(iter_sampling=200, show_console=True)
    assert datagen2_fit.chains == 4
    summary = datagen2_fit.summary()

    assert datagen2_fit.step_size is not None
    assert np.isnan(datagen2_fit.step_size).all()
    assert 'lp__' in list(summary.index)


def test_index_bounds_error() -> None:
    oob_stan = os.path.join(DATAFILES_PATH, 'out_of_bounds.stan')
    oob_model = CmdStanModel(stan_file=oob_stan)
    with pytest.raises(RuntimeError):
        oob_model.sample()


def test_show_console(stanfile: str = 'bernoulli.stan') -> None:
    stan = os.path.join(DATAFILES_PATH, stanfile)
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    sys_stdout = io.StringIO()
    with contextlib.redirect_stdout(sys_stdout):
        bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_warmup=100,
            iter_sampling=100,
            show_console=True,
        )
    console = sys_stdout.getvalue()
    assert 'Chain [1] method = sample' in console
    assert 'Chain [2] method = sample' in console


def test_show_progress(stanfile: str = 'bernoulli.stan') -> None:
    stan = os.path.join(DATAFILES_PATH, stanfile)
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    sys_stderr = io.StringIO()  # tqdm prints to stderr
    with contextlib.redirect_stderr(sys_stderr):
        bern_model.sample(
            data=jdata,
            chains=2,
            iter_warmup=100,
            iter_sampling=100,
            show_progress=True,
        )
    console = sys_stderr.getvalue()
    assert 'chain 1' in console
    assert 'chain 2' in console
    assert 'Sampling completed' in console

    sys_stderr = io.StringIO()  # tqdm prints to stderr
    with contextlib.redirect_stderr(sys_stderr):
        bern_model.sample(
            data=jdata,
            chains=7,
            iter_warmup=100,
            iter_sampling=100,
            show_progress=True,
        )
    console = sys_stderr.getvalue()
    assert 'chain 6' in console
    assert 'chain 7' in console
    assert 'Sampling completed' in console
    sys_stderr = io.StringIO()  # tqdm prints to stderr

    with contextlib.redirect_stderr(sys_stderr):
        bern_model.sample(
            data=jdata,
            chains=2,
            chain_ids=[6, 7],
            iter_warmup=100,
            iter_sampling=100,
            force_one_process_per_chain=True,
            show_progress=True,
        )
    console = sys_stderr.getvalue()
    assert 'chain 6' in console
    assert 'chain 7' in console
    assert 'Sampling completed' in console


def test_validate_good_run() -> None:
    # construct fit using existing sampler output
    csv_files = [
        os.path.join(DATAFILES_PATH, 'runset-good', f'bern-{i}.csv')
        for i in range(1, 5)
    ]
    config_files = [
        os.path.join(DATAFILES_PATH, 'runset-good', f'bern-{i}_config.json')
        for i in range(1, 5)
    ]
    fit = CmdStanMCMC.from_files(
        csv_files=csv_files,
        config_files=config_files,
    )
    assert 1000 == fit.num_draws_warmup
    assert 100 == fit.num_draws_sampling
    assert len(BERNOULLI_COLS) == len(fit.column_names)
    assert 'lp__' == fit.column_names[0]

    draws_pd = fit.draws_pd()
    assert draws_pd.shape == (
        fit.chains * fit.num_draws_sampling,
        len(fit.column_names) + 3,
    )
    assert fit.draws_pd(vars=['theta']).shape == (400, 1)
    assert fit.draws_pd(vars=['lp__', 'theta']).shape == (400, 2)
    assert fit.draws_pd(vars=['theta', 'lp__']).shape == (400, 2)
    assert fit.draws_pd(vars='theta').shape == (400, 1)

    assert list(fit.draws_pd(vars=['theta', 'lp__']).columns) == [
        'theta',
        'lp__',
    ]
    assert list(fit.draws_pd(vars=['lp__', 'theta', 'iter__']).columns) == [
        'lp__',
        'theta',
        'iter__',
    ]

    summary = fit.summary()
    assert '5%' in list(summary.columns)
    assert '50%' in list(summary.columns)
    assert '95%' in list(summary.columns)
    assert '1%' not in list(summary.columns)
    assert '99%' not in list(summary.columns)
    assert summary.index.name is None
    assert 'lp__' in list(summary.index)
    assert 'theta' in list(summary.index)

    summary = fit.summary(percentiles=[1, 45, 99])
    assert '1%' in list(summary.columns)
    assert '45%' in list(summary.columns)
    assert '99%' in list(summary.columns)
    assert '5%' not in list(summary.columns)
    assert '50%' not in list(summary.columns)
    assert '95%' not in list(summary.columns)

    with pytest.raises(ValueError):
        fit.summary(percentiles=[])

    with pytest.raises(ValueError):
        fit.summary(percentiles=[-1])

    diagnostics = fit.diagnose()
    assert diagnostics is not None
    assert 'Treedepth satisfactory for all transitions.' in diagnostics
    assert 'No divergent transitions found.' in diagnostics
    assert 'E-BFMI satisfactory' in diagnostics
    assert 'effective sample size satisfactory' in diagnostics.lower()


def test_validate_big_run() -> None:
    csv_files = [
        os.path.join(DATAFILES_PATH, 'runset-big', f'output_icar_nyc-{i}.csv')
        for i in (1, 2)
    ]
    config_files = [
        os.path.join(
            DATAFILES_PATH, 'runset-big', f'output_icar_nyc-{i}_config.json'
        )
        for i in (1, 2)
    ]
    metric_files = [
        os.path.join(
            DATAFILES_PATH, 'runset-big', f'output_icar_nyc-{i}_metric.json'
        )
        for i in (1, 2)
    ]
    fit = CmdStanMCMC.from_files(
        csv_files=csv_files,
        config_files=config_files,
        metric_files=metric_files,
    )
    phis = ['phi[{}]'.format(str(x + 1)) for x in range(2095)]
    column_names = list(fit.metadata.method_vars.keys()) + phis
    assert fit.num_draws_sampling == 1000
    assert fit.column_names == tuple(column_names)
    assert fit.metric_type == 'diag_e'
    assert fit.step_size is not None
    assert fit.step_size.shape == (2,)
    assert fit.inv_metric is not None
    assert fit.inv_metric.shape == (2, 2095)
    assert fit.draws().shape == (1000, 2, 2102)
    assert fit.draws_pd(vars=['phi']).shape == (2000, 2095)
    with raises_nested(ValueError, r'Unknown variable: gamma'):
        fit.draws_pd(vars=['gamma'])


def test_instantiate_from_output_filesfiles() -> None:
    csvfiles_path = os.path.join(DATAFILES_PATH, 'runset-good')
    bern_fit = from_output_files(path=csvfiles_path)
    assert isinstance(bern_fit, CmdStanMCMC)
    draws_pd = bern_fit.draws_pd()
    assert draws_pd.shape == (
        bern_fit.chains * bern_fit.num_draws_sampling,
        len(bern_fit.column_names) + 3,
    )
    csvfiles_path = os.path.join(DATAFILES_PATH, 'runset-big')
    big_fit = from_output_files(path=csvfiles_path)
    assert isinstance(big_fit, CmdStanMCMC)
    draws_pd = big_fit.draws_pd()
    assert draws_pd.shape == (
        big_fit.chains * big_fit.num_draws_sampling,
        len(big_fit.column_names) + 3,
    )
    # explicit list naming every file of the fit
    csvfiles_path = os.path.join(DATAFILES_PATH, 'runset-good')
    outfiles = []
    for file in os.listdir(csvfiles_path):
        if not file.endswith('.txt'):
            outfiles.append(os.path.join(csvfiles_path, file))
    bern_fit = from_output_files(path=outfiles)
    assert isinstance(bern_fit, CmdStanMCMC)
    assert bern_fit.chains == 4
    draws_pd = bern_fit.draws_pd()
    assert draws_pd.shape == (
        bern_fit.chains * bern_fit.num_draws_sampling,
        len(bern_fit.column_names) + 3,
    )
    # a config file identifies the whole fit
    config_file = os.path.join(csvfiles_path, 'bern-1_config.json')
    bern_fit = from_output_files(path=config_file)
    assert isinstance(bern_fit, CmdStanMCMC)
    assert bern_fit.chains == 4
    # so does a single CSV file, through the config alongside it
    csv_file = os.path.join(csvfiles_path, 'bern-2.csv')
    bern_fit = from_output_files(path=csv_file)
    assert isinstance(bern_fit, CmdStanMCMC)
    assert bern_fit.chains == 4
    draws_pd = bern_fit.draws_pd()
    assert draws_pd.shape == (
        bern_fit.chains * bern_fit.num_draws_sampling,
        len(bern_fit.column_names) + 3,
    )


@pytest.mark.parametrize(
    'path',
    [
        [
            os.path.join(DATAFILES_PATH, 'runset-good', f'bern-{chain}.csv')
            for chain in range(1, 5)
        ],
        os.path.join(DATAFILES_PATH, 'runset-good', 'bern-*.csv'),
    ],
)
def test_from_csv_deprecated_alias(
    path: str | list[str],
) -> None:
    with pytest.deprecated_call(match='use from_output_files instead'):
        fit = cmdstanpy.from_csv(path)

    assert isinstance(fit, CmdStanMCMC)
    assert fit.chains == 4


def test_pd_xr_agreement() -> None:
    csvfiles_path = os.path.join(DATAFILES_PATH, 'runset-good')
    bern_fit = from_output_files(path=csvfiles_path)
    assert isinstance(bern_fit, CmdStanMCMC)
    draws_pd = bern_fit.draws_pd()
    draws_xr = bern_fit.draws_xr()

    # check that the indexing is the same between the two
    np.testing.assert_equal(
        draws_pd[draws_pd['chain__'] == 2]['theta'],
        draws_xr.theta.sel(chain=2).values,
    )
    # "draw" is 0-indexed in xarray, equiv. "iter__" is 1-indexed in pandas
    np.testing.assert_equal(
        draws_pd[draws_pd['iter__'] == 100]['theta'],
        draws_xr.theta.sel(draw=99).values,
    )


def test_from_output_files_single_process_layout(tmp_path: Path) -> None:
    # when all chains run in one process CmdStan writes a single config for
    # the run, named for the first chain's output file; the config names
    # every chain's CSV in its ``output file`` argument
    outdir = os.path.join(tmp_path, 'threaded')
    os.makedirs(outdir)
    chain_ids = range(7, 11)
    for source_index, chain_id in enumerate(chain_ids, start=1):
        shutil.copy(
            os.path.join(GOODFILES_PATH, f'bern-{source_index}.csv'),
            os.path.join(outdir, f'bern_{chain_id}.csv'),
        )
        shutil.copy(
            os.path.join(GOODFILES_PATH, f'bern-{source_index}_metric.json'),
            os.path.join(outdir, f'bern_{chain_id}_metric.json'),
        )
    with open(os.path.join(GOODFILES_PATH, 'bern-1_config.json')) as f:
        config = json.load(f)
    config['id'] = 7
    config['method']['sample']['num_chains'] = 4
    # the recorded paths may be stale (e.g. the fit was moved); only the
    # file names are used, resolved next to the config file
    config['output']['file'] = ','.join(
        f'/some/stale/dir/bern_{i}.csv' for i in chain_ids
    )
    with open(os.path.join(outdir, 'bern_7_config.json'), 'w') as f:
        json.dump(config, f)

    fit = from_output_files(path=outdir)
    assert isinstance(fit, CmdStanMCMC)
    assert fit.chains == 4
    assert fit.config_files == [os.path.join(outdir, 'bern_7_config.json')]
    assert fit.chain_ids == list(chain_ids)
    assert fit.metric_files == [
        os.path.join(outdir, f'bern_{i}_metric.json') for i in chain_ids
    ]
    assert fit.metric_type == 'diag_e'
    assert fit.step_size is not None and fit.step_size.shape == (4,)

    fit_from_later_chain = from_output_files(os.path.join(outdir, 'bern_8.csv'))
    assert isinstance(fit_from_later_chain, CmdStanMCMC)
    assert fit_from_later_chain.chain_ids == list(chain_ids)

    draws_pd = fit.draws_pd()
    assert draws_pd.shape == (
        fit.chains * fit.num_draws_sampling,
        len(fit.column_names) + 3,
    )


def test_from_output_files_ignores_non_draw_csvs(tmp_path: Path) -> None:
    # latent dynamics and profile CSVs sit alongside the draws in an output
    # directory and must not be picked up as extra chains
    for chain_id in range(1, 5):
        _copy_bern_chain(chain_id, os.fspath(tmp_path), chain_id)
        shutil.copy(
            os.path.join(GOODFILES_PATH, f'bern-{chain_id}.csv'),
            os.path.join(tmp_path, f'bern_diagnostic_{chain_id}.csv'),
        )
    shutil.copy(
        os.path.join(GOODFILES_PATH, 'bern-1.csv'),
        os.path.join(tmp_path, 'bern_profile.csv'),
    )

    fit = from_output_files(path=os.fspath(tmp_path))
    assert isinstance(fit, CmdStanMCMC)
    assert fit.chains == 4


def _copy_bern_chain(source_index: int, dest_dir: str, chain_id: int) -> None:
    """Copy one runset-good chain's CSV and config, renumbered to
    ``chain_id``, into ``dest_dir``."""
    shutil.copy(
        os.path.join(GOODFILES_PATH, f'bern-{source_index}.csv'),
        os.path.join(dest_dir, f'bern_{chain_id}.csv'),
    )
    with open(
        os.path.join(GOODFILES_PATH, f'bern-{source_index}_config.json')
    ) as f:
        config = json.load(f)
    config['id'] = chain_id
    config['output']['file'] = f'bern_{chain_id}.csv'
    with open(os.path.join(dest_dir, f'bern_{chain_id}_config.json'), 'w') as f:
        json.dump(config, f)


def test_from_output_files_single_chain_timestamp_name(tmp_path: Path) -> None:
    stem = 'bernoulli-20260831225044'
    csv_file = tmp_path / f'{stem}.csv'
    config_file = tmp_path / f'{stem}_config.json'
    shutil.copy(os.path.join(GOODFILES_PATH, 'bern-1.csv'), csv_file)
    with open(os.path.join(GOODFILES_PATH, 'bern-1_config.json')) as fd:
        config = json.load(fd)
    config['id'] = 7
    config['output']['file'] = os.fspath(csv_file)
    with open(config_file, 'w') as fd:
        json.dump(config, fd)

    fit = from_output_files(config_file)
    assert isinstance(fit, CmdStanMCMC)
    assert fit.chain_ids == [7]
    assert fit.csv_files == [os.fspath(csv_file)]


def test_from_output_files_recovers_chain_ids(tmp_path: Path) -> None:
    # chain ids need not start at 1; they are recorded in each chain's config
    chain_ids = [7, 8]
    for index, chain_id in enumerate(chain_ids, start=1):
        _copy_bern_chain(index, os.fspath(tmp_path), chain_id)

    fit = from_output_files(path=os.fspath(tmp_path))
    assert isinstance(fit, CmdStanMCMC)
    assert list(fit.chain_ids) == chain_ids


def test_from_output_files_orders_chains_by_id(tmp_path: Path) -> None:
    # chain 10 sorts before chain 2 lexically; ids must be compared numerically
    for chain_id in (1, 2, 10):
        _copy_bern_chain(1, os.fspath(tmp_path), chain_id)

    fit = from_output_files(path=os.fspath(tmp_path))
    assert isinstance(fit, CmdStanMCMC)
    assert list(fit.chain_ids) == [1, 2, 10]
    assert [os.path.basename(f) for f in fit.csv_files] == [
        'bern_1.csv',
        'bern_2.csv',
        'bern_10.csv',
    ]


def test_instantiate_from_output_filesfiles_fail() -> None:
    with pytest.raises(ValueError, match=r'Must specify path'):
        from_output_files(None)

    csvfiles_path = os.path.join(DATAFILES_PATH, 'runset-good')
    with pytest.raises(ValueError, match=r'Bad method argument'):
        from_output_files(csvfiles_path, 'not-a-method')

    with pytest.raises(
        ValueError,
        match='Expecting CmdStan output files from method optimize',
    ):
        from_output_files(csvfiles_path, 'optimize')

    csvfiles: list[str] = []
    with pytest.raises(ValueError, match=r'No output files provided'):
        from_output_files(csvfiles, 'sample')

    # a list must not contain files that are not part of the fit
    for file in os.listdir(csvfiles_path):
        csvfiles.append(os.path.join(csvfiles_path, file))
    with pytest.raises(ValueError, match=r'Unrecognized output file'):
        from_output_files(csvfiles, 'sample')

    # a list must name the config JSON(s) of the fit
    csvfiles = [
        os.path.join(csvfiles_path, file)
        for file in os.listdir(csvfiles_path)
        if file.endswith('.csv')
    ]
    with pytest.raises(ValueError, match=r'No CmdStan config JSON'):
        from_output_files(csvfiles, 'sample')

    # globs are no longer supported
    glob_path = os.path.join(csvfiles_path, '*')
    with pytest.raises(ValueError, match=r'Invalid path specification'):
        from_output_files(glob_path, 'sample')

    csvfiles_path = os.path.join(DATAFILES_PATH, 'no-such-directory')
    with pytest.raises(ValueError, match=r'Invalid path specification'):
        from_output_files(path=csvfiles_path)

    no_csvfiles_path = os.path.join(DATAFILES_PATH, 'test-fail-empty-directory')
    if os.path.exists(no_csvfiles_path):
        shutil.rmtree(no_csvfiles_path, ignore_errors=True)
    os.mkdir(no_csvfiles_path)
    with pytest.raises(ValueError, match=r'No CmdStan config files found'):
        from_output_files(path=no_csvfiles_path)
    if os.path.exists(no_csvfiles_path):
        shutil.rmtree(no_csvfiles_path, ignore_errors=True)


def test_from_output_files_multiple_fits_in_directory(
    tmp_path: Path,
) -> None:
    # a directory holding more than one fit is ambiguous; the config file
    # of the desired fit must be passed instead
    for chain_id in (1, 2):
        _copy_bern_chain(chain_id, os.fspath(tmp_path), chain_id)
    shutil.copy(
        os.path.join(GOODFILES_PATH, 'bern-1.csv'),
        os.path.join(tmp_path, 'other.csv'),
    )
    with open(os.path.join(GOODFILES_PATH, 'bern-1_config.json')) as fd:
        other_config = json.load(fd)
    other_config['output']['file'] = 'other.csv'
    with open(os.path.join(tmp_path, 'other_config.json'), 'w') as fd:
        json.dump(other_config, fd)

    with pytest.raises(ValueError, match=r'more than one fit'):
        from_output_files(path=os.fspath(tmp_path))

    fit = from_output_files(path=os.path.join(tmp_path, 'other_config.json'))
    assert isinstance(fit, CmdStanMCMC)
    assert fit.chains == 1


def test_from_output_files_same_base_different_methods(tmp_path: Path) -> None:
    _copy_bern_chain(1, os.fspath(tmp_path), 1)
    shutil.copy(
        os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv'),
        os.path.join(tmp_path, 'bern_2.csv'),
    )
    with open(
        os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle_config.json')
    ) as fd:
        optimize_config = json.load(fd)
    optimize_config['output']['file'] = 'bern_2.csv'
    with open(os.path.join(tmp_path, 'bern_2_config.json'), 'w') as fd:
        json.dump(optimize_config, fd)

    with pytest.raises(ValueError, match=r'more than one fit'):
        from_output_files(tmp_path)


def test_from_output_files_ignores_incomplete_optional_files(
    tmp_path: Path,
) -> None:
    for chain_id in (1, 2):
        _copy_bern_chain(chain_id, os.fspath(tmp_path), chain_id)
    shutil.copy(
        os.path.join(GOODFILES_PATH, 'bern-1_metric.json'),
        os.path.join(tmp_path, 'bern_1_metric.json'),
    )
    shutil.copy(
        os.path.join(GOODFILES_PATH, 'bern-3.csv'),
        os.path.join(tmp_path, 'bern_3.csv'),
    )

    fit = from_output_files(tmp_path)
    assert isinstance(fit, CmdStanMCMC)
    assert fit.chains == 2
    assert fit.metric_files is None


def test_from_output_files_explicit_does_not_discover_siblings(
    tmp_path: Path,
) -> None:
    for chain_id in (1, 2):
        _copy_bern_chain(chain_id, os.fspath(tmp_path), chain_id)

    fit = from_output_files(
        [tmp_path / 'bern_1.csv', tmp_path / 'bern_1_config.json']
    )
    assert isinstance(fit, CmdStanMCMC)
    assert fit.chains == 1


def test_from_output_files_explicit_allows_nonstandard_csv_name(
    tmp_path: Path,
) -> None:
    csv_file = tmp_path / 'model_profile.csv'
    config_file = tmp_path / 'other_config.json'
    shutil.copy(os.path.join(GOODFILES_PATH, 'bern-1.csv'), csv_file)
    with open(os.path.join(GOODFILES_PATH, 'bern-1_config.json')) as fd:
        config = json.load(fd)
    config['output']['file'] = csv_file.name
    config_file.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=r'Cannot discover a fit'):
        from_output_files(config_file)

    fit = from_output_files([csv_file, config_file])
    assert isinstance(fit, CmdStanMCMC)
    assert fit.chains == 1

    config['output']['file'] = 'different.csv'
    config_file.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=r'Configured output name'):
        from_output_files([csv_file, config_file])


@pytest.mark.parametrize('num_chains', [0, -1])
def test_from_output_files_rejects_invalid_num_chains(
    tmp_path: Path, num_chains: int
) -> None:
    csv_file = tmp_path / 'bern.csv'
    config_file = tmp_path / 'bern_config.json'
    shutil.copy(os.path.join(GOODFILES_PATH, 'bern-1.csv'), csv_file)
    with open(os.path.join(GOODFILES_PATH, 'bern-1_config.json')) as fd:
        config = json.load(fd)
    config['method']['sample']['num_chains'] = num_chains
    config['output']['file'] = csv_file.name
    with open(config_file, 'w') as fd:
        json.dump(config, fd)

    with pytest.raises(ValueError, match=r'Cannot parse CmdStan config'):
        from_output_files(config_file)


def test_from_output_files_rejects_nonpositive_single_process_id(
    tmp_path: Path,
) -> None:
    for chain_id in (0, 1):
        shutil.copy(
            os.path.join(GOODFILES_PATH, f'bern-{chain_id + 1}.csv'),
            tmp_path / f'bern_{chain_id}.csv',
        )
    with open(os.path.join(GOODFILES_PATH, 'bern-1_config.json')) as fd:
        config = json.load(fd)
    config['id'] = 0
    config['method']['sample']['num_chains'] = 2
    config['output']['file'] = 'bern_0.csv,bern_1.csv'
    config_file = tmp_path / 'bern_0_config.json'
    config_file.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=r'non-positive chain ID'):
        from_output_files(config_file)


def test_from_output_files_rejects_multichain_per_chain_configs(
    tmp_path: Path,
) -> None:
    files: list[Path] = []
    for chain_id in (1, 2):
        _copy_bern_chain(chain_id, os.fspath(tmp_path), chain_id)
        config_file = tmp_path / f'bern_{chain_id}_config.json'
        config = json.loads(config_file.read_text())
        config['method']['sample']['num_chains'] = 2
        config_file.write_text(json.dumps(config))
        files.extend([tmp_path / f'bern_{chain_id}.csv', config_file])

    with pytest.raises(ValueError, match=r'must each record num_chains=1'):
        from_output_files(files)


def test_from_output_files_csv_without_config(tmp_path: Path) -> None:
    # a bare CSV without the config JSON CmdStan writes alongside it does
    # not follow CmdStanPy naming and cannot be used for discovery
    csv_file = os.path.join(tmp_path, 'bern_1.csv')
    shutil.copy(os.path.join(GOODFILES_PATH, 'bern-1.csv'), csv_file)

    with pytest.raises(ValueError, match=r'Cannot identify one config JSON'):
        from_output_files(path=csv_file)


def test_from_output_files_raw_cmdstan_num_chains(tmp_path: Path) -> None:
    # running CmdStan directly with ``num_chains`` and a single output name
    # writes ``output_<id>.csv`` per chain and one ``output_config.json``;
    # that config is not named for any CSV, so the files cannot be
    # discovered and must be passed explicitly
    for chain_id in range(1, 5):
        shutil.copy(
            os.path.join(GOODFILES_PATH, f'bern-{chain_id}.csv'),
            os.path.join(tmp_path, f'output_{chain_id}.csv'),
        )
    with open(os.path.join(GOODFILES_PATH, 'bern-1_config.json')) as f:
        config = json.load(f)
    config['method']['sample']['num_chains'] = 4
    config['output']['file'] = 'output.csv'
    with open(os.path.join(tmp_path, 'output_config.json'), 'w') as f:
        json.dump(config, f)

    with pytest.raises(ValueError, match=r'No CmdStan config files found'):
        from_output_files(path=os.fspath(tmp_path))

    fit = from_output_files(
        path=[os.path.join(tmp_path, f) for f in os.listdir(tmp_path)]
    )
    assert isinstance(fit, CmdStanMCMC)
    assert fit.chains == 4
    assert list(fit.chain_ids) == [1, 2, 3, 4]


def test_from_output_files_fixed_param() -> None:
    csv_path = os.path.join(DATAFILES_PATH, 'fixed_param_sample.csv')
    fixed_param_sample = from_output_files(path=csv_path)
    assert isinstance(fixed_param_sample, CmdStanMCMC)
    assert fixed_param_sample.draws_pd().shape == (100, 88)


def test_from_output_files_no_param_hmc() -> None:
    csv_path = os.path.join(DATAFILES_PATH, 'no_param_hmc_sample.csv')
    no_parameters_sample = from_output_files(path=csv_path)
    assert isinstance(no_parameters_sample, CmdStanMCMC)
    assert no_parameters_sample.draws_pd().shape == (100, 93)


@pytest.mark.parametrize('force_one_process_per_chain', [True, False])
def test_custom_metric(force_one_process_per_chain: bool) -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    jmetric = os.path.join(DATAFILES_PATH, 'bernoulli.metric.json')
    jmetric2 = os.path.join(DATAFILES_PATH, 'bernoulli.metric-2.json')
    # read json in as dict
    with open(jmetric) as fd:
        metric_dict_1 = json.load(fd)
    with open(jmetric2) as fd:
        metric_dict_2 = json.load(fd)
    # just test that it runs without error
    fit1 = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=10,
        iter_sampling=10,
        inv_metric=jmetric,
        force_one_process_per_chain=force_one_process_per_chain,
    )
    assert fit1.inv_metric is not None
    np.testing.assert_allclose(
        fit1.inv_metric[0], metric_dict_1['inv_metric'], atol=1e-6
    )
    np.testing.assert_allclose(
        fit1.inv_metric[1], metric_dict_1['inv_metric'], atol=1e-6
    )

    fit2 = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=10,
        iter_sampling=10,
        inv_metric=[jmetric, jmetric2],
        force_one_process_per_chain=force_one_process_per_chain,
    )
    assert fit2.inv_metric is not None
    np.testing.assert_allclose(
        fit2.inv_metric[0], metric_dict_1['inv_metric'], atol=1e-6
    )
    np.testing.assert_allclose(
        fit2.inv_metric[1], metric_dict_2['inv_metric'], atol=1e-6
    )

    fit3 = bern_model.sample(
        data=jdata,
        chains=4,
        parallel_chains=2,
        seed=12345,
        iter_warmup=10,
        iter_sampling=10,
        inv_metric=metric_dict_1,
        force_one_process_per_chain=force_one_process_per_chain,
    )
    assert fit3.inv_metric is not None
    for i in range(4):
        np.testing.assert_allclose(
            fit3.inv_metric[i], metric_dict_1['inv_metric'], atol=1e-6
        )
    fit4 = bern_model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=10,
        iter_sampling=10,
        inv_metric=[metric_dict_1, metric_dict_2],
        force_one_process_per_chain=force_one_process_per_chain,
    )
    assert fit4.inv_metric is not None
    np.testing.assert_allclose(
        fit4.inv_metric[0], metric_dict_1['inv_metric'], atol=1e-6
    )
    np.testing.assert_allclose(
        fit4.inv_metric[1], metric_dict_2['inv_metric'], atol=1e-6
    )

    fit5 = bern_model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=10,
        iter_sampling=10,
        inv_metric=[np.array(metric_dict_1['inv_metric']), jmetric2],
        force_one_process_per_chain=force_one_process_per_chain,
    )
    assert fit5.inv_metric is not None
    np.testing.assert_allclose(
        fit5.inv_metric[0], metric_dict_1['inv_metric'], atol=1e-6
    )
    np.testing.assert_allclose(
        fit5.inv_metric[1], metric_dict_2['inv_metric'], atol=1e-6
    )

    with pytest.raises(
        ValueError,
        match='Number of metric files must match number of chains,',
    ):
        bern_model.sample(
            data=jdata,
            chains=4,
            parallel_chains=2,
            seed=12345,
            iter_warmup=10,
            iter_sampling=10,
            inv_metric=[metric_dict_1, metric_dict_2],
            force_one_process_per_chain=force_one_process_per_chain,
        )
    # metric mismatches - (not appropriate for bernoulli)
    with open(os.path.join(DATAFILES_PATH, 'metric_diag.data.json')) as fd:
        metric_dict_1 = json.load(fd)
    with open(os.path.join(DATAFILES_PATH, 'metric_dense.data.json')) as fd:
        metric_dict_2 = json.load(fd)
    with pytest.raises(RuntimeError, match='Error during sampling'):
        bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            iter_warmup=10,
            iter_sampling=10,
            inv_metric=[metric_dict_1, metric_dict_2],
            force_one_process_per_chain=force_one_process_per_chain,
        )
    # metric dict, no "inv_metric":
    some_dict = {"foo": [1, 2, 3]}
    with pytest.raises(
        ValueError, match='Entry "inv_metric" not found in metric dict.'
    ):
        bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            iter_warmup=100,
            iter_sampling=200,
            inv_metric=some_dict,
            force_one_process_per_chain=force_one_process_per_chain,
        )


def test_custom_step_size() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    # just test that it runs without error
    bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
        step_size=1,
    )

    bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
        step_size=[1, 2],
    )


def test_custom_seed() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    # just test that it runs without error
    bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=[44444, 55555],
        iter_warmup=100,
        iter_sampling=200,
    )


def test_adapt_schedule() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata,
        chains=1,
        seed=12345,
        iter_sampling=200,
        iter_warmup=200,
        adapt_init_phase=11,
        adapt_metric_window=12,
        adapt_step_size=13,
    )
    txt_file = bern_fit.stdout_files[0]  # type: ignore
    with open(txt_file, 'r') as fd:
        lines = fd.readlines()
        stripped = [line.strip() for line in lines]
        assert 'init_buffer = 11' in stripped
        assert 'window = 12' in stripped
        assert 'term_buffer = 13' in stripped


def test_save_csv() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
    )
    for i in range(bern_fit.chains):
        csv_file = bern_fit.csv_files[i]
        stdout_file = bern_fit.stdout_files[i]  # type: ignore
        assert os.path.exists(csv_file)
        assert os.path.exists(stdout_file)

    # save files to good dir
    bern_fit.save_output_files(dir=DATAFILES_PATH)
    for i in range(bern_fit.chains):
        csv_file = bern_fit.csv_files[i]
        assert os.path.exists(csv_file)
    with pytest.raises(ValueError, match='File exists, not overwriting: '):
        bern_fit.save_output_files(dir=DATAFILES_PATH)

    tmp2_dir = os.path.join(HERE, 'tmp2')
    os.mkdir(tmp2_dir)
    bern_fit.save_output_files(dir=tmp2_dir)
    for i in range(bern_fit.chains):
        csv_file = bern_fit.csv_files[i]
        assert os.path.exists(csv_file)
    for attr in (  # cleanup datafile_path dir
        'csv_files',
        'stdout_files',
        'config_files',
        'metric_files',
    ):
        files = getattr(bern_fit, attr)
        if files is None:
            continue
        for f in files:
            if os.path.exists(f):
                os.remove(f)
    shutil.rmtree(tmp2_dir, ignore_errors=True)

    # regenerate to tmpdir, save to good dir
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_sampling=200,
    )
    bern_fit.save_output_files()  # default dir
    for i in range(bern_fit.chains):
        csv_file = bern_fit.csv_files[i]
        assert os.path.exists(csv_file)
    for attr in (
        'csv_files',
        'stdout_files',
        'config_files',
        'metric_files',
    ):
        files = getattr(bern_fit, attr)
        if files is None:
            continue
        for f in files:
            if os.path.exists(f):
                os.remove(f)

    with pytest.raises(ValueError, match='Cannot access CSV file'):
        bern_fit.save_output_files(dir=DATAFILES_PATH)

    if platform.system() != 'Windows':
        with pytest.raises(RuntimeError, match='Cannot save to path: '):
            dir = tempfile.mkdtemp(dir=_TMPDIR)
            os.chmod(dir, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            bern_fit.save_output_files(dir=dir)


def test_diagnose_divergences() -> None:
    csv_file = os.path.join(
        DATAFILES_PATH, 'diagnose-good', 'corr_gauss_depth8-1.csv'
    )
    config_file = os.path.join(
        DATAFILES_PATH, 'diagnose-good', 'corr_gauss_depth8-1_config.json'
    )
    fit = CmdStanMCMC.from_files(
        csv_files=[csv_file],
        config_files=[config_file],
    )
    # TODO - use cmdstan test files instead
    expected = [
        'Checking sampler transitions treedepth.',
        '424 of 1000',
        'treedepth limit of 8, or 2^8 leapfrog steps.',
        'Trajectories that are prematurely terminated '
        'due to this limit will result in slow exploration.',
        'For optimal performance, increase this limit.',
    ]

    diagnose = fit.diagnose()
    assert diagnose is not None
    for e in expected:
        assert e in diagnose


def test_validate_bad_run() -> None:
    def fixtures(prefix: str) -> tuple[list[str], list[str]]:
        csvs = [
            os.path.join(DATAFILES_PATH, 'runset-bad', f'{prefix}-bern-{i}.csv')
            for i in range(1, 5)
        ]
        configs = [
            os.path.join(
                DATAFILES_PATH, 'runset-bad', f'{prefix}-bern-{i}_config.json'
            )
            for i in range(1, 5)
        ]
        return csvs, configs

    # csv file headers inconsistent
    csvs, configs = fixtures('bad-hdr')
    with raises_nested(ValueError, 'CmdStan config mismatch'):
        CmdStanMCMC.from_files(csv_files=csvs, config_files=configs)

    # bad draws
    csvs, configs = fixtures('bad-draws')
    with raises_nested(ValueError, 'draws'):
        CmdStanMCMC.from_files(csv_files=csvs, config_files=configs)

    # mismatch - column headers, draws
    csvs, configs = fixtures('bad-cols')
    with raises_nested(ValueError, 'bad draw, expecting 9 items, found 8'):
        CmdStanMCMC.from_files(csv_files=csvs, config_files=configs)


def _good_runset_files() -> tuple[list[str], list[str], list[str]]:
    csvs = [
        os.path.join(DATAFILES_PATH, 'runset-good', f'bern-{i}.csv')
        for i in range(1, 5)
    ]
    configs = [
        os.path.join(DATAFILES_PATH, 'runset-good', f'bern-{i}_config.json')
        for i in range(1, 5)
    ]
    metrics = [
        os.path.join(DATAFILES_PATH, 'runset-good', f'bern-{i}_metric.json')
        for i in range(1, 5)
    ]
    return csvs, configs, metrics


def test_metric_info_unavailable_returns_none() -> None:
    # No metric files (e.g. adaptation disabled, so CmdStan wrote none) means
    # the metric properties report None rather than raising. Covers both an
    # explicit absence and metric paths that were listed but never written.
    csvs, configs, _ = _good_runset_files()
    absent = [f'/no/such/bern-{i}_metric.json' for i in range(1, 5)]
    for metric_files in (None, absent):
        fit = CmdStanMCMC.from_files(
            csv_files=csvs, config_files=configs, metric_files=metric_files
        )
        assert fit.metric_type is None
        assert fit.step_size is None
        assert fit.inv_metric is None


def test_metric_files_misaligned() -> None:
    csvs, configs, metrics = _good_runset_files()
    # too few metric files -> rejected at construction, not silently accepted
    with pytest.raises(ValueError, match='one metric file per chain'):
        CmdStanMCMC.from_files(
            csv_files=csvs, config_files=configs, metric_files=metrics[:3]
        )
    # right count but one absent -> partial set rejected on access
    fit = CmdStanMCMC.from_files(
        csv_files=csvs,
        config_files=configs,
        metric_files=metrics[:3] + ['/no/such/bern-4_metric.json'],
    )
    with pytest.raises(ValueError, match='missing for some chains'):
        _ = fit.metric_type


def test_sample_sporadic_exception(caplog: pytest.LogCaptureFixture) -> None:
    stan = os.path.join(DATAFILES_PATH, 'linear_regression.stan')
    jdata = os.path.join(DATAFILES_PATH, 'linear_regression.data.json')
    linear_model = CmdStanModel(stan_file=stan)
    # will produce a failure due to calling normal_lpdf with 0 for scale
    # but then continue sampling normally
    with caplog.at_level(logging.WARNING):
        linear_model.sample(data=jdata, inits=0)
    check_present(
        caplog, ('cmdstanpy', 'WARNING', re.compile(r"Non-fatal error.*"))
    )


def test_save_warmup() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=200,
        iter_sampling=100,
        save_warmup=True,
    )
    assert bern_fit.column_names == tuple(BERNOULLI_COLS)
    assert bern_fit.num_draws_warmup == 200
    assert bern_fit.num_draws_sampling == 100
    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))
    assert bern_fit.draws(inc_warmup=False).shape == (
        100,
        2,
        len(BERNOULLI_COLS),
    )
    assert bern_fit.draws(concat_chains=True).shape == (
        200,
        len(BERNOULLI_COLS),
    )
    assert bern_fit.draws(inc_warmup=True).shape == (
        300,
        2,
        len(BERNOULLI_COLS),
    )
    assert bern_fit.draws(inc_warmup=True, concat_chains=True).shape == (
        600,
        len(BERNOULLI_COLS),
    )

    assert bern_fit.draws_pd().shape == (200, len(BERNOULLI_COLS) + 3)
    assert bern_fit.draws_pd(inc_warmup=False).shape == (
        200,
        len(BERNOULLI_COLS) + 3,
    )
    assert bern_fit.draws_pd(inc_warmup=True).shape == (
        600,
        len(BERNOULLI_COLS) + 3,
    )


def test_save_warmup_thin() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=200,
        iter_sampling=100,
        thin=5,
        save_warmup=True,
    )
    assert bern_fit.column_names == tuple(BERNOULLI_COLS)
    assert bern_fit.draws().shape == (20, 2, len(BERNOULLI_COLS))
    assert bern_fit.draws(concat_chains=True).shape == (40, len(BERNOULLI_COLS))
    assert bern_fit.draws(inc_warmup=True).shape == (60, 2, len(BERNOULLI_COLS))


def test_dont_save_warmup(caplog: pytest.LogCaptureFixture) -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=200,
        iter_sampling=100,
        save_warmup=False,
    )
    assert bern_fit.column_names == tuple(BERNOULLI_COLS)
    assert bern_fit.num_draws_sampling == 100
    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))
    with caplog.at_level(logging.WARNING):
        assert bern_fit.draws(inc_warmup=True).shape == (
            100,
            2,
            len(BERNOULLI_COLS),
        )
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            "Sample doesn't contain draws from warmup iterations,"
            ' rerun sampler with "save_warmup=True".',
        ),
    )
    with caplog.at_level(logging.WARNING):
        assert bern_fit.draws(inc_warmup=True, concat_chains=True).shape == (
            200,
            len(BERNOULLI_COLS),
        )
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            "Sample doesn't contain draws from warmup iterations,"
            ' rerun sampler with "save_warmup=True".',
        ),
    )
    with caplog.at_level(logging.WARNING):
        assert bern_fit.draws_pd(inc_warmup=True).shape == (
            200,
            len(BERNOULLI_COLS) + 3,
        )
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            "Sample doesn't contain draws from warmup iterations,"
            ' rerun sampler with "save_warmup=True".',
        ),
    )


def test_warmup_no_adapt() -> None:
    # we may want to have a "burn-in" period, even without adaptation
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=200,
        iter_sampling=100,
        adapt_engaged=False,
    )

    assert bern_fit.column_names == tuple(BERNOULLI_COLS)
    assert bern_fit.num_draws_sampling == 100
    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))


def test_sampler_diags() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata, chains=2, seed=12345, iter_warmup=100, iter_sampling=100
    )
    diags = bern_fit.method_variables()
    assert SAMPLER_STATE == list(diags)
    for diag in diags.values():
        assert diag.shape == (100, 2)

    diags = bern_fit.method_variables()
    assert SAMPLER_STATE == list(diags)
    for diag in diags.values():
        assert diag.shape == (100, 2)
    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))


def test_variable_bern() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata, chains=2, seed=12345, iter_warmup=100, iter_sampling=100
    )
    assert 1 == len(bern_fit.metadata.stan_vars)
    assert 'theta' in bern_fit.metadata.stan_vars
    assert bern_fit.metadata.stan_vars['theta'].dimensions == ()
    assert bern_fit.stan_variable(var='theta').shape == (200,)
    with pytest.raises(ValueError):
        bern_fit.stan_variable(var='eta')
    with pytest.raises(ValueError):
        bern_fit.stan_variable(var='lp__')


def test_variables_2d() -> None:
    csvfiles_path = os.path.join(DATAFILES_PATH, 'lotka-volterra.csv')
    fit = from_output_files(path=csvfiles_path)
    assert isinstance(fit, CmdStanMCMC)
    assert 20 == fit.num_draws_sampling
    assert 8 == len(fit.metadata.stan_vars)
    assert 'z' in fit.metadata.stan_vars
    assert fit.metadata.stan_vars['z'].dimensions == (20, 2)
    vars = fit.stan_variables()
    assert len(vars) == len(fit.metadata.stan_vars)
    assert 'z' in vars
    assert vars['z'].shape == (20, 20, 2)
    assert 'theta' in vars
    assert vars['theta'].shape == (20, 4)


def test_variables_3d() -> None:
    # construct fit using existing sampler output
    csvfiles_path = os.path.join(DATAFILES_PATH, 'multidim_vars.csv')
    fit = from_output_files(path=csvfiles_path)
    assert isinstance(fit, CmdStanMCMC)
    assert 20 == fit.num_draws_sampling
    assert 3 == len(fit.metadata.stan_vars)
    assert 'y_rep' in fit.metadata.stan_vars
    assert fit.metadata.stan_vars['y_rep'].dimensions == (5, 4, 3)
    var_y_rep = fit.stan_variable(var='y_rep')
    assert var_y_rep.shape == (20, 5, 4, 3)
    var_beta = fit.stan_variable(var='beta')
    assert var_beta.shape, (20, 2)
    var_frac_60 = fit.stan_variable(var='frac_60')
    assert var_frac_60.shape == (20,)
    vars = fit.stan_variables()
    assert len(vars) == len(fit.metadata.stan_vars)
    assert 'y_rep' in vars
    assert vars['y_rep'].shape == (20, 5, 4, 3)
    assert 'beta' in vars
    assert vars['beta'].shape == (20, 2)
    assert 'frac_60' in vars
    assert vars['frac_60'].shape == (20,)


def test_variables_issue_361() -> None:
    # tests that array ordering is preserved
    stan = os.path.join(DATAFILES_PATH, 'container_vars.stan')
    container_vars_model = CmdStanModel(stan_file=stan)
    chain_1_fit = container_vars_model.sample(
        chains=1, iter_sampling=4, fixed_param=True
    )
    v_2d_arr = chain_1_fit.stan_variable('v_2d_arr')
    assert v_2d_arr.shape == (4, 2, 3)
    # stan 1-based indexing vs. python 0-based indexing
    for i in range(2):
        for j in range(3):
            assert v_2d_arr[0, i, j] == ((i + 1) * 10) + j + 1
    chain_2_fit = container_vars_model.sample(
        chains=2, iter_sampling=4, fixed_param=True
    )
    v_2d_arr = chain_2_fit.stan_variable('v_2d_arr')
    assert v_2d_arr.shape == (8, 2, 3)
    # stan 1-based indexing vs. python 0-based indexing
    for i in range(2):
        for j in range(3):
            assert v_2d_arr[0, i, j] == ((i + 1) * 10) + j + 1


def test_validate() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=200,
        iter_sampling=100,
        thin=2,
        save_warmup=True,
    )
    # _validate_csv_files called during instantiation
    assert bern_fit.num_draws_warmup == 100
    assert bern_fit.num_draws_sampling == 50
    assert len(bern_fit.column_names) == 8
    assert len(bern_fit.metadata.stan_vars) == 1
    assert bern_fit.metric_type == 'diag_e'


def test_validate_sample_sig_figs(stanfile: str = 'bernoulli.stan') -> None:
    stan = os.path.join(DATAFILES_PATH, stanfile)
    bern_model = CmdStanModel(stan_file=stan)

    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit = bern_model.sample(
        data=jdata,
        chains=1,
        seed=12345,
        iter_sampling=100,
    )
    bern_draws = bern_fit.draws()
    theta = format(bern_draws[99, 0, 7], '.18g')
    assert not theta.startswith('0.21238045821757600')

    bern_fit_17 = bern_model.sample(
        data=jdata,
        chains=1,
        seed=12345,
        iter_sampling=100,
        sig_figs=17,
    )
    assert bern_fit_17.draws().size

    with pytest.raises(ValueError):
        bern_model.sample(
            data=jdata,
            chains=1,
            seed=12345,
            iter_sampling=100,
            sig_figs=27,
        )
        with pytest.raises(ValueError):
            bern_model.sample(
                data=jdata,
                chains=1,
                seed=12345,
                iter_sampling=100,
                sig_figs=-1,
            )


def test_validate_summary_sig_figs() -> None:
    # construct CmdStanMCMC from logistic model output
    fit = from_output_files(
        [
            os.path.join(DATAFILES_PATH, f'logistic_output_{i}{suffix}')
            for i in range(1, 5)
            for suffix in ('.csv', '_config.json')
        ]
    )
    assert isinstance(fit, CmdStanMCMC)

    sum_default = fit.summary()

    beta1_default = format(sum_default.iloc[1, 0], '.18g')
    assert beta1_default.startswith('1.3')

    sum_17 = fit.summary(sig_figs=17)
    beta1_17 = format(sum_17.iloc[1, 0], '.18g')
    assert beta1_17.startswith('1.343377085648')

    sum_10 = fit.summary(sig_figs=10)
    beta1_10 = format(sum_10.iloc[1, 0], '.18g')
    assert beta1_10.startswith('1.34337708')

    with pytest.raises(ValueError):
        fit.summary(sig_figs=20)
    with pytest.raises(ValueError):
        fit.summary(sig_figs=-1)


def test_metadata() -> None:
    # construct CmdStanMCMC from logistic model output, config
    csv_files = [
        os.path.join(DATAFILES_PATH, f'logistic_output_{i}.csv')
        for i in range(1, 5)
    ]
    config_files = [
        os.path.join(DATAFILES_PATH, f'logistic_output_{i}_config.json')
        for i in range(1, 5)
    ]
    metric_files = [
        os.path.join(DATAFILES_PATH, f'logistic_output_{i}_metric.json')
        for i in range(1, 5)
    ]
    fit = CmdStanMCMC.from_files(
        csv_files=csv_files,
        config_files=config_files,
        metric_files=metric_files,
        sig_figs=17,
    )
    assert fit.model_name == 'logistic_model'
    col_names = (
        'lp__',
        'accept_stat__',
        'stepsize__',
        'treedepth__',
        'n_leapfrog__',
        'divergent__',
        'energy__',
        'beta[1]',
        'beta[2]',
    )

    assert fit.chains == 4
    assert fit.chain_ids == [1, 2, 3, 4]
    assert fit.num_draws_warmup == 1000
    assert fit.num_draws_sampling == 100
    assert fit.column_names == col_names
    assert fit.metric_type == 'diag_e'

    assert fit.config.method_config.num_samples == 100
    assert fit.config.method_config.thin == 1
    assert fit.config.method_config.algorithm == 'hmc'

    assert 'n_leapfrog__' in fit.metadata.method_vars
    assert 'energy__' in fit.metadata.method_vars
    assert 'beta' not in fit.metadata.method_vars
    assert 'energy__' not in fit.metadata.stan_vars
    assert 'beta' in fit.metadata.stan_vars
    assert fit.metadata.stan_vars['beta'].dimensions == (2,)
    assert tuple(fit.metadata.stan_vars['beta'].columns()) == (7, 8)


def test_save_latent_dynamics() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
        save_latent_dynamics=True,
    )
    for i in range(bern_fit.chains):
        diagnostics_file = bern_fit.diagnostic_files[i]  # type: ignore
        assert os.path.exists(diagnostics_file)


def test_save_profile() -> None:
    stan = os.path.join(DATAFILES_PATH, 'profile_likelihood.stan')
    profile_model = CmdStanModel(
        stan_file=stan, cpp_options={"STAN_THREADS": '1'}, force_compile=True
    )

    profile_fit = profile_model.sample(
        chains=2,
        parallel_chains=2,
        force_one_process_per_chain=True,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
        save_profile=True,
    )
    assert len(profile_fit.profile_files) == 2  # type: ignore
    for profile_file in profile_fit.profile_files:  # type: ignore
        assert os.path.exists(profile_file)

    profile_fit = profile_model.sample(
        chains=2,
        parallel_chains=2,
        force_one_process_per_chain=False,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
        save_profile=True,
    )

    assert len(profile_fit.profile_files) == 1  # type: ignore
    for profile_file in profile_fit.profile_files:  # type: ignore
        assert os.path.exists(profile_file)


def test_xarray_draws() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata, chains=2, seed=12345, iter_warmup=100, iter_sampling=100
    )
    xr_data = bern_fit.draws_xr()
    assert xr_data.theta.dims == ('chain', 'draw')
    np.testing.assert_allclose(
        xr_data.theta.transpose('draw', ...).values,
        bern_fit.draws()[:, :, -1],
    )
    assert xr_data.theta.values.shape == (2, 100)

    xr_data = bern_fit.draws_xr(vars=['theta'])
    assert xr_data.theta.values.shape == (2, 100)

    with pytest.raises(KeyError):
        xr_data = bern_fit.draws_xr(vars=['eta'])

    # test inc_warmup
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        save_warmup=True,
    )
    xr_data = bern_fit.draws_xr(inc_warmup=True)
    assert xr_data.theta.values.shape == (2, 200)

    # test that array[1] and chains=1 are properly handled dimension-wise
    stan = os.path.join(DATAFILES_PATH, 'bernoulli_array.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    bern_fit = bern_model.sample(
        data=jdata, chains=1, seed=12345, iter_warmup=100, iter_sampling=100
    )
    xr_data = bern_fit.draws_xr()
    assert xr_data.theta.dims == ('chain', 'draw', 'theta_dim_0')
    assert xr_data.theta.values.shape == (1, 100, 1)

    xr_var = bern_fit.draws_xr(vars='theta')
    assert xr_var.theta.dims == ('chain', 'draw', 'theta_dim_0')
    assert xr_var.theta.values.shape == (1, 100, 1)

    xr_var = bern_fit.draws_xr(vars=['theta'])
    assert xr_var.theta.dims == ('chain', 'draw', 'theta_dim_0')
    assert xr_var.theta.values.shape == (1, 100, 1)


def test_no_xarray() -> None:
    with without_import('xarray', cmdstanpy.stanfit.mcmc):
        with pytest.raises(ImportError):
            # if this fails the testing framework is the problem
            import xarray as _  # noqa

        stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
        jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
        bern_model = CmdStanModel(stan_file=stan)
        bern_fit = bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            iter_warmup=100,
            iter_sampling=100,
        )

        with pytest.raises(RuntimeError):
            bern_fit.draws_xr()


def test_single_row_csv() -> None:
    stan = os.path.join(DATAFILES_PATH, 'matrix_var.stan')
    model = CmdStanModel(stan_file=stan)
    fit = model.sample(iter_sampling=1, chains=1)
    z_as_ndarray = fit.stan_variable(var="z")
    assert z_as_ndarray.shape == (1, 4, 3)  # flattens chains
    z_as_xr = fit.draws_xr(vars="z")
    assert z_as_xr.z.data.shape == (1, 1, 4, 3)  # keeps chains
    for i in range(4):
        for j in range(3):
            assert int(z_as_ndarray[0, i, j]) == i + 1
            assert int(z_as_xr.z.data[0, 0, i, j]) == i + 1


def test_overlapping_names() -> None:
    stan = os.path.join(DATAFILES_PATH, 'normal-rng.stan')

    mod = CmdStanModel(stan_file=stan)
    # %Y to force same names
    fits = [
        mod.sample(data={}, time_fmt="%Y", iter_sampling=1, iter_warmup=1)
        for i in range(10)
    ]

    assert len(np.unique([fit.stan_variables()["x"][0] for fit in fits])) == 10


def test_complex_output() -> None:
    stan = os.path.join(DATAFILES_PATH, 'complex_var.stan')
    model = CmdStanModel(stan_file=stan)
    fit = model.sample(chains=1, iter_sampling=10)

    assert fit.stan_variable('zs').shape == (10, 2, 3)
    assert fit.stan_variable('z')[0] == 3 + 4j
    # make sure the name 'imag' isn't magic
    assert fit.stan_variable('imag').shape == (10, 2)

    np.testing.assert_allclose(
        fit.stan_variable('zs')[0], np.array([[3, 4j, 5], [1j, 2j, 3j]])
    )
    np.testing.assert_allclose(
        fit.stan_variable('zs_mat')[0],
        np.array([[3, 4j, 5], [1j, 2j, 3j]]),
    )

    assert "zs_dim_2" not in fit.draws_xr()
    # getting a raw scalar out of xarray is heavy
    assert fit.draws_xr().z.isel(chain=0, draw=1).data[()] == 3 + 4j
    np.testing.assert_allclose(
        fit.draws_xr().zs.isel(chain=0, draw=1).data,
        np.array([[3, 4j, 5], [1j, 2j, 3j]]),
    )


def test_attrs() -> None:
    stan = os.path.join(DATAFILES_PATH, 'named_output.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    fit = model.sample(chains=1, iter_sampling=10, data=jdata)

    assert fit.a[0] == 4.5
    assert fit.b.shape == (10, 3)
    assert fit.theta.shape == (10,)

    assert fit.thin == 1
    assert fit.stan_variable('thin')[0] == 3.5

    fit.draws()
    assert fit.stan_variable('draws')[0] == 0

    with pytest.raises(AttributeError, match='Unknown variable name:'):
        dummy = fit.c


def test_diagnostics(caplog: pytest.LogCaptureFixture) -> None:
    # centered 8 schools hits funnel
    stan = os.path.join(DATAFILES_PATH, 'eight_schools.stan')
    model = CmdStanModel(stan_file=stan)
    rdata = os.path.join(DATAFILES_PATH, 'eight_schools.data.R')
    with caplog.at_level(logging.WARNING):
        logging.getLogger()
        fit = model.sample(
            data=rdata,
            seed=55157,
        )
        assert not np.all(fit.divergences == 0)
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            re.compile(r'(?s).*Some chains may have failed to converge.*'),
        ),
    )

    with caplog.at_level(logging.WARNING):
        logging.getLogger()
        fit = model.sample(
            data=rdata,
            seed=40508,
            max_treedepth=3,
        )
        assert not np.all(fit.max_treedepths == 0)
    check_present(
        caplog,
        (
            'cmdstanpy',
            'WARNING',
            re.compile(r'(?s).*max treedepth*'),
        ),
    )

    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    fit = model.sample(
        data=jdata,
        iter_warmup=200,
        iter_sampling=100,
    )
    assert np.all(fit.divergences == 0)
    assert np.all(fit.max_treedepths == 0)

    # fixed_param returns None
    stan = os.path.join(DATAFILES_PATH, 'container_vars.stan')
    container_vars_model = CmdStanModel(stan_file=stan)
    fit = container_vars_model.sample(
        chains=1,
        iter_sampling=4,
        fixed_param=True,
        show_progress=False,
        show_console=False,
    )
    assert fit.max_treedepths is None
    assert fit.divergences is None


def test_timeout() -> None:
    stan = os.path.join(DATAFILES_PATH, 'timeout.stan')
    timeout_model = CmdStanModel(stan_file=stan)
    with pytest.raises(TimeoutError):
        timeout_model.sample(timeout=0.1, chains=1, data={'loop': 1})


def test_json_edges() -> None:
    stan = os.path.join(DATAFILES_PATH, 'data-test.stan')
    data_model = CmdStanModel(stan_file=stan)
    data = {"inf": float("inf"), "nan": float("NaN")}
    fit = data_model.sample(data, chains=1, iter_warmup=1, iter_sampling=1)
    assert np.isnan(fit.stan_variable("nan_out")[0])
    assert np.isinf(fit.stan_variable("inf_out")[0])

    data = {"inf": np.inf, "nan": np.nan}
    fit = data_model.sample(data, chains=1, iter_warmup=1, iter_sampling=1)
    assert np.isnan(fit.stan_variable("nan_out")[0])
    assert np.isinf(fit.stan_variable("inf_out")[0])


def test_json_junk_alongside_data() -> None:
    stan = os.path.join(DATAFILES_PATH, 'data-test.stan')
    data_model = CmdStanModel(stan_file=stan)
    data = {
        "inf": float("inf"),
        "nan": float("NaN"),
        "_foo": "this should be harmless!",
    }
    data_model.sample(data, chains=1, iter_warmup=1, iter_sampling=1)


def test_tuple_data_in() -> None:
    stan = os.path.join(DATAFILES_PATH, 'tuple_data.stan')
    data_model = CmdStanModel(stan_file=stan)
    data = {"x": (1, 2, 3), 'y': [(i, np.random.randn(4, 5)) for i in range(3)]}
    data_model.sample(data, chains=1, iter_warmup=1, iter_sampling=1)


def test_csv_roundtrip() -> None:
    stan = os.path.join(DATAFILES_PATH, 'matrix_var.stan')
    model = CmdStanModel(stan_file=stan)
    fit = model.sample(
        iter_sampling=10, iter_warmup=9, chains=2, save_warmup=True
    )
    z = fit.stan_variable(var="z")
    assert z.shape == (20, 4, 3)
    z_with_warmup = fit.stan_variable(var="z", inc_warmup=True)
    assert z_with_warmup.shape == (38, 4, 3)

    # mostly just asserting that from_output_files always succeeds
    # in parsing latest cmdstan headers
    assert fit.config_files is not None
    fit_from_output_files = from_output_files(fit.csv_files + fit.config_files)
    assert isinstance(fit_from_output_files, CmdStanMCMC)
    z_from_output_files = fit_from_output_files.stan_variable(var="z")
    assert z_from_output_files.shape == (20, 4, 3)
    z_with_warmup_from_output_files = fit.stan_variable(
        var="z", inc_warmup=True
    )
    assert z_with_warmup_from_output_files.shape == (38, 4, 3)


@pytest.mark.order(before="test_no_xarray")
def test_serialization(stanfile: str = 'bernoulli.stan') -> None:
    # This test must before any test that uses the `without_import` context
    # manager because the latter uses `reload` with side effects that affect
    # the consistency of classes.
    stan = os.path.join(DATAFILES_PATH, stanfile)
    bern_model = CmdStanModel(stan_file=stan)

    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_fit1 = bern_model.sample(
        data=jdata,
        chains=1,
        iter_warmup=200,
        iter_sampling=100,
        show_progress=False,
    )
    # Dump the result (which assembles draws) and delete the source files.
    dumped = pickle.dumps(bern_fit1)
    shutil.rmtree(os.path.dirname(bern_fit1.csv_files[0]))
    # Load the serialized result and compare results.
    bern_fit2: CmdStanMCMC = pickle.loads(dumped)
    variables1 = bern_fit1.stan_variables()
    variables2 = bern_fit2.stan_variables()
    assert set(variables1) == set(variables2)
    for key, value1 in variables1.items():
        np.testing.assert_array_equal(value1, variables2[key])


def test_mcmc_create_inits() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = cmdstanpy.CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    mcmc = bern_model.sample(data=jdata)

    inits = mcmc.create_inits()
    assert isinstance(inits, list)
    assert len(inits) == 4
    assert isinstance(inits[0], dict)
    assert 'theta' in inits[0]

    inits_10 = mcmc.create_inits(chains=10)
    assert isinstance(inits_10, list)
    assert len(inits_10) == 10

    inits_1 = mcmc.create_inits(chains=1)
    assert isinstance(inits_1, dict)
    assert 'theta' in inits_1
    assert len(inits_1) == 1

    seeded = mcmc.create_inits(seed=1234)
    seeded2 = mcmc.create_inits(seed=1234)
    assert isinstance(seeded, list)
    assert isinstance(seeded2, list)
    assert all(
        init1['theta'] == init2['theta']
        for init1, init2 in zip(seeded, seeded2)
    )


def test_mcmc_init_sampling() -> None:
    stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
    logistic_model = cmdstanpy.CmdStanModel(stan_file=stan)
    logistic_data = os.path.join(DATAFILES_PATH, 'logistic.data.R')

    initial_mcmc = logistic_model.sample(data=logistic_data)
    inits = initial_mcmc.create_inits()

    fit = logistic_model.sample(data=logistic_data, inits=inits)

    assert fit.chains == 4
    assert fit.draws().shape == (1000, 4, 9)


def test_sample_dense_mass_matrix() -> None:
    stan = os.path.join(DATAFILES_PATH, 'linear_regression.stan')
    jdata = os.path.join(DATAFILES_PATH, 'linear_regression.data.json')
    linear_model = CmdStanModel(stan_file=stan)

    fit = linear_model.sample(data=jdata, metric="dense_e", chains=2)
    assert fit.inv_metric is not None
    assert fit.inv_metric.shape == (2, 3, 3)


def test_no_output_draws() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    model = cmdstanpy.CmdStanModel(stan_file=stan)
    data = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    mcmc = model.sample(data=data, iter_sampling=0, save_warmup=False, chains=2)
    draws = mcmc.draws()
    assert np.array_equal(draws, np.empty((0, 2, len(mcmc.column_names))))


def test_config_output() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    model = CmdStanModel(stan_file=stan)
    fit = model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
    )
    assert all(os.path.exists(cf) for cf in fit.config_files)  # type: ignore

    # Config file naming differs when only a single chain is output
    fit_one_chain = model.sample(
        data=jdata,
        chains=1,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
    )
    assert all(
        os.path.exists(cf) for cf in fit_one_chain.config_files  # type: ignore
    )
