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
from test import check_present, raises_nested, without_import
from time import time

import numpy as np
import pytest

import cmdstanpy.stanfit
from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import CmdStanArgs, Method, SamplerArgs
from cmdstanpy.model import CmdStanModel
from cmdstanpy.stanfit import CmdStanMCMC, RunSet, from_csv
from cmdstanpy.utils import EXTENSION, cmdstan_version_before

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
def test_bernoulli_good(stanfile: str):
    stan = os.path.join(DATAFILES_PATH, stanfile)
    bern_model = CmdStanModel(stan_file=stan)

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
    assert 'CmdStanMCMC: model=bernoulli' in repr(bern_fit)
    assert 'method=sample' in repr(bern_fit)

    assert bern_fit.runset._args.method == Method.SAMPLE

    for i in range(bern_fit.runset.chains):
        csv_file = bern_fit.runset.csv_files[i]
        stdout_file = bern_fit.runset.stdout_files[i]
        assert os.path.exists(csv_file)
        assert os.path.exists(stdout_file)

    assert bern_fit.chains == 2
    assert bern_fit.thin == 1
    assert bern_fit.num_draws_warmup == 200
    assert bern_fit.num_draws_sampling == 100
    assert bern_fit.column_names == tuple(BERNOULLI_COLS)

    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))
    assert bern_fit.metric_type == 'diag_e'
    assert bern_fit.step_size.shape == (2,)
    assert bern_fit.metric.shape == (2, 1)

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
    assert 'CmdStanMCMC: model=bernoulli' in repr(bern_fit)
    assert 'method=sample' in repr(bern_fit)

    assert bern_fit.runset._args.method == Method.SAMPLE

    for i in range(bern_fit.runset.chains):
        csv_file = bern_fit.runset.csv_files[i]
        stdout_file = bern_fit.runset.stdout_files[i]
        assert os.path.exists(csv_file)
        assert os.path.exists(stdout_file)

    assert bern_fit.runset.chains == 2
    assert bern_fit.num_draws_sampling == 100
    assert bern_fit.column_names == tuple(BERNOULLI_COLS)

    bern_sample = bern_fit.draws()
    assert bern_sample.shape == (100, 2, len(BERNOULLI_COLS))
    assert bern_fit.metric_type == 'dense_e'
    assert bern_fit.step_size.shape == (2,)
    assert bern_fit.metric.shape == (2, 1, 1)

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
    for i in range(bern_fit.runset.chains):
        csv_file = bern_fit.runset.csv_files[i]
        stdout_file = bern_fit.runset.stdout_files[i]
        assert os.path.exists(csv_file)
        assert os.path.exists(stdout_file)
    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))
    for i in range(bern_fit.runset.chains):  # cleanup datafile_path dir
        os.remove(bern_fit.runset.csv_files[i])
        if os.path.exists(bern_fit.runset.stdout_files[i]):
            os.remove(bern_fit.runset.stdout_files[i])
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
def test_bernoulli_unit_e(
    stanfile: str, caplog: pytest.LogCaptureFixture
) -> None:
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
    assert bern_fit.step_size.shape == (2,)
    with caplog.at_level(logging.INFO):
        logging.getLogger()
        assert bern_fit.metric is None
    check_present(
        caplog,
        (
            'cmdstanpy',
            'INFO',
            'Unit diagnonal metric, inverse mass matrix size unknown.',
        ),
    )
    assert bern_fit.draws().shape == (100, 2, len(BERNOULLI_COLS))


def test_init_types() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        inits=1.1,
        show_progress=False,
    )
    assert 'init=1.1' in repr(bern_fit.runset)

    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        inits=1,
        show_progress=False,
    )
    assert 'init=1' in repr(bern_fit.runset)

    # Save init to json
    inits_path1 = os.path.join(_TMPDIR, 'inits_test_1.json')
    with open(inits_path1, 'w') as fd:
        json.dump({'theta': 0.1}, fd)
    inits_path2 = os.path.join(_TMPDIR, 'inits_test_2.json')
    with open(inits_path2, 'w') as fd:
        json.dump({'theta': 0.9}, fd)

    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        inits=inits_path1,
        show_progress=False,
    )
    assert 'init={}'.format(inits_path1.replace('\\', '\\\\')) in repr(
        bern_fit.runset
    )

    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=100,
        inits=[inits_path1, inits_path2],
        show_progress=False,
    )
    assert 'init={}'.format(inits_path1.replace('\\', '\\\\')) in repr(
        bern_fit.runset
    )

    with pytest.raises(ValueError):
        bern_model.sample(
            data=jdata,
            chains=2,
            parallel_chains=2,
            seed=12345,
            iter_warmup=100,
            iter_sampling=100,
            inits=(1, 2),
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
    # 2.28 compile with cpp_options={'STAN_THREADS':'true'}
    if not cmdstan_version_before(2, 28):
        logistic_stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
        logistic_model = CmdStanModel(stan_file=logistic_stan)

        os.remove(logistic_model.exe_file)
        logistic_model.compile(
            force=True,
            cpp_options={'STAN_THREADS': 'TRUE'},
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
    assert datagen_fit.runset._args.method == Method.SAMPLE
    assert datagen_fit.metric_type is None
    assert datagen_fit.metric is None
    assert datagen_fit.step_size is None
    assert datagen_fit.divergences is None
    assert datagen_fit.max_treedepths is None

    for i in range(datagen_fit.runset.chains):
        csv_file = datagen_fit.runset.csv_files[i]
        stdout_file = datagen_fit.runset.stdout_files[i]
        assert os.path.exists(csv_file)
        assert os.path.exists(stdout_file)

    assert datagen_fit.runset.chains == 1

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
    assert datagen_fit.metric is None
    assert datagen_fit.metric_type is None
    assert datagen_fit.step_size is None


def test_fixed_param_unspecified() -> None:
    stan = os.path.join(DATAFILES_PATH, 'datagen_poisson_glm.stan')
    datagen_model = CmdStanModel(stan_file=stan)
    datagen_fit = datagen_model.sample(iter_sampling=100, show_progress=False)
    assert datagen_fit.step_size is None
    summary = datagen_fit.summary()
    assert 'lp__' not in list(summary.index)

    exe_only = os.path.join(DATAFILES_PATH, 'exe_only')
    shutil.copyfile(datagen_model.exe_file, exe_only)
    os.chmod(exe_only, 0o755)
    datagen2_model = CmdStanModel(exe_file=exe_only)
    datagen2_fit = datagen2_model.sample(iter_sampling=200, show_console=True)
    assert datagen2_fit.chains == 4
    assert datagen2_fit.step_size is None
    summary = datagen2_fit.summary()
    assert 'lp__' not in list(summary.index)


def test_index_bounds_error() -> None:
    if not cmdstan_version_before(2, 27):
        oob_stan = os.path.join(DATAFILES_PATH, 'out_of_bounds.stan')
        oob_model = CmdStanModel(stan_file=oob_stan)
        with pytest.raises(RuntimeError):
            oob_model.sample()


def test_show_console(stanfile='bernoulli.stan'):
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


def test_show_progress(stanfile='bernoulli.stan'):
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
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs(
        iter_sampling=100, max_treedepth=11, adapt_delta=0.95
    )
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[1, 2, 3, 4],
        seed=12345,
        data=jdata,
        output_dir=DATAFILES_PATH,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=4)
    runset._csv_files = [
        os.path.join(DATAFILES_PATH, 'runset-good', 'bern-1.csv'),
        os.path.join(DATAFILES_PATH, 'runset-good', 'bern-2.csv'),
        os.path.join(DATAFILES_PATH, 'runset-good', 'bern-3.csv'),
        os.path.join(DATAFILES_PATH, 'runset-good', 'bern-4.csv'),
    ]
    assert 4 == runset.chains
    retcodes = runset._retcodes
    for i in range(len(retcodes)):
        runset._set_retcode(i, 0)
    assert runset._check_retcodes()

    fit = CmdStanMCMC(runset)
    assert 1000 == fit.num_draws_warmup
    assert 100 == fit.num_draws_sampling
    assert len(BERNOULLI_COLS) == len(fit.column_names)
    assert 'lp__' == fit.column_names[0]

    draws_pd = fit.draws_pd()
    assert draws_pd.shape == (
        fit.runset.chains * fit.num_draws_sampling,
        len(fit.column_names),
    )
    assert fit.draws_pd(vars=['theta']).shape == (400, 1)
    assert fit.draws_pd(vars=['lp__', 'theta']).shape == (400, 2)
    assert fit.draws_pd(vars=['theta', 'lp__']).shape == (400, 2)
    assert fit.draws_pd(vars='theta').shape == (400, 1)

    assert list(fit.draws_pd(vars=['theta', 'lp__']).columns) == [
        'theta',
        'lp__',
    ]
    assert list(fit.draws_pd(vars=['lp__', 'theta']).columns) == [
        'lp__',
        'theta',
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
    assert 'Treedepth satisfactory for all transitions.' in diagnostics
    assert 'No divergent transitions found.' in diagnostics
    assert 'E-BFMI satisfactory' in diagnostics
    assert 'Effective sample size satisfactory.' in diagnostics


def test_validate_big_run() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    sampler_args = SamplerArgs(iter_warmup=1500, iter_sampling=1000)
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[1, 2],
        seed=12345,
        output_dir=DATAFILES_PATH,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=2)
    runset._csv_files = [
        os.path.join(DATAFILES_PATH, 'runset-big', 'output_icar_nyc-1.csv'),
        os.path.join(DATAFILES_PATH, 'runset-big', 'output_icar_nyc-1.csv'),
    ]
    fit = CmdStanMCMC(runset)
    phis = ['phi[{}]'.format(str(x + 1)) for x in range(2095)]
    column_names = list(fit.metadata.method_vars_cols.keys()) + phis
    assert fit.num_draws_sampling == 1000
    assert fit.column_names == tuple(column_names)
    assert fit.metric_type == 'diag_e'
    assert fit.step_size.shape == (2,)
    assert fit.metric.shape == (2, 2095)
    assert fit.draws().shape == (1000, 2, 2102)
    assert fit.draws_pd(vars=['phi']).shape == (2000, 2095)
    with raises_nested(ValueError, r'Unknown variable: gamma'):
        fit.draws_pd(vars=['gamma'])


def test_instantiate_from_csvfiles() -> None:
    csvfiles_path = os.path.join(DATAFILES_PATH, 'runset-good')
    bern_fit = from_csv(path=csvfiles_path)
    draws_pd = bern_fit.draws_pd()
    assert draws_pd.shape == (
        bern_fit.runset.chains * bern_fit.num_draws_sampling,
        len(bern_fit.column_names),
    )
    csvfiles_path = os.path.join(DATAFILES_PATH, 'runset-big')
    big_fit = from_csv(path=csvfiles_path)
    draws_pd = big_fit.draws_pd()
    assert draws_pd.shape == (
        big_fit.runset.chains * big_fit.num_draws_sampling,
        len(big_fit.column_names),
    )
    # list
    csvfiles_path = os.path.join(DATAFILES_PATH, 'runset-good')
    csvfiles = []
    for file in os.listdir(csvfiles_path):
        if file.endswith(".csv"):
            csvfiles.append(os.path.join(csvfiles_path, file))
    bern_fit = from_csv(path=csvfiles)

    draws_pd = bern_fit.draws_pd()
    assert draws_pd.shape == (
        bern_fit.runset.chains * bern_fit.num_draws_sampling,
        len(bern_fit.column_names),
    )
    # single csvfile
    bern_fit = from_csv(path=csvfiles[0])
    draws_pd = bern_fit.draws_pd()
    assert draws_pd.shape == (
        bern_fit.num_draws_sampling,
        len(bern_fit.column_names),
    )
    # glob
    csvfiles_path = os.path.join(csvfiles_path, '*.csv')
    big_fit = from_csv(path=csvfiles_path)
    draws_pd = big_fit.draws_pd()
    assert draws_pd.shape == (
        big_fit.runset.chains * big_fit.num_draws_sampling,
        len(big_fit.column_names),
    )


def test_instantiate_from_csvfiles_fail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(ValueError, match=r'Must specify path'):
        from_csv(None)

    csvfiles_path = os.path.join(DATAFILES_PATH, 'runset-good')
    with pytest.raises(ValueError, match=r'Bad method argument'):
        from_csv(csvfiles_path, 'not-a-method')

    with pytest.raises(
        ValueError,
        match='Expecting Stan CSV output files from method ' 'optimize',
    ):
        from_csv(csvfiles_path, 'optimize')

    csvfiles = []
    with pytest.raises(ValueError, match=r'No CSV files found'):
        from_csv(csvfiles, 'sample')

    for file in os.listdir(csvfiles_path):
        csvfiles.append(os.path.join(csvfiles_path, file))
    with pytest.raises(ValueError, match=r'Bad CSV file path spec'):
        from_csv(csvfiles, 'sample')

    csvfiles_path = os.path.join(csvfiles_path, '*')
    with pytest.raises(ValueError, match=r'Bad CSV file path spec'):
        from_csv(csvfiles_path, 'sample')

    csvfiles_path = os.path.join(csvfiles_path, '*')
    with pytest.raises(ValueError, match=r'Invalid path specification'):
        from_csv(csvfiles_path, 'sample')

    csvfiles_path = os.path.join(DATAFILES_PATH, 'no-such-directory')
    with pytest.raises(ValueError, match=r'Invalid path specification'):
        from_csv(path=csvfiles_path)

    wrong_method_path = os.path.join(DATAFILES_PATH, 'from_csv')
    with caplog.at_level(logging.INFO):
        logging.getLogger()
        from_csv(path=wrong_method_path)
    check_present(
        caplog,
        (
            'cmdstanpy',
            'INFO',
            'Unable to process CSV output files from method diagnose.',
        ),
    )

    no_csvfiles_path = os.path.join(DATAFILES_PATH, 'test-fail-empty-directory')
    if os.path.exists(no_csvfiles_path):
        shutil.rmtree(no_csvfiles_path, ignore_errors=True)
    os.mkdir(no_csvfiles_path)
    with pytest.raises(ValueError, match=r'No CSV files found'):
        from_csv(path=no_csvfiles_path)
    if os.path.exists(no_csvfiles_path):
        shutil.rmtree(no_csvfiles_path, ignore_errors=True)


def test_from_csv_fixed_param() -> None:
    csv_path = os.path.join(DATAFILES_PATH, 'fixed_param_sample.csv')
    fixed_param_sample = from_csv(path=csv_path)
    assert fixed_param_sample.draws_pd().shape == (100, 85)


def test_custom_metric() -> None:
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    bern_model = CmdStanModel(stan_file=stan)
    jmetric = os.path.join(DATAFILES_PATH, 'bernoulli.metric.json')
    # just test that it runs without error
    bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
        metric=jmetric,
    )
    jmetric2 = os.path.join(DATAFILES_PATH, 'bernoulli.metric-2.json')
    bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
        metric=[jmetric, jmetric2],
    )
    # read json in as dict
    with open(jmetric) as fd:
        metric_dict_1 = json.load(fd)
    with open(jmetric2) as fd:
        metric_dict_2 = json.load(fd)
    bern_model.sample(
        data=jdata,
        chains=4,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
        metric=metric_dict_1,
    )
    bern_model.sample(
        data=jdata,
        chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
        metric=[metric_dict_1, metric_dict_2],
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
            iter_warmup=100,
            iter_sampling=200,
            metric=[metric_dict_1, metric_dict_2],
        )
    # metric mismatches - (not appropriate for bernoulli)
    with open(os.path.join(DATAFILES_PATH, 'metric_diag.data.json')) as fd:
        metric_dict_1 = json.load(fd)
    with open(os.path.join(DATAFILES_PATH, 'metric_dense.data.json')) as fd:
        metric_dict_2 = json.load(fd)
    with pytest.raises(
        ValueError, match='Found inconsistent "inv_metric" entry'
    ):
        bern_model.sample(
            data=jdata,
            chains=2,
            seed=12345,
            iter_warmup=100,
            iter_sampling=200,
            metric=[metric_dict_1, metric_dict_2],
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
            metric=some_dict,
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
    txt_file = bern_fit.runset.stdout_files[0]
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
    for i in range(bern_fit.runset.chains):
        csv_file = bern_fit.runset.csv_files[i]
        stdout_file = bern_fit.runset.stdout_files[i]
        assert os.path.exists(csv_file)
        assert os.path.exists(stdout_file)

    # save files to good dir
    bern_fit.save_csvfiles(dir=DATAFILES_PATH)
    for i in range(bern_fit.runset.chains):
        csv_file = bern_fit.runset.csv_files[i]
        assert os.path.exists(csv_file)
    with pytest.raises(ValueError, match='File exists, not overwriting: '):
        bern_fit.save_csvfiles(dir=DATAFILES_PATH)

    tmp2_dir = os.path.join(HERE, 'tmp2')
    os.mkdir(tmp2_dir)
    bern_fit.save_csvfiles(dir=tmp2_dir)
    for i in range(bern_fit.runset.chains):
        csv_file = bern_fit.runset.csv_files[i]
        assert os.path.exists(csv_file)
    for i in range(bern_fit.runset.chains):  # cleanup datafile_path dir
        os.remove(bern_fit.runset.csv_files[i])
        if os.path.exists(bern_fit.runset.stdout_files[i]):
            os.remove(bern_fit.runset.stdout_files[i])
    shutil.rmtree(tmp2_dir, ignore_errors=True)

    # regenerate to tmpdir, save to good dir
    bern_fit = bern_model.sample(
        data=jdata,
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_sampling=200,
    )
    bern_fit.save_csvfiles()  # default dir
    for i in range(bern_fit.runset.chains):
        csv_file = bern_fit.runset.csv_files[i]
        assert os.path.exists(csv_file)
    for i in range(bern_fit.runset.chains):  # cleanup default dir
        os.remove(bern_fit.runset.csv_files[i])
        if os.path.exists(bern_fit.runset.stdout_files[i]):
            os.remove(bern_fit.runset.stdout_files[i])

    with pytest.raises(ValueError, match='Cannot access CSV file'):
        bern_fit.save_csvfiles(dir=DATAFILES_PATH)

    if platform.system() != 'Windows':
        with pytest.raises(RuntimeError, match='Cannot save to path: '):
            dir = tempfile.mkdtemp(dir=_TMPDIR)
            os.chmod(dir, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            bern_fit.save_csvfiles(dir=dir)


def test_diagnose_divergences() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    sampler_args = SamplerArgs()
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[1],
        output_dir=DATAFILES_PATH,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=1)
    runset._csv_files = [
        os.path.join(DATAFILES_PATH, 'diagnose-good', 'corr_gauss_depth8-1.csv')
    ]
    fit = CmdStanMCMC(runset)
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
    for e in expected:
        assert e in diagnose


def test_validate_bad_run() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs(max_treedepth=11, adapt_delta=0.95)

    # some chains had errors
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[1, 2, 3, 4],
        seed=12345,
        data=jdata,
        output_dir=DATAFILES_PATH,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=4)
    for i in range(4):
        runset._set_retcode(i, 0)
    assert runset._check_retcodes()

    # errors reported
    runset._stdout_files = [
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-transcript-bern-1.txt'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-transcript-bern-2.txt'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-transcript-bern-3.txt'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-transcript-bern-4.txt'),
    ]
    assert 'Exception' in runset.get_err_msgs()

    # csv file headers inconsistent
    runset._csv_files = [
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-hdr-bern-1.csv'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-hdr-bern-2.csv'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-hdr-bern-3.csv'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-hdr-bern-4.csv'),
    ]
    with raises_nested(ValueError, 'CmdStan config mismatch'):
        CmdStanMCMC(runset)

    # bad draws
    runset._csv_files = [
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-draws-bern-1.csv'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-draws-bern-2.csv'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-draws-bern-3.csv'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-draws-bern-4.csv'),
    ]
    with raises_nested(ValueError, 'draws'):
        CmdStanMCMC(runset)

    # mismatch - column headers, draws
    runset._csv_files = [
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-cols-bern-1.csv'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-cols-bern-2.csv'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-cols-bern-3.csv'),
        os.path.join(DATAFILES_PATH, 'runset-bad', 'bad-cols-bern-4.csv'),
    ]
    with raises_nested(ValueError, 'bad draw, expecting 9 items, found 8'):
        CmdStanMCMC(runset)


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

    assert bern_fit.draws_pd().shape == (200, len(BERNOULLI_COLS))
    assert bern_fit.draws_pd(inc_warmup=False).shape == (
        200,
        len(BERNOULLI_COLS),
    )
    assert bern_fit.draws_pd(inc_warmup=True).shape == (
        600,
        len(BERNOULLI_COLS),
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
    assert 1 == len(bern_fit.metadata.stan_vars_dims)
    assert 'theta' in bern_fit.metadata.stan_vars_dims
    assert bern_fit.metadata.stan_vars_dims['theta'] == ()
    assert bern_fit.stan_variable(var='theta').shape == (200,)
    with pytest.raises(ValueError):
        bern_fit.stan_variable(var='eta')
    with pytest.raises(ValueError):
        bern_fit.stan_variable(var='lp__')


def test_variables_2d() -> None:
    csvfiles_path = os.path.join(DATAFILES_PATH, 'lotka-volterra.csv')
    fit = from_csv(path=csvfiles_path)
    assert 20 == fit.num_draws_sampling
    assert 8 == len(fit.metadata.stan_vars_dims)
    assert 'z' in fit.metadata.stan_vars_dims
    assert fit.metadata.stan_vars_dims['z'] == (20, 2)
    vars = fit.stan_variables()
    assert len(vars) == len(fit.metadata.stan_vars_dims)
    assert 'z' in vars
    assert vars['z'].shape == (20, 20, 2)
    assert 'theta' in vars
    assert vars['theta'].shape == (20, 4)


def test_variables_3d() -> None:
    # construct fit using existing sampler output
    csvfiles_path = os.path.join(DATAFILES_PATH, 'multidim_vars.csv')
    fit = from_csv(path=csvfiles_path)
    assert 20 == fit.num_draws_sampling
    assert 3 == len(fit.metadata.stan_vars_dims)
    assert 'y_rep' in fit.metadata.stan_vars_dims
    assert fit.metadata.stan_vars_dims['y_rep'] == (5, 4, 3)
    var_y_rep = fit.stan_variable(var='y_rep')
    assert var_y_rep.shape == (20, 5, 4, 3)
    var_beta = fit.stan_variable(var='beta')
    assert var_beta.shape, (20, 2)
    var_frac_60 = fit.stan_variable(var='frac_60')
    assert var_frac_60.shape == (20,)
    vars = fit.stan_variables()
    assert len(vars) == len(fit.metadata.stan_vars_dims)
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
    assert len(bern_fit.metadata.stan_vars_dims) == 1
    assert len(bern_fit.metadata.stan_vars_cols.keys()) == 1
    assert bern_fit.metric_type == 'diag_e'


def test_validate_sample_sig_figs(stanfile='bernoulli.stan'):
    if not cmdstan_version_before(2, 25):
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
    # construct CmdStanMCMC from logistic model output, config
    exe = os.path.join(DATAFILES_PATH, 'logistic' + EXTENSION)
    rdata = os.path.join(DATAFILES_PATH, 'logistic.data.R')
    sampler_args = SamplerArgs(iter_sampling=100)
    cmdstan_args = CmdStanArgs(
        model_name='logistic',
        model_exe=exe,
        chain_ids=[1, 2, 3, 4],
        seed=12345,
        data=rdata,
        output_dir=DATAFILES_PATH,
        sig_figs=17,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=4)
    runset._csv_files = [
        os.path.join(DATAFILES_PATH, 'logistic_output_1.csv'),
        os.path.join(DATAFILES_PATH, 'logistic_output_2.csv'),
        os.path.join(DATAFILES_PATH, 'logistic_output_3.csv'),
        os.path.join(DATAFILES_PATH, 'logistic_output_4.csv'),
    ]
    retcodes = runset._retcodes
    for i in range(len(retcodes)):
        runset._set_retcode(i, 0)
    fit = CmdStanMCMC(runset)

    sum_default = fit.summary()
    beta1_default = format(sum_default.iloc[1, 0], '.18g')
    assert beta1_default.startswith('1.3')

    if not cmdstan_version_before(2, 25):
        sum_17 = fit.summary(sig_figs=17)
        beta1_17 = format(sum_17.iloc[1, 0], '.18g')
        assert beta1_17.startswith('1.345767078273')

        sum_10 = fit.summary(sig_figs=10)
        beta1_10 = format(sum_10.iloc[1, 0], '.18g')
        assert beta1_10.startswith('1.34576707')

    with pytest.raises(ValueError):
        fit.summary(sig_figs=20)
    with pytest.raises(ValueError):
        fit.summary(sig_figs=-1)


def test_metadata() -> None:
    # construct CmdStanMCMC from logistic model output, config
    exe = os.path.join(DATAFILES_PATH, 'logistic' + EXTENSION)
    rdata = os.path.join(DATAFILES_PATH, 'logistic.data.R')
    sampler_args = SamplerArgs(iter_sampling=100)
    cmdstan_args = CmdStanArgs(
        model_name='logistic',
        model_exe=exe,
        chain_ids=[1, 2, 3, 4],
        seed=12345,
        data=rdata,
        output_dir=DATAFILES_PATH,
        sig_figs=17,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=4)
    runset._csv_files = [
        os.path.join(DATAFILES_PATH, 'logistic_output_1.csv'),
        os.path.join(DATAFILES_PATH, 'logistic_output_2.csv'),
        os.path.join(DATAFILES_PATH, 'logistic_output_3.csv'),
        os.path.join(DATAFILES_PATH, 'logistic_output_4.csv'),
    ]
    retcodes = runset._retcodes
    for i in range(len(retcodes)):
        runset._set_retcode(i, 0)
    fit = CmdStanMCMC(runset)
    meta = fit.metadata
    assert meta.cmdstan_config['model'] == 'logistic_model'
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

    assert fit.metadata.cmdstan_config['num_samples'] == 100
    assert fit.metadata.cmdstan_config['thin'] == 1
    assert fit.metadata.cmdstan_config['algorithm'] == 'hmc'
    assert fit.metadata.cmdstan_config['metric'] == 'diag_e'
    np.testing.assert_almost_equal(fit.metadata.cmdstan_config['delta'], 0.80)

    assert 'n_leapfrog__' in fit.metadata.method_vars_cols
    assert 'energy__' in fit.metadata.method_vars_cols
    assert 'beta' not in fit.metadata.method_vars_cols
    assert 'energy__' not in fit.metadata.stan_vars_dims
    assert 'beta' in fit.metadata.stan_vars_dims
    assert 'beta' in fit.metadata.stan_vars_cols
    assert fit.metadata.stan_vars_dims['beta'] == (2,)
    assert fit.metadata.stan_vars_cols['beta'] == (7, 8)


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
    for i in range(bern_fit.runset.chains):
        diagnostics_file = bern_fit.runset.diagnostic_files[i]
        assert os.path.exists(diagnostics_file)


def test_save_profile() -> None:
    stan = os.path.join(DATAFILES_PATH, 'profile_likelihood.stan')
    profile_model = CmdStanModel(stan_file=stan)
    profile_fit = profile_model.sample(
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_warmup=100,
        iter_sampling=200,
        save_profile=True,
    )
    for i in range(profile_fit.runset.chains):
        profile_file = profile_fit.runset.profile_files[i]
        assert os.path.exists(profile_file)

    profile_fit = profile_model.sample(
        chains=2,
        parallel_chains=2,
        seed=12345,
        iter_sampling=200,
        save_latent_dynamics=True,
        save_profile=True,
    )

    for i in range(profile_fit.runset.chains):
        profile_file = profile_fit.runset.profile_files[i]
        assert os.path.exists(profile_file)
        diagnostics_file = profile_fit.runset.diagnostic_files[i]
        assert os.path.exists(diagnostics_file)


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


@pytest.mark.order(before="test_no_xarray")
def test_serialization(stanfile='bernoulli.stan'):
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
    shutil.rmtree(bern_fit1.runset._output_dir)
    # Load the serialized result and compare results.
    bern_fit2: CmdStanMCMC = pickle.loads(dumped)
    variables1 = bern_fit1.stan_variables()
    variables2 = bern_fit2.stan_variables()
    assert set(variables1) == set(variables2)
    for key, value1 in variables1.items():
        np.testing.assert_array_equal(value1, variables2[key])
