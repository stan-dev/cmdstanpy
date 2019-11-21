import math
import os
import platform
import re
import subprocess
import shutil
import sys
import logging

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Union

from cmdstanpy.cmdstan_args import (
    CmdStanArgs,
    SamplerArgs,
    OptimizeArgs,
    GenerateQuantitiesArgs,
    VariationalArgs,
)
from cmdstanpy.stanfit import (
    RunSet,
    CmdStanMCMC,
    CmdStanMLE,
    CmdStanGQ,
    CmdStanVB,
)
from cmdstanpy.utils import (
    do_command,
    EXTENSION,
    cmdstan_path,
    MaybeDictToFilePath,
    TemporaryCopiedFile,
    get_logger,
)


class CmdStanModel(object):
    """
    Stan model.

    + Stores pathnames to Stan program, compiled executable, and list of
        paths for directories which contain included Stan programs.

    + Provides functions to compile the model and perform inference on the
        model given data.

    + By default, compiles model on instantiation - override with argument
        ``compile=False``
    """

    def __init__(
        self,
        stan_file: str = None,
        exe_file: str = None,
        include_paths: List[str] = None,
        compile: bool = True,
        logger: logging.Logger = None,
    ) -> None:
        """Initialize object."""
        self._stan_file = stan_file
        self._name = None
        self._exe_file = None
        self._include_paths = None
        self._logger = logger or get_logger()

        if stan_file is None:
            if exe_file is None:
                raise ValueError(
                    'must specify Stan source or executable program file'
                )
        else:
            if not os.path.exists(stan_file):
                raise ValueError('no such file {}'.format(self._stan_file))
            _, filename = os.path.split(stan_file)
            if len(filename) < 6 or not filename.endswith('.stan'):
                raise ValueError(
                    'invalid stan filename {}'.format(self._stan_file)
                )
            self._name, _ = os.path.splitext(filename)
            self._exe_file = None
            # if program has #includes, search program dir
            with open(stan_file, 'r') as fp:
                program = fp.read()
            if '#include' in program:
                path, _ = os.path.split(stan_file)
                if include_paths is None:
                    include_paths = []
                if path not in include_paths:
                    include_paths.append(path)

        if exe_file is not None:
            if not os.path.exists(exe_file):
                raise ValueError('no such file {}'.format(exe_file))
            _, exename = os.path.split(exe_file)
            if self._name is None:
                self._name, _ = os.path.splitext(exename)
            else:
                if self._name != os.path.splitext(exename)[0]:
                    raise ValueError(
                        'name mismatch between Stan file and compiled'
                        ' executable, expecting basename: {}'
                        ' found: {}'.format(self._name, exename)
                    )
            self._exe_file = exe_file

        if include_paths is not None:
            bad_paths = [d for d in include_paths if not os.path.exists(d)]
            if any(bad_paths):
                raise ValueError(
                    'invalid include paths: {}'.format(', '.join(bad_paths))
                )
            self._include_paths = include_paths

        if platform.system() == 'Windows':
            # Add tbb to the $PATH on Windows
            libtbb = os.getenv('STAN_TBB')
            if libtbb is None:
                libtbb = os.path.join(
                    cmdstan_path(), 'stan', 'lib', 'stan_math', 'lib', 'tbb'
                )
            os.environ['PATH'] = ';'.join(list(OrderedDict.fromkeys(
                [libtbb, ] + os.getenv('PATH', '').split(';')
            )))

        if compile and self._exe_file is None:
            self.compile()
            if self._exe_file is None:
                raise ValueError(
                    'unable to compile Stan model file: {}'.format(
                        self._stan_file
                    )
                )

    def __repr__(self) -> str:
        return 'CmdStanModel(name={},  stan_file="{}", exe_file="{}")'.format(
            self._name, self._stan_file, self._exe_file
        )

    def code(self) -> str:
        """Return Stan program as a string."""
        if not self._stan_file:
            raise RuntimeError('Please specify source file')

        code = None
        try:
            with open(self._stan_file, 'r') as fd:
                code = fd.read()
        except IOError:
            self._logger.error(
                'Cannot read file Stan file: {}'.format(self._stan_file)
            )
        return code

    @property
    def name(self) -> str:
        return self._name

    @property
    def stan_file(self) -> str:
        return self._stan_file

    @property
    def exe_file(self) -> str:
        return self._exe_file

    @property
    def include_paths(self) -> List[str]:
        return self._include_paths

    def compile(self, opt_lvl: int = 3, force: bool = False) -> None:
        """
        Compile the given Stan program file.  Translates the Stan code to
        C++, then calls the C++ compiler.

        By default, this function compares the timestamps on the source and
        executable files; if the executable is newer than the source file, it
        will not recompile the file, unless argument ``force`` is True.

        :param opt_lvl: Optimization level used by the C++ compiler, one of
            {0, 1, 2, 3}.  Defaults to level 2. Level 0 optimization results
            in the shortest compilation time with code that may run slowly.
            Higher optimization levels increase runtime performance but will
            take longer to compile.

        :param force: When ``True``, always compile, even if the executable file
            is newer than the source file.  Used for Stan models which have
            ``#include`` directives in order to force recompilation when changes
            are made to the included files.
        """
        if not self._stan_file:
            raise RuntimeError('Please specify source file')

        compilation_failed = False

        with TemporaryCopiedFile(self._stan_file) as (stan_file, is_copied):
            exe_file, _ = os.path.splitext(os.path.abspath(stan_file))
            exe_file = Path(exe_file).as_posix()
            exe_file += EXTENSION
            do_compile = True
            if os.path.exists(exe_file):
                src_time = os.path.getmtime(self._stan_file)
                exe_time = os.path.getmtime(exe_file)
                if exe_time > src_time and not force:
                    do_compile = False
                    self._logger.info('found newer exe file, not recompiling')

            if do_compile:
                make = os.getenv(
                    'MAKE',
                    'make'
                    if platform.system() != 'Windows'
                    else 'mingw32-make',
                )
                hpp_file = os.path.splitext(stan_file)[0] + '.hpp'
                hpp_file = Path(hpp_file).as_posix()
                if not os.path.exists(hpp_file):
                    self._logger.info('stan to c++ (%s)', hpp_file)
                    cmd = [
                        make,
                        Path(exe_file).as_posix(),
                        'STANCFLAGS+=--o={}'.format(hpp_file)
                    ]
                    if self._include_paths is not None:
                        bad_paths = [
                            d
                            for d in self._include_paths
                            if not os.path.exists(d)
                        ]
                        if any(bad_paths):
                            raise ValueError(
                                'invalid include paths: {}'.format(
                                    ', '.join(bad_paths)
                                )
                            )
                        cmd.append(
                            'STANCFLAGS+=--include_paths=' +
                            ','.join(
                                (
                                    Path(p).as_posix()
                                    for p in self._include_paths
                                )
                            )
                        )
                    try:
                        do_command(cmd, cmdstan_path(), logger=self._logger)
                    except Exception as e:
                        self._logger.error('file {}, {}'.format(stan_file, e))
                        compilation_failed = True

                if not compilation_failed:
                    cmd = [make, 'O={}'.format(opt_lvl), exe_file]
                    self._logger.info('compiling c++')
                    try:
                        do_command(cmd, cmdstan_path(), logger=self._logger)
                    except Exception as e:
                        self._logger.error('make cmd failed %s', e)
                        compilation_failed = True

            if not compilation_failed:
                if is_copied:
                    original_target_dir = os.path.dirname(
                        os.path.abspath(self._stan_file)
                    )
                    new_exec_name = (
                        os.path.basename(os.path.splitext(self._stan_file)[0])
                        + EXTENSION
                    )
                    self._exe_file = os.path.join(
                        original_target_dir, new_exec_name
                    )
                    shutil.copy(exe_file, self._exe_file)
                else:
                    self._exe_file = exe_file
                self._logger.info('compiled model file: %s', self._exe_file)
            else:
                self._logger.error('model compilation failed')

    def optimize(
        self,
        data: Union[Dict, str] = None,
        seed: int = None,
        inits: Union[Dict, float, str] = None,
        csv_basename: str = None,
        algorithm: str = None,
        init_alpha: float = None,
        iter: int = None,
    ) -> CmdStanMLE:
        """
        Wrapper for optimize call

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param seed: The seed for random number generator. Must be an integer
            between ``0`` and ``2^32 - 1``. If unspecified,
            ``numpy.random.RandomState()``
            is used to generate a seed which will be used for all chains.

        :param inits:  Specifies how the sampler initializes parameter values.
            Initializiation is either uniform random on a range centered on 0,
            exactly 0, or a dictionary or file of initial values for some or
            all parameters in the model.  The default initialization behavoir
            will initialize all parameter values on range [-2, 2] on the
            _unconstrained_ support.  If the expected parameter values are
            too far from this range, this option may improve estimation.
            The following value types are allowed:

            * Single number ``n > 0`` - initialization range is [-n, n].
            * ``0`` - all parameters are initialized to 0.
            * dictionary - pairs parameter name : initial value.
            * string - pathname to a JSON or Rdump data file.

        :param csv_basename:  A path or file name which will be used as the
            basename for the CmdStan output files.  The csv output files
            are written to file ``<basename>-0.csv`` and the console output
            and error messages are written to file ``<basename>-0.txt``.

        :param algorithm: Algorithm to use. One of: "BFGS", "LBFGS", "Newton"

        :param init_alpha: Line search step size for first iteration

        :param iter: Total number of iterations

        :return: CmdStanMLE object
        """
        optimize_args = OptimizeArgs(
            algorithm=algorithm, init_alpha=init_alpha, iter=iter
        )

        with MaybeDictToFilePath(data, inits) as (_data, _inits):
            args = CmdStanArgs(
                self._name,
                self._exe_file,
                chain_ids=None,
                data=_data,
                seed=seed,
                inits=_inits,
                output_basename=csv_basename,
                method_args=optimize_args,
            )

            dummy_chain_id = 0
            runset = RunSet(args=args, chains=1)
            self._run_cmdstan(runset, dummy_chain_id)

        if not runset._check_retcodes():
            msg = 'Error during optimizing'
            if runset._retcode(dummy_chain_id) != 0:
                msg = '{}, error code {}'.format(
                    msg, runset._retcode(dummy_chain_id)
                )
                raise RuntimeError(msg)
        mle = CmdStanMLE(runset)
        mle._set_mle_attrs(runset.csv_files[0])
        return mle

    def sample(
        self,
        data: Union[Dict, str] = None,
        chains: Union[int, None] = None,
        cores: Union[int, None] = None,
        seed: Union[int, List[int]] = None,
        chain_ids: Union[int, List[int]] = None,
        inits: Union[Dict, float, str, List[str]] = None,
        warmup_iters: int = None,
        sampling_iters: int = None,
        save_warmup: bool = False,
        thin: int = None,
        max_treedepth: float = None,
        metric: Union[str, List[str]] = None,
        step_size: Union[float, List[float]] = None,
        adapt_engaged: bool = True,
        adapt_delta: float = None,
        fixed_param: bool = False,
        csv_basename: str = None,
        show_progress: Union[bool, str] = False,
    ) -> CmdStanMCMC:
        """
        Run or more chains of the NUTS sampler to produce a set of draws
        from the posterior distribution of a model conditioned on some data.

        This function validates the specified configuration, composes a call to
        the CmdStan ``sample`` method and spawns one subprocess per chain to run
        the sampler and waits for all chains to run to completion.
        Unspecified arguments are not included in the call to CmdStan, i.e.,
        those arguments will have CmdStan default values.

        For each chain, the ``CmdStanMCMC`` object records the command,
        the return code, the sampler output file paths, and the corresponding
        subprocess console outputs, if any.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param chains: Number of sampler chains, should be > 1.

        :param cores: Number of processes to run in parallel. Must be an
            integer between 1 and the number of CPUs in the system.
            If none then set automatically to `chains` but no more
            than `total_cpu_count - 2`

        :param seed: The seed for random number generator. Must be an integer
            between ``0`` and ``2^32 - 1``. If unspecified,
            ``numpy.random.RandomState()``
            is used to generate a seed which will be used for all chains.
            When the same seed is used across all chains,
            the chain-id is used to advance the RNG to avoid dependent samples.

        :param chain_ids: The offset for the random number generator, either
            an integer or a list of unique per-chain offsets.  If unspecified,
            chain ids are numbered sequentially starting from 1.

        :param inits: Specifies how the sampler initializes parameter values.
            Initializiation is either uniform random on a range centered on 0,
            exactly 0, or a dictionary or file of initial values for some or all
            parameters in the model.  The default initialization behavoir will
            initialize all parameter values on range [-2, 2] on the
            _unconstrained_ support.  If the expected parameter values are
            too far from this range, this option may improve adaptation.
            The following value types are allowed:

            * Single number ``n > 0`` - initialization range is [-n, n].
            * ``0`` - all parameters are initialized to 0.
            * dictionary - pairs parameter name : initial value.
            * string - pathname to a JSON or Rdump data file.
            * list of strings - per-chain pathname to data file.

        :param warmup_iters: Number of warmup iterations for each chain.

        :param sampling_iters: Number of draws from the posterior for each
            chain.

        :param save_warmup: When True, sampler saves warmup draws as part of
            the Stan csv output file.

        :param thin: Period between saved samples.

        :param max_treedepth: Maximum depth of trees evaluated by NUTS sampler
            per iteration.

        :param metric: Specification of the mass matrix, either as a
            vector consisting of the diagonal elements of the covariance
            matrix (``diag`` or ``diag_e``) or the full covariance matrix
            (``dense`` or ``dense_e``).

            If the value of the metric argument is a string other than
            ``diag``, ``diag_e``, ``dense``, or ``dense_e``, it must be
            a valid filepath to a JSON or Rdump file which contains an entry
            ``inv_metric`` whose value is either the diagonal vector or
            the full covariance matrix.

            If the value of the metric argument is a list of paths, its
            length must match the number of chains and all paths must be
            unique.

        :param step_size: Initial stepsize for HMC sampler.  The value is either
            a single number or a list of numbers which will be used as the
            global or per-chain initial step_size, respectively.
            The length of the list of step sizes must match the number of
            chains.

        :param adapt_engaged: When True, adapt stepsize and metric.
            *Note: If True, ``warmup_iters`` must be > 0.*

        :param adapt_delta: Adaptation target Metropolis acceptance rate.
            The default value is 0.8.  Increasing this value, which must be
            strictly less than 1, causes adaptation to use smaller step sizes.
            It improves the effective sample size, but may increase the time
            per iteration.

        :param fixed_param: When True, call CmdStan with argument
            "algorithm=fixed_param" which runs the sampler without
            updating the Markov Chain, thus the values of all parameters and
            transformed parameters are constant across all draws and
            only those values in the generated quantities block that are
            produced by RNG functions may change.  This provides
            a way to use Stan programs to generate simulated data via the
            generated quantities block.  This option must be used when the
            parameters block is empty.  Default value is False.

        :param csv_basename: A path or file name which will be used as the
            basename for the sampler output files.  The csv output files
            for each chain are written to file ``<basename>-<chain_id>.csv``
            and the console output and error messages are written to file
            ``<basename>-<chain_id>.txt``.

        :param show_progress: Use tqdm progress bar to show sampling progress.
            If show_progress=='notebook' use tqdm_notebook
            (needs nodejs for jupyter).

        :return: CmdStanMCMC object
        """

        if chains is None:
            if fixed_param:
                chains = 1
            else:
                chains = 4
        if chains < 1:
            raise ValueError(
                'chains must be a positive integer value, found {}'.format(
                    chains
                )
            )

        if chain_ids is None:
            chain_ids = [x + 1 for x in range(chains)]
        else:
            if type(chain_ids) is int:
                if chain_ids < 1:
                    raise ValueError(
                        'chain_id must be a positive integer value,'
                        ' found {}'.format(chain_ids)
                    )
                offset = chain_ids
                chain_ids = [x + offset + 1 for x in range(chains)]
            else:
                if not len(chain_ids) == chains:
                    raise ValueError(
                        'chain_ids must correspond to number of chains'
                        ' specified {} chains, found {} chain_ids'.format(
                            chains, len(chain_ids)
                        )
                    )
                for i in len(chain_ids):
                    if chain_ids[i] < 1:
                        raise ValueError(
                            'chain_id must be a positive integer value,'
                            ' found {}'.format(chain_ids[i])
                        )

        cores_avail = cpu_count()
        if cores is None:
            cores = max(min(cores_avail - 2, chains), 1)
        if cores < 1:
            raise ValueError(
                'cores must be a positive integer value, found {}'.format(cores)
            )
        if cores > cores_avail:
            self._logger.warning(
                'requested %u cores, only %u available', cores, cpu_count()
            )
            cores = cores_avail

        refresh = None
        if show_progress:
            try:
                import tqdm

                # progress bar updates - 100 per warmup, sampling
                if fixed_param or not adapt_engaged or warmup_iters == 0:
                    num_updates = 100
                    w_iters = 0
                else:
                    num_updates = 200
                    if warmup_iters is None:
                        w_iters = 1000
                    else:
                        w_iters = warmup_iters
                s_iters = sampling_iters
                if s_iters is None:
                    s_iters = 1000
                refresh = max(int((s_iters + w_iters) // num_updates), 1)
                # disable logger for console (temporary) - use tqdm
                self._logger.propagate = False
            except ImportError:
                self._logger.warning(
                    (
                        'tqdm not installed, progress information is not '
                        'shown. Please install tqdm with '
                        "'pip install tqdm'"
                    )
                )
                show_progress = False

        # TODO:  issue 49: inits can be initialization function

        sampler_args = SamplerArgs(
            warmup_iters=warmup_iters,
            sampling_iters=sampling_iters,
            save_warmup=save_warmup,
            thin=thin,
            max_treedepth=max_treedepth,
            metric=metric,
            step_size=step_size,
            adapt_engaged=adapt_engaged,
            adapt_delta=adapt_delta,
            fixed_param=fixed_param,
        )
        with MaybeDictToFilePath(data, inits) as (_data, _inits):
            args = CmdStanArgs(
                self._name,
                self._exe_file,
                chain_ids=chain_ids,
                data=_data,
                seed=seed,
                inits=_inits,
                output_basename=csv_basename,
                method_args=sampler_args,
                refresh=refresh,
            )

            runset = RunSet(args=args, chains=chains)

            pbar = None
            pbar_dict = {}
            with ThreadPoolExecutor(max_workers=cores) as executor:
                for i in range(chains):
                    if show_progress:
                        if (
                            isinstance(show_progress, str)
                            and show_progress.lower() == 'notebook'
                        ):
                            try:
                                tqdm_pbar = tqdm.tqdm_notebook
                            except ImportError:
                                msg = (
                                    'Cannot import tqdm.tqdm_notebook.\n'
                                    'Functionality is only supported on the '
                                    'Jupyter Notebook and compatible platforms'
                                    '.\nPlease follow the instructions in '
                                    'https://github.com/tqdm/tqdm/issues/394#'
                                    'issuecomment-384743637 and remember to '
                                    'stop & start your jupyter server.'
                                )
                                self._logger.warning(msg)
                                tqdm_pbar = tqdm.tqdm
                        else:
                            tqdm_pbar = tqdm.tqdm
                        # enable dynamic_ncols for advanced users
                        # currently hidden feature
                        dynamic_ncols = os.environ.get(
                            'TQDM_DYNAMIC_NCOLS', 'False'
                        )
                        if dynamic_ncols.lower() in ['0', 'false']:
                            dynamic_ncols = False
                        else:
                            dynamic_ncols = True
                        pbar = [
                            # warmup
                            tqdm_pbar(
                                desc='Chain {} - warmup'.format(i + 1),
                                position=i * 2,
                                total=sampler_args.warmup_iters,
                                dynamic_ncols=dynamic_ncols,
                            ),
                            # sampling
                            tqdm_pbar(
                                desc='Chain {} - sample'.format(i + 1),
                                position=i * 2 + 1,
                                total=sampler_args.sampling_iters,
                                dynamic_ncols=dynamic_ncols,
                            ),
                        ]

                    future = executor.submit(self._run_cmdstan, runset, i, pbar)
                    pbar_dict[future] = pbar
                if show_progress:
                    for future in as_completed(pbar_dict):
                        pbar = pbar_dict[future]
                        for pbar_item in pbar:
                            # close here to just to be sure
                            pbar_item.close()

            if show_progress:
                # re-enable logger for console
                self._logger.propagate = True
            if not runset._check_retcodes():
                msg = 'Error during sampling'
                for i in range(chains):
                    if runset._retcode(i) != 0:
                        msg = '{}, chain {} returned error code {}'.format(
                            msg, i, runset._retcode(i)
                        )
                raise RuntimeError(msg)
            mcmc = CmdStanMCMC(runset, fixed_param)
            mcmc._validate_csv_files()
        return mcmc

    def generate_quantities(
        self,
        data: Union[Dict, str] = None,
        mcmc_sample: Union[CmdStanMCMC, List[str]] = None,
        seed: int = None,
        gq_csv_basename: str = None,
    ) -> CmdStanGQ:
        """
        Wrapper for generated quantities call.  Given a CmdStanMCMC object
        containing a sample from the fitted model, along with the
        corresponding dataset for that fit, run just the generated quantities
        block of the model in order to get additional quantities of interest.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param mcmc_sample: Can be either a CmdStanMCMC object returned by
            CmdStanPy's `sample` method or a list of stan-csv files generated
            by fitting the model to the data using any Stan interface.

        :param seed: The seed for random number generator. Must be an integer
            between ``0`` and ``2^32 - 1``. If unspecified,
            ``numpy.random.RandomState()``
            is used to generate a seed which will be used for all chains.
            *NOTE: Specifying the seed will guarantee the same result for
            multiple invocations of this method with the same inputs.  However
            this will not reproduce results from the sample method given
            the same inputs because the RNG will be in a different state.*

        :param gq_csv_basename: A path or file name which will be used as the
            basename for the sampler output files.  The csv output files
            for each chain are written to file ``<basename>-<chain_id>.csv``
            and the console output and error messages are written to file
            ``<basename>-<chain_id>.txt``.

        :return: CmdStanGQ object
        """
        sample_csv_files = []
        sample_drawset = None
        chains = 0

        if isinstance(mcmc_sample, CmdStanMCMC):
            sample_csv_files = mcmc_sample.runset.csv_files
            sample_drawset = mcmc_sample.get_drawset()
            chains = mcmc_sample.chains
        elif isinstance(mcmc_sample, list):
            sample_csv_files = mcmc_sample
        else:
            raise ValueError(
                'mcmc_sample must be either CmdStanMCMC object'
                ' or list of paths to sample csv_files'
            )

        try:
            chains = len(sample_csv_files)
            if sample_drawset is None:  # assemble sample from csv files
                sampler_args = SamplerArgs()
                args = CmdStanArgs(
                    self._name,
                    self._exe_file,
                    chain_ids=[x + 1 for x in range(chains)],
                    method_args=sampler_args,
                )
                runset = RunSet(args=args, chains=chains)
                runset._csv_files = sample_csv_files
                sample_fit = CmdStanMCMC(runset)
                sample_fit._validate_csv_files()
                sample_drawset = sample_fit.get_drawset()
        except Exception as e:
            raise ValueError(
                'Invalid mcmc_sample, error:\n\t{}\n\t'
                ' while processing files\n\t{}'.format(
                    str(e), '\n\t'.join(sample_csv_files)
                )
            )

        generate_quantities_args = GenerateQuantitiesArgs(
            csv_files=sample_csv_files
        )
        generate_quantities_args.validate(chains)
        with MaybeDictToFilePath(data, None) as (_data, _inits):
            args = CmdStanArgs(
                self._name,
                self._exe_file,
                chain_ids=[x + 1 for x in range(chains)],
                data=_data,
                seed=seed,
                output_basename=gq_csv_basename,
                method_args=generate_quantities_args,
            )
            runset = RunSet(args=args, chains=chains)

            cores_avail = cpu_count()
            cores = max(min(cores_avail - 2, chains), 1)
            with ThreadPoolExecutor(max_workers=cores) as executor:
                for i in range(chains):
                    executor.submit(self._run_cmdstan, runset, i)

            if not runset._check_retcodes():
                msg = 'Error during generate_quantities'
                for i in range(chains):
                    if runset._retcode(i) != 0:
                        msg = '{}, chain {} returned error code {}'.format(
                            msg, i, runset._retcode(i)
                        )
                raise RuntimeError(msg)
            quantities = CmdStanGQ(runset=runset, mcmc_sample=sample_drawset)
            quantities._set_attrs_gq_csv_files(sample_csv_files[0])
        return quantities

    def variational(
        self,
        data: Union[Dict, str] = None,
        seed: int = None,
        inits: float = None,
        csv_basename: str = None,
        algorithm: str = None,
        iter: int = None,
        grad_samples: int = None,
        elbo_samples: int = None,
        eta: Real = None,
        adapt_iter: int = None,
        tol_rel_obj: Real = None,
        eval_elbo: int = None,
        output_samples: int = None,
    ) -> CmdStanVB:
        """
        Run CmdStan's variational inference algorithm to approximate
        the posterior distribution of the model conditioned on the data.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param seed: The seed for random number generator. Must be an integer
            between ``0`` and ``2^32 - 1``. If unspecified,
            ``numpy.random.RandomState()``
            is used to generate a seed which will be used for all chains.

        :param inits:  Specifies how the sampler initializes parameter values.
            Initializiation is uniform random on a range centered on ``0`` with
            default range of ``2``. Specifying a single number ``n > 0`` changes
            the initialization range to ``[-n, n]``.

        :param csv_basename:  A path or file name which will be used as the
            basename for the CmdStan output files.  The csv output files
            are written to file ``<basename>-0.csv`` and the console output
            and error messages are written to file ``<basename>-0.txt``.

        :param algorithm: Algorithm to use. One of: "meanfield", "fullrank".

        :param iter: Maximum number of ADVI iterations.

        :param grad_samples: Number of MC draws for computing the gradient.

        :param elbo_samples: Number of MC draws for estimate of ELBO.

        :param eta: Stepsize scaling parameter.

        :param adapt_iter: Number of iterations for eta adaptation.

        :param tol_rel_obj: Relative tolerance parameter for convergence.

        :param eval_elbo: Number of interations between ELBO evaluations.

        :param output_samples: Number of approximate posterior output draws
            to save.

        :return: CmdStanVB object
        """
        variational_args = VariationalArgs(
            algorithm=algorithm,
            iter=iter,
            grad_samples=grad_samples,
            elbo_samples=elbo_samples,
            eta=eta,
            adapt_iter=adapt_iter,
            tol_rel_obj=tol_rel_obj,
            eval_elbo=eval_elbo,
            output_samples=output_samples,
        )

        with MaybeDictToFilePath(data, inits) as (_data, _inits):
            args = CmdStanArgs(
                self._name,
                self._exe_file,
                chain_ids=None,
                data=_data,
                seed=seed,
                inits=_inits,
                output_basename=csv_basename,
                method_args=variational_args,
            )

            dummy_chain_id = 0
            runset = RunSet(args=args, chains=1)
            self._run_cmdstan(runset, dummy_chain_id)

        # treat failure to converge as failure
        transcript_file = runset.console_files[dummy_chain_id]
        valid = True
        pat = re.compile(r'The algorithm may not have converged.', re.M)
        with open(transcript_file, 'r') as transcript:
            contents = transcript.read()
            errors = re.findall(pat, contents)
            if len(errors) > 0:
                valid = False
        if not valid:
            raise RuntimeError('The algorithm may not have converged.')
        if not runset._check_retcodes():
            msg = 'Error during variational inference'
            if runset._retcode(dummy_chain_id) != 0:
                msg = '{}, error code {}'.format(
                    msg, runset._retcode(dummy_chain_id)
                )
                raise RuntimeError(msg)
        vi = CmdStanVB(runset)
        vi._set_variational_attrs(runset.csv_files[0])
        return vi

    def _run_cmdstan(
        self, runset: RunSet, idx: int = 0, pbar: List[Any] = None
    ) -> None:
        """
        Encapsulates call to cmdstan.
        Spawn process, capture console output to file, record returncode.
        """
        cmd = runset.cmds[idx]
        self._logger.info('start chain %u', idx + 1)
        self._logger.debug('sampling: %s', cmd)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        if pbar:
            stdout_pbar = self._read_progress(proc, pbar, idx)
        stdout, stderr = proc.communicate()
        if pbar:
            stdout = stdout_pbar + stdout
        transcript_file = runset.console_files[idx]
        self._logger.info('finish chain %u', idx + 1)
        with open(transcript_file, 'w+') as transcript:
            if stdout:
                transcript.write(stdout.decode('utf-8'))
            if stderr:
                transcript.write('ERROR')
                transcript.write(stderr.decode('utf-8'))
        runset._set_retcode(idx, proc.returncode)

    def _read_progress(
        self, proc: subprocess.Popen, pbar: List[Any], idx: int
    ) -> bytes:
        """
        Update tqdm progress bars according to CmdStan console progress msgs.
        Poll process to get CmdStan console outputs,
        check for output lines that start with 'Iteration: '.
        NOTE: if CmdStan output messages change, this will break.
        """
        pattern = (
            r'^Iteration\:\s*(\d+)\s*/\s*(\d+)\s*\[\s*\d+%\s*\]\s*\((\S*)\)$'
        )
        pattern_compiled = re.compile(pattern, flags=re.IGNORECASE)
        pbar_warmup, pbar_sampling = pbar
        num_warmup = pbar_warmup.total
        num_sampling = pbar_sampling.total
        count_warmup = 0
        count_sampling = 0
        stdout = b''

        try:
            # iterate while process is sampling
            while proc.poll() is None:
                output = proc.stdout.readline()
                stdout += output
                output = output.decode('utf-8').strip()
                refresh_warmup = True
                if output.startswith('Iteration'):
                    match = re.search(pattern_compiled, output)
                    if match:
                        # check if pbars need reset
                        if num_warmup is None or num_sampling is None:
                            total_count = int(match.group(2))
                            if num_warmup is None and num_sampling is None:
                                num_warmup = total_count // 2
                                num_sampling = total_count - num_warmup
                                pbar_warmup.total = num_warmup
                                pbar_sampling.total = num_sampling
                            elif num_warmup is None:
                                num_warmup = total_count - num_sampling
                                pbar_warmup.total = num_warmup
                            else:
                                num_sampling = total_count - num_warmup
                                pbar_sampling.total = num_sampling
                        # raw_count = warmup + sampling
                        raw_count = int(match.group(1))
                        if match.group(3).lower() == 'warmup':
                            count, count_warmup = (
                                raw_count - count_warmup,
                                raw_count,
                            )
                            pbar_warmup.update(count)
                        elif match.group(3).lower() == 'sampling':
                            # refresh warmup and close the progress bar
                            if refresh_warmup:
                                pbar_warmup.update(num_warmup - count_warmup)
                                pbar_warmup.refresh()
                                pbar_warmup.close()
                                refresh_warmup = False
                            # update values to full
                            count, count_sampling = (
                                raw_count - num_warmup - count_sampling,
                                raw_count - num_warmup,
                            )
                            pbar_sampling.update(count)

            # read and process rest of the stdout if needed
            warmup_cumulative_count = 0
            sampling_cumulative_count = 0
            for output in proc.stdout:
                stdout += output
                output = output.decode('utf-8').strip()
                if output.startswith('Iteration'):
                    match = re.search(pattern_compiled, output)
                    if match:
                        # check if pbars need reset
                        if num_warmup is None or num_sampling is None:
                            total_count = int(match.group(2))
                            if num_warmup is None and num_sampling is None:
                                num_warmup = total_count // 2
                                num_sampling = total_count - num_warmup
                                pbar_warmup.total = num_warmup
                                pbar_sampling.total = num_sampling
                            elif num_warmup is None:
                                num_warmup = total_count - num_sampling
                                pbar_warmup.total = num_warmup
                            else:
                                num_sampling = total_count - num_warmup
                                pbar_sampling.total = num_sampling
                        # raw_count = warmup + sampling
                        raw_count = int(match.group(1))
                        if match.group(3).lower() == 'warmup':
                            count, count_warmup = (
                                raw_count - count_warmup,
                                raw_count,
                            )
                            warmup_cumulative_count += count
                        elif match.group(3).lower() == 'sampling':
                            count, count_sampling = (
                                raw_count - num_warmup - count_sampling,
                                raw_count - num_warmup,
                            )
                            sampling_cumulative_count += count
            # update warmup pbar if needed
            if warmup_cumulative_count:
                pbar_warmup.update(warmup_cumulative_count)
                pbar_warmup.refresh()
            # update sampling pbar if needed
            if sampling_cumulative_count:
                pbar_sampling.update(sampling_cumulative_count)
                pbar_sampling.refresh()

        except Exception as e:
            self._logger.warning(
                'Chain %s: Failed to read the progress on the fly. Error: %s',
                idx,
                e,
            )
        # close both pbar
        pbar_warmup.close()
        pbar_sampling.close()

        # return stdout
        return stdout
