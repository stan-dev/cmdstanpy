"""CmdStanModel"""

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from multiprocessing import cpu_count
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_WARMUP, _TMPDIR, compilation
from cmdstanpy.cmdstan_args import (
    CmdStanArgs,
    GenerateQuantitiesArgs,
    LaplaceArgs,
    OptimizeArgs,
    PathfinderArgs,
    SamplerArgs,
    VariationalArgs,
)
from cmdstanpy.stanfit import (
    CmdStanGQ,
    CmdStanLaplace,
    CmdStanMCMC,
    CmdStanMLE,
    CmdStanPathfinder,
    CmdStanVB,
    PrevFit,
    RunSet,
    from_output_files,
)
from cmdstanpy.stanfit.base import SingleFileFit
from cmdstanpy.utils import do_command, get_logger
from cmdstanpy.utils.cmdstan import cmdstan_version_before, windows_tbb_path
from cmdstanpy.utils.filesystem import (
    accompanying_json,
    temp_inits,
    temp_metrics,
    temp_single_json,
)
from cmdstanpy.utils.stancsv import try_deduce_metric_type

from . import progress as progbar

OptionalPath = str | os.PathLike | None


def _output_files_for_generate_quantities(files: list[str]) -> list[str]:
    """Add corresponding config files for the CSV-only input"""
    output_files = list(files)
    if all(os.path.splitext(path)[1] == '.csv' for path in files):
        output_files.extend(
            config_file
            for path in files
            if os.path.isfile(config_file := accompanying_json(path, 'config'))
        )
    return output_files


class CmdStanModel:
    # overview, omitted from doc comment in order to improve Sphinx docs.
    #    A CmdStanModel object encapsulates the Stan program and provides
    #    methods for compilation and inference.
    """
    The constructor method allows model instantiation given either the
    Stan program source file or the compiled executable, or both.
    This will compile the model if provided a Stan file and no executable,

    :param stan_file: Path to Stan program file.

    :param exe_file: Path to compiled executable file.  Optional, unless
        no Stan program file is specified.  If both the program file and
        the compiled executable file are specified, the base filenames
        must match, (but different directory locations are allowed).

    :param force_compile: If ``True``, always compile, even if there
        is an existing executable file for this model.

    :param stanc_options: Options for stanc compiler, specified as a Python
        dictionary containing Stanc3 compiler option name, value pairs.
        Optional.

    :param cpp_options: Options for C++ compiler, specified as a Python
        dictionary containing C++ compiler option name, value pairs.
        Optional.

    :param user_header: A path to a header file to include during C++
        compilation.
        Optional.
    """

    def __init__(
        self,
        stan_file: OptionalPath = None,
        exe_file: OptionalPath = None,
        force_compile: bool = False,
        stanc_options: dict[str, Any] | None = None,
        cpp_options: dict[str, Any] | None = None,
        user_header: OptionalPath = None,
        *,
        multithreading: bool = False,
        stanc_optimizations: bool = False,
    ) -> None:
        """
        Initialize object given constructor args.

        :param stan_file: Path to Stan program file.
        :param exe_file: Path to compiled executable file.
        :param force_compile: Whether or not to force recompilation if
            executable file already exists.
        :param multithreading: Enables multithreading in a Stan model.
            Equivalent to `cpp_options = {"STAN_THREADS": "TRUE"}`.
            Defaults to False.
        :param stanc_optimizations: Enables O1 optimizations in the
            stanc compiler. Equivalent to `stanc_options = {"O": 1}`.
            Defaults to False.
        :param stanc_options: Options for stanc compiler. Note, this
            will override the `stanc_optimizations` if in conflict.
        :param cpp_options: Options for C++ compiler. Note, this will
            override the `multithreading` option if in conflict.
        :param user_header: A path to a header file to include during C++
            compilation.
        """
        self._name = ''
        self._stan_file = None
        self._stanc_options = compilation.resolve_stanc_options(
            stanc_options, stanc_optimizations
        )
        cpp_options = compilation.resolve_cpp_options(
            cpp_options, multithreading
        )

        self._fixed_param = False

        windows_tbb_path()

        if exe_file is not None:
            self._exe_file = os.path.realpath(os.path.expanduser(exe_file))
            if not os.path.exists(self._exe_file):
                raise ValueError('no such file {}'.format(self._exe_file))
            _, exename = os.path.split(self._exe_file)
            self._name, _ = os.path.splitext(exename)

        if stan_file is None:
            if exe_file is None:
                raise ValueError(
                    'Missing model file arguments, you must specify '
                    'either Stan source or executable program file or both.'
                )
            if force_compile:
                raise ValueError(
                    'Cannot force compilation when no Stan source file '
                    'is specified.'
                )
        else:
            self._stan_file = os.path.realpath(os.path.expanduser(stan_file))
            if not os.path.exists(self._stan_file):
                raise ValueError('no such file {}'.format(self._stan_file))
            _, filename = os.path.split(stan_file)
            if len(filename) < 6 or not filename.endswith('.stan'):
                raise ValueError(
                    'invalid stan filename {}'.format(self._stan_file)
                )
            if not self._name:
                self._name, _ = os.path.splitext(filename)
            else:
                if self._name != os.path.splitext(filename)[0]:
                    raise ValueError(
                        'Name mismatch between Stan file and compiled'
                        ' executable, expecting basename: {}'
                        ' found: {}.'.format(self._name, filename)
                    )

            self._exe_file = compilation.compile_stan_file(
                str(self.stan_file),
                force=force_compile,
                stanc_options=self._stanc_options,
                cpp_options=cpp_options,
                user_header=user_header,
            )

        # check CmdStan version compatibility
        exe_info = None
        try:
            exe_info = self.exe_info()
        # pylint: disable=broad-except
        except Exception as e:
            get_logger().warning(
                'Could not get exe info for model %s, error: %s',
                self._name,
                str(e),
            )
        if cmdstan_version_before(2, 37, exe_info):
            raise RuntimeError(
                "This version of CmdStanPy requires CmdStan 2.37 or higher."
            )

    def __repr__(self) -> str:
        return (
            f"CmdStanModel(stan_file={self.stan_file!r}, "
            f"exe_file={self.exe_file!r})"
        )

    @property
    def name(self) -> str:
        """
        Model name used in output filename templates.
        """
        return self._name

    @property
    def stan_file(self) -> OptionalPath:
        """Full path to Stan program file."""
        return self._stan_file

    @property
    def exe_file(self) -> str:
        """Full path to Stan exe file."""
        return self._exe_file

    def exe_info(self) -> dict[str, str]:
        """
        Run model with option 'info'. Parse output statements, which all
        have form 'key = value' into a Dict.
        """
        result: dict[str, str] = {}
        info = StringIO()
        do_command(cmd=[self.exe_file, 'info'], fd_out=info)
        lines = info.getvalue().split('\n')
        for line in lines:
            kv_pair = [x.strip() for x in line.split('=')]
            if len(kv_pair) != 2:
                continue
            result[kv_pair[0]] = kv_pair[1]
        return result

    def src_info(self) -> dict[str, Any]:
        """
        Run stanc with option '--info'.

        If stanc is older than 2.27 or if the stan
        file cannot be found, returns an empty dictionary.
        """
        if self.stan_file is None:
            return {}
        return compilation.src_info(str(self.stan_file), self._stanc_options)

    def code(self) -> str | None:
        """Return Stan program as a string."""
        if not self._stan_file:
            raise RuntimeError('Please specify source file')

        code = None
        try:
            with open(self._stan_file, 'r') as fd:
                code = fd.read()
        except IOError:
            get_logger().error(
                'Cannot read file Stan file: %s', self._stan_file
            )
        return code

    def optimize(
        self,
        data: Mapping[str, Any] | str | os.PathLike | None = None,
        seed: int | None = None,
        inits: Mapping[str, Any] | float | str | os.PathLike | None = None,
        output_dir: OptionalPath = None,
        sig_figs: int | None = None,
        save_profile: bool = False,
        algorithm: str | None = None,
        init_alpha: float | None = None,
        tol_obj: float | None = None,
        tol_rel_obj: float | None = None,
        tol_grad: float | None = None,
        tol_rel_grad: float | None = None,
        tol_param: float | None = None,
        history_size: int | None = None,
        iter: int | None = None,
        save_iterations: bool = False,
        require_converged: bool = True,
        show_console: bool = False,
        refresh: int | None = None,
        time_fmt: str = "%Y%m%d%H%M%S",
        timeout: float | None = None,
        jacobian: bool = False,
        # would be nice to move this further up, but that's a breaking change
    ) -> CmdStanMLE:
        """
        Run the specified CmdStan optimize algorithm to produce a
        penalized maximum likelihood estimate of the model parameters.

        This function validates the specified configuration, composes a call to
        the CmdStan ``optimize`` method and spawns one subprocess to run the
        optimizer and waits for it to run to completion.
        Unspecified arguments are not included in the call to CmdStan, i.e.,
        those arguments will have CmdStan default values.

        The :class:`CmdStanMLE` object records the command, the return code,
        and the paths to the optimize method output CSV and console files.
        The output files are written either to a specified output directory
        or to a temporary directory which is deleted upon session exit.

        Output files are either written to a temporary directory or to the
        specified output directory. Optimize output filenames correspond to
        the template '<model_name>-<YYYYMMDDHHMM>' plus the file suffix which is
        either '.csv' for the CmdStan output or '_stdout.txt' for
        the console messages, e.g. 'bernoulli-20251107142835.csv'.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param seed: The seed for random number generator. Must be an integer
            between 0 and 2^32 - 1. If unspecified,
            :func:`numpy.random.default_rng` is used to generate a seed.

        :param inits:  Specifies how the sampler initializes parameter values.
            Initialization is either uniform random on a range centered on 0,
            exactly 0, or a dictionary or file of initial values for some or
            all parameters in the model.  The default initialization behavior
            will initialize all parameter values on range [-2, 2] on the
            *unconstrained* support.  If the expected parameter values are
            too far from this range, this option may improve estimation.
            The following value types are allowed:

            * Single number, n > 0 - initialization range is [-n, n].
            * 0 - all parameters are initialized to 0.
            * dictionary - pairs parameter name : initial value.
            * string - pathname to a JSON or Rdump data file.

        :param output_dir: Name of the directory to which CmdStan output
            files are written. If unspecified, output files will be written
            to a temporary directory which is deleted upon session exit.

        :param sig_figs: Numerical precision used for output CSV and text files.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.
            Introduced in CmdStan-2.25.

        :param save_profile: Whether or not to profile auto-diff operations in
            labelled blocks of code.  If ``True``, CSV outputs are written to
            file '<model_name>-<YYYYMMDDHHMM>_profile.csv'.
            Introduced in CmdStan-2.26.

        :param algorithm: Algorithm to use. One of: 'BFGS', 'LBFGS', 'Newton'

        :param init_alpha: Line search step size for first iteration

        :param tol_obj: Convergence tolerance on changes in objective
            function value

        :param tol_rel_obj: Convergence tolerance on relative changes
            in objective function value

        :param tol_grad: Convergence tolerance on the norm of the gradient

        :param tol_rel_grad: Convergence tolerance on the relative
            norm of the gradient

        :param tol_param: Convergence tolerance on changes in parameter value

        :param history_size: Size of the history for LBFGS Hessian
            approximation. The value should be less than the dimensionality
            of the parameter space. 5-10 usually sufficient

        :param iter: Total number of iterations

        :param save_iterations: When ``True``, save intermediate approximations
            to the output CSV file.  Default is ``False``.

        :param require_converged: Whether or not to raise an error if the
            optimizer fails to meet a convergence criterion. With CmdStan
            2.39 and newer this is determined from the ``converged__`` output
            column; older versions use the process return code.

        :param show_console: If ``True``, stream CmdStan messages sent to
            stdout and stderr to the console.  Default is ``False``.

        :param refresh: Specify the number of iterations cmdstan will take
            between progress messages. Default value is 100.

        :param time_fmt: A format string passed to
            :meth:`~datetime.datetime.strftime` to decide the file names for
            output CSVs. Defaults to "%Y%m%d%H%M%S"

        :param timeout: Duration at which optimization times out in seconds.

        :param jacobian: Whether or not to use the Jacobian adjustment for
            constrained variables in optimization. By default this is false,
            meaning optimization yields the Maximum Likehood Estimate (MLE).
            Setting it to true yields the Maximum A Posteriori Estimate (MAP).

        :return: CmdStanMLE object
        """
        optimize_args = OptimizeArgs(
            algorithm=algorithm,
            init_alpha=init_alpha,
            tol_obj=tol_obj,
            tol_rel_obj=tol_rel_obj,
            tol_grad=tol_grad,
            tol_rel_grad=tol_rel_grad,
            tol_param=tol_param,
            history_size=history_size,
            iter=iter,
            save_iterations=save_iterations,
            jacobian=jacobian,
        )

        with (
            temp_single_json(data) as _data,
            temp_inits(inits, allow_multiple=False) as _inits,
        ):
            args = CmdStanArgs(
                self._name,
                self._exe_file,
                chain_ids=None,
                data=_data,
                seed=seed,
                inits=_inits,
                output_dir=output_dir,
                sig_figs=sig_figs,
                save_profile=save_profile,
                method_args=optimize_args,
                refresh=refresh,
            )
            dummy_chain_id = 0
            runset = RunSet(args=args, chains=1, time_fmt=time_fmt)
            self._run_cmdstan(
                runset,
                dummy_chain_id,
                show_console=show_console,
                timeout=timeout,
            )
        runset.raise_for_timeouts()

        process_succeeded = runset._check_retcodes()
        if not process_succeeded:
            msg = "Error during optimization! Command '{}' failed: {}".format(
                ' '.join(runset.cmd(0)), runset.get_err_msgs()
            )
            if 'Line search failed' in msg and not require_converged:
                get_logger().warning(msg)
            else:
                raise RuntimeError(msg)
        mle = CmdStanMLE.from_files(
            csv_file=runset.csv_files[0],
            config_file=runset.config_files[0],
            stdout_file=runset.stdout_files[0],
            converged=process_succeeded,
        )
        if process_succeeded and not mle.converged:
            msg = 'Error during optimization! The algorithm did not converge.'
            if require_converged:
                raise RuntimeError(msg)
            get_logger().warning(msg)
        return mle

    # pylint: disable=too-many-arguments
    def sample(
        self,
        data: Mapping[str, Any] | str | os.PathLike | None = None,
        chains: int | None = None,
        parallel_chains: int | None = None,
        threads_per_chain: int | None = None,
        seed: int | list[int] | None = None,
        chain_ids: int | list[int] | None = None,
        inits: (
            Mapping[str, Any]
            | float
            | str
            | Sequence[str | Mapping[str, Any]]
            | None
        ) = None,
        iter_warmup: int | None = None,
        iter_sampling: int | None = None,
        save_warmup: bool = False,
        thin: int | None = None,
        max_treedepth: int | None = None,
        metric: str | None = None,
        step_size: float | list[float] | None = None,
        adapt_engaged: bool = True,
        adapt_delta: float | None = None,
        adapt_init_phase: int | None = None,
        adapt_metric_window: int | None = None,
        adapt_step_size: int | None = None,
        fixed_param: bool = False,
        output_dir: OptionalPath = None,  # Update OptionalPath if needed
        sig_figs: int | None = None,
        save_latent_dynamics: bool = False,
        save_profile: bool = False,
        show_progress: bool = True,
        show_console: bool = False,
        refresh: int | None = None,
        time_fmt: str = "%Y%m%d%H%M%S",
        timeout: float | None = None,
        *,
        force_one_process_per_chain: bool | None = None,
        inv_metric: (
            str
            | np.ndarray
            | Mapping[str, Any]
            | Sequence[str | np.ndarray | Mapping[str, Any]]
            | None
        ) = None,
    ) -> CmdStanMCMC:
        """
        Run or more chains of the NUTS-HMC sampler to produce a set of draws
        from the posterior distribution of a model conditioned on some data.

        This function validates the specified configuration, composes a call to
        the CmdStan ``sample`` method and spawns one subprocess per chain to run
        the sampler and waits for all chains to run to completion.
        Unspecified arguments are not included in the call to CmdStan, i.e.,
        those arguments will have CmdStan default values.

        For each chain, the :class:`CmdStanMCMC` object records the command,
        the return code, the sampler output file paths, and the corresponding
        console outputs, if any. The output files are written either to a
        specified output directory or to a temporary directory which is deleted
        upon session exit.

        Output files are either written to a temporary directory or to the
        specified output directory.  Ouput filenames correspond to the template
        '<model_name>-<YYYYMMDDHHMM>' plus additional bits to identify which
        output file it corresponds to. CmdStan output will suffix with
        '_<chain_id>.csv' if there is more than one chain, and simply'.csv'
        in the single-chain case. For example, 'bernoulli-20251107144515_1.csv'.
        Console message output is written to a text file suffixed
        `_stdout_<chain_id>.txt` if each chain executes in a separate process
        (default behavior) or simply `_stdout.txt` if done so in a single
        process, such as when STAN_THREADS is enabled and you are sampling
        more than one chain.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param chains: Number of sampler chains, must be a positive integer.

        :param parallel_chains: Number of processes to run in parallel. Must be
            a positive integer.  Defaults to :func:`multiprocessing.cpu_count`,
            i.e., it will only run as many chains in parallel as there are
            cores on the machine.   Note that CmdStan can run all chains in
            parallel providing that the model was compiled with threading
            support.

        :param threads_per_chain: The number of threads to use in parallelized
            sections within an MCMC chain (e.g., when using the Stan functions
            ``reduce_sum()``  or ``map_rect()``).  This will only have an effect
            if the model was compiled with threading support.  For such models,
            CmdStan will run all chains in parallel from within a single
            process.  The total number of threads used
            will be ``parallel_chains * threads_per_chain``, where the default
            value for parallel_chains is the number of cpus, not chains.

        :param seed: The seed for random number generator. Must be an integer
            between 0 and 2^32 - 1. If unspecified,
            :func:`numpy.random.default_rng`
            is used to generate a seed which will be used for all chains.
            When the same seed is used across all chains,
            the chain-id is used to advance the RNG to avoid dependent samples.

        :param chain_ids: The offset for the random number generator, either
            an integer or a list of unique per-chain offsets.  If unspecified,
            chain ids are numbered sequentially starting from 1.

        :param inits: Specifies how the sampler initializes parameter values.
            Initialization is either uniform random on a range centered on 0,
            exactly 0, or a dictionary or file of initial values for some or all
            parameters in the model.  The default initialization behavior will
            initialize all parameter values on range [-2, 2] on the
            *unconstrained* support.  If the expected parameter values are
            too far from this range, this option may improve adaptation.
            The following value types are allowed:

            * Single number n > 0 - initialization range is [-n, n].
            * 0 - all parameters are initialized to 0.
            * dictionary - pairs parameter name : initial value.
            * string - pathname to a JSON or Rdump data file.
            * list of strings - per-chain pathname to data file.
            * list of dictionaries - per-chain initial values.

        :param iter_warmup: Number of warmup iterations for each chain.

        :param iter_sampling: Number of draws from the posterior for each
            chain.

        :param save_warmup: When ``True``, sampler saves warmup draws as part of
            the Stan CSV output file.

        :param thin: Period between recorded iterations.  Default is 1, i.e.,
             all iterations are recorded.

        :param max_treedepth: Maximum depth of trees evaluated by NUTS sampler
            per iteration.

        :param metric: Specify the type of the inverse mass matrix. Options are
            'diag' or 'diag_e' for diagonal matrix, 'dense' or 'dense_e'
            for a dense matrix, or 'unit_e' an identity mass matrix. To provide
            an initial value for the inverse mass matrix, use the ``inv_metric``
            argument.

        :param inv_metric: Provide an initial value for the inverse
            mass matrix.

            Valid options include:
             - a string, which must be a valid filepath to a JSON or
               Rdump file which contains an entry 'inv_metric' whose value
               is either a diagonal vector or dense matrix.
             - a numpy array containing either the diagonal vector or dense
               matrix.
             - a dictionary containing an entry 'inv_metric' whose value
                is either a diagonal vector or dense matrix.
             - a list of any of the above, of length num_chains, with
               the same shape of metric in each entry.

        :param step_size: Initial step size for HMC sampler.  The value is
            either a single number or a list of numbers which will be used
            as the global or per-chain initial step size, respectively.
            The length of the list of step sizes must match the number of
            chains.

        :param adapt_engaged: When ``True``, adapt step size and metric.

        :param adapt_delta: Adaptation target Metropolis acceptance rate.
            The default value is 0.8.  Increasing this value, which must be
            strictly less than 1, causes adaptation to use smaller step sizes
            which improves the effective sample size, but may increase the time
            per iteration.

        :param adapt_init_phase: Iterations for initial phase of adaptation
            during which step size is adjusted so that the chain converges
            towards the typical set.

        :param adapt_metric_window: The second phase of adaptation tunes
            the metric and step size in a series of intervals.  This parameter
            specifies the number of iterations used for the first tuning
            interval; window size increases for each subsequent interval.

        :param adapt_step_size: Number of iterations given over to adjusting
            the step size given the tuned metric during the final phase of
            adaptation.

        :param fixed_param: When ``True``, call CmdStan with argument
            ``algorithm=fixed_param`` which runs the sampler without
            updating the Markov Chain, thus the values of all parameters and
            transformed parameters are constant across all draws and
            only those values in the generated quantities block that are
            produced by RNG functions may change.  This provides
            a way to use Stan programs to generate simulated data via the
            generated quantities block.  Default value is ``False``.

        :param output_dir: Name of the directory to which CmdStan output
            files are written. If unspecified, output files will be written
            to a temporary directory which is deleted upon session exit.

        :param sig_figs: Numerical precision used for output CSV and text files.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.
            Introduced in CmdStan-2.25.

        :param save_latent_dynamics: Whether or not to output the position and
            momentum information for the model parameters (unconstrained).
            If ``True``, CSV outputs are written to an output file
            '<model_name>-<YYYYMMDDHHMM>_diagnostic_<chain_id>',
            e.g. 'bernoulli-201912081451_diagnostic_1.csv', see
            https://mc-stan.org/docs/cmdstan-guide/stan_csv.html,
            section "Diagnostic CSV output file" for details.

        :param save_profile: Whether or not to profile auto-diff operations in
            labelled blocks of code.  If ``True``, CSV outputs are written to
            file '<model_name>-<YYYYMMDDHHMM>_profile_<chain_id>.csv' if each
            chain runs in its own process, otherwise
            '<model_name>-<YYYYMMDDHHMM>_profile.csv' if all chains run in a
            single process.
            Introduced in CmdStan-2.26, see
            https://mc-stan.org/docs/cmdstan-guide/stan_csv.html,
            section "Profiling CSV output file" for details.

        :param show_progress: If ``True``, display progress bar to track
            progress for warmup and sampling iterations.  Default is ``True``,
            unless package tqdm progress bar encounter errors.

        :param show_console: If ``True``, stream CmdStan messages sent to stdout
            and stderr to the console.  Default is ``False``.

        :param refresh: Specify the number of iterations CmdStan will take
            between progress messages. Default value is 100.

        :param time_fmt: A format string passed to
            :meth:`~datetime.datetime.strftime` to decide the file names for
            output CSVs. Defaults to "%Y%m%d%H%M%S"

        :param force_one_process_per_chain: If ``True``, run multiple chains in
            distinct processes regardless of model ability to run parallel
            chains. If ``False``, always run multiple chains in one process
            (does not check that this is valid).

            If None (Default): Check that the model was compiled with
            STAN_THREADS=True, and utilize the parallel chain functionality
            if so.

        :param timeout: Duration at which sampling times out in seconds.

        :return: CmdStanMCMC object
        """
        if fixed_param is None:
            fixed_param = self._fixed_param

        if chains is None:
            chains = 4
        if chains < 1:
            raise ValueError(
                'Chains must be a positive integer value, found {}.'.format(
                    chains
                )
            )

        if parallel_chains is None:
            parallel_chains = max(min(cpu_count(), chains), 1)
        elif parallel_chains > chains:
            get_logger().info(
                'Requested %u parallel_chains but only %u required, '
                'will run all chains in parallel.',
                parallel_chains,
                chains,
            )
            parallel_chains = chains
        elif parallel_chains < 1:
            raise ValueError(
                'Argument parallel_chains must be a positive '
                'integer, found {}.'.format(parallel_chains)
            )
        if threads_per_chain is None:
            threads_per_chain = 1
        if threads_per_chain < 1:
            raise ValueError(
                'Argument threads_per_chain must be a positive integer, '
                'found {}.'.format(threads_per_chain)
            )

        parallel_procs = parallel_chains
        num_threads = threads_per_chain
        one_process_per_chain = True
        info_dict = self.exe_info()
        stan_threads = info_dict.get('STAN_THREADS', 'false').lower()
        # run multi-chain sampler unless algo is fixed_param or 1 chain
        if chains == 1:
            force_one_process_per_chain = True

        if force_one_process_per_chain is None and stan_threads == 'true':
            one_process_per_chain = False
            num_threads = parallel_chains * num_threads
            parallel_procs = 1
        if force_one_process_per_chain is False:
            one_process_per_chain = False
            num_threads = parallel_chains * num_threads
            parallel_procs = 1
            if stan_threads == 'false':
                get_logger().warning(
                    'Stan program not compiled for threading, '
                    'process will run chains sequentially. '
                    'For multi-chain parallelization, recompile '
                    'the model with argument '
                    '"cpp_options={\'STAN_THREADS\':\'TRUE\'}.'
                )

        os.environ['STAN_NUM_THREADS'] = str(num_threads)

        if chain_ids is None:
            chain_ids = [i + 1 for i in range(chains)]
        else:
            if isinstance(chain_ids, int):
                if chain_ids < 1:
                    raise ValueError(
                        'Chain_id must be a positive integer value, '
                        'found {}.'.format(chain_ids)
                    )
                chain_ids = [i + chain_ids for i in range(chains)]
            else:
                if not one_process_per_chain:
                    for i, j in zip(chain_ids, chain_ids[1:]):
                        if i != j - 1:
                            raise ValueError(
                                'chain_ids must be sequential list of integers,'
                                ' found {}.'.format(chain_ids)
                            )
                if not len(chain_ids) == chains:
                    raise ValueError(
                        'Chain_ids must correspond to number of chains'
                        ' specified {} chains, found {} chain_ids.'.format(
                            chains, len(chain_ids)
                        )
                    )
                for chain_id in chain_ids:
                    if chain_id < 0:
                        raise ValueError(
                            'Chain_id must be a non-negative integer value,'
                            ' found {}.'.format(chain_id)
                        )

        if metric is None and inv_metric is not None:
            metric = try_deduce_metric_type(inv_metric)

        if isinstance(inv_metric, list):
            if not len(inv_metric) == chains:
                raise ValueError(
                    'Number of metric files must match number of chains,'
                    ' found {} metric files for {} chains.'.format(
                        len(inv_metric), chains
                    )
                )

        with (
            temp_single_json(data) as _data,
            temp_inits(inits, id=chain_ids[0]) as _inits,
            temp_metrics(inv_metric, id=chain_ids[0]) as _inv_metric,
        ):
            cmdstan_inits: str | list[str] | int | float | None
            cmdstan_metrics: str | list[str] | None

            if one_process_per_chain and isinstance(inits, list):  # legacy
                cmdstan_inits = [
                    f"{_inits[:-5]}_{i}.json" for i in chain_ids  # type: ignore
                ]
            else:
                cmdstan_inits = _inits
            if one_process_per_chain and isinstance(inv_metric, list):  # legacy
                cmdstan_metrics = [
                    f"{_inv_metric[:-5]}_{i}.json"  # type: ignore
                    for i in chain_ids
                ]
            else:
                cmdstan_metrics = _inv_metric

            sampler_args = SamplerArgs(
                num_chains=1 if one_process_per_chain else chains,
                iter_warmup=iter_warmup,
                iter_sampling=iter_sampling,
                save_warmup=save_warmup,
                thin=thin,
                max_treedepth=max_treedepth,
                metric_type=metric,
                metric_file=cmdstan_metrics,
                step_size=step_size,
                adapt_engaged=adapt_engaged,
                adapt_delta=adapt_delta,
                adapt_init_phase=adapt_init_phase,
                adapt_metric_window=adapt_metric_window,
                adapt_step_size=adapt_step_size,
                fixed_param=fixed_param,
            )

            args = CmdStanArgs(
                self._name,
                self._exe_file,
                chain_ids=chain_ids,
                data=_data,
                seed=seed,
                inits=cmdstan_inits,
                output_dir=output_dir,
                sig_figs=sig_figs,
                save_latent_dynamics=save_latent_dynamics,
                save_profile=save_profile,
                method_args=sampler_args,
                refresh=refresh,
            )

            if show_console:
                show_progress = False
            else:
                show_progress = show_progress and progbar.allow_show_progress()
                get_logger().info('CmdStan start processing')

            progress_hook: Callable[[str, int], None] | None = None
            if show_progress:
                iter_total = 0
                if iter_warmup is None:
                    iter_total += _CMDSTAN_WARMUP
                else:
                    iter_total += iter_warmup
                if iter_sampling is None:
                    iter_total += _CMDSTAN_SAMPLING
                else:
                    iter_total += iter_sampling

                progress_hook = self._wrap_sampler_progress_hook(
                    chain_ids=chain_ids,
                    total=iter_total,
                )
            runset = RunSet(
                args=args,
                chains=chains,
                chain_ids=chain_ids,
                time_fmt=time_fmt,
                one_process_per_chain=one_process_per_chain,
            )
            with ThreadPoolExecutor(max_workers=parallel_procs) as executor:
                for i in range(runset.num_procs):
                    executor.submit(
                        self._run_cmdstan,
                        runset=runset,
                        idx=i,
                        show_progress=show_progress,
                        show_console=show_console,
                        progress_hook=progress_hook,
                        timeout=timeout,
                    )
            if show_progress and progress_hook is not None:
                progress_hook("Done", -1)  # -1 == all chains finished

                # advance terminal window cursor past progress bars
                term_size: os.terminal_size = shutil.get_terminal_size(
                    fallback=(80, 24)
                )
                if term_size is not None and term_size[0] > 0:
                    for i in range(chains):
                        sys.stdout.write(' ' * term_size[0])
                        sys.stdout.flush()
                sys.stdout.write('\n')
                get_logger().info('CmdStan done processing.')

            runset.raise_for_timeouts()

            get_logger().debug('runset\n%s', repr(runset))

            # hack needed to parse CSV files if model has no params
            # needed if exe is supplied without stan file
            with open(runset.stdout_files[0], 'r') as fd:
                console_msgs = fd.read()
                get_logger().debug('Chain 1 console:\n%s', console_msgs)
                if 'running fixed_param sampler' in console_msgs:
                    get_logger().debug("Detected fixed param model")
                    sampler_args.fixed_param = True
                    runset._args.method_args = sampler_args

            errors = runset.get_err_msgs()
            if not runset._check_retcodes():
                msg = (
                    f'Error during sampling:\n{errors}\n'
                    f'Command and output files:\n{repr(runset)}'
                )
                if not show_console:
                    msg += (
                        '\nConsider re-running with show_console=True if the'
                        ' above output is unclear!'
                    )
                raise RuntimeError(msg)
            if errors:
                msg = f'Non-fatal error during sampling:\n{errors}'
                if not show_console:
                    msg += (
                        '\nConsider re-running with show_console=True if the'
                        ' above output is unclear!'
                    )
                get_logger().warning(msg)

            mcmc = CmdStanMCMC.from_files(
                csv_files=runset.csv_files,
                config_files=runset.config_files,
                metric_files=runset.metric_files or None,
                stdout_files=runset.stdout_files,
                diagnostic_files=runset.diagnostic_files or None,
                profile_files=runset.profile_files or None,
                chain_ids=runset.chain_ids,
                sig_figs=runset._args.sig_figs,
            )
        return mcmc

    def generate_quantities(
        self,
        data: Mapping[str, Any] | str | os.PathLike | None = None,
        previous_fit: PrevFit | list[str] | None = None,
        seed: int | None = None,
        gq_output_dir: OptionalPath = None,
        sig_figs: int | None = None,
        show_console: bool = False,
        refresh: int | None = None,
        time_fmt: str = "%Y%m%d%H%M%S",
        timeout: float | None = None,
    ) -> CmdStanGQ[PrevFit]:
        """
        Run CmdStan's generate_quantities method which runs the generated
        quantities block of a model given an existing sample.

        This function takes one of the Stan fit objects
        :class:`CmdStanMCMC`, :class:`CmdStanMLE`, or :class:`CmdStanVB`
        and the data required for the model and calls to the CmdStan
        ``generate_quantities`` method to generate additional quantities of
        interest.

        The :class:`CmdStanGQ` object records the command, the return code,
        and the paths to the generate method output CSV and console files.
        The output files are written either to a specified output directory
        or to a temporary directory which is deleted upon session exit.

        Output files are either written to a temporary directory or to the
        specified output directory.  Ouput filenames correspond to the template
        '<model_name>-<YYYYMMDDHHMM>' plus additional bits to identify which
        output file it corresponds to. CmdStan output will suffix with
        '_<chain_id>.csv' if there is more than one chain, and simply'.csv'
        in the single-chain case. For example, 'bernoulli-20251107144515_1.csv'.
        Console message output is written to a text file suffixed
        `_stdout_<chain_id>.txt` if each chain executes in a separate process
        (default behavior) or simply `_stdout.txt` if done so in a single
        process, such as when STAN_THREADS is enabled and you are sampling
        more than one chain.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param previous_fit: Can be either a :class:`CmdStanMCMC`,
            :class:`CmdStanMLE`, or :class:`CmdStanVB`, or a list of Stan CSV
            files.

        :param seed: The seed for random number generator. Must be an integer
            between 0 and 2^32 - 1. If unspecified,
            :func:`numpy.random.default_rng`
            is used to generate a seed which will be used for all chains.
            *NOTE: Specifying the seed will guarantee the same result for
            multiple invocations of this method with the same inputs.  However
            this will not reproduce results from the sample method given
            the same inputs because the RNG will be in a different state.*

        :param gq_output_dir:  Name of the directory in which the CmdStan output
            files are saved.  If unspecified, files will be written to a
            temporary directory which is deleted upon session exit.

        :param sig_figs: Numerical precision used for output CSV and text files.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.
            Introduced in CmdStan-2.25.

        :param show_console: If ``True``, stream CmdStan messages sent to
            stdout and stderr to the console.  Default is ``False``.

        :param refresh: Specify the number of iterations CmdStan will take
            between progress messages. Default value is 100.

        :param time_fmt: A format string passed to
            :meth:`~datetime.datetime.strftime` to decide the file names for
            output CSVs. Defaults to "%Y%m%d%H%M%S"

        :param timeout: Duration at which generation times out in seconds.

        :return: CmdStanGQ object
        """

        if isinstance(
            previous_fit,
            (
                CmdStanMCMC,
                CmdStanMLE,
                CmdStanVB,
                CmdStanLaplace,
                CmdStanPathfinder,
            ),
        ):
            fit_object = previous_fit
        elif isinstance(previous_fit, list):
            if len(previous_fit) < 1:
                raise ValueError(
                    'Expecting list of output files, found empty list'
                )
            try:
                output_files = _output_files_for_generate_quantities(
                    previous_fit
                )
                fit_object: PrevFit = from_output_files(  # type: ignore
                    output_files
                )
            except ValueError as e:
                raise ValueError(
                    'Invalid sample from Stan CSV files, error:\n\t{}\n\t'
                    ' while processing files\n\t{}'.format(
                        repr(e), '\n\t'.join(previous_fit)
                    )
                ) from e
        else:
            raise ValueError(
                'Previous fit must be either CmdStanPy fit object'
                ' or list of paths to the output files of a fit.'
            )
        if isinstance(fit_object, SingleFileFit):
            fit_csv_files = [fit_object.csv_file]
        else:
            fit_csv_files = fit_object.csv_files
        if isinstance(fit_object, CmdStanMCMC):
            chains = fit_object.chains
            chain_ids = fit_object.chain_ids
            if fit_object._save_warmup:
                get_logger().warning(
                    'Sample contains saved warmup draws which will be used '
                    'to generate additional quantities of interest.'
                )
        elif isinstance(fit_object, CmdStanMLE):
            chains = 1
            chain_ids = [1]
            if fit_object.config.method_config.save_iterations:
                get_logger().warning(
                    'MLE contains saved iterations which will be used '
                    'to generate additional quantities of interest.'
                )
        else:  # isinstance(fit_object, CmdStanVB)
            chains = 1
            chain_ids = [1]

        generate_quantities_args = GenerateQuantitiesArgs(
            csv_files=fit_csv_files
        )
        generate_quantities_args.validate(chains)
        with temp_single_json(data) as _data:
            args = CmdStanArgs(
                self._name,
                self._exe_file,
                chain_ids=chain_ids,
                data=_data,
                seed=seed,
                output_dir=gq_output_dir,
                sig_figs=sig_figs,
                method_args=generate_quantities_args,
                refresh=refresh,
            )
            runset = RunSet(
                args=args, chains=chains, chain_ids=chain_ids, time_fmt=time_fmt
            )

            parallel_chains_avail = cpu_count()
            parallel_chains = max(min(parallel_chains_avail - 2, chains), 1)
            with ThreadPoolExecutor(max_workers=parallel_chains) as executor:
                for i in range(chains):
                    executor.submit(
                        self._run_cmdstan,
                        runset,
                        i,
                        show_console=show_console,
                        timeout=timeout,
                    )

            runset.raise_for_timeouts()
            errors = runset.get_err_msgs()
            if errors:
                msg = (
                    f'Error during generate_quantities:\n{errors}\n'
                    f'Command and output files:\n{repr(runset)}'
                )
                if not show_console:
                    msg += (
                        '\nConsider re-running with show_console=True if the'
                        ' above output is unclear!'
                    )
                raise RuntimeError(msg)
            quantities = CmdStanGQ.from_files(
                csv_files=runset.csv_files,
                config_files=runset.config_files,
                previous_fit=fit_object,
                stdout_files=runset.stdout_files,
                chain_ids=runset.chain_ids,
            )
        return quantities

    def variational(
        self,
        data: Mapping[str, Any] | str | os.PathLike | None = None,
        seed: int | None = None,
        inits: float | None = None,
        output_dir: OptionalPath = None,
        sig_figs: int | None = None,
        save_latent_dynamics: bool = False,
        save_profile: bool = False,
        algorithm: str | None = None,
        iter: int | None = None,
        grad_samples: int | None = None,
        elbo_samples: int | None = None,
        eta: float | None = None,
        adapt_engaged: bool = True,
        adapt_iter: int | None = None,
        tol_rel_obj: float | None = None,
        eval_elbo: int | None = None,
        draws: int | None = None,
        require_converged: bool = True,
        show_console: bool = False,
        refresh: int | None = None,
        time_fmt: str = "%Y%m%d%H%M%S",
        timeout: float | None = None,
    ) -> CmdStanVB:
        """
        Run CmdStan's variational inference algorithm to approximate
        the posterior distribution of the model conditioned on the data.

        This function validates the specified configuration, composes a call to
        the CmdStan ``variational`` method and spawns one subprocess to run the
        optimizer and waits for it to run to completion.
        Unspecified arguments are not included in the call to CmdStan, i.e.,
        those arguments will have CmdStan default values.

        The :class:`CmdStanVB` object records the command, the return code,
        and the paths to the variational method output CSV and console files.
        The output files are written either to a specified output directory
        or to a temporary directory which is deleted upon session exit.

        Output files are either written to a temporary directory or to the
        specified output directory.  Output filenames correspond to the template
        '<model_name>-<YYYYMMDDHHMM>' plus the file suffix which is
        either '.csv' for the CmdStan output or '_stdout.txt' for
        the console messages, e.g. 'bernoulli-201912081451.csv'.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param seed: The seed for random number generator. Must be an integer
            between 0 and 2^32 - 1. If unspecified,
            :func:`numpy.random.default_rng`
            is used to generate a seed which will be used for all chains.

        :param inits:  Specifies how the sampler initializes parameter values.
            Initialization is uniform random on a range centered on 0 with
            default range of 2. Specifying a single number n > 0 changes
            the initialization range to [-n, n].

        :param output_dir: Name of the directory to which CmdStan output
            files are written. If unspecified, output files will be written
            to a temporary directory which is deleted upon session exit.

        :param sig_figs: Numerical precision used for output CSV and text files.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.
            Introduced in CmdStan-2.25.

        :param save_latent_dynamics: Whether or not to save diagnostics.
            If ``True``, CSV outputs are written to output file
            '<model_name>-<YYYYMMDDHHMM>-diagnostic-<chain_id>',
            e.g. 'bernoulli-201912081451-diagnostic-1.csv'.

        :param save_profile: Whether or not to profile auto-diff operations in
            labelled blocks of code.  If ``True``, CSV outputs are written to
            file '<model_name>-<YYYYMMDDHHMM>-profile-<chain_id>'.
            Introduced in CmdStan-2.26.

        :param algorithm: Algorithm to use. One of: 'meanfield', 'fullrank'.

        :param iter: Maximum number of ADVI iterations.

        :param grad_samples: Number of MC draws for computing the gradient.
            Default is 10.  If problems arise, try doubling current value.

        :param elbo_samples: Number of MC draws for estimate of ELBO.

        :param eta: Step size scaling parameter.

        :param adapt_engaged: Whether eta adaptation is engaged.

        :param adapt_iter: Number of iterations for eta adaptation.

        :param tol_rel_obj: Relative tolerance parameter for convergence.

        :param eval_elbo: Number of iterations between ELBO evaluations.

        :param draws: Number of approximate posterior output draws
            to save.

        :param require_converged: Whether or not to raise an error if Stan
            reports that "The algorithm may not have converged".

        :param show_console: If ``True``, stream CmdStan messages sent to
            stdout and stderr to the console.  Default is ``False``.

        :param refresh: Specify the number of iterations CmdStan will take
            between progress messages. Default value is 100.

        :param time_fmt: A format string passed to
            :meth:`~datetime.datetime.strftime` to decide the file names for
            output CSVs. Defaults to "%Y%m%d%H%M%S"

        :param timeout: Duration at which variational Bayesian inference times
            out in seconds.

        :return: CmdStanVB object
        """

        variational_args = VariationalArgs(
            algorithm=algorithm,
            iter=iter,
            grad_samples=grad_samples,
            elbo_samples=elbo_samples,
            eta=eta,
            adapt_engaged=adapt_engaged,
            adapt_iter=adapt_iter,
            tol_rel_obj=tol_rel_obj,
            eval_elbo=eval_elbo,
            output_samples=draws,
        )

        with (
            temp_single_json(data) as _data,
            temp_inits(inits, allow_multiple=False) as _inits,
        ):
            args = CmdStanArgs(
                self._name,
                self._exe_file,
                chain_ids=None,
                data=_data,
                seed=seed,
                inits=_inits,
                output_dir=output_dir,
                sig_figs=sig_figs,
                save_latent_dynamics=save_latent_dynamics,
                save_profile=save_profile,
                method_args=variational_args,
                refresh=refresh,
            )

            dummy_chain_id = 0
            runset = RunSet(args=args, chains=1, time_fmt=time_fmt)
            self._run_cmdstan(
                runset,
                dummy_chain_id,
                show_console=show_console,
                timeout=timeout,
            )
        runset.raise_for_timeouts()

        # treat failure to converge as failure
        transcript_file = runset.stdout_files[dummy_chain_id]
        pat = re.compile(r'The algorithm may not have converged.', re.M)
        with open(transcript_file, 'r') as transcript:
            contents = transcript.read()
        if len(re.findall(pat, contents)) > 0:
            if require_converged:
                raise RuntimeError(
                    'The algorithm may not have converged.\n'
                    'If you would like to inspect the output, '
                    're-call with require_converged=False'
                )
            # else:
            get_logger().warning(
                '%s\n%s',
                'The algorithm may not have converged.',
                'Proceeding because require_converged is set to False',
            )
        if not runset._check_retcodes():
            transcript_file = runset.stdout_files[dummy_chain_id]
            with open(transcript_file, 'r') as transcript:
                contents = transcript.read()
            pat = re.compile(
                r'stan::variational::normal_meanfield::calc_grad:', re.M
            )
            if len(re.findall(pat, contents)) > 0:
                if grad_samples is None:
                    grad_samples = 10
                msg = (
                    'Variational algorithm gradient calculation failed. '
                    'Double the value of argument "grad_samples", '
                    'current value is {}.'.format(grad_samples)
                )
            else:
                msg = 'Error during variational inference: {}'.format(
                    runset.get_err_msgs()
                )
            raise RuntimeError(msg)
        return CmdStanVB.from_files(
            csv_file=runset.csv_files[0],
            config_file=runset.config_files[0],
            stdout_file=runset.stdout_files[0],
        )

    def pathfinder(
        self,
        data: Mapping[str, Any] | str | os.PathLike | None = None,
        *,
        init_alpha: float | None = None,
        tol_obj: float | None = None,
        tol_rel_obj: float | None = None,
        tol_grad: float | None = None,
        tol_rel_grad: float | None = None,
        tol_param: float | None = None,
        history_size: int | None = None,
        num_paths: int | None = None,
        max_lbfgs_iters: int | None = None,
        draws: int | None = None,
        num_single_draws: int | None = None,
        num_elbo_draws: int | None = None,
        psis_resample: bool = True,
        calculate_lp: bool = True,
        # arguments standard to all methods
        seed: int | None = None,
        inits: (
            Mapping[str, Any]
            | float
            | str
            | Sequence[str | Mapping[str, Any]]
            | None
        ) = None,
        output_dir: OptionalPath = None,
        sig_figs: int | None = None,
        save_profile: bool = False,
        show_console: bool = False,
        refresh: int | None = None,
        time_fmt: str = "%Y%m%d%H%M%S",
        timeout: float | None = None,
        num_threads: int | None = None,
    ) -> CmdStanPathfinder:
        """
        Run CmdStan's Pathfinder variational inference algorithm.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param num_paths: Number of single-path Pathfinders to run.
            Default is 4, when the number of paths is 1 then no importance
            sampling is done.

        :param draws: Number of approximate draws to return.

        :param num_single_draws: Number of draws each single-pathfinder will
            draw.
            If ``num_paths`` is 1, only one of this and ``draws`` should be
            used.

        :param max_lbfgs_iters: Maximum number of L-BFGS iterations.

        :param num_elbo_draws: Number of Monte Carlo draws to evaluate ELBO.

        :param psis_resample: Whether or not to use Pareto Smoothed Importance
            Sampling on the result of the individual Pathfinders. If False, the
            result contains the draws from each path.

        :param calculate_lp: Whether or not to calculate the log probability
            for approximate draws. If False, this also implies that
            ``psis_resample`` will be set to False.

        :param seed: The seed for random number generator. Must be an integer
            between 0 and 2^32 - 1. If unspecified,
            :func:`numpy.random.default_rng` is used to generate a seed.

        :param inits: Specifies how the algorithm initializes parameter values.
            Initialization is either uniform random on a range centered on 0,
            exactly 0, or a dictionary or file of initial values for some or all
            parameters in the model.  The default initialization behavior will
            initialize all parameter values on range [-2, 2] on the
            *unconstrained* support.  If the expected parameter values are
            too far from this range, this option may improve adaptation.
            The following value types are allowed:

            * Single number n > 0 - initialization range is [-n, n].
            * 0 - all parameters are initialized to 0.
            * dictionary - pairs parameter name : initial value.
            * string - pathname to a JSON or Rdump data file.
            * list of strings - per-path pathname to data file.
            * list of dictionaries - per-path initial values.

        :param init_alpha: For internal L-BFGS: Line search step size for
            first iteration

        :param tol_obj: For internal L-BFGS: Convergence tolerance on changes
            in objective function value

        :param tol_rel_obj: For internal L-BFGS: Convergence tolerance on
            relative changes in objective function value

        :param tol_grad: For internal L-BFGS: Convergence tolerance on the
            norm of the gradient

        :param tol_rel_grad: For internal L-BFGS: Convergence tolerance on
            the relative norm of the gradient

        :param tol_param: For internal L-BFGS: Convergence tolerance on changes
            in parameter value

        :param history_size: For internal L-BFGS: Size of the history for LBFGS
            Hessian approximation. The value should be less than the
            dimensionality of the parameter space. 5-10 is usually sufficient

        :param output_dir: Name of the directory to which CmdStan output
            files are written. If unspecified, output files will be written
            to a temporary directory which is deleted upon session exit.

        :param sig_figs: Numerical precision used for output CSV and text files.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.
            Introduced in CmdStan-2.25.

        :param save_profile: Whether or not to profile auto-diff operations in
            labelled blocks of code.  If ``True``, CSV outputs are written to
            file '<model_name>-<YYYYMMDDHHMM>_profile.csv'.
            Introduced in CmdStan-2.26, see
            https://mc-stan.org/docs/cmdstan-guide/stan_csv.html,
            section "Profiling CSV output file" for details.

        :param show_console: If ``True``, stream CmdStan messages sent to stdout
            and stderr to the console.  Default is ``False``.

        :param refresh: Specify the number of iterations CmdStan will take
            between progress messages. Default value is 100.

        :param time_fmt: A format string passed to
            :meth:`~datetime.datetime.strftime` to decide the file names for
            output CSVs. Defaults to "%Y%m%d%H%M%S"

        :param timeout: Duration at which Pathfinder times
            out in seconds. Defaults to None.

        :param num_threads: Number of threads to request for parallel execution.
            A number other than ``1`` requires the model to have been compiled
            with STAN_THREADS=True.

        :return: A :class:`CmdStanPathfinder` object

        References
        ----------

        Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder:
        Parallel quasi-Newton variational inference. Journal of Machine Learning
        Research, 23(306), 1–49. Retrieved from
        http://jmlr.org/papers/v23/21-0889.html
        """

        exe_info = self.exe_info()

        if num_threads is not None:
            if (
                num_threads != 1
                and exe_info.get('STAN_THREADS', '').lower() != 'true'
            ):
                raise ValueError(
                    "Model must be compiled with 'STAN_THREADS=true' to use"
                    " 'num_threads' argument"
                )
            os.environ['STAN_NUM_THREADS'] = str(num_threads)

        if num_paths == 1:
            if num_single_draws is None:
                num_single_draws = draws
            if draws is not None and num_single_draws != draws:
                raise ValueError(
                    "Cannot specify both 'draws' and 'num_single_draws'"
                    " when 'num_paths' is 1"
                )

        pathfinder_args = PathfinderArgs(
            init_alpha=init_alpha,
            tol_obj=tol_obj,
            tol_rel_obj=tol_rel_obj,
            tol_grad=tol_grad,
            tol_rel_grad=tol_rel_grad,
            tol_param=tol_param,
            history_size=history_size,
            num_psis_draws=draws,
            num_paths=num_paths,
            max_lbfgs_iters=max_lbfgs_iters,
            num_draws=num_single_draws,
            num_elbo_draws=num_elbo_draws,
            psis_resample=psis_resample,
            calculate_lp=calculate_lp,
        )

        with temp_single_json(data) as _data, temp_inits(inits) as _inits:
            args = CmdStanArgs(
                self._name,
                self._exe_file,
                chain_ids=None,
                data=_data,
                seed=seed,
                inits=_inits,
                output_dir=output_dir,
                sig_figs=sig_figs,
                save_profile=save_profile,
                method_args=pathfinder_args,
                refresh=refresh,
            )
            dummy_chain_id = 0
            runset = RunSet(args=args, chains=1, time_fmt=time_fmt)
            self._run_cmdstan(
                runset,
                dummy_chain_id,
                show_console=show_console,
                timeout=timeout,
            )
        runset.raise_for_timeouts()

        if not runset._check_retcodes():
            msg = "Error during Pathfinder! Command '{}' failed: {}".format(
                ' '.join(runset.cmd(0)), runset.get_err_msgs()
            )
            raise RuntimeError(msg)
        return CmdStanPathfinder.from_files(
            csv_file=runset.csv_files[0],
            config_file=runset.config_files[0],
            stdout_file=runset.stdout_files[0],
        )

    def log_prob(
        self,
        params: dict[str, Any] | str | os.PathLike,
        data: Mapping[str, Any] | str | os.PathLike | None = None,
        *,
        jacobian: bool = True,
        sig_figs: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the log probability and gradient at the given parameter
        values.

        .. note:: This function is **NOT** an efficient way to evaluate the log
            density of the model. It should be used for diagnostics ONLY.
            Please, do not use this for other purposes such as testing new
            sampling algorithms!

        :param params: Values for all parameters in the model, specified
            either as a dictionary with entries matching the parameter
            variables, or as the path of a data file in JSON or Rdump format.

            These should be given on the constrained (natural) scale.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param jacobian: Whether or not to enable the Jacobian adjustment
            for constrained parameters. Defaults to ``True``.

        :param sig_figs: Numerical precision used for output CSV and text files.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.

        :return: A pandas.DataFrame containing columns "lp__" and additional
            columns for the gradient values. These gradients will be for the
            unconstrained parameters of the model.
        """

        with (
            temp_single_json(data) as _data,
            temp_single_json(params) as _params,
        ):
            cmd = [
                str(self.exe_file),
                "log_prob",
                f"constrained_params={_params}",
                f"jacobian={int(jacobian)}",
            ]
            if _data is not None:
                cmd += ["data", f"file={_data}"]

            output_dir = tempfile.mkdtemp(prefix=self.name, dir=_TMPDIR)

            output = os.path.join(output_dir, "output.csv")
            cmd += ["output", f"file={output}"]
            if sig_figs is not None:
                cmd.append(f"sig_figs={sig_figs}")

            get_logger().debug("Cmd: %s", str(cmd))

            proc = subprocess.run(
                cmd, capture_output=True, check=False, text=True
            )
            if proc.returncode:
                get_logger().error(
                    "'log_prob' command failed!\nstdout:%s\nstderr:%s",
                    proc.stdout,
                    proc.stderr,
                )
                raise RuntimeError(
                    "Method 'log_prob' failed with return code "
                    + str(proc.returncode)
                )

            result = pd.read_csv(output, comment="#")
            return result

    def laplace_sample(
        self,
        data: Mapping[str, Any] | str | os.PathLike | None = None,
        mode: CmdStanMLE | str | os.PathLike | None = None,
        draws: int | None = None,
        *,
        jacobian: bool = True,  # NB: Different than optimize!
        seed: int | None = None,
        output_dir: OptionalPath = None,
        sig_figs: int | None = None,
        save_profile: bool = False,
        show_console: bool = False,
        refresh: int | None = None,
        time_fmt: str = "%Y%m%d%H%M%S",
        timeout: float | None = None,
        opt_args: dict[str, Any] | None = None,
    ) -> CmdStanLaplace:
        """
        Run a Laplace approximation around the posterior mode.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param mode: The mode around which to place the approximation, either

            * A :class:`CmdStanMLE` object
            * A path to a CSV file containing the output of an
              optimization run, with the config JSON file written by
              CmdStan alongside it.
            * ``None`` - use default optimizer settings and/or any ``opt_args``.

        :param draws: Number of approximate draws to return.
            Defaults to 1000

        :param jacobian: Whether or not to enable the Jacobian adjustment
            for constrained parameters. Defaults to ``True``.
            Note: This must match the argument used in the creation of
            ``mode``, if supplied.

        :param output_dir: Name of the directory to which CmdStan output
            files are written. If unspecified, output files will be written
            to a temporary directory which is deleted upon session exit.

        :param sig_figs: Numerical precision used for output CSV and text files.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.
            Introduced in CmdStan-2.25.

        :param save_profile: Whether or not to profile auto-diff operations in
            labelled blocks of code.  If ``True``, CSV outputs are written to
            file '<model_name>-<YYYYMMDDHHMM>_profile.csv'.
            Introduced in CmdStan-2.26, see
            https://mc-stan.org/docs/cmdstan-guide/stan_csv.html,
            section "Profiling CSV output file" for details.

        :param show_console: If ``True``, stream CmdStan messages sent to stdout
            and stderr to the console.  Default is ``False``.

        :param refresh: Specify the number of iterations CmdStan will take
            between progress messages. Default value is 100.

        :param time_fmt: A format string passed to
            :meth:`~datetime.datetime.strftime` to decide the file names for
            output CSVs. Defaults to "%Y%m%d%H%M%S"

        :param timeout: Duration at which Pathfinder times
            out in seconds. Defaults to None.

        :param opt_args: Dictionary of additional arguments
            which will be passed to :meth:`~CmdStanModel.optimize`

        :return: A :class:`CmdStanLaplace` object.
        """

        if opt_args is not None and mode is not None:
            raise ValueError(
                "Cannot specify both 'opt_args' and 'mode' arguments"
            )
        if mode is None:
            optimize_args = {
                "seed": seed,
                "sig_figs": sig_figs,
                "jacobian": jacobian,
                "save_profile": save_profile,
                "show_console": show_console,
                "refresh": refresh,
                "time_fmt": time_fmt,
                "timeout": timeout,
                "output_dir": output_dir,
            }
            optimize_args.update(opt_args or {})
            optimize_args['time_fmt'] = 'opt-' + time_fmt
            try:
                cmdstan_mode: CmdStanMLE = self.optimize(
                    data=data,
                    **optimize_args,  # type: ignore
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to run optimizer on model. "
                    "Consider supplying a mode or additional optimizer args"
                ) from e
        elif not isinstance(mode, CmdStanMLE):
            # we check the type below
            cmdstan_mode = from_output_files(mode)  # type: ignore
        else:
            cmdstan_mode = mode

        if not isinstance(cmdstan_mode, CmdStanMLE):
            raise ValueError(
                "Mode must be a CmdStanMLE or a path to an optimize CSV"
            )

        mode_jacobian = cmdstan_mode.config.method_config.jacobian
        if mode_jacobian != jacobian:
            raise ValueError(
                "Jacobian argument to optimize and laplace must match!\n"
                f"Laplace was run with jacobian={jacobian},\n"
                f"but optimize was run with jacobian={mode_jacobian}"
            )

        laplace_args = LaplaceArgs(cmdstan_mode.csv_file, draws, jacobian)

        with temp_single_json(data) as _data:
            args = CmdStanArgs(
                self._name,
                self._exe_file,
                chain_ids=None,
                data=_data,
                seed=seed,
                output_dir=output_dir,
                sig_figs=sig_figs,
                save_profile=save_profile,
                method_args=laplace_args,
                refresh=refresh,
            )
            dummy_chain_id = 0
            runset = RunSet(args=args, chains=1, time_fmt=time_fmt)
            self._run_cmdstan(
                runset,
                dummy_chain_id,
                show_console=show_console,
                timeout=timeout,
            )
        runset.raise_for_timeouts()
        return CmdStanLaplace.from_files(
            csv_file=runset.csv_files[0],
            config_file=runset.config_files[0],
            stdout_file=runset.stdout_files[0],
            mode=cmdstan_mode,
        )

    def _run_cmdstan(
        self,
        runset: RunSet,
        idx: int,
        show_progress: bool = False,
        show_console: bool = False,
        progress_hook: Callable[[str, int], None] | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Helper function which encapsulates call to CmdStan.
        Uses subprocess POpen object to run the process.
        Records stdout, stderr messages, and process returncode.
        Args 'show_progress' and 'show_console' allow use of progress bar,
        streaming output to console, respectively.
        """
        get_logger().debug('idx %d', idx)
        get_logger().debug(
            'running CmdStan, num_threads: %s',
            str(os.environ.get('STAN_NUM_THREADS')),
        )

        logger_prefix = 'CmdStan'
        console_prefix = ''
        if runset.one_process_per_chain:
            logger_prefix = 'Chain [{}]'.format(runset.chain_ids[idx])
            console_prefix = 'Chain [{}] '.format(runset.chain_ids[idx])

        cmd = runset.cmd(idx)
        get_logger().debug('CmdStan args: %s', cmd)

        if not show_progress:
            get_logger().info('%s start processing', logger_prefix)
        try:
            fd_out = open(runset.stdout_files[idx], 'w')
            proc = subprocess.Popen(
                cmd,
                bufsize=1,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # avoid buffer overflow
                env=os.environ,
                universal_newlines=True,
            )
            timer: threading.Timer | None
            if timeout:

                def _timer_target() -> None:
                    # Abort if the process has already terminated.
                    if proc.poll() is not None:
                        return
                    proc.terminate()
                    runset._set_timeout_flag(idx, True)

                timer = threading.Timer(timeout, _timer_target)
                timer.daemon = True
                timer.start()
            else:
                timer = None
            while proc.poll() is None:
                if proc.stdout is not None:
                    line = proc.stdout.readline()
                    fd_out.write(line)
                    line = line.strip()
                    if show_console:
                        print(f'{console_prefix}{line}')
                    elif progress_hook is not None:
                        progress_hook(line, idx)

            stdout, _ = proc.communicate()
            retcode = proc.returncode
            runset._set_retcode(idx, retcode)
            if timer:
                timer.cancel()

            if stdout:
                fd_out.write(stdout)
                if show_console:
                    lines = stdout.split('\n')
                    for line in lines:
                        print(f'{console_prefix}{line}')
            fd_out.close()
        except OSError as e:
            msg = 'Failed with error {}\n'.format(str(e))
            raise RuntimeError(msg) from e
        finally:
            fd_out.close()

        if not show_progress:
            get_logger().info('%s done processing', logger_prefix)

        if retcode != 0:
            serror = ''
            try:
                serror = os.strerror(retcode)
            except (ArithmeticError, ValueError):
                pass
            get_logger().error(
                "%s error: code '%d' %s", logger_prefix, retcode, serror
            )

    @staticmethod
    @progbar.wrap_callback
    def _wrap_sampler_progress_hook(
        chain_ids: list[int],
        total: int,
    ) -> Callable[[str, int], None] | None:
        """
        Sets up tqdm callback for CmdStan sampler console msgs.
        CmdStan progress messages start with "Iteration", for single chain
        process, "Chain [id] Iteration" for multi-chain processing.
        For the latter, manage array of pbars, update accordingly.
        """
        chain_pat = re.compile(r'(Chain \[(\d+)\] )?Iteration:\s+(\d+)')
        pbars: dict[int, tqdm] = {
            chain_id: tqdm(
                total=total,
                desc=f'chain {chain_id}',
                postfix='(Warmup)',
                colour='yellow',
            )
            for chain_id in chain_ids
        }

        def progress_hook(line: str, idx: int) -> None:
            if line == "Done":
                for pbar in pbars.values():
                    pbar.set_postfix_str('(Sampling completed)')
                    pbar.update(total - pbar.n)
                    pbar.close()
            elif (match := chain_pat.match(line)) is not None:
                idx = int(match.group(2) or chain_ids[idx])
                current_iter = int(match.group(3))

                pbar = pbars[idx]
                if pbar.colour == 'yellow' and 'Sampling' in line:
                    pbar.colour = 'blue'
                    pbar.set_postfix_str('(Sampling)')

                pbar.update(current_iter - pbar.n)

        return progress_hook

    def diagnose(
        self,
        inits: dict[str, Any] | str | os.PathLike | None = None,
        data: Mapping[str, Any] | str | os.PathLike | None = None,
        *,
        epsilon: float | None = None,
        error: float | None = None,
        require_gradients_ok: bool = True,
        sig_figs: int | None = None,
    ) -> pd.DataFrame:
        """
        Run diagnostics to calculate the gradients at the specified parameter
        values and compare them with gradients calculated by finite differences.

        :param inits: Specifies how the sampler initializes parameter values.
            Initialization is either uniform random on a range centered on 0,
            exactly 0, or a dictionary or file of initial values for some or
            all parameters in the model. The default initialization behavior
            will initialize all parameter values on range [-2, 2] on the
            *unconstrained* support. The following value types are allowed:
            * Single number, n > 0 - initialization range is [-n, n].
            * 0 - all parameters are initialized to 0.
            * dictionary - pairs parameter name : initial value.
            * string - pathname to a JSON or Rdump data file.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param sig_figs: Numerical precision used for output CSV and text files.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.

        :param epsilon: Step size for finite difference gradients.

        :param error: Absolute error threshold for comparing autodiff and finite
            difference gradients.

        :param require_gradients_ok: Whether or not to raise an error if Stan
            reports that the difference between autodiff gradients and finite
            difference gradients exceed the error threshold.

        :return: A pandas.DataFrame containing columns
            * "param_idx": increasing parameter index.
            * "value": Parameter value.
            * "model": Gradients evaluated using autodiff.
            * "finite_diff": Gradients evaluated using finite differences.
            * "error": Delta between autodiff and finite difference gradients.

            Gradients are evaluated in the unconstrained space.
        """

        with temp_single_json(data) as _data, temp_single_json(inits) as _inits:
            cmd = [
                str(self.exe_file),
                "diagnose",
                "test=gradient",
            ]
            if epsilon is not None:
                cmd.append(f"epsilon={epsilon}")
            if error is not None:
                cmd.append(f"epsilon={error}")
            if _data is not None:
                cmd += ["data", f"file={_data}"]
            if _inits is not None:
                cmd.append(f"init={_inits}")

            output_dir = tempfile.mkdtemp(prefix=self.name, dir=_TMPDIR)

            output = os.path.join(output_dir, "output.csv")
            cmd += ["output", f"file={output}"]
            if sig_figs is not None:
                cmd.append(f"sig_figs={sig_figs}")

            get_logger().debug("Cmd: %s", str(cmd))

            proc = subprocess.run(
                cmd, capture_output=True, check=False, text=True
            )
            if proc.returncode:
                get_logger().error(
                    "'diagnose' command failed!\nstdout:%s\nstderr:%s",
                    proc.stdout,
                    proc.stderr,
                )
                if require_gradients_ok:
                    raise RuntimeError(
                        "The difference between autodiff and finite difference "
                        "gradients may exceed the error threshold. If you "
                        "would like to inspect the output, re-call with "
                        "`require_gradients_ok=False`."
                    )
                get_logger().warning(
                    "The difference between autodiff and finite difference "
                    "gradients may exceed the error threshold. Proceeding "
                    "because `require_gradients_ok` is set to `False`."
                )

            # Read the text and get the last chunk separated by a single # char.
            try:
                with open(output) as handle:
                    text = handle.read()
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Output of 'diagnose' command does not exist."
                ) from exc
            *_, table = re.split(r"#\s*\n", text)
            table = (
                re.sub(r"^#\s*", "", table, flags=re.M)
                .replace("param idx", "param_idx")
                .replace("finite diff", "finite_diff")
            )
            return pd.read_csv(io.StringIO(table), sep=r"\s+")
