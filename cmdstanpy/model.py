"""CmdStanModel"""

import io
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from tqdm.auto import tqdm

from cmdstanpy import _CMDSTAN_REFRESH, _CMDSTAN_SAMPLING, _CMDSTAN_WARMUP
from cmdstanpy.cmdstan_args import (
    CmdStanArgs,
    GenerateQuantitiesArgs,
    OptimizeArgs,
    SamplerArgs,
    VariationalArgs,
)
from cmdstanpy.compiler_opts import CompilerOptions
from cmdstanpy.stanfit import (
    CmdStanGQ,
    CmdStanMCMC,
    CmdStanMLE,
    CmdStanVB,
    RunSet,
    from_csv,
)
from cmdstanpy.utils import (
    EXTENSION,
    MaybeDictToFilePath,
    TemporaryCopiedFile,
    cmdstan_path,
    do_command,
    get_logger,
    returncode_msg,
)

from . import progress as progbar


class CmdStanModel:
    # overview, omitted from doc comment in order to improve Sphinx docs.
    #    A CmdStanModel object encapsulates the Stan program and provides
    #    methods for compilation and inference.
    """
    The constructor method allows model instantiation given either the
    Stan program source file or the compiled executable, or both.
    By default, the constructor will compile the Stan program on instantiation
    unless the argument ``compile=False`` is specified.
    The set of constructor arguments are:

    :param model_name: Model name, used for output file names.
        Optional, default is the base filename of the Stan program file.

    :param stan_file: Path to Stan program file.

    :param exe_file: Path to compiled executable file.  Optional, unless
        no Stan program file is specified.  If both the program file and
        the compiled executable file are specified, the base filenames
        must match, (but different directory locations are allowed).

    :param compile: Whether or not to compile the model.  Default is ``True``.

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
        model_name: Optional[str] = None,
        stan_file: Optional[str] = None,
        exe_file: Optional[str] = None,
        compile: bool = True,
        stanc_options: Optional[Dict[str, Any]] = None,
        cpp_options: Optional[Dict[str, Any]] = None,
        user_header: Optional[str] = None,
    ) -> None:
        """
        Initialize object given constructor args.

        :param model_name: Model name, used for output file names.
        :param stan_file: Path to Stan program file.
        :param exe_file: Path to compiled executable file.
        :param compile: Whether or not to compile the model.
        :param stanc_options: Options for stanc compiler.
        :param cpp_options: Options for C++ compiler.
        :param user_header: A path to a header file to include during C++
            compilation.
        """
        self._name = ''
        self._stan_file = None
        self._exe_file = None
        self._compiler_options = CompilerOptions(
            stanc_options=stanc_options,
            cpp_options=cpp_options,
            user_header=user_header,
        )

        if model_name is not None:
            if not model_name.strip():
                raise ValueError(
                    'Invalid value for argument model name, found "{}"'.format(
                        model_name
                    )
                )
            self._name = model_name.strip()

        if stan_file is None:
            if exe_file is None:
                raise ValueError(
                    'Missing model file arguments, you must specify '
                    'either Stan source or executable program file or both.'
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
            # if program has include directives, record path
            with open(self._stan_file, 'r') as fd:
                program = fd.read()
            if '#include' in program:
                path, _ = os.path.split(self._stan_file)
                if self._compiler_options is None:
                    self._compiler_options = CompilerOptions(
                        stanc_options={'include_paths': [path]}
                    )
                elif self._compiler_options._stanc_options is None:
                    self._compiler_options._stanc_options = {
                        'include_paths': [path]
                    }
                else:
                    self._compiler_options.add_include_path(path)

        if exe_file is not None:
            self._exe_file = os.path.realpath(os.path.expanduser(exe_file))
            if not os.path.exists(self._exe_file):
                raise ValueError('no such file {}'.format(self._exe_file))
            _, exename = os.path.split(self._exe_file)
            if not self._name:
                self._name, _ = os.path.splitext(exename)
            else:
                if self._name != os.path.splitext(exename)[0]:
                    raise ValueError(
                        'Name mismatch between Stan file and compiled'
                        ' executable, expecting basename: {}'
                        ' found: {}.'.format(self._name, exename)
                    )

        self._compiler_options.validate()

        if platform.system() == 'Windows':
            try:
                do_command(['where.exe', 'tbb.dll'], fd_out=None)
            except RuntimeError:
                # Add tbb to the $PATH on Windows
                libtbb = os.environ.get('STAN_TBB')
                if libtbb is None:
                    libtbb = os.path.join(
                        cmdstan_path(), 'stan', 'lib', 'stan_math', 'lib', 'tbb'
                    )
                get_logger().debug("Adding TBB (%s) to PATH", libtbb)
                os.environ['PATH'] = ';'.join(
                    list(
                        OrderedDict.fromkeys(
                            [libtbb] + os.environ.get('PATH', '').split(';')
                        )
                    )
                )
            else:
                get_logger().debug("TBB already found in load path")

        if compile and self._exe_file is None:
            self.compile()
            if self._exe_file is None:
                raise ValueError(
                    'Unable to compile Stan model file: {}.'.format(
                        self._stan_file
                    )
                )

    def __repr__(self) -> str:
        repr = 'CmdStanModel: name={}'.format(self._name)
        repr = '{}\n\t stan_file={}'.format(repr, self._stan_file)
        repr = '{}\n\t exe_file={}'.format(repr, self._exe_file)
        repr = '{}\n\t compiler_options={}'.format(repr, self._compiler_options)
        return repr

    @property
    def name(self) -> str:
        """
        Model name used in output filename templates. Default is basename
        of Stan program or exe file, unless specified in call to constructor
        via argument ``model_name``.
        """
        return self._name

    @property
    def stan_file(self) -> Optional[str]:
        """Full path to Stan program file."""
        return self._stan_file

    @property
    def exe_file(self) -> Optional[str]:
        """Full path to Stan exe file."""
        return self._exe_file

    @property
    def stanc_options(self) -> Dict[str, Union[bool, int, str]]:
        """Options to stanc compilers."""
        return self._compiler_options._stanc_options

    @property
    def cpp_options(self) -> Dict[str, Union[bool, int]]:
        """Options to C++ compilers."""
        return self._compiler_options._cpp_options

    @property
    def user_header(self) -> str:
        """The user header file if it exists, otherwise empty"""
        return self._compiler_options._user_header

    def code(self) -> Optional[str]:
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

    def compile(
        self,
        force: bool = False,
        stanc_options: Optional[Dict[str, Any]] = None,
        cpp_options: Optional[Dict[str, Any]] = None,
        user_header: Optional[str] = None,
        override_options: bool = False,
    ) -> None:
        """
        Compile the given Stan program file.  Translates the Stan code to
        C++, then calls the C++ compiler.

        By default, this function compares the timestamps on the source and
        executable files; if the executable is newer than the source file, it
        will not recompile the file, unless argument ``force`` is ``True``.

        :param force: When ``True``, always compile, even if the executable file
            is newer than the source file.  Used for Stan models which have
            ``#include`` directives in order to force recompilation when changes
            are made to the included files.

        :param stanc_options: Options for stanc compiler.
        :param cpp_options: Options for C++ compiler.
        :param user_header: A path to a header file to include during C++
            compilation.

        :param override_options: When ``True``, override existing option.
            When ``False``, add/replace existing options.  Default is ``False``.
        """
        if not self._stan_file:
            raise RuntimeError('Please specify source file')

        compiler_options = None
        if not (
            stanc_options is None
            and cpp_options is None
            and user_header is None
        ):
            compiler_options = CompilerOptions(
                stanc_options=stanc_options,
                cpp_options=cpp_options,
                user_header=user_header,
            )
            compiler_options.validate()
            if self._compiler_options is None:
                self._compiler_options = compiler_options
            elif override_options:
                self._compiler_options = compiler_options
            else:
                self._compiler_options.add(compiler_options)

        # check if exe file exists in original location
        exe_file, _ = os.path.splitext(os.path.abspath(self._stan_file))
        exe_file = Path(exe_file).as_posix() + EXTENSION
        do_compile = True
        if os.path.exists(exe_file):
            src_time = os.path.getmtime(self._stan_file)
            exe_time = os.path.getmtime(exe_file)
            if exe_time > src_time and not force:
                do_compile = False
                get_logger().info('found newer exe file, not recompiling')
                self._exe_file = exe_file
                get_logger().info('compiled model file: %s', self._exe_file)
        if do_compile:
            compilation_failed = False
            with TemporaryCopiedFile(self._stan_file) as (stan_file, is_copied):
                exe_file, _ = os.path.splitext(os.path.abspath(stan_file))
                exe_file = Path(exe_file).as_posix() + EXTENSION
                do_compile = True
                if os.path.exists(exe_file):
                    src_time = os.path.getmtime(self._stan_file)
                    exe_time = os.path.getmtime(exe_file)
                    if exe_time > src_time and not force:
                        do_compile = False
                        get_logger().info(
                            'found newer exe file, not recompiling'
                        )

                if do_compile:
                    get_logger().info(
                        'compiling stan program, exe file: %s', exe_file
                    )
                    if self._compiler_options is not None:
                        self._compiler_options.validate()
                        get_logger().info(
                            'compiler options: %s', self._compiler_options
                        )
                    make = os.getenv(
                        'MAKE',
                        'make'
                        if platform.system() != 'Windows'
                        else 'mingw32-make',
                    )
                    cmd = [make]
                    if self._compiler_options is not None:
                        cmd.extend(self._compiler_options.compose())
                    cmd.append(Path(exe_file).as_posix())

                    sout = io.StringIO()
                    try:
                        do_command(cmd=cmd, cwd=cmdstan_path(), fd_out=sout)
                    except RuntimeError as e:
                        sout.write(f'\n{str(e)}\n')
                        compilation_failed = True
                    finally:
                        console = sout.getvalue()

                    if compilation_failed or 'Warning:' in console:
                        lines = console.split('\n')
                        warnings = [x for x in lines if x.startswith('Warning')]
                        syntax_errors = [
                            x for x in lines if x.startswith('Syntax error')
                        ]
                        semantic_errors = [
                            x for x in lines if x.startswith('Semantic error')
                        ]
                        exceptions = [
                            x
                            for x in lines
                            if x.startswith('Uncaught exception')
                        ]
                        if (
                            len(syntax_errors) > 0
                            or len(semantic_errors) > 0
                            or len(exceptions) > 0
                        ):
                            get_logger().error(
                                'Stan program failed to compile:'
                            )
                            get_logger().warning(console)
                        elif len(warnings) > 0:
                            get_logger().warning(
                                'Stan compiler has produced %d warnings:',
                                len(warnings),
                            )
                            get_logger().warning(console)

                        if 'PCH file' in console:
                            get_logger().warning(
                                "CmdStan's precompiled header (PCH) files "
                                "may need to be rebuilt."
                                "If your model failed to compile please run "
                                "cmdstanpy.rebuild_cmdstan().\nIf the "
                                "issue persists please open a bug report"
                            )

                if not compilation_failed:
                    if is_copied:
                        original_target_dir = os.path.dirname(
                            os.path.abspath(self._stan_file)
                        )
                        new_exec_name = (
                            os.path.basename(
                                os.path.splitext(self._stan_file)[0]
                            )
                            + EXTENSION
                        )
                        self._exe_file = os.path.join(
                            original_target_dir, new_exec_name
                        )
                        shutil.copy(exe_file, self._exe_file)
                    else:
                        self._exe_file = exe_file
                    get_logger().info('compiled model file: %s', self._exe_file)
                else:
                    get_logger().error('model compilation failed')

    def optimize(
        self,
        data: Union[Mapping[str, Any], str, None] = None,
        seed: Optional[int] = None,
        inits: Union[Dict[str, float], float, str, None] = None,
        output_dir: Optional[str] = None,
        sig_figs: Optional[int] = None,
        save_profile: bool = False,
        algorithm: Optional[str] = None,
        init_alpha: Optional[float] = None,
        tol_obj: Optional[float] = None,
        tol_rel_obj: Optional[float] = None,
        tol_grad: Optional[float] = None,
        tol_rel_grad: Optional[float] = None,
        tol_param: Optional[float] = None,
        history_size: Optional[int] = None,
        iter: Optional[int] = None,
        save_iterations: bool = False,
        require_converged: bool = True,
        show_console: bool = False,
        refresh: Optional[int] = None,
        time_fmt: str = "%Y%m%d%H%M%S",
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
        specified output directory.  Output filenames correspond to the template
        '<model_name>-<YYYYMMDDHHMM>-<chain_id>' plus the file suffix which is
        either '.csv' for the CmdStan output or '.txt' for
        the console messages, e.g. 'bernoulli-201912081451-1.csv'.
        Output files written to the temporary directory contain an additional
        8-character random string, e.g. 'bernoulli-201912081451-1-5nm6as7u.csv'.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param seed: The seed for random number generator. Must be an integer
            between 0 and 2^32 - 1. If unspecified,
            :class:`numpy.random.RandomState` is used to generate a seed.

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
            file '<model_name>-<YYYYMMDDHHMM>-profile-<chain_id>'.
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

        :param require_converged: Whether or not to raise an error if Stan
            reports that "The algorithm may not have converged".

        :param show_console: If ``True``, stream CmdStan messages sent to
            stdout and stderr to the console.  Default is ``False``.

        :param refresh: Specify the number of iterations cmdstan will take
            between progress messages. Default value is 100.

        :param time_fmt: A format string passed to
            :meth:`~datetime.datetime.strftime` to decide the file names for
            output CSVs. Defaults to "%Y%m%d%H%M%S"

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
        )

        with MaybeDictToFilePath(data, inits) as (_data, _inits):
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
            self._run_cmdstan(runset, dummy_chain_id, False, show_console)

        if not runset._check_retcodes():
            msg = 'Error during optimization: {}'.format(runset.get_err_msgs())
            if 'Line search failed' in msg and not require_converged:
                get_logger().warning(msg)
            else:
                raise RuntimeError(msg)
        mle = CmdStanMLE(runset)
        return mle

    # pylint: disable=too-many-arguments
    def sample(
        self,
        data: Union[Mapping[str, Any], str, None] = None,
        chains: Optional[int] = None,
        parallel_chains: Optional[int] = None,
        threads_per_chain: Optional[int] = None,
        seed: Union[int, List[int], None] = None,
        chain_ids: Union[int, List[int], None] = None,
        inits: Union[Dict[str, float], float, str, List[str], None] = None,
        iter_warmup: Optional[int] = None,
        iter_sampling: Optional[int] = None,
        save_warmup: bool = False,
        thin: Optional[int] = None,
        max_treedepth: Optional[int] = None,
        metric: Union[
            str, Dict[str, Any], List[str], List[Dict[str, Any]], None
        ] = None,
        step_size: Union[float, List[float], None] = None,
        adapt_engaged: bool = True,
        adapt_delta: Optional[float] = None,
        adapt_init_phase: Optional[int] = None,
        adapt_metric_window: Optional[int] = None,
        adapt_step_size: Optional[int] = None,
        fixed_param: bool = False,
        output_dir: Optional[str] = None,
        sig_figs: Optional[int] = None,
        save_latent_dynamics: bool = False,
        save_profile: bool = False,
        show_progress: bool = True,
        show_console: bool = False,
        refresh: Optional[int] = None,
        time_fmt: str = "%Y%m%d%H%M%S",
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
        '<model_name>-<YYYYMMDDHHMM>-<chain_id>' plus the file suffix which is
        either '.csv' for the CmdStan output or '.txt' for
        the console messages, e.g. 'bernoulli-201912081451-1.csv'.
        Output files written to the temporary directory contain an additional
        8-character random string, e.g. 'bernoulli-201912081451-1-5nm6as7u.csv'.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param chains: Number of sampler chains, must be a positive integer.

        :param parallel_chains: Number of processes to run in parallel. Must be
            a positive integer.  Defaults to :func:`multiprocessing.cpu_count`.

        :param threads_per_chain: The number of threads to use in parallelized
            sections within an MCMC chain (e.g., when using the Stan functions
            ``reduce_sum()``  or ``map_rect()``).  This will only have an effect
            if the model was compiled with threading support. The total number
            of threads used will be ``parallel_chains * threads_per_chain``.

        :param seed: The seed for random number generator. Must be an integer
            between 0 and 2^32 - 1. If unspecified,
            :class:`numpy.random.RandomState`
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

        :param iter_warmup: Number of warmup iterations for each chain.

        :param iter_sampling: Number of draws from the posterior for each
            chain.

        :param save_warmup: When ``True``, sampler saves warmup draws as part of
            the Stan CSV output file.

        :param thin: Period between recorded iterations.  Default is 1, i.e.,
             all iterations are recorded.

        :param max_treedepth: Maximum depth of trees evaluated by NUTS sampler
            per iteration.

        :param metric: Specification of the mass matrix, either as a
            vector consisting of the diagonal elements of the covariance
            matrix ('diag' or 'diag_e') or the full covariance matrix
            ('dense' or 'dense_e').

            If the value of the metric argument is a string other than
            'diag', 'diag_e', 'dense', or 'dense_e', it must be
            a valid filepath to a JSON or Rdump file which contains an entry
            'inv_metric' whose value is either the diagonal vector or
            the full covariance matrix.

            If the value of the metric argument is a list of paths, its
            length must match the number of chains and all paths must be
            unique.

            If the value of the metric argument is a Python dict object, it
            must contain an entry 'inv_metric' which specifies either the
            diagnoal or dense matrix.

            If the value of the metric argument is a list of Python dicts,
            its length must match the number of chains and all dicts must
            containan entry 'inv_metric' and all 'inv_metric' entries must
            have the same shape.

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
            generated quantities block.  This option must be used when the
            parameters block is empty.  Default value is ``False``.

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
            '<model_name>-<YYYYMMDDHHMM>-diagnostic-<chain_id>',
            e.g. 'bernoulli-201912081451-diagnostic-1.csv', see
            https://mc-stan.org/docs/cmdstan-guide/stan-csv.html,
            section "Diagnostic CSV output file" for details.

        :param save_profile: Whether or not to profile auto-diff operations in
            labelled blocks of code.  If ``True``, CSV outputs are written to
            file '<model_name>-<YYYYMMDDHHMM>-profile-<chain_id>'.
            Introduced in CmdStan-2.26, see
            https://mc-stan.org/docs/cmdstan-guide/stan-csv.html,
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

        :return: CmdStanMCMC object
        """
        if chains is None:
            if fixed_param:
                chains = 1
            else:
                chains = 4
        if chains < 1:
            raise ValueError(
                'Chains must be a positive integer value, found {}.'.format(
                    chains
                )
            )
        if chain_ids is None:
            chain_ids = [x + 1 for x in range(chains)]
        else:
            if isinstance(chain_ids, int):
                if chain_ids < 1:
                    raise ValueError(
                        'Chain_id must be a positive integer value,'
                        ' found {}.'.format(chain_ids)
                    )
                chain_ids = [chain_ids + i for i in range(chains)]
            else:
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
        if parallel_chains is None:
            parallel_chains = max(min(cpu_count(), chains), 1)
        elif parallel_chains > chains:
            get_logger().info(
                'Requesting %u parallel_chains for %u chains,'
                ' running all chains in parallel.',
                parallel_chains,
                chains,
            )
            parallel_chains = chains
        elif parallel_chains < 1:
            raise ValueError(
                'Argument parallel_chains must be a positive integer value, '
                'found {}.'.format(parallel_chains)
            )
        if threads_per_chain is None:
            threads_per_chain = 1
        if threads_per_chain < 1:
            raise ValueError(
                'Argument threads_per_chain must be a positive integer value, '
                'found {}.'.format(threads_per_chain)
            )
        get_logger().debug(
            'total threads: %u', parallel_chains * threads_per_chain
        )
        os.environ['STAN_NUM_THREADS'] = str(threads_per_chain)

        # TODO:  issue 49: inits can be initialization function
        sampler_args = SamplerArgs(
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            save_warmup=save_warmup,
            thin=thin,
            max_treedepth=max_treedepth,
            metric=metric,
            step_size=step_size,
            adapt_engaged=adapt_engaged,
            adapt_delta=adapt_delta,
            adapt_init_phase=adapt_init_phase,
            adapt_metric_window=adapt_metric_window,
            adapt_step_size=adapt_step_size,
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
                output_dir=output_dir,
                sig_figs=sig_figs,
                save_latent_dynamics=save_latent_dynamics,
                save_profile=save_profile,
                method_args=sampler_args,
                refresh=refresh,
            )
            runset = RunSet(
                args=args, chains=chains, chain_ids=chain_ids, time_fmt=time_fmt
            )  # bookkeeping object for command, result, filepaths

            # progress reporting
            iter_total = 0
            if iter_warmup is None:
                iter_total += _CMDSTAN_WARMUP
            else:
                iter_total += iter_warmup
            if iter_sampling is None:
                iter_total += _CMDSTAN_SAMPLING
            else:
                iter_total += iter_sampling
            if refresh is None:
                refresh = _CMDSTAN_REFRESH
            iter_total = iter_total // refresh + 2

            if show_console:
                show_progress = False
            else:
                show_progress = show_progress and progbar.allow_show_progress()

            get_logger().info('sampling: %s', runset.cmds[0])
            with ThreadPoolExecutor(max_workers=parallel_chains) as executor:
                for i in range(chains):
                    executor.submit(
                        self._run_cmdstan,
                        runset,
                        i,
                        show_progress,
                        show_console,
                        iter_total,
                    )
            if show_progress and progbar.allow_show_progress():
                # advance terminal window cursor past progress bars
                term_size: os.terminal_size = shutil.get_terminal_size(
                    fallback=(80, 24)
                )
                if term_size is not None and term_size[0] > 0:
                    for i in range(chains):
                        sys.stdout.write(' ' * term_size[0])
                        sys.stdout.flush()
                sys.stdout.write('\n')
            get_logger().info('sampling completed')
            if not runset._check_retcodes():
                msg = 'Error during sampling:\n{}'.format(runset.get_err_msgs())
                msg = '{}Command and output files:\n{}'.format(
                    msg, runset.__repr__()
                )
                raise RuntimeError(msg)

            mcmc = CmdStanMCMC(runset)
        return mcmc

    def generate_quantities(
        self,
        data: Union[Mapping[str, Any], str, None] = None,
        mcmc_sample: Union[CmdStanMCMC, List[str], None] = None,
        seed: Optional[int] = None,
        gq_output_dir: Optional[str] = None,
        sig_figs: Optional[int] = None,
        show_console: bool = False,
        refresh: Optional[int] = None,
        time_fmt: str = "%Y%m%d%H%M%S",
    ) -> CmdStanGQ:
        """
        Run CmdStan's generate_quantities method which runs the generated
        quantities block of a model given an existing sample.

        This function takes a :class:`CmdStanMCMC` object and the dataset used
        to generate that sample and calls to the CmdStan ``generate_quantities``
        method to generate additional quantities of interest.

        The :class:`CmdStanGQ` object records the command, the return code,
        and the paths to the generate method output CSV and console files.
        The output files are written either to a specified output directory
        or to a temporary directory which is deleted upon session exit.

        Output files are either written to a temporary directory or to the
        specified output directory.  Output filenames correspond to the template
        '<model_name>-<YYYYMMDDHHMM>-<chain_id>' plus the file suffix which is
        either '.csv' for the CmdStan output or '.txt' for
        the console messages, e.g. 'bernoulli-201912081451-1.csv'.
        Output files written to the temporary directory contain an additional
        8-character random string, e.g. 'bernoulli-201912081451-1-5nm6as7u.csv'.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param mcmc_sample: Can be either a :class:`CmdStanMCMC` object returned
            by the :meth:`sample` method or a list of stan-csv files generated
            by fitting the model to the data using any Stan interface.

        :param seed: The seed for random number generator. Must be an integer
            between 0 and 2^32 - 1. If unspecified,
            :class:`numpy.random.RandomState`
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

        :return: CmdStanGQ object
        """
        if isinstance(mcmc_sample, CmdStanMCMC):
            mcmc_fit = mcmc_sample
            sample_csv_files = mcmc_sample.runset.csv_files
        elif isinstance(mcmc_sample, list):
            if len(mcmc_sample) < 1:
                raise ValueError(
                    'Expecting list of Stan CSV files, found empty list'
                )
            try:
                sample_csv_files = mcmc_sample
                sample_fit = from_csv(sample_csv_files)
                mcmc_fit = sample_fit  # type: ignore
            except ValueError as e:
                raise ValueError(
                    'Invalid sample from Stan CSV files, error:\n\t{}\n\t'
                    ' while processing files\n\t{}'.format(
                        repr(e), '\n\t'.join(mcmc_sample)
                    )
                ) from e
        else:
            raise ValueError(
                'MCMC sample must be either CmdStanMCMC object'
                ' or list of paths to sample Stan CSV files.'
            )
        chains = mcmc_fit.chains
        chain_ids = mcmc_fit.chain_ids
        if mcmc_fit.metadata.cmdstan_config['save_warmup']:
            get_logger().warning(
                'Sample contains saved warmup draws which will be used '
                'to generate additional quantities of interest.'
            )
        generate_quantities_args = GenerateQuantitiesArgs(
            csv_files=sample_csv_files
        )
        generate_quantities_args.validate(chains)
        with MaybeDictToFilePath(data, None) as (_data, _inits):
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
                        self._run_cmdstan, runset, i, False, show_console
                    )

            if not runset._check_retcodes():
                msg = 'Error during generate_quantities:\n{}'.format(
                    runset.get_err_msgs()
                )
                msg = '{}Command and output files:\n{}'.format(
                    msg, runset.__repr__()
                )
                raise RuntimeError(msg)
            quantities = CmdStanGQ(runset=runset, mcmc_sample=mcmc_fit)
        return quantities

    def variational(
        self,
        data: Union[Mapping[str, Any], str, None] = None,
        seed: Optional[int] = None,
        inits: Optional[float] = None,
        output_dir: Optional[str] = None,
        sig_figs: Optional[int] = None,
        save_latent_dynamics: bool = False,
        save_profile: bool = False,
        algorithm: Optional[str] = None,
        iter: Optional[int] = None,
        grad_samples: Optional[int] = None,
        elbo_samples: Optional[int] = None,
        eta: Optional[float] = None,
        adapt_engaged: bool = True,
        adapt_iter: Optional[int] = None,
        tol_rel_obj: Optional[float] = None,
        eval_elbo: Optional[int] = None,
        output_samples: Optional[int] = None,
        require_converged: bool = True,
        show_console: bool = False,
        refresh: Optional[int] = None,
        time_fmt: str = "%Y%m%d%H%M%S",
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
        '<model_name>-<YYYYMMDDHHMM>-<chain_id>' plus the file suffix which is
        either '.csv' for the CmdStan output or '.txt' for
        the console messages, e.g. 'bernoulli-201912081451-1.csv'.
        Output files written to the temporary directory contain an additional
        8-character random string, e.g. 'bernoulli-201912081451-1-5nm6as7u.csv'.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param seed: The seed for random number generator. Must be an integer
            between 0 and 2^32 - 1. If unspecified,
            :class:`numpy.random.RandomState`
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

        :param elbo_samples: Number of MC draws for estimate of ELBO.

        :param eta: Step size scaling parameter.

        :param adapt_engaged: Whether eta adaptation is engaged.

        :param adapt_iter: Number of iterations for eta adaptation.

        :param tol_rel_obj: Relative tolerance parameter for convergence.

        :param eval_elbo: Number of iterations between ELBO evaluations.

        :param output_samples: Number of approximate posterior output draws
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
                output_dir=output_dir,
                sig_figs=sig_figs,
                save_latent_dynamics=save_latent_dynamics,
                save_profile=save_profile,
                method_args=variational_args,
                refresh=refresh,
            )

            dummy_chain_id = 0
            runset = RunSet(args=args, chains=1, time_fmt=time_fmt)
            self._run_cmdstan(runset, dummy_chain_id, False, show_console)

        # treat failure to converge as failure
        transcript_file = runset.stdout_files[dummy_chain_id]
        valid = True
        pat = re.compile(r'The algorithm may not have converged.', re.M)
        with open(transcript_file, 'r') as transcript:
            contents = transcript.read()
            errors = re.findall(pat, contents)
            if len(errors) > 0:
                valid = False
        if not valid:
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
            msg = 'Error during variational inference:\n{}'.format(
                runset.get_err_msgs()
            )
            msg = '{}Command and output files:\n{}'.format(
                msg, runset.__repr__()
            )
            raise RuntimeError(msg)
        # pylint: disable=invalid-name
        vb = CmdStanVB(runset)
        return vb

    # pylint: disable=no-self-use
    def _run_cmdstan(
        self,
        runset: RunSet,
        idx: int = 0,
        show_progress: bool = False,
        show_console: bool = False,
        iter_total: int = 0,
    ) -> None:
        """
        Helper function which encapsulates call to CmdStan.
        Uses subprocess POpen object to run the process.
        Records stdout, stderr messages, and process returncode.
        Args 'show_progress' and 'show_console' allow use of progress bar,
        streaming output to console, respectively.
        """

        cmd = runset.cmds[idx]
        get_logger().debug(
            'threads: %s', str(os.environ.get('STAN_NUM_THREADS'))
        )
        if show_progress and progbar.allow_show_progress():
            progress_hook: Optional[
                Callable[[str], None]
            ] = self._wrap_sampler_progress_hook(idx + 1, iter_total)
        else:
            get_logger().info('start chain %d', idx + 1)
            progress_hook = None
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
            while proc.poll() is None:
                if proc.stdout is not None:
                    line = proc.stdout.readline()
                    fd_out.write(line)
                    line = line.strip()
                    if show_console:
                        print('chain {}: {}'.format(idx + 1, line))
                    elif show_progress and progress_hook is not None:
                        progress_hook(line)
            if show_progress and progress_hook is not None:
                progress_hook("Done")

            stdout, _ = proc.communicate()
            if stdout:
                fd_out.write(stdout)
                if show_console:
                    lines = stdout.split('\n')
                    for line in lines:
                        print('chain {}: {}'.format(idx + 1, line))

            fd_out.close()
        except OSError as e:
            msg = 'Failed with error {}\n'.format(str(e))
            raise RuntimeError(msg) from e
        finally:
            fd_out.close()

        if not show_progress:
            get_logger().info('finish chain %d', idx + 1)
        runset._set_retcode(idx, proc.returncode)
        if proc.returncode != 0:
            retcode_summary = returncode_msg(proc.returncode)
            serror = ''
            try:
                serror = os.strerror(proc.returncode)
            except (ArithmeticError, ValueError):
                pass
            get_logger().error(
                'Chain %d error: %s %s', idx + 1, retcode_summary, serror
            )
        else:
            with open(runset.stdout_files[idx], 'r') as fd:
                console = fd.read()
                if 'running fixed_param sampler' in console:
                    sampler_args = runset._args.method_args
                    assert isinstance(
                        sampler_args, SamplerArgs
                    )  # make the typechecker happy
                    sampler_args.fixed_param = True

    @staticmethod
    @progbar.wrap_callback
    def _wrap_sampler_progress_hook(
        chain_id: int, total: int
    ) -> Optional[Callable[[str], None]]:
        """Sets up tqdm callback for CmdStan sampler console msgs."""
        pbar: tqdm = tqdm(
            total=total,
            bar_format="{desc} |{bar}| {elapsed} {postfix[0][value]}",
            postfix=[dict(value="Status")],
            desc=f'chain {chain_id}',
            colour='yellow',
        )

        def sampler_progress_hook(line: str) -> None:
            if line == "Done":
                pbar.postfix[0]["value"] = 'Sampling completed'
                pbar.update(total - pbar.n)
                pbar.close()
            elif line.startswith("Iteration"):
                if 'Sampling' in line:
                    pbar.colour = 'blue'
                pbar.update(1)
                pbar.postfix[0]["value"] = line

        return sampler_progress_hook
