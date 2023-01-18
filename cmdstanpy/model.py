"""CmdStanModel"""

import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import StringIO
from multiprocessing import cpu_count
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

import pandas as pd
from tqdm.auto import tqdm

from cmdstanpy import (
    _CMDSTAN_REFRESH,
    _CMDSTAN_SAMPLING,
    _CMDSTAN_WARMUP,
    _TMPDIR,
)
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
    SanitizedOrTmpFilePath,
    cmdstan_path,
    cmdstan_version,
    cmdstan_version_before,
    do_command,
    get_logger,
    returncode_msg,
)

from . import progress as progbar

OptionalPath = Union[str, os.PathLike, None]
Fit = TypeVar('Fit', CmdStanMCMC, CmdStanMLE, CmdStanVB)


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
        If set to the string ``"force"``, it will always compile even if
        an existing executable is found.

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
        stan_file: OptionalPath = None,
        exe_file: OptionalPath = None,
        # TODO should be Literal['force'] not str
        compile: Union[bool, str] = True,
        stanc_options: Optional[Dict[str, Any]] = None,
        cpp_options: Optional[Dict[str, Any]] = None,
        user_header: OptionalPath = None,
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
        self._fixed_param = False

        if model_name is not None:
            if not model_name.strip():
                raise ValueError(
                    'Invalid value for argument model name, found "{}"'.format(
                        model_name
                    )
                )
            self._name = model_name.strip()

        self._compiler_options.validate()

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
                if self._compiler_options._stanc_options is None:
                    self._compiler_options._stanc_options = {
                        'include-paths': [path]
                    }
                else:
                    self._compiler_options.add_include_path(path)

            # try to detect models w/out parameters, needed for sampler
            if not cmdstan_version_before(
                2, 27
            ):  # unknown end of version range
                try:
                    model_info = self.src_info()
                    if 'parameters' in model_info:
                        self._fixed_param |= len(model_info['parameters']) == 0
                except ValueError as e:
                    if compile:
                        raise
                    get_logger().debug(e)

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
            self.compile(force=str(compile).lower() == 'force')

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
    def stan_file(self) -> OptionalPath:
        """Full path to Stan program file."""
        return self._stan_file

    @property
    def exe_file(self) -> OptionalPath:
        """Full path to Stan exe file."""
        return self._exe_file

    def exe_info(self) -> Dict[str, str]:
        """
        Run model with option 'info'. Parse output statements, which all
        have form 'key = value' into a Dict.
        If exe file compiled with CmdStan < 2.27, option 'info' isn't
        available and the method returns an empty dictionary.
        """
        result: Dict[str, str] = {}
        if self.exe_file is None:
            return result
        try:
            info = StringIO()
            do_command(cmd=[str(self.exe_file), 'info'], fd_out=info)
            lines = info.getvalue().split('\n')
            for line in lines:
                kv_pair = [x.strip() for x in line.split('=')]
                if len(kv_pair) != 2:
                    continue
                result[kv_pair[0]] = kv_pair[1]
            return result
        except RuntimeError as e:
            get_logger().debug(e)
            return result

    def src_info(self) -> Dict[str, Any]:
        """
        Run stanc with option '--info'.

        If stanc is older than 2.27 or if the stan
        file cannot be found, returns an empty dictionary.
        """
        if self.stan_file is None or cmdstan_version_before(2, 27):
            return {}
        cmd = (
            [os.path.join(cmdstan_path(), 'bin', 'stanc' + EXTENSION)]
            # handle include-paths, allow-undefined etc
            + self._compiler_options.compose_stanc()
            + ['--info', str(self.stan_file)]
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode:
            raise ValueError(
                f"Failed to get source info for Stan model "
                f"'{self._stan_file}'. Console:\n{proc.stderr}"
            )
        result: Dict[str, Any] = json.loads(proc.stdout)
        return result

    def format(
        self,
        overwrite_file: bool = False,
        canonicalize: Union[bool, str, Iterable[str]] = False,
        max_line_length: int = 78,
        *,
        backup: bool = True,
    ) -> None:
        """
        Run stanc's auto-formatter on the model code. Either saves directly
        back to the file or prints for inspection


        :param overwrite_file: If True, save the updated code to disk, rather
            than printing it. By default False
        :param canonicalize: Whether or not the compiler should 'canonicalize'
            the Stan model, removing things like deprecated syntax. Default is
            False. If True, all canonicalizations are run. If it is a list of
            strings, those options are passed to stanc (new in Stan 2.29)
        :param max_line_length: Set the wrapping point for the formatter. The
            default value is 78, which wraps most lines by the 80th character.
        :param backup: If True, create a stanfile.bak backup before
            writing to the file. Only disable this if you're sure you have other
            copies of the file or are using a version control system like Git.
        """
        if self.stan_file is None or not os.path.isfile(self.stan_file):
            raise ValueError("No Stan file found for this module")
        try:
            cmd = (
                [os.path.join(cmdstan_path(), 'bin', 'stanc' + EXTENSION)]
                # handle include-paths, allow-undefined etc
                + self._compiler_options.compose_stanc()
                + [str(self.stan_file)]
            )

            if canonicalize:
                if cmdstan_version_before(2, 29):
                    if isinstance(canonicalize, bool):
                        cmd.append('--print-canonical')
                    else:
                        raise ValueError(
                            "Invalid arguments passed for current CmdStan"
                            + " version({})\n".format(
                                cmdstan_version() or "Unknown"
                            )
                            + "--canonicalize requires 2.29 or higher"
                        )
                else:
                    if isinstance(canonicalize, str):
                        cmd.append('--canonicalize=' + canonicalize)
                    elif isinstance(canonicalize, Iterable):
                        cmd.append('--canonicalize=' + ','.join(canonicalize))
                    else:
                        cmd.append('--print-canonical')

            # before 2.29, having both --print-canonical
            # and --auto-format printed twice
            if not (cmdstan_version_before(2, 29) and canonicalize):
                cmd.append('--auto-format')

            if not cmdstan_version_before(2, 29):
                cmd.append(f'--max-line-length={max_line_length}')
            elif max_line_length != 78:
                raise ValueError(
                    "Invalid arguments passed for current CmdStan version"
                    + " ({})\n".format(cmdstan_version() or "Unknown")
                    + "--max-line-length requires 2.29 or higher"
                )

            out = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            if out.stderr:
                get_logger().warning(out.stderr)
            result = out.stdout
            if overwrite_file:
                if result:
                    if backup:
                        shutil.copyfile(
                            self.stan_file,
                            str(self.stan_file)
                            + '.bak-'
                            + datetime.now().strftime("%Y%m%d%H%M%S"),
                        )
                    with (open(self.stan_file, 'w')) as file_handle:
                        file_handle.write(result)
            else:
                print(result)

        except (ValueError, RuntimeError) as e:
            raise RuntimeError("Stanc formatting failed") from e

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
        user_header: OptionalPath = None,
        override_options: bool = False,
    ) -> None:
        """
        Compile the given Stan program file.  Translates the Stan code to
        C++, then calls the C++ compiler.

        By default, this function compares the timestamps on the source and
        executable files; if the executable is newer than the source file, it
        will not recompile the file, unless argument ``force`` is ``True``
        or unless the compiler options have been changed.

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
        if (
            stanc_options is not None
            or cpp_options is not None
            or user_header is not None
        ):
            compiler_options = CompilerOptions(
                stanc_options=stanc_options,
                cpp_options=cpp_options,
                user_header=user_header,
            )
            compiler_options.validate()

            if compiler_options != self._compiler_options:
                force = True
                if self._compiler_options is None:
                    self._compiler_options = compiler_options
                elif override_options:
                    self._compiler_options = compiler_options
                else:
                    self._compiler_options.add(compiler_options)
        exe_target = os.path.splitext(self._stan_file)[0] + EXTENSION
        if os.path.exists(exe_target):
            exe_time = os.path.getmtime(exe_target)
            included_files = [self._stan_file]
            included_files.extend(self.src_info().get('included_files', []))
            out_of_date = any(
                os.path.getmtime(included_file) > exe_time
                for included_file in included_files
            )
            if not out_of_date and not force:
                get_logger().debug('found newer exe file, not recompiling')
                if self._exe_file is None:  # called from constructor
                    self._exe_file = exe_target
                return

        compilation_failed = False
        # if target path has space, use copy in a tmpdir (GNU-Make constraint)
        with SanitizedOrTmpFilePath(self._stan_file) as (stan_file, is_copied):
            exe_file = os.path.splitext(stan_file)[0] + EXTENSION

            hpp_file = os.path.splitext(exe_file)[0] + '.hpp'
            if os.path.exists(hpp_file):
                os.remove(hpp_file)
            if os.path.exists(exe_file):
                get_logger().debug('Removing %s', exe_file)
                os.remove(exe_file)

            get_logger().info(
                'compiling stan file %s to exe file %s',
                self._stan_file,
                exe_target,
            )

            make = os.getenv(
                'MAKE',
                'make' if platform.system() != 'Windows' else 'mingw32-make',
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

            get_logger().debug('Console output:\n%s', console)
            if not compilation_failed:
                if is_copied:
                    shutil.copy(exe_file, exe_target)
                self._exe_file = exe_target
                get_logger().info(
                    'compiled model executable: %s', self._exe_file
                )
            if 'Warning' in console:
                lines = console.split('\n')
                warnings = [x for x in lines if x.startswith('Warning')]
                get_logger().warning(
                    'Stan compiler has produced %d warnings:',
                    len(warnings),
                )
                get_logger().warning(console)
            if compilation_failed:
                if 'PCH' in console or 'precompiled header' in console:
                    get_logger().warning(
                        "CmdStan's precompiled header (PCH) files "
                        "may need to be rebuilt."
                        "Please run cmdstanpy.rebuild_cmdstan().\n"
                        "If the issue persists please open a bug report"
                    )
                raise ValueError(
                    f"Failed to compile Stan model '{self._stan_file}'. "
                    f"Console:\n{console}"
                )

    def optimize(
        self,
        data: Union[Mapping[str, Any], str, os.PathLike, None] = None,
        seed: Optional[int] = None,
        inits: Union[Dict[str, float], float, str, os.PathLike, None] = None,
        output_dir: OptionalPath = None,
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
        timeout: Optional[float] = None,
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

        :param timeout: Duration at which optimization times out in seconds.

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
            self._run_cmdstan(
                runset,
                dummy_chain_id,
                show_console=show_console,
                timeout=timeout,
            )
        runset.raise_for_timeouts()

        if not runset._check_retcodes():
            msg = "Error during optimization! Command '{}' failed: {}".format(
                ' '.join(runset.cmd(0)), runset.get_err_msgs()
            )
            if 'Line search failed' in msg and not require_converged:
                get_logger().warning(msg)
            else:
                raise RuntimeError(msg)
        mle = CmdStanMLE(runset)
        return mle

    # pylint: disable=too-many-arguments
    def sample(
        self,
        data: Union[Mapping[str, Any], str, os.PathLike, None] = None,
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
        output_dir: OptionalPath = None,
        sig_figs: Optional[int] = None,
        save_latent_dynamics: bool = False,
        save_profile: bool = False,
        show_progress: bool = True,
        show_console: bool = False,
        refresh: Optional[int] = None,
        time_fmt: str = "%Y%m%d%H%M%S",
        timeout: Optional[float] = None,
        *,
        force_one_process_per_chain: Optional[bool] = None,
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
            a positive integer.  Defaults to :func:`multiprocessing.cpu_count`,
            i.e., it will only run as many chains in parallel as there are
            cores on the machine.   Note that CmdStan 2.28 and higher can run
            all chains in parallel providing that the model was compiled with
            threading support.

        :param threads_per_chain: The number of threads to use in parallelized
            sections within an MCMC chain (e.g., when using the Stan functions
            ``reduce_sum()``  or ``map_rect()``).  This will only have an effect
            if the model was compiled with threading support.  For such models,
            CmdStan version 2.28 and higher will run all chains in parallel
            from within a single process.  The total number of threads used
            will be ``parallel_chains * threads_per_chain``, where the default
            value for parallel_chains is the number of cpus, not chains.

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

        :param force_one_process_per_chain: If ``True``, run multiple chains in
            distinct processes regardless of model ability to run parallel
            chains (CmdStan 2.28+ feature). If ``False``, always run multiple
            chains in one process (does not check that this is valid).

            If None (Default): Check that CmdStan version is >=2.28, and that
            model was compiled with STAN_THREADS=True, and utilize the
            parallel chain functionality if those conditions are met.

        :param timeout: Duration at which sampling times out in seconds.

        :return: CmdStanMCMC object
        """
        if fixed_param is None:
            fixed_param = self._fixed_param

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
            chain_ids = [i + 1 for i in range(chains)]
        else:
            if isinstance(chain_ids, int):
                if chain_ids < 1:
                    raise ValueError(
                        'Chain_id must be a positive integer value,'
                        ' found {}.'.format(chain_ids)
                    )
                chain_ids = [i + chain_ids for i in range(chains)]
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
                    'Argument parallel_chains must be a positive integer, '
                    'found {}.'.format(parallel_chains)
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
            if fixed_param or (chains == 1):
                force_one_process_per_chain = True

            if (
                force_one_process_per_chain is None
                and not cmdstan_version_before(2, 28, info_dict)
                and stan_threads == 'true'
            ):
                one_process_per_chain = False
                num_threads = parallel_chains * num_threads
                parallel_procs = 1
            if force_one_process_per_chain is False:
                if not cmdstan_version_before(2, 28, info_dict):
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
                else:
                    get_logger().warning(
                        'Installed version of CmdStan cannot multi-process '
                        'chains, will run %d processes. '
                        'Run "install_cmdstan" to upgrade to latest version.',
                        chains,
                    )
            os.environ['STAN_NUM_THREADS'] = str(num_threads)

            if show_console:
                show_progress = False
            else:
                show_progress = show_progress and progbar.allow_show_progress()
                get_logger().info('CmdStan start processing')

            progress_hook: Optional[Callable[[str, int], None]] = None
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
                if refresh is None:
                    refresh = _CMDSTAN_REFRESH
                iter_total = iter_total // refresh + 2

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

            # if there was an exe-file only initialization,
            # this could happen, so throw a nice error
            if (
                sampler_args.fixed_param
                and not one_process_per_chain
                and chains > 1
            ):
                raise RuntimeError(
                    "Cannot use single-process multichain parallelism"
                    " with algorithm fixed_param.\nTry setting argument"
                    " force_one_process_per_chain to True"
                )

            errors = runset.get_err_msgs()
            if not runset._check_retcodes():
                msg = (
                    f'Error during sampling:\n{errors}\n'
                    + f'Command and output files:\n{repr(runset)}\n'
                    + 'Consider re-running with show_console=True if the above'
                    + ' output is unclear!'
                )
                raise RuntimeError(msg)
            if errors:
                msg = (
                    f'Non-fatal error during sampling:\n{errors}\n'
                    + 'Consider re-running with show_console=True if the above'
                    + ' output is unclear!'
                )
                get_logger().warning(msg)

            mcmc = CmdStanMCMC(runset)
        return mcmc

    def generate_quantities(
        self,
        data: Union[Mapping[str, Any], str, os.PathLike, None] = None,
        previous_fit: Union[Fit, List[str], None] = None,
        seed: Optional[int] = None,
        gq_output_dir: OptionalPath = None,
        sig_figs: Optional[int] = None,
        show_console: bool = False,
        refresh: Optional[int] = None,
        time_fmt: str = "%Y%m%d%H%M%S",
        timeout: Optional[float] = None,
        *,
        mcmc_sample: Union[CmdStanMCMC, List[str], None] = None,
    ) -> CmdStanGQ[Fit]:
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
        specified output directory.  Output filenames correspond to the template
        '<model_name>-<YYYYMMDDHHMM>-<chain_id>' plus the file suffix which is
        either '.csv' for the CmdStan output or '.txt' for
        the console messages, e.g. 'bernoulli-201912081451-1.csv'.
        Output files written to the temporary directory contain an additional
        8-character random string, e.g. 'bernoulli-201912081451-1-5nm6as7u.csv'.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param previous_fit: Can be either a :class:`CmdStanMCMC`,
            :class:`CmdStanMLE`, or :class:`CmdStanVB` or a list of
            stan-csv files generated by fitting the model to the data
            using any Stan interface.

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

        :param timeout: Duration at which generation times out in seconds.

        :return: CmdStanGQ object
        """
        if mcmc_sample is not None:
            if previous_fit:
                raise ValueError(
                    "Cannot supply both 'previous_fit' and "
                    "deprecated argument 'mcmc_sample'"
                )
            get_logger().warning(
                "Argument name `mcmc_sample` is deprecated, please "
                "rename to `previous_fit`."
            )

            previous_fit = mcmc_sample  # type: ignore

        if isinstance(previous_fit, (CmdStanMCMC, CmdStanMLE, CmdStanVB)):
            fit_object = previous_fit
            fit_csv_files = previous_fit.runset.csv_files
        elif isinstance(previous_fit, list):
            if len(previous_fit) < 1:
                raise ValueError(
                    'Expecting list of Stan CSV files, found empty list'
                )
            try:
                fit_csv_files = previous_fit
                fit_object = from_csv(fit_csv_files)  # type: ignore
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
                ' or list of paths to Stan CSV files.'
            )
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
            if fit_object._save_iterations:
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
                    + f'Command and output files:\n{repr(runset)}\n'
                    + 'Consider re-running with show_console=True if the above'
                    + ' output is unclear!'
                )
                raise RuntimeError(msg)
            quantities = CmdStanGQ(runset=runset, previous_fit=fit_object)
        return quantities

    def variational(
        self,
        data: Union[Mapping[str, Any], str, os.PathLike, None] = None,
        seed: Optional[int] = None,
        inits: Optional[float] = None,
        output_dir: OptionalPath = None,
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
        timeout: Optional[float] = None,
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
            Default is 10.  If problems arise, try doubling current value.

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
        # pylint: disable=invalid-name
        vb = CmdStanVB(runset)
        return vb

    def log_prob(
        self,
        params: Union[Dict[str, Any], str, os.PathLike],
        data: Union[Mapping[str, Any], str, os.PathLike, None] = None,
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

        :return: A pandas.DataFrame containing columns "lp_" and additional
            columns for the gradient values. These gradients will be for the
            unconstrained parameters of the model.
        """

        if cmdstan_version_before(2, 31, self.exe_info()):
            raise ValueError(
                "Method 'log_prob' not available for CmdStan versions "
                "before 2.31"
            )
        with MaybeDictToFilePath(data, params) as (_data, _params):
            cmd = [
                str(self.exe_file),
                "log_prob",
                f"constrained_params={_params}",
            ]
            if _data is not None:
                cmd += ["data", f"file={_data}"]

            output_dir = tempfile.mkdtemp(prefix=self.name, dir=_TMPDIR)

            output = os.path.join(output_dir, "output.csv")
            cmd += ["output", f"file={output}"]

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

    def _run_cmdstan(
        self,
        runset: RunSet,
        idx: int,
        show_progress: bool = False,
        show_console: bool = False,
        progress_hook: Optional[Callable[[str, int], None]] = None,
        timeout: Optional[float] = None,
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
            timer: Optional[threading.Timer]
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
            retcode_summary = returncode_msg(retcode)
            serror = ''
            try:
                serror = os.strerror(retcode)
            except (ArithmeticError, ValueError):
                pass
            get_logger().error(
                '%s error: %s %s', logger_prefix, retcode_summary, serror
            )

    @staticmethod
    @progbar.wrap_callback
    def _wrap_sampler_progress_hook(
        chain_ids: List[int],
        total: int,
    ) -> Optional[Callable[[str, int], None]]:
        """
        Sets up tqdm callback for CmdStan sampler console msgs.
        CmdStan progress messages start with "Iteration", for single chain
        process, "Chain [id] Iteration" for multi-chain processing.
        For the latter, manage array of pbars, update accordingly.
        """
        pat = re.compile(r'Chain \[(\d*)\] (Iteration.*)')
        pbars: Dict[int, tqdm] = {
            chain_id: tqdm(
                total=total,
                bar_format="{desc} |{bar}| {elapsed} {postfix[0][value]}",
                postfix=[dict(value="Status")],
                desc=f'chain {chain_id}',
                colour='yellow',
            )
            for chain_id in chain_ids
        }

        def progress_hook(line: str, idx: int) -> None:
            if line == "Done":
                for pbar in pbars.values():
                    pbar.postfix[0]["value"] = 'Sampling completed'
                    pbar.update(total - pbar.n)
                    pbar.close()
            else:
                match = pat.match(line)
                if match:
                    idx = int(match.group(1))
                    mline = match.group(2).strip()
                elif line.startswith("Iteration"):
                    mline = line
                    idx = chain_ids[idx]
                else:
                    return
                if 'Sampling' in mline:
                    pbars[idx].colour = 'blue'
                pbars[idx].update(1)
                pbars[idx].postfix[0]["value"] = mline

        return progress_hook
