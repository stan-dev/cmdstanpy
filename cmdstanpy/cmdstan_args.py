"""
CmdStan arguments
"""
import logging
import os
from enum import Enum, auto
from numbers import Integral, Real
from time import time
from typing import List, Union

from numpy.random import RandomState

from cmdstanpy.utils import (
    cmdstan_path,
    cmdstan_version_at,
    get_logger,
    read_metric,
)


class Method(Enum):
    """Supported CmdStan method names."""

    SAMPLE = auto()
    OPTIMIZE = auto()
    GENERATE_QUANTITIES = auto()
    VARIATIONAL = auto()

    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


class SamplerArgs:
    """Arguments for the NUTS adaptive sampler."""

    def __init__(
        self,
        iter_warmup: int = None,
        iter_sampling: int = None,
        save_warmup: bool = False,
        thin: int = None,
        max_treedepth: int = None,
        metric: Union[str, List[str]] = None,
        step_size: Union[float, List[float]] = None,
        adapt_engaged: bool = True,
        adapt_delta: float = None,
        adapt_init_phase: int = None,
        adapt_metric_window: int = None,
        adapt_step_size: int = None,
        fixed_param: bool = False,
    ) -> None:
        """Initialize object."""
        self.iter_warmup = iter_warmup
        self.iter_sampling = iter_sampling
        self.save_warmup = save_warmup
        self.thin = thin
        self.max_treedepth = max_treedepth
        self.metric = metric
        self.metric_file = None
        self.step_size = step_size
        self.adapt_engaged = adapt_engaged
        self.adapt_delta = adapt_delta
        self.adapt_init_phase = adapt_init_phase
        self.adapt_metric_window = adapt_metric_window
        self.adapt_step_size = adapt_step_size
        self.fixed_param = fixed_param
        self.diagnostic_file = None

    def validate(self, chains: int) -> None:
        """
        Check arguments correctness and consistency.

        * adaptation and warmup args are consistent
        * if file(s) for metric are supplied, check contents.
        * length of per-chain lists equals specified # of chains
        """
        if not isinstance(chains, Integral) or chains < 1:
            raise ValueError(
                'sampler expects number of chains to be greater than 0'
            )
        if not (
            self.adapt_delta is None
            and self.adapt_init_phase is None
            and self.adapt_metric_window is None
            and self.adapt_step_size is None
        ):
            if self.adapt_engaged is False:
                msg = 'conflicting arguments: adapt_engaged: False'
                if self.adapt_delta is not None:
                    msg = '{}, adapt_delta: {}'.format(msg, self.adapt_delta)
                if self.adapt_init_phase is not None:
                    msg = '{}, adapt_init_phase: {}'.format(
                        msg, self.adapt_init_phase
                    )
                if self.adapt_metric_window is not None:
                    msg = '{}, adapt_metric_window: {}'.format(
                        msg, self.adapt_metric_window
                    )
                if self.adapt_step_size is not None:
                    msg = '{}, adapt_step_size: {}'.format(
                        msg, self.adapt_step_size
                    )
                raise ValueError(msg)

        if self.iter_warmup is not None:
            if self.iter_warmup < 0 or not isinstance(
                self.iter_warmup, Integral
            ):
                raise ValueError(
                    'iter_warmup must be a non-negative integer,'
                    ' found {}'.format(self.iter_warmup)
                )
            if self.iter_warmup > 0 and not self.adapt_engaged:
                raise ValueError(
                    'adapt_engaged is False, cannot specify warmup iterations'
                )
        if self.iter_sampling is not None:
            if self.iter_sampling < 0 or not isinstance(
                self.iter_sampling, Integral
            ):
                raise ValueError(
                    'iter_sampling must be a non-negative integer,'
                    ' found {}'.format(self.iter_sampling)
                )
        if self.thin is not None:
            if self.thin < 1 or not isinstance(self.thin, Integral):
                raise ValueError(
                    'thin must be a positive integer,'
                    'found {}'.format(self.thin)
                )
        if self.max_treedepth is not None:
            if self.max_treedepth < 1 or not isinstance(
                self.max_treedepth, Integral
            ):
                raise ValueError(
                    'max_treedepth must be a positive integer,'
                    ' found {}'.format(self.max_treedepth)
                )
        if self.step_size is not None:
            if isinstance(self.step_size, Real):
                if self.step_size <= 0:
                    raise ValueError(
                        'step_size must be > 0, found {}'.format(self.step_size)
                    )
            else:
                if len(self.step_size) != chains:
                    raise ValueError(
                        'number of step_sizes must match number of chains,'
                        ' found {} step_sizes for {} chains'.format(
                            len(self.step_size), chains
                        )
                    )
                for step_size in self.step_size:
                    if step_size < 0:
                        raise ValueError(
                            'step_size must be > 0, found {}'.format(step_size)
                        )
        if self.metric is not None:
            dims = []
            if isinstance(self.metric, str):
                if self.metric in ['diag', 'diag_e']:
                    self.metric = 'diag_e'
                elif self.metric in ['dense', 'dense_e']:
                    self.metric = 'dense_e'
                else:
                    if not os.path.exists(self.metric):
                        raise ValueError('no such file {}'.format(self.metric))
                    dims = read_metric(self.metric)
            elif isinstance(self.metric, (list, tuple)):
                if len(self.metric) != chains:
                    raise ValueError(
                        'number of metric files must match number of chains,'
                        ' found {} metric files for {} chains'.format(
                            len(self.metric), chains
                        )
                    )
                names_set = set(self.metric)
                if len(names_set) != len(self.metric):
                    raise ValueError(
                        'each chain must have its own metric file,'
                        ' found duplicates in metric files list.'
                    )
                for i, metric in enumerate(self.metric):
                    if not os.path.exists(metric):
                        raise ValueError('no such file {}'.format(metric))
                    if i == 0:
                        dims = read_metric(metric)
                    else:
                        dims2 = read_metric(metric)
                        if len(dims) != len(dims2):
                            raise ValueError(
                                'metrics files {}, {},'
                                ' inconsistent metrics'.format(
                                    self.metric[0], metric
                                )
                            )
                        for dim, dim2 in zip(dims, dims2):
                            if dim != dim2:
                                raise ValueError(
                                    'metrics files {}, {},'
                                    ' inconsistent metrics'.format(
                                        self.metric[0], metric
                                    )
                                )
            if any(dims):
                if len(dims) > 2 or (len(dims) == 2 and dims[0] != dims[1]):
                    raise ValueError('bad metric specifiation')
                self.metric_file = self.metric
                if len(dims) == 1:
                    self.metric = 'diag_e'
                elif len(dims) == 2:
                    self.metric = 'dense_e'
        if self.adapt_delta is not None:
            if not 0 < self.adapt_delta < 1:
                raise ValueError(
                    'adapt_delta must be between 0 and 1,'
                    ' found {}'.format(self.adapt_delta)
                )
        if self.adapt_init_phase is not None:
            if self.adapt_init_phase < 0 or not isinstance(
                self.adapt_init_phase, Integral
            ):
                raise ValueError(
                    'adapt_init_phase must be a non-negative integer,'
                    'found {}'.format(self.adapt_init_phase)
                )
        if self.adapt_metric_window is not None:
            if self.adapt_metric_window < 0 or not isinstance(
                self.adapt_metric_window, Integral
            ):
                raise ValueError(
                    'adapt_metric_window must be a non-negative integer,'
                    'found {}'.format(self.adapt_metric_window)
                )
        if self.adapt_step_size is not None:
            if self.adapt_step_size < 0 or not isinstance(
                self.adapt_step_size, Integral
            ):
                raise ValueError(
                    'adapt_step_size must be a non-negative integer,'
                    'found {}'.format(self.adapt_step_size)
                )

        if self.fixed_param and (
            (self.iter_warmup is not None and self.iter_warmup > 0)
            or self.save_warmup
            or self.max_treedepth is not None
            or self.metric is not None
            or self.step_size is not None
            or not (
                self.adapt_delta is None
                and self.adapt_init_phase is None
                and self.adapt_metric_window is None
                and self.adapt_step_size is None
            )
        ):
            raise ValueError(
                'when fixed_param=True, cannot specify warmup'
                ' or adaptation parameters.'
            )

    def compose(self, idx: int, cmd: List) -> str:
        """
        Compose CmdStan command for method-specific non-default arguments.
        """
        cmd.append('method=sample')
        if self.iter_sampling is not None:
            cmd.append('num_samples={}'.format(self.iter_sampling))
        if self.iter_warmup is not None:
            cmd.append('num_warmup={}'.format(self.iter_warmup))
        if self.save_warmup:
            cmd.append('save_warmup=1')
        if self.thin is not None:
            cmd.append('thin={}'.format(self.thin))
        if self.fixed_param:
            cmd.append('algorithm=fixed_param')
            return cmd
        else:
            cmd.append('algorithm=hmc')
        if self.max_treedepth is not None:
            cmd.append('engine=nuts')
            cmd.append('max_depth={}'.format(self.max_treedepth))
        if self.step_size is not None:
            if not isinstance(self.step_size, list):
                cmd.append('stepsize={}'.format(self.step_size))
            else:
                cmd.append('stepsize={}'.format(self.step_size[idx]))
        if self.metric is not None:
            cmd.append('metric={}'.format(self.metric))
        if self.metric_file is not None:
            if not isinstance(self.metric_file, list):
                cmd.append('metric_file={}'.format(self.metric_file))
            else:
                cmd.append('metric_file={}'.format(self.metric_file[idx]))
        cmd.append('adapt')
        if self.adapt_engaged:
            cmd.append('engaged=1')
        else:
            cmd.append('engaged=0')
        if self.adapt_delta is not None:
            cmd.append('delta={}'.format(self.adapt_delta))
        if self.adapt_init_phase is not None:
            cmd.append('init_buffer={}'.format(self.adapt_init_phase))
        if self.adapt_metric_window is not None:
            cmd.append('window={}'.format(self.adapt_metric_window))
        if self.adapt_step_size is not None:
            cmd.append('term_buffer={}'.format(self.adapt_step_size))

        return cmd


class OptimizeArgs:
    """Container for arguments for the optimizer."""

    OPTIMIZE_ALGOS = {'BFGS', 'LBFGS', 'Newton'}

    def __init__(
        self, algorithm: str = None, init_alpha: Real = None, iter: int = None
    ) -> None:

        self.algorithm = algorithm
        self.init_alpha = init_alpha
        self.iter = iter

    def validate(self, chains=None) -> None:  # pylint: disable=unused-argument
        """
        Check arguments correctness and consistency.
        """
        if (
            self.algorithm is not None
            and self.algorithm not in self.OPTIMIZE_ALGOS
        ):
            raise ValueError(
                'Please specify optimizer algorithms as one of [{}]'.format(
                    ', '.join(self.OPTIMIZE_ALGOS)
                )
            )

        if self.init_alpha is not None:
            if self.algorithm == 'Newton':
                raise ValueError(
                    'init_alpha must not be set when algorithm is Newton'
                )
            if isinstance(self.init_alpha, Real):
                if self.init_alpha <= 0:
                    raise ValueError('init_alpha must be greater than 0')
            else:
                raise ValueError('init_alpha must be type of float')

        if self.iter is not None:
            if isinstance(self.iter, Integral):
                if self.iter < 0:
                    raise ValueError('iter must be greater than 0')
            else:
                raise ValueError('iter must be type of int')

    # pylint: disable=unused-argument
    def compose(self, idx: int, cmd: List) -> str:
        """compose command string for CmdStan for non-default arg values."""
        cmd.append('method=optimize')
        if self.algorithm:
            cmd.append('algorithm={}'.format(self.algorithm.lower()))
        if self.init_alpha is not None:
            cmd.append('init_alpha={}'.format(self.init_alpha))
        if self.iter is not None:
            cmd.append('iter={}'.format(self.iter))
        return cmd


class GenerateQuantitiesArgs:
    """Arguments needed for generate_quantities method."""

    def __init__(self, csv_files: List[str]) -> None:
        """Initialize object."""
        self.sample_csv_files = csv_files

    def validate(self, chains: int) -> None:  # pylint: disable=unused-argument
        """
        Check arguments correctness and consistency.

        * check that sample csv files exist
        """
        for csv in self.sample_csv_files:
            if not os.path.exists(csv):
                raise ValueError(
                    'Invalid path for sample csv file: {}'.format(csv)
                )

    def compose(self, idx: int, cmd: List) -> str:
        """
        Compose CmdStan command for method-specific non-default arguments.
        """
        cmd.append('method=generate_quantities')
        cmd.append('fitted_params={}'.format(self.sample_csv_files[idx - 1]))
        return cmd


class VariationalArgs:
    """Arguments needed for variational method."""

    VARIATIONAL_ALGOS = {'meanfield', 'fullrank'}

    def __init__(
        self,
        algorithm: str = None,
        iter: int = None,
        grad_samples: int = None,
        elbo_samples: int = None,
        eta: Real = None,
        adapt_iter: int = None,
        adapt_engaged: bool = True,
        tol_rel_obj: Real = None,
        eval_elbo: int = None,
        output_samples: int = None,
    ) -> None:
        self.algorithm = algorithm
        self.iter = iter
        self.grad_samples = grad_samples
        self.elbo_samples = elbo_samples
        self.eta = eta
        self.adapt_iter = adapt_iter
        self.adapt_engaged = adapt_engaged
        self.tol_rel_obj = tol_rel_obj
        self.eval_elbo = eval_elbo
        self.output_samples = output_samples

    def validate(self, chains=None) -> None:  # pylint: disable=unused-argument
        """
        Check arguments correctness and consistency.
        """
        if (
            self.algorithm is not None
            and self.algorithm not in self.VARIATIONAL_ALGOS
        ):
            raise ValueError(
                'Please specify variational algorithms as one of [{}]'.format(
                    ', '.join(self.VARIATIONAL_ALGOS)
                )
            )
        if self.iter is not None:
            if self.iter < 1 or not isinstance(self.iter, Integral):
                raise ValueError(
                    'iter must be a positive integer,'
                    ' found {}'.format(self.iter)
                )
        if self.grad_samples is not None:
            if self.grad_samples < 1 or not isinstance(
                self.grad_samples, Integral
            ):
                raise ValueError(
                    'grad_samples must be a positive integer,'
                    ' found {}'.format(self.grad_samples)
                )
        if self.elbo_samples is not None:
            if self.elbo_samples < 1 or not isinstance(
                self.elbo_samples, Integral
            ):
                raise ValueError(
                    'elbo_samples must be a positive integer,'
                    ' found {}'.format(self.elbo_samples)
                )
        if self.eta is not None:
            if self.eta < 0 or not isinstance(self.eta, (Integral, Real)):
                raise ValueError(
                    'eta must be a non-negative number,'
                    ' found {}'.format(self.eta)
                )
        if self.adapt_iter is not None:
            if self.adapt_iter < 1 or not isinstance(self.adapt_iter, Integral):
                raise ValueError(
                    'adapt_iter must be a positive integer,'
                    ' found {}'.format(self.adapt_iter)
                )
        if self.tol_rel_obj is not None:
            if self.tol_rel_obj <= 0 or not isinstance(
                self.tol_rel_obj, (Integral, Real)
            ):
                raise ValueError(
                    'tol_rel_obj must be a positive number,'
                    ' found {}'.format(self.tol_rel_obj)
                )
        if self.eval_elbo is not None:
            if self.eval_elbo < 1 or not isinstance(self.eval_elbo, Integral):
                raise ValueError(
                    'eval_elbo must be a positive integer,'
                    ' found {}'.format(self.eval_elbo)
                )
        if self.output_samples is not None:
            if self.output_samples < 1 or not isinstance(
                self.output_samples, Integral
            ):
                raise ValueError(
                    'output_samples must be a positive integer,'
                    ' found {}'.format(self.output_samples)
                )

    # pylint: disable=unused-argument
    def compose(self, idx: int, cmd: List) -> str:
        """
        Compose CmdStan command for method-specific non-default arguments.
        """
        cmd.append('method=variational')
        if self.algorithm is not None:
            cmd.append('algorithm={}'.format(self.algorithm))
        if self.iter is not None:
            cmd.append('iter={}'.format(self.iter))
        if self.grad_samples is not None:
            cmd.append('grad_samples={}'.format(self.grad_samples))
        if self.elbo_samples is not None:
            cmd.append('elbo_samples={}'.format(self.elbo_samples))
        if self.eta is not None:
            cmd.append('eta={}'.format(self.eta))
        cmd.append('adapt')
        if self.adapt_engaged:
            cmd.append('engaged=1')
            if self.adapt_iter is not None:
                cmd.append('iter={}'.format(self.adapt_iter))
        else:
            cmd.append('engaged=0')
        if self.tol_rel_obj is not None:
            cmd.append('tol_rel_obj={}'.format(self.tol_rel_obj))
        if self.eval_elbo is not None:
            cmd.append('eval_elbo={}'.format(self.eval_elbo))
        if self.output_samples is not None:
            cmd.append('output_samples={}'.format(self.output_samples))
        return cmd


class CmdStanArgs:
    """
    Container for CmdStan command line arguments.
    Consists of arguments common to all methods and
    and an object which contains the method-specific arguments.
    """

    def __init__(
        self,
        model_name: str,
        model_exe: str,
        chain_ids: Union[List[int], None],
        method_args: Union[
            SamplerArgs, OptimizeArgs, GenerateQuantitiesArgs, VariationalArgs
        ],
        data: Union[str, dict] = None,
        seed: Union[int, List[int]] = None,
        inits: Union[int, float, str, List[str]] = None,
        output_dir: str = None,
        sig_figs: str = None,
        save_diagnostics: bool = False,
        refresh: str = None,
        logger: logging.Logger = None,
    ) -> None:
        """Initialize object."""
        self.model_name = model_name
        self.model_exe = model_exe
        self.chain_ids = chain_ids
        self.data = data
        self.seed = seed
        self.inits = inits
        self.output_dir = output_dir
        self.sig_figs = sig_figs
        self.save_diagnostics = save_diagnostics
        self.refresh = refresh
        self.method_args = method_args
        if isinstance(method_args, SamplerArgs):
            self.method = Method.SAMPLE
        elif isinstance(method_args, OptimizeArgs):
            self.method = Method.OPTIMIZE
        elif isinstance(method_args, GenerateQuantitiesArgs):
            self.method = Method.GENERATE_QUANTITIES
        elif isinstance(method_args, VariationalArgs):
            self.method = Method.VARIATIONAL
        self.method_args.validate(len(chain_ids) if chain_ids else None)
        self._logger = logger or get_logger()
        self.validate()

    def validate(self) -> None:
        """
        Check arguments correctness and consistency.

        * input files must exist
        * output files must be in a writeable directory
        * if no seed specified, set random seed.
        * length of per-chain lists equals specified # of chains
        """
        if self.model_name is None:
            raise ValueError('no stan model specified')
        if self.model_exe is None:
            raise ValueError('model not compiled')

        if self.chain_ids is not None:
            for i in range(len(self.chain_ids)):
                if self.chain_ids[i] < 1:
                    raise ValueError(
                        'invalid chain_id {}'.format(self.chain_ids[i])
                    )
        if self.output_dir is not None:
            self.output_dir = os.path.realpath(
                os.path.expanduser(self.output_dir)
            )
            if not os.path.exists(self.output_dir):
                try:
                    os.makedirs(self.output_dir)
                    self._logger.info(
                        'created output directory: %s', self.output_dir
                    )
                except (RuntimeError, PermissionError) as exc:
                    raise ValueError(
                        'invalid path for output files, no such dir: {}'.format(
                            self.output_dir
                        )
                    ) from exc
            if not os.path.isdir(self.output_dir):
                raise ValueError(
                    'specified output_dir not a directory: {}'.format(
                        self.output_dir
                    )
                )
            try:
                testpath = os.path.join(self.output_dir, str(time()))
                with open(testpath, 'w+'):
                    pass
                os.remove(testpath)  # cleanup
            except Exception as exc:
                raise ValueError(
                    'invalid path for output files,'
                    ' cannot write to dir: {}'.format(self.output_dir)
                ) from exc

        if self.sig_figs is not None:
            if (
                not isinstance(self.sig_figs, int)
                or self.sig_figs < 1
                or self.sig_figs > 18
            ):
                raise ValueError(
                    'sig_figs must be an integer between 1 and 18,'
                    ' found {}'.format(self.sig_figs)
                )
            if not cmdstan_version_at(2, 25):
                self.sig_figs = None
                self._logger.warning(
                    'arg sig_figs not valid, CmdStan version must be 2.25 '
                    'or higher, using verson %s in directory %s',
                    os.path.basename(cmdstan_path()),
                    os.path.dirname(cmdstan_path()),
                )

        if self.seed is None:
            rng = RandomState()
            self.seed = rng.randint(1, 99999 + 1)
        else:
            if not isinstance(self.seed, (int, list)):
                raise ValueError(
                    'seed must be an integer between 0 and 2**32-1,'
                    ' found {}'.format(self.seed)
                )
            if isinstance(self.seed, int):
                if self.seed < 0 or self.seed > 2 ** 32 - 1:
                    raise ValueError(
                        'seed must be an integer between 0 and 2**32-1,'
                        ' found {}'.format(self.seed)
                    )
            else:
                if self.chain_ids is None:
                    raise ValueError(
                        'seed must not be a list when no chains used'
                    )

                if len(self.seed) != len(self.chain_ids):
                    raise ValueError(
                        'number of seeds must match number of chains,'
                        ' found {} seed for {} chains '.format(
                            len(self.seed), len(self.chain_ids)
                        )
                    )
                for i in range(len(self.seed)):
                    if self.seed[i] < 0 or self.seed[i] > 2 ** 32 - 1:
                        raise ValueError(
                            'seed must be an integer value'
                            ' between 0 and 2**32-1,'
                            ' found {}'.format(self.seed[i])
                        )

        if isinstance(self.data, str):
            if not os.path.exists(self.data):
                raise ValueError('no such file {}'.format(self.data))
        elif self.data is None:
            if isinstance(self.method_args, OptimizeArgs):
                raise ValueError('data must be set when optimizing')
        elif not isinstance(self.data, (str, dict)):
            raise ValueError('data must be string or dict')

        if self.inits is not None:
            if isinstance(self.inits, (Integral, Real)):
                if self.inits < 0:
                    raise ValueError(
                        'inits must be > 0, found {}'.format(self.inits)
                    )
            elif isinstance(self.inits, str):
                if not os.path.exists(self.inits):
                    raise ValueError('no such file {}'.format(self.inits))
            elif isinstance(self.inits, list):
                if self.chain_ids is None:
                    raise ValueError(
                        'inits must not be a list when no chains are used'
                    )

                if len(self.inits) != len(self.chain_ids):
                    raise ValueError(
                        'number of inits files must match number of chains,'
                        ' found {} inits files for {} chains '.format(
                            len(self.inits), len(self.chain_ids)
                        )
                    )
                names_set = set(self.inits)
                if len(names_set) != len(self.inits):
                    raise ValueError(
                        'each chain must have its own init file,'
                        ' found duplicates in inits files list.'
                    )
                for i in range(len(self.inits)):
                    if not os.path.exists(self.inits[i]):
                        raise ValueError(
                            'no such file {}'.format(self.inits[i])
                        )

    def compose_command(
        self, idx: int, csv_file: str, diagnostic_file: str = None
    ) -> str:
        """
        Compose CmdStan command for non-default arguments.
        """
        cmd = []
        if idx is not None and self.chain_ids is not None:
            if idx < 0 or idx > len(self.chain_ids) - 1:
                raise ValueError(
                    'index ({}) exceeds number of chains ({})'.format(
                        idx, len(self.chain_ids)
                    )
                )
            cmd.append(self.model_exe)
            cmd.append('id={}'.format(self.chain_ids[idx]))
        else:
            cmd.append(self.model_exe)

        if self.seed is not None:
            if not isinstance(self.seed, list):
                cmd.append('random')
                cmd.append('seed={}'.format(self.seed))
            else:
                cmd.append('random')
                cmd.append('seed={}'.format(self.seed[idx]))
        if self.data is not None:
            cmd.append('data')
            cmd.append('file={}'.format(self.data))
        if self.inits is not None:
            if not isinstance(self.inits, list):
                cmd.append('init={}'.format(self.inits))
            else:
                cmd.append('init={}'.format(self.inits[idx]))
        cmd.append('output')
        cmd.append('file={}'.format(csv_file))
        if diagnostic_file is not None:
            cmd.append('diagnostic_file={}'.format(diagnostic_file))
        if self.refresh is not None:
            cmd.append('refresh={}'.format(self.refresh))
        if self.sig_figs is not None:
            cmd.append('sig_figs={}'.format(self.sig_figs))
        cmd = self.method_args.compose(idx, cmd)
        return cmd
