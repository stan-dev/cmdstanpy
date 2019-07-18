"""
CmdStan arguments
"""
import os
import numpy as np

from numbers import Integral, Real
from typing import List, Union
from cmdstanpy.utils import read_metric


class SamplerArgs(object):
    """Arguments for the NUTS adaptive sampler."""

    def __init__(
        self,
        warmup_iters: int = None,
        sampling_iters: int = None,
        save_warmup: bool = False,
        thin: int = None,
        max_treedepth: int = None,
        metric: Union[str, List[str]] = None,
        step_size: Union[float, List[float]] = None,
        adapt_engaged: bool = None,
        adapt_delta: float = None,
    ) -> None:
        """Initialize object."""
        self.warmup_iters = warmup_iters
        self.sampling_iters = sampling_iters
        self.save_warmup = save_warmup
        self.thin = thin
        self.max_treedepth = max_treedepth
        self.metric = metric
        self.metric_file = None
        self.step_size = step_size
        self.adapt_engaged = adapt_engaged
        self.adapt_delta = adapt_delta

    def validate(self, chains: int) -> None:
        """
        Check arguments correctness and consistency.

        * adaptation and warmup args are consistent
        * if file(s) for metric are supplied, check contents.
        * length of per-chain lists equals specified # of chains
        """
        if not isinstance(chains, Integral) or chains < 1:
            raise ValueError(
                "sampler expects number of chains to be greater than 0"
            )

        if self.warmup_iters is not None:
            if self.warmup_iters < 0:
                raise ValueError(
                    'warmup_iters must be a non-negative integer'.format(
                        self.warmup_iters
                    )
                )
            if self.adapt_engaged and self.warmup_iters == 0:
                raise ValueError(
                    'adaptation requested but 0 warmup iterations specified, '
                    'must run warmup iterations'
                )

        if self.sampling_iters is not None:
            if self.sampling_iters < 0:
                raise ValueError(
                    'sampling_iters must be a non-negative integer'.format(
                        self.sampling_iters
                    )
                )

        if self.thin is not None:
            if self.thin < 1:
                raise ValueError(
                    'thin must be at least 1, found {}'.format(self.thin)
                )

        if self.max_treedepth is not None:
            if self.max_treedepth < 1:
                raise ValueError(
                    'max_treedepth must be at least 1, found {}'.format(
                        self.max_treedepth
                    )
                )

        if self.step_size is not None:
            if isinstance(self.step_size, Real):
                if self.step_size < 0:
                    raise ValueError(
                        'step_size must be > 0, found {}'.format(self.step_size)
                    )
            else:
                if len(self.step_size) != chains:
                    raise ValueError(
                        'number of step_sizes must match number of chains '
                        ' found {} step_sizes for {} chains '.format(
                            len(self.step_size), chains
                        )
                    )
                for step_size in self.step_size:
                    if step_size < 0:
                        raise ValueError(
                            'step_size must be > 0, found {}'.format(step_size)
                        )

        if self.metric is not None:
            dims = None
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
                        'number of metric files must match number of chains '
                        ' found {} metric files for {} chains '.format(
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
                        raise ValueError(
                            'no such file {}'.format(metric)
                        )
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
                        for j, dim in enumerate(dims):
                            if dims[j] != dims2[j]:
                                raise ValueError(
                                    'metrics files {}, {},'
                                    ' inconsistent metrics'.format(
                                        self.metric[0], metric
                                    )
                                )
            if dims is not None:
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

    def compose(self, idx: int, cmd: str) -> str:
        """
        Compose CmdStan command for method-specific non-default arguments.
        """
        cmd = cmd + ' method=sample'
        if self.sampling_iters is not None:
            cmd = '{} num_samples={}'.format(cmd, self.sampling_iters)
        if self.warmup_iters is not None:
            cmd = '{} num_warmup={}'.format(cmd, self.warmup_iters)
        if self.save_warmup:
            cmd = cmd + ' save_warmup=1'
        if self.thin is not None:
            cmd = '{} thin={}'.format(cmd, self.thin)
        cmd = cmd + ' algorithm=hmc'
        if self.max_treedepth is not None:
            cmd = '{} engine=nuts max_depth={}'.format(cmd, self.max_treedepth)
        if self.step_size is not None:
            if not isinstance(self.step_size, list):
                cmd = '{} stepsize={}'.format(cmd, self.step_size)
            else:
                cmd = '{} stepsize={}'.format(cmd, self.step_size[idx])
        if self.metric is not None:
            cmd = '{} metric={}'.format(cmd, self.metric)
        if self.metric_file is not None:
            if not isinstance(self.metric_file, list):
                cmd = '{} metric_file="{}"'.format(cmd, self.metric_file)
            else:
                cmd = '{} metric_file="{}"'.format(cmd, self.metric_file[idx])
        if self.adapt_engaged is not None or self.adapt_delta is not None:
            cmd = cmd + ' adapt'
        if self.adapt_engaged is not None:
            if self.adapt_engaged:
                cmd = cmd + ' engaged=1'
            else:
                cmd = cmd + ' engaged=0'
        if self.adapt_delta is not None:
            cmd = '{} delta={}'.format(cmd, self.adapt_delta)
        return cmd


class FixedParamArgs(object):
    """Arguments for the NUTS adaptive sampler."""

    def compose(self, idx: int, cmd: str) -> str:
        cmd = cmd + ' method=fixed_param'
        return cmd

    def validate(self, chains):
        pass


class OptimizeArgs(object):
    """Container for arguments for the optimizer."""

    OPTIMIZE_ALGOS = {"BFGS", "LBFGS", "Newton"}

    def __init__(
            self,
            algorithm: str = None,
            init_alpha: Real = None,
            iter: int = None
            ) -> None:

        self.algorithm = algorithm
        self.init_alpha = init_alpha
        self.iter = iter

    def validate(self, chains=None) -> None:
        """
        Check arguments correctness and consistency.
        """
        if self.algorithm is not None and \
                self.algorithm not in self.OPTIMIZE_ALGOS:
            raise ValueError(
                "Please specify optimizer algorithms as one of [{}]"
                .format(", ".join(self.OPTIMIZE_ALGOS))
            )

        if self.init_alpha is not None:
            if self.algorithm == "Newton":
                raise ValueError(
                    "init_alpha must not be set when algorithm is Newton"
                )
            if isinstance(self.init_alpha, Real):
                if self.init_alpha < 0:
                    raise ValueError("init_alpha must be greater than 0")
            else:
                raise ValueError("init_alpha must be type of float")

        if self.iter is not None:
            if isinstance(self.iter, Integral):
                if self.iter < 0:
                    raise ValueError("iter must be greater than 0")
            else:
                raise ValueError("iter must be type of int")

    def compose(self, idx: int, cmd: str) -> str:
        """compose command string for CmdStan for non-default arg values.
        """

        cmd = cmd + ' method=optimize'
        if self.algorithm:
            cmd += ' algorithm={}'.format(self.algorithm.lower())
        if self.init_alpha is not None:
            cmd += ' init_alpha={}'.format(self.init_alpha)
        if self.iter is not None:
            cmd += ' iter={}'.format(self.iter)
        return cmd


class CmdStanArgs(object):
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
        method_args: Union[SamplerArgs, FixedParamArgs, OptimizeArgs],
        data: Union[str, dict] = None,
        seed: Union[int, List[int]] = None,
        inits: Union[float, str, List[str]] = None,
        output_basename: str = None,
    ) -> None:
        """Initialize object."""
        self.model_name = model_name
        self.model_exe = model_exe
        self.chain_ids = chain_ids
        self.method_args = method_args
        self.method_args.validate(len(chain_ids) if chain_ids else None)
        self.data = data
        self.seed = seed
        self.inits = inits
        self.output_basename = output_basename
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

        if self.output_basename is not None:
            if not os.path.exists(os.path.dirname(self.output_basename)):
                raise ValueError(
                    'invalid path for output files: {}'.format(
                        self.output_basename)
                )
            try:
                with open(self.output_basename, 'w+') as fd:
                    pass
                os.remove(self.output_basename)  # cleanup
            except Exception:
                raise ValueError(
                    'invalid path for output files: {}'.format(
                        self.output_basename)
                )
            self.output_basename, _ = os.path.splitext(self.output_basename)

        if self.seed is None:
            rng = np.random.RandomState()
            self.seed = rng.randint(1, 99999 + 1)
        else:
            if not isinstance(self.seed, (int, list)):
                raise ValueError(
                    'seed must be an integer between 0 and 2**32-1,'
                    ' found {}'.format(self.seed)
                )
            elif isinstance(self.seed, int):
                if self.seed < 0 or self.seed > 2 ** 32 - 1:
                    raise ValueError(
                        'seed must be an integer between 0 and 2**32-1,'
                        ' found {}'.format(self.seed)
                    )
            else:
                if self.chain_ids is None:
                    raise ValueError(
                        "seed must not be a list when no chains used"
                    )

                if len(self.seed) != len(self.chain_ids):
                    raise ValueError(
                        'number of seeds must match number of chains '
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
        elif not isinstance(self.data, dict):
            raise ValueError('data must be string or dict')

        if self.inits is not None:
            if isinstance(self.inits, Real):
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
                        "inits must not be a list when no chains are used"
                    )

                if len(self.inits) != len(self.chain_ids):
                    raise ValueError(
                        'number of inits files must match number of chains '
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

    def compose_command(self, idx: int, csv_file: str) -> str:
        """
        Compose CmdStan command for non-default arguments.
        """
        if idx is not None and self.chain_ids is not None:
            if idx < 0 or idx > len(self.chain_ids) - 1:
                raise ValueError(
                    'index ({}) exceeds number of chains ({})'.format(
                        idx, len(self.chain_ids))
                )
            cmd = '{} id={}'.format(self.model_exe, self.chain_ids[idx])
        else:
            cmd = self.model_exe

        if self.seed is not None:
            if not isinstance(self.seed, list):
                cmd = '{} random seed={}'.format(cmd, self.seed)
            else:
                cmd = '{} random seed={}'.format(cmd, self.seed[idx])
        if self.data is not None:
            cmd = '{} data file={}'.format(cmd, self.data)
        if self.inits is not None:
            if not isinstance(self.inits, list):
                cmd = '{} init={}'.format(cmd, self.inits)
            else:
                cmd = '{} init={}'.format(cmd, self.inits[idx])
        cmd = '{} output file={}'.format(cmd, csv_file)
        cmd = self.method_args.compose(idx, cmd)
        return cmd
