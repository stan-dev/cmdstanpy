"""Arguments for the sample subcommand"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

from cmdstanpy import _TMPDIR
from cmdstanpy.utils import create_named_text_file, read_metric, write_stan_json


class SampleArgs:
    """Arguments for the NUTS adaptive sampler."""

    def __init__(
        self,
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
    ) -> None:
        """Initialize object."""
        self.iter_warmup = iter_warmup
        self.iter_sampling = iter_sampling
        self.save_warmup = save_warmup
        self.thin = thin
        self.max_treedepth = max_treedepth
        self.metric = metric
        self.metric_type: Optional[str] = None
        self.metric_file: Union[str, List[str], None] = None
        self.step_size = step_size
        self.adapt_engaged = adapt_engaged
        self.adapt_delta = adapt_delta
        self.adapt_init_phase = adapt_init_phase
        self.adapt_metric_window = adapt_metric_window
        self.adapt_step_size = adapt_step_size
        self.fixed_param = fixed_param
        self.diagnostic_file = None

    def validate(self, chains: Optional[int]) -> None:
        """
        Check arguments correctness and consistency.

        * adaptation and warmup args are consistent
        * if file(s) for metric are supplied, check contents.
        * length of per-chain lists equals specified # of chains
        """
        if not isinstance(chains, int) or chains < 1:
            raise ValueError(
                'Sampler expects number of chains to be greater than 0.'
            )
        if not (
            self.adapt_delta is None
            and self.adapt_init_phase is None
            and self.adapt_metric_window is None
            and self.adapt_step_size is None
        ):
            if self.adapt_engaged is False:
                msg = 'Conflicting arguments: adapt_engaged: False'
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
            if self.iter_warmup < 0 or not isinstance(self.iter_warmup, int):
                raise ValueError(
                    'Value for iter_warmup must be a non-negative integer,'
                    ' found {}.'.format(self.iter_warmup)
                )
            if self.iter_warmup > 0 and not self.adapt_engaged:
                raise ValueError(
                    'Argument "adapt_engaged" is False, '
                    'cannot specify warmup iterations.'
                )
        if self.iter_sampling is not None:
            if self.iter_sampling < 0 or not isinstance(
                self.iter_sampling, int
            ):
                raise ValueError(
                    'Argument "iter_sampling" must be a non-negative integer,'
                    ' found {}.'.format(self.iter_sampling)
                )
        if self.thin is not None:
            if self.thin < 1 or not isinstance(self.thin, int):
                raise ValueError(
                    'Argument "thin" must be a positive integer,'
                    'found {}.'.format(self.thin)
                )
        if self.max_treedepth is not None:
            if self.max_treedepth < 1 or not isinstance(
                self.max_treedepth, int
            ):
                raise ValueError(
                    'Argument "max_treedepth" must be a positive integer,'
                    ' found {}.'.format(self.max_treedepth)
                )
        if self.step_size is not None:
            if isinstance(self.step_size, (float, int)):
                if self.step_size <= 0:
                    raise ValueError(
                        'Argument "step_size" must be > 0, '
                        'found {}.'.format(self.step_size)
                    )
            else:
                if len(self.step_size) != chains:
                    raise ValueError(
                        'Expecting {} per-chain step_size specifications, '
                        ' found {}.'.format(chains, len(self.step_size))
                    )
                for i, step_size in enumerate(self.step_size):
                    if step_size < 0:
                        raise ValueError(
                            'Argument "step_size" must be > 0, '
                            'chain {}, found {}.'.format(i + 1, step_size)
                        )
        if self.metric is not None:
            if isinstance(self.metric, str):
                if self.metric in ['diag', 'diag_e']:
                    self.metric_type = 'diag_e'
                elif self.metric in ['dense', 'dense_e']:
                    self.metric_type = 'dense_e'
                elif self.metric in ['unit', 'unit_e']:
                    self.metric_type = 'unit_e'
                else:
                    if not os.path.exists(self.metric):
                        raise ValueError('no such file {}'.format(self.metric))
                    dims = read_metric(self.metric)
                    if len(dims) == 1:
                        self.metric_type = 'diag_e'
                    else:
                        self.metric_type = 'dense_e'
                    self.metric_file = self.metric
            elif isinstance(self.metric, Dict):
                if 'inv_metric' not in self.metric:
                    raise ValueError(
                        'Entry "inv_metric" not found in metric dict.'
                    )
                dims = list(np.asarray(self.metric['inv_metric']).shape)
                if len(dims) == 1:
                    self.metric_type = 'diag_e'
                else:
                    self.metric_type = 'dense_e'
                dict_file = create_named_text_file(
                    dir=_TMPDIR, prefix="metric", suffix=".json"
                )
                write_stan_json(dict_file, self.metric)
                self.metric_file = dict_file
            elif isinstance(self.metric, (list, tuple)):
                if len(self.metric) != chains:
                    raise ValueError(
                        'Number of metric files must match number of chains,'
                        ' found {} metric files for {} chains.'.format(
                            len(self.metric), chains
                        )
                    )
                if all(isinstance(elem, dict) for elem in self.metric):
                    metric_files: List[str] = []
                    for i, metric in enumerate(self.metric):
                        assert isinstance(
                            metric, dict
                        )  # make the typechecker happy
                        metric_dict: Dict[str, Any] = metric
                        if 'inv_metric' not in metric_dict:
                            raise ValueError(
                                'Entry "inv_metric" not found in metric dict '
                                'for chain {}.'.format(i + 1)
                            )
                        if i == 0:
                            dims = list(
                                np.asarray(metric_dict['inv_metric']).shape
                            )
                        else:
                            dims2 = list(
                                np.asarray(metric_dict['inv_metric']).shape
                            )
                            if dims != dims2:
                                raise ValueError(
                                    'Found inconsistent "inv_metric" entry '
                                    'for chain {}: entry has dims '
                                    '{}, expected {}.'.format(
                                        i + 1, dims, dims2
                                    )
                                )
                        dict_file = create_named_text_file(
                            dir=_TMPDIR, prefix="metric", suffix=".json"
                        )
                        write_stan_json(dict_file, metric_dict)
                        metric_files.append(dict_file)
                    if len(dims) == 1:
                        self.metric_type = 'diag_e'
                    else:
                        self.metric_type = 'dense_e'
                    self.metric_file = metric_files
                elif all(isinstance(elem, str) for elem in self.metric):
                    metric_files = []
                    for i, metric in enumerate(self.metric):
                        assert isinstance(metric, str)  # typecheck
                        if not os.path.exists(metric):
                            raise ValueError('no such file {}'.format(metric))
                        if i == 0:
                            dims = read_metric(metric)
                        else:
                            dims2 = read_metric(metric)
                            if len(dims) != len(dims2):
                                raise ValueError(
                                    'Metrics files {}, {},'
                                    ' inconsistent metrics'.format(
                                        self.metric[0], metric
                                    )
                                )
                            if dims != dims2:
                                raise ValueError(
                                    'Metrics files {}, {},'
                                    ' inconsistent metrics'.format(
                                        self.metric[0], metric
                                    )
                                )
                        metric_files.append(metric)
                    if len(dims) == 1:
                        self.metric_type = 'diag_e'
                    else:
                        self.metric_type = 'dense_e'
                    self.metric_file = metric_files
                else:
                    raise ValueError(
                        'Argument "metric" must be a list of pathnames or '
                        'Python dicts, found list of {}.'.format(
                            type(self.metric[0])
                        )
                    )
            else:
                raise ValueError(
                    'Invalid metric specified, not a recognized metric type, '
                    'must be either a metric type name, a filepath, dict, '
                    'or list of per-chain filepaths or dicts.  Found '
                    'an object of type {}.'.format(type(self.metric))
                )

        if self.adapt_delta is not None:
            if not 0 < self.adapt_delta < 1:
                raise ValueError(
                    'Argument "adapt_delta" must be between 0 and 1,'
                    ' found {}'.format(self.adapt_delta)
                )
        if self.adapt_init_phase is not None:
            if self.adapt_init_phase < 0 or not isinstance(
                self.adapt_init_phase, int
            ):
                raise ValueError(
                    'Argument "adapt_init_phase" must be a non-negative '
                    'integer, found {}'.format(self.adapt_init_phase)
                )
        if self.adapt_metric_window is not None:
            if self.adapt_metric_window < 0 or not isinstance(
                self.adapt_metric_window, int
            ):
                raise ValueError(
                    'Argument "adapt_metric_window" must be a non-negative '
                    ' integer, found {}'.format(self.adapt_metric_window)
                )
        if self.adapt_step_size is not None:
            if self.adapt_step_size < 0 or not isinstance(
                self.adapt_step_size, int
            ):
                raise ValueError(
                    'Argument "adapt_step_size" must be a non-negative integer,'
                    'found {}'.format(self.adapt_step_size)
                )

        if self.fixed_param and (
            self.max_treedepth is not None
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
                'When fixed_param=True, cannot specify adaptation parameters.'
            )

    def compose(self, idx: int, cmd: List[str]) -> List[str]:
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
            cmd.append('metric={}'.format(self.metric_type))
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
