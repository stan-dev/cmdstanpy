"""Container objects for results of CmdStan run(s)."""

import os
import re
import shutil
import logging
from typing import List, Tuple
from collections import Counter, OrderedDict
from datetime import datetime
from time import time
import numpy as np
import pandas as pd

from cmdstanpy import _TMPDIR
from cmdstanpy.utils import (
    check_sampler_csv,
    scan_optimize_csv,
    scan_generated_quantities_csv,
    scan_variational_csv,
    create_named_text_file,
    EXTENSION,
    cmdstan_path,
    do_command,
    get_logger,
)
from cmdstanpy.cmdstan_args import Method, CmdStanArgs


class RunSet:
    """
    Record of CmdStan run for a specified configuration and number of chains.
    """

    def __init__(
        self, args: CmdStanArgs, chains: int = 4, logger: logging.Logger = None
    ) -> None:
        """Initialize object."""
        self._args = args
        self._chains = chains
        self._logger = logger or get_logger()
        if chains < 1:
            raise ValueError(
                'chains must be positive integer value, '
                'found {}'.format(chains)
            )

        self._retcodes = [-1 for _ in range(chains)]

        # stdout, stderr are written to text files
        # prefix: ``<model_name>-<YYYYMMDDHHMM>-<chain_id>``
        # suffixes: ``-stdout.txt``, ``-stderr.txt``
        now = datetime.now()
        now_str = now.strftime('%Y%m%d%H%M')
        file_basename = '-'.join([args.model_name, now_str])
        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            output_dir = _TMPDIR
        self._csv_files = [None for _ in range(chains)]
        self._diagnostic_files = [None for _ in range(chains)]
        self._stdout_files = [None for _ in range(chains)]
        self._stderr_files = [None for _ in range(chains)]
        self._cmds = []
        for i in range(chains):
            if args.output_dir is None:
                csv_file = create_named_text_file(
                    dir=output_dir,
                    prefix='{}-{}-'.format(file_basename, i + 1),
                    suffix='.csv',
                )
            else:
                csv_file = os.path.join(
                    output_dir, '{}-{}.{}'.format(file_basename, i + 1, 'csv')
                )
            self._csv_files[i] = csv_file
            stdout_file = ''.join(
                [os.path.splitext(csv_file)[0], '-stdout.txt']
            )
            self._stdout_files[i] = stdout_file
            stderr_file = ''.join(
                [os.path.splitext(csv_file)[0], '-stderr.txt']
            )
            self._stderr_files[i] = stderr_file
            if args.save_diagnostics:
                if args.output_dir is None:
                    diag_file = create_named_text_file(
                        dir=_TMPDIR,
                        prefix='{}-diagnostic-{}-'.format(file_basename, i + 1),
                        suffix='.csv',
                    )
                else:
                    diag_file = os.path.join(
                        output_dir,
                        '{}-diagnostic-{}.{}'.format(
                            file_basename, i + 1, 'csv'
                        ),
                    )
                self._diagnostic_files[i] = diag_file
                self._cmds.append(
                    args.compose_command(
                        i, self._csv_files[i], self._diagnostic_files[i]
                    )
                )
            else:
                self._cmds.append(args.compose_command(i, self._csv_files[i]))

    def __repr__(self) -> str:
        repr = 'RunSet: chains={}'.format(self._chains)
        repr = '{}\n cmd:\n\t{}'.format(repr, self._cmds[0])
        repr = '{}\n csv_files:\n\t{}\n output_files:\n\t{}'.format(
            repr,
            '\n\t'.join(self._csv_files),
            '\n\t'.join(self._stdout_files),
        )
        return repr

    @property
    def model(self) -> str:
        """Stan model name."""
        return self._args.model_name

    @property
    def method(self) -> Method:
        """Returns the CmdStan method used to generate this fit."""
        return self._args.method

    @property
    def chains(self) -> int:
        """Number of sampler chains."""
        return self._chains

    @property
    def cmds(self) -> List[str]:
        """Per-chain call to CmdStan."""
        return self._cmds

    @property
    def csv_files(self) -> List[str]:
        """
        List of paths to CmdStan output files.
        """
        return self._csv_files

    @property
    def stdout_files(self) -> List[str]:
        """
        List of paths to CmdStan stdout transcripts.
        """
        return self._stdout_files

    @property
    def stderr_files(self) -> List[str]:
        """
        List of paths to CmdStan stderr transcripts.
        """
        return self._stderr_files

    def _check_retcodes(self) -> bool:
        """True when all chains have retcode 0."""
        for i in range(self._chains):
            if self._retcodes[i] != 0:
                return False
        return True

    @property
    def diagnostic_files(self) -> List[str]:
        """
        List of paths to CmdStan diagnostic output files.
        """
        return self._diagnostic_files

    def _retcode(self, idx: int) -> int:
        """Get retcode for chain[idx]."""
        return self._retcodes[idx]

    def _set_retcode(self, idx: int, val: int) -> None:
        """Set retcode for chain[idx] to val."""
        self._retcodes[idx] = val

    def _get_err_msgs(self) -> List[str]:
        """Checks console messages for each chain."""
        msgs = []
        for i in range(self._chains):
            if (
                os.path.exists(self._stderr_files[i])
                and os.stat(self._stderr_files[i]).st_size > 0
            ):
                with open(self._stderr_files[i], 'r') as fd:
                    msgs.append('chain {}:\n{}\n'.format(i + 1, fd.read()))
            if (
                os.path.exists(self._stdout_files[i])
                and os.stat(self._stdout_files[i]).st_size > 0
            ):
                with open(self._stdout_files[i], 'r') as fd:
                    contents = fd.read()
                    pat = re.compile(r'^Exception.*$', re.M)
                errors = re.findall(pat, contents)
                if len(errors) > 0:
                    msgs.append('chain {}: {}\n'.format(i + 1, errors))
        return msgs

    def save_csvfiles(self, dir: str = None) -> None:
        """
        Moves csvfiles to specified directory.

        :param dir: directory path
        """
        if dir is None:
            dir = os.path.realpath('.')
        test_path = os.path.join(dir, str(time()))
        try:
            os.makedirs(dir, exist_ok=True)
            with open(test_path, 'w') as fd:
                pass
            os.remove(test_path)  # cleanup
        except OSError:
            raise Exception('cannot save to path: {}'.format(dir))

        for i in range(self.chains):
            if not os.path.exists(self._csv_files[i]):
                raise ValueError(
                    'cannot access csv file {}'.format(self._csv_files[i])
                )

            path, filename = os.path.split(self._csv_files[i])
            if path == _TMPDIR:  # cleanup tmpstr in filename
                root, ext = os.path.splitext(filename)
                rlist = root.split('-')
                root = '-'.join(rlist[:-1])
                filename = ''.join([root, ext])

            to_path = os.path.join(dir, filename)
            if os.path.exists(to_path):
                raise ValueError(
                    'file exists, not overwriting: {}'.format(to_path)
                )
            try:
                self._logger.debug(
                    'saving tmpfile: "%s" as: "%s"', self._csv_files[i], to_path
                )
                shutil.move(self._csv_files[i], to_path)
                self._csv_files[i] = to_path
            except (IOError, OSError, PermissionError) as e:
                raise ValueError(
                    'cannot save to file: {}'.format(to_path)
                ) from e


class CmdStanMCMC:
    """
    Container for outputs from CmdStan sampler run.
    """

    def __init__(self, runset: RunSet, is_fixed_param: bool = False) -> None:
        """Initialize object."""
        if not runset.method == Method.SAMPLE:
            raise ValueError(
                'Wrong runset method, expecting sample runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self._draws = None
        self._column_names = ()
        self._num_params = None  # metric dim(s)
        self._metric_type = None
        self._metric = None
        self._stepsize = None
        self._sample = None
        self._is_fixed_param = is_fixed_param

    def __repr__(self) -> str:
        repr = 'CmdStanMCMC: model={} chains={}{}'.format(
            self.runset.model,
            self.runset.chains,
            self.runset._args.method_args.compose(0, cmd=[]),
        )
        repr = '{}\n csv_files:\n\t{}\n output_files:\n\t{}'.format(
            repr,
            '\n\t'.join(self.runset.csv_files),
            '\n\t'.join(self.runset.stdout_files),
        )
        return repr

    @property
    def chains(self) -> int:
        """Number of chains."""
        return self.runset.chains

    @property
    def draws(self) -> int:
        """Number of draws per chain."""
        return self._draws

    @property
    def columns(self) -> int:
        """
        Total number of information items returned by sampler for each draw.
        Consists of sampler state, model parameters and computed quantities.
        """
        return len(self._column_names)

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of information items returned by sampler for each draw.
        Includes for sampler state labels and
        names of model parameters and computed quantities.
        """
        return self._column_names

    @property
    def metric_type(self) -> str:
        """
        Metric type used for adaptation, either 'diag_e' or 'dense_e'.
        When sampler algorithm 'fixed_param' is specified, metric_type is None.
        """
        return self._metric_type

    @property
    def metric(self) -> np.ndarray:
        """
        Metric used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, metric is None.
        """
        if not self._is_fixed_param and self._metric is None:
            self._assemble_sample()
        return self._metric

    @property
    def stepsize(self) -> np.ndarray:
        """
        Stepsize used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, stepsize is None.
        """
        if not self._is_fixed_param and self._stepsize is None:
            self._assemble_sample()
        return self._stepsize

    @property
    def sample(self) -> np.ndarray:
        """
        A 3-D numpy ndarray which contains all draws across all chains
        arranged as (draws, chains, columns) stored column major
        so that the values for each parameter are stored contiguously
        in memory, likewise all draws from a chain are contiguous.
        """
        if self._sample is None:
            self._assemble_sample()
        return self._sample

    def _validate_csv_files(self) -> None:
        """
        Checks that csv output files for all chains are consistent.
        Populates attributes for draws, column_names, num_params, metric_type.
        Raises exception when inconsistencies detected.
        """
        dzero = {}
        for i in range(self.runset.chains):
            if i == 0:
                dzero = check_sampler_csv(
                    self.runset.csv_files[i], self._is_fixed_param
                )
            else:
                drest = check_sampler_csv(
                    self.runset.csv_files[i], self._is_fixed_param
                )
                for key in dzero:
                    if (
                        key not in ['id', 'diagnostic_file']
                        and dzero[key] != drest[key]
                    ):
                        raise ValueError(
                            'csv file header mismatch, '
                            'file {}, key {} is {}, expected {}'.format(
                                self.runset.csv_files[i],
                                key,
                                dzero[key],
                                drest[key],
                            )
                        )
        self._draws = dzero['draws']
        self._column_names = dzero['column_names']
        if not self._is_fixed_param:
            self._num_params = dzero['num_params']
            self._metric_type = dzero.get('metric')

    def _assemble_sample(self) -> None:
        """
        Allocates and populates the stepsize, metric, and sample arrays
        by parsing the validated stan_csv files.
        """
        if self._sample is not None:
            return
        self._sample = np.empty(
            (self._draws, self.runset.chains, len(self._column_names)),
            dtype=float,
            order='F',
        )
        if not self._is_fixed_param:
            self._stepsize = np.empty(self.runset.chains, dtype=float)
            if self._metric_type == 'diag_e':
                self._metric = np.empty(
                    (self.runset.chains, self._num_params), dtype=float
                )
            else:
                self._metric = np.empty(
                    (self.runset.chains, self._num_params, self._num_params),
                    dtype=float,
                )
        for chain in range(self.runset.chains):
            with open(self.runset.csv_files[chain], 'r') as fd:
                # skip initial comments, reads thru column header
                line = fd.readline().strip()
                while len(line) > 0 and line.startswith('#'):
                    line = fd.readline().strip()
                if not self._is_fixed_param:
                    # skip warmup draws, if any, read to adaptation msg
                    line = fd.readline().strip()
                    if line != '# Adaptation terminated':
                        while line != '# Adaptation terminated':
                            line = fd.readline().strip()
                    line = fd.readline().strip()  # stepsize
                    _, stepsize = line.split('=')
                    self._stepsize[chain] = float(stepsize.strip())
                    line = fd.readline().strip()  # metric header
                    # process metric
                    if self._metric_type == 'diag_e':
                        line = fd.readline().lstrip(' #\t')
                        xs = line.split(',')
                        self._metric[chain, :] = [float(x) for x in xs]
                    else:
                        for i in range(self._num_params):
                            line = fd.readline().lstrip(' #\t')
                            xs = line.split(',')
                            self._metric[chain, i, :] = [float(x) for x in xs]
                # process draws
                for i in range(self._draws):
                    line = fd.readline().lstrip(' #\t')
                    xs = line.split(',')
                    self._sample[i, chain, :] = [float(x) for x in xs]

    def summary(self) -> pd.DataFrame:
        """
        Run cmdstan/bin/stansummary over all output csv files.
        Echo stansummary stdout/stderr to console.
        Assemble csv tempfile contents into pandasDataFrame.
        """
        cmd_path = os.path.join(
            cmdstan_path(), 'bin', 'stansummary' + EXTENSION
        )
        tmp_csv_file = 'stansummary-{}-{}-chain-'.format(
            self.runset._args.model_name, self.runset.chains
        )
        tmp_csv_path = create_named_text_file(
            dir=_TMPDIR, prefix=tmp_csv_file, suffix='.csv'
        )
        cmd = [
            cmd_path,
            '--csv_file={}'.format(tmp_csv_path),
        ] + self.runset.csv_files
        do_command(cmd, logger=self.runset._logger)
        with open(tmp_csv_path, 'rb') as fd:
            summary_data = pd.read_csv(
                fd, delimiter=',', header=0, index_col=0, comment='#'
            )
        mask = [x == 'lp__' or not x.endswith('__') for x in summary_data.index]
        return summary_data[mask]

    def diagnose(self) -> str:
        """
        Run cmdstan/bin/diagnose over all output csv files.
        Returns output of diagnose (stdout/stderr).

        The diagnose utility reads the outputs of all chains
        and checks for the following potential problems:

        + Transitions that hit the maximum treedepth
        + Divergent transitions
        + Low E-BFMI values (sampler transitions HMC potential energy)
        + Low effective sample sizes
        + High R-hat values
        """
        cmd_path = os.path.join(cmdstan_path(), 'bin', 'diagnose' + EXTENSION)
        cmd = [cmd_path] + self.runset.csv_files
        result = do_command(cmd=cmd, logger=self.runset._logger)
        if result:
            self.runset._logger.info(result)
        return result

    def get_drawset(self, params: List[str] = None) -> pd.DataFrame:
        """
        Returns the assembled sample as a pandas DataFrame consisting of
        one column per parameter and one row per draw.

        :param params: list of model parameter names.
        """
        pnames_base = [name.split('.')[0] for name in self.column_names]
        if params is not None:
            for param in params:
                if not (param in self._column_names or param in pnames_base):
                    raise ValueError('unknown parameter: {}'.format(param))
        self._assemble_sample()
        # pylint: disable=redundant-keyword-arg
        data = self.sample.reshape(
            (self.draws * self.runset.chains), len(self.column_names), order='A'
        )
        drawset = pd.DataFrame(data=data, columns=self.column_names)
        if params is None:
            return drawset
        mask = []
        for param in params:
            for name in self.column_names:
                if param == name or param == name.split('.')[0]:
                    mask.append(name)
        return drawset[mask]

    def save_csvfiles(self, dir: str = None) -> None:
        """
        Moves csvfiles to specified directory using specified basename,
        appending suffix '-<id>.csv' to each.

        :param dir: directory path
        """
        self.runset.save_csvfiles(dir)


class CmdStanMLE:
    """
    Container for outputs from CmdStan optimization.
    """

    def __init__(self, runset: RunSet) -> None:
        """Initialize object."""
        if not runset.method == Method.OPTIMIZE:
            raise ValueError(
                'Wrong runset method, expecting optimize runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self._column_names = ()
        self._mle = {}

    def __repr__(self) -> str:
        repr = 'CmdStanMLE: model={}{}'.format(
            self.runset.model, self.runset._args.method_args.compose(0, cmd=[])
        )
        repr = '{}\n csv_file:\n\t{}\n output_file:\n\t{}'.format(
            repr,
            '\n\t'.join(self.runset.csv_files),
            '\n\t'.join(self.runset.stdout_files),
        )
        return repr

    def _set_mle_attrs(self, sample_csv_0: str) -> None:
        meta = scan_optimize_csv(sample_csv_0)
        self._column_names = meta['column_names']
        self._mle = meta['mle']

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of estimated quantities, includes joint log probability,
        and all parameters, transformed parameters, and generated quantitites.
        """
        return self._column_names

    @property
    def optimized_params_np(self) -> np.array:
        """Returns optimized params as numpy array."""
        if self._mle is None:
            self._set_mle_attrs(self.runset.csv_files[0])
        return np.asarray(self._mle)

    @property
    def optimized_params_pd(self) -> pd.DataFrame:
        """Returns optimized params as pandas DataFrame."""
        if self._mle is None:
            self._set_mle_attrs(self.runset.csv_files[0])
        return pd.DataFrame([self._mle], columns=self.column_names)

    @property
    def optimized_params_dict(self) -> OrderedDict:
        """Returns optimized params as Dict."""
        if self._mle is None:
            self._set_mle_attrs(self.runset.csv_files[0])
        return OrderedDict(zip(self.column_names, self._mle))

    def save_csvfiles(self, dir: str = None) -> None:
        """
        Moves csvfiles to specified directory using specified basename,
        appending suffix '-<id>.csv' to each.

        :param dir: directory path
        """
        self.runset.save_csvfiles(dir)


class CmdStanGQ:
    """
    Container for outputs from CmdStan generate_quantities run.
    """

    def __init__(self, runset: RunSet, mcmc_sample: pd.DataFrame) -> None:
        """Initialize object."""
        if not runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError(
                'Wrong runset method, expecting generate_quantities runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self.mcmc_sample = mcmc_sample
        self._column_names = None
        self._generated_quantities = None

    def __repr__(self) -> str:
        repr = 'CmdStanGQ: model={} chains={}{}'.format(
            self.runset.model,
            self.runset.chains,
            self.runset._args.method_args.compose(0, cmd=[]),
        )
        repr = '{}\n csv_files:\n\t{}\n output_files:\n\t{}'.format(
            repr,
            '\n\t'.join(self.runset.csv_files),
            '\n\t'.join(self.runset.stdout_files),
        )
        return repr

    @property
    def chains(self) -> int:
        """Number of chains."""
        return self.runset.chains

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of generated quantities of interest.
        """
        return self._column_names

    @property
    def generated_quantities(self) -> np.ndarray:
        """
        A 2-D numpy ndarray which contains generated quantities draws
        for all chains where the columns correspond to the generated quantities
        block variables and the rows correspond to the draws from all chains,
        where first M draws are the first M draws of chain 1 and the
        last M draws are the last M draws of chain N, i.e.,
        flattened chain, draw ordering.
        """
        if not self.runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError('Bad runset method {}.'.format(self.runset.method))
        if self._generated_quantities is None:
            self._assemble_generated_quantities()
        return self._generated_quantities

    @property
    def generated_quantities_pd(self) -> pd.DataFrame:
        """
        Returns the generated quantities as a pandas DataFrame consisting of
        one column per quantity of interest and one row per draw.
        """
        if not self.runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError('Bad runset method {}.'.format(self.runset.method))
        if self._generated_quantities is None:
            self._assemble_generated_quantities()
        return pd.DataFrame(
            data=self._generated_quantities, columns=self.column_names
        )

    @property
    def sample_plus_quantities(self) -> pd.DataFrame:
        """
        Returns the column-wise concatenation of the input drawset
        with generated quantities drawset.  If there are duplicate
        columns in both the input and the generated quantities,
        the input column is dropped in favor of the recomputed
        values in the generate quantities drawset.
        """
        if not self.runset.method == Method.GENERATE_QUANTITIES:
            raise ValueError('Bad runset method {}.'.format(self.runset.method))
        if self._generated_quantities is None:
            self._assemble_generated_quantities()

        cols_1 = self.mcmc_sample.columns.tolist()
        cols_2 = self.generated_quantities_pd.columns.tolist()

        dups = [
            item
            for item, count in Counter(cols_1 + cols_2).items()
            if count > 1
        ]
        return pd.concat(
            [self.mcmc_sample.drop(columns=dups), self.generated_quantities_pd],
            axis=1,
        )

    def _set_attrs_gq_csv_files(self, sample_csv_0: str) -> None:
        """
        Propogate information from original sample to additional sample
        returned by generate_quantities.
        """
        check_sampler_csv(sample_csv_0)
        dzero = scan_generated_quantities_csv(self.runset.csv_files[0])
        self._column_names = dzero['column_names']

    def _assemble_generated_quantities(self) -> None:
        drawset_list = []
        for chain in range(self.runset.chains):
            drawset_list.append(
                pd.read_csv(self.runset.csv_files[chain], comment='#')
            )
        self._generated_quantities = pd.concat(drawset_list).values

    def save_csvfiles(self, dir: str = None) -> None:
        """
        Moves csvfiles to specified directory using specified basename,
        appending suffix '-<id>.csv' to each.

        :param dir: directory path
        """
        self.runset.save_csvfiles(dir)


class CmdStanVB:
    """
    Container for outputs from CmdStan variational run.
    """

    def __init__(self, runset: RunSet) -> None:
        """Initialize object."""
        if not runset.method == Method.VARIATIONAL:
            raise ValueError(
                'Wrong runset method, expecting variational inference, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self._column_names = ()
        self._variational_mean = {}
        self._variational_sample = None

    def __repr__(self) -> str:
        repr = 'CmdStanVB: model={}{}'.format(
            self.runset.model, self.runset._args.method_args.compose(0, cmd=[])
        )
        repr = '{}\n csv_file:\n\t{}\n output_file:\n\t{}'.format(
            repr,
            '\n\t'.join(self.runset.csv_files),
            '\n\t'.join(self.runset.stdout_files),
        )
        return repr

    def _set_variational_attrs(self, sample_csv_0: str) -> None:
        meta = scan_variational_csv(sample_csv_0)
        self._column_names = meta['column_names']
        self._variational_mean = meta['variational_mean']
        self._variational_sample = meta['variational_sample']

    @property
    def columns(self) -> int:
        """
        Total number of information items returned by sampler.
        Includes approximation information and names of model parameters
        and computed quantities.
        """
        return len(self._column_names)

    @property
    def column_names(self) -> Tuple[str, ...]:
        """
        Names of information items returned by sampler for each draw.
        Includes approximation information and names of model parameters
        and computed quantities.
        """
        return self._column_names

    @property
    def variational_params_np(self) -> np.array:
        """Returns inferred parameter means as numpy array."""
        if self._variational_mean is None:
            self._set_variational_attrs(self.runset.csv_files[0])
        return self._variational_mean

    @property
    def variational_params_pd(self) -> pd.DataFrame:
        """Returns inferred parameter means as pandas DataFrame."""
        if self._variational_mean is None:
            self._set_variational_attrs(self.runset.csv_files[0])
        return pd.DataFrame([self._variational_mean], columns=self.column_names)

    @property
    def variational_params_dict(self) -> OrderedDict:
        """Returns inferred parameter means as Dict."""
        if self._variational_mean is None:
            self._set_variational_attrs(self.runset.csv_files[0])
        return OrderedDict(zip(self.column_names, self._variational_mean))

    @property
    def variational_sample(self) -> np.array:
        """Returns the set of approximate posterior output draws."""
        if self._variational_sample is None:
            self._set_variational_attrs(self.runset.csv_files[0])
        return self._variational_sample

    def save_csvfiles(self, dir: str = None) -> None:
        """
        Moves csvfiles to specified directory using specified basename,
        appending suffix '-<id>.csv' to each.

        :param dir: directory path
        """
        self.runset.save_csvfiles(dir)
