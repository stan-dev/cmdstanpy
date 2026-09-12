"""
Container for the result of running the sample (MCMC) method
"""

from __future__ import annotations

import math
import os
from collections.abc import Hashable, MutableMapping, Sequence
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, ClassVar

import numpy as np
import pandas as pd

try:
    import xarray as xr

    XARRAY_INSTALLED = True
except ImportError:
    XARRAY_INSTALLED = False

from cmdstanpy import _TMPDIR
from cmdstanpy.utils import (
    EXTENSION,
    build_xarray_data,
    cmdstan_path,
    create_named_text_file,
    do_command,
    flatten_chains,
    get_logger,
    stancsv,
)

from .base import MultiChainFit
from .metadata import MetricInfo, SampleConfig, SampleRunConfig, StanConfig


@dataclass(kw_only=True)
class CmdStanMCMC(MultiChainFit[SampleConfig]):
    """
    Container for outputs from CmdStan sampler run.
    Provides methods to summarize and diagnose the model fit
    and accessor methods to access the entire sample or
    individual items. Created by :meth:`CmdStanModel.sample`.

    The sample is lazily instantiated on first access of either
    the resulting sample or the HMC tuning parameters, i.e., the
    step size and metric.
    """

    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    metric_files: list[str] | None = None
    diagnostic_files: list[str] | None = None
    profile_files: list[str] | None = None
    sig_figs: int | None = None

    _FILE_ATTRS: ClassVar[tuple[str, ...]] = (
        'csv_files',
        'config_files',
        'metric_files',
        'stdout_files',
        'diagnostic_files',
        'profile_files',
    )

    _draws: np.ndarray = field(default_factory=lambda: np.array(()), init=False)
    _metric_type: str | None = field(default=None, init=False)
    _metric: np.ndarray = field(
        default_factory=lambda: np.array(()), init=False
    )
    _step_size: np.ndarray = field(
        default_factory=lambda: np.array(()), init=False
    )
    _divergences: np.ndarray = field(
        default_factory=lambda: np.array(()), init=False
    )
    _max_treedepths: np.ndarray = field(
        default_factory=lambda: np.array(()), init=False
    )

    def __post_init__(self) -> None:
        self._divergences = np.zeros(self.chains, dtype=int)
        self._max_treedepths = np.zeros(self.chains, dtype=int)
        super().__post_init__()
        self._validate_csv_files()
        if not self._is_fixed_param:
            self._check_sampler_diagnostics()

    @classmethod
    def from_files(
        cls,
        csv_files: Sequence[str | os.PathLike],
        config_files: Sequence[str | os.PathLike] | str | os.PathLike,
        metric_files: Sequence[str | os.PathLike] | None = None,
        stdout_files: Sequence[str | os.PathLike] | None = None,
        diagnostic_files: Sequence[str | os.PathLike] | None = None,
        profile_files: Sequence[str | os.PathLike] | None = None,
        chain_ids: Sequence[int] | None = None,
        sig_figs: int | None = None,
    ) -> CmdStanMCMC:
        """Build a CmdStanMCMC from output files.

        ``config_files`` may be a single path (when CmdStan ran multiple chains
        in one process) or a per-chain list.
        """
        kwargs = cls._from_files_kwargs(
            csv_files, config_files, stdout_files, chain_ids, SampleRunConfig
        )
        stan_config = kwargs['config']
        chains = len(kwargs['csv_files'])

        if sig_figs is None and stan_config.output.sig_figs > 0:
            # A value of -1 is CmdStan's default (six digits).
            sig_figs = stan_config.output.sig_figs

        def _maybe_list(
            files: Sequence[str | os.PathLike] | None,
        ) -> list[str] | None:
            return [os.fspath(f) for f in files] if files is not None else None

        metric_files_list = _maybe_list(metric_files)
        if metric_files_list is not None and len(metric_files_list) != chains:
            raise ValueError(
                f'Expected one metric file per chain ({chains}), '
                f'got {len(metric_files_list)}.'
            )

        return cls(
            metric_files=metric_files_list,
            diagnostic_files=_maybe_list(diagnostic_files),
            profile_files=_maybe_list(profile_files),
            sig_figs=sig_figs,
            **kwargs,
        )

    @property
    def _iter_sampling(self) -> int:
        return self.config.method_config.num_samples

    @property
    def _iter_warmup(self) -> int:
        return self.config.method_config.num_warmup

    @property
    def _thin(self) -> int:
        return self.config.method_config.thin

    @property
    def _save_warmup(self) -> bool:
        return self.config.method_config.save_warmup

    @property
    def _is_fixed_param(self) -> bool:
        return self.config.method_config.algorithm == 'fixed_param'

    def create_inits(
        self, seed: int | None = None, chains: int = 4
    ) -> list[dict[str, np.ndarray]] | dict[str, np.ndarray]:
        """
        Create initial values for the parameters of the model by
        randomly selecting draws from the MCMC samples. If the samples
        contain draws from multiple chains, each draw will be from
        a different chain, if possible. Otherwise the chain is randomly
        selected.

        :param seed: Used for random selection, defaults to None
        :param chains: Number of initial values to return, defaults to 4
        :return: The initial values for the parameters of the model.

        If ``chains`` is 1, a dictionary is returned, otherwise a list
        of dictionaries is returned, in the format expected for the
        ``inits`` argument of :meth:`CmdStanModel.sample`.
        """
        self._assemble()
        rng = np.random.default_rng(seed)
        n_draws, n_chains = self._draws.shape[:2]
        draw_idxs = rng.choice(n_draws, size=chains, replace=False)
        chain_idxs = rng.choice(
            n_chains, size=chains, replace=(n_chains <= chains)
        )
        if chains == 1:
            draw = self._draws[draw_idxs[0], chain_idxs[0]]
            return {
                name: var.extract_reshape(draw)
                for name, var in self.metadata.stan_vars.items()
            }
        else:
            return [
                {
                    name: var.extract_reshape(self._draws[d, i])
                    for name, var in self.metadata.stan_vars.items()
                }
                for d, i in zip(draw_idxs, chain_idxs)
            ]

    def __repr__(self) -> str:
        mc = self.config.method_config
        lines = [
            f'CmdStanMCMC: model={self.model_name} chains={self.chains}'
            f' method={mc.method} algorithm={mc.algorithm}',
            ' csv_files:\n\t' + '\n\t'.join(self.csv_files),
        ]
        if self.config_files is not None:
            lines.append(' config_files:\n\t' + '\n\t'.join(self.config_files))
        if self.stdout_files is not None:
            lines.append(' output_files:\n\t' + '\n\t'.join(self.stdout_files))
        return '\n'.join(lines)

    @property
    def num_draws_warmup(self) -> int:
        """Number of warmup draws per chain, i.e., thinned warmup iterations."""
        return int(math.ceil((self._iter_warmup) / self._thin))

    @property
    def num_draws_sampling(self) -> int:
        """
        Number of sampling (post-warmup) draws per chain, i.e.,
        thinned sampling iterations.
        """
        return int(math.ceil((self._iter_sampling) / self._thin))

    @property
    def metric_type(self) -> str | None:
        """
        Metric type used for adaptation, either 'diag_e' or 'dense_e', according
        to CmdStan arg 'metric'.
        Returns None when sampler algorithm 'fixed_param' is specified, or when
        no metric files are available (e.g. adaptation was disabled, so CmdStan
        wrote no metric output).
        """
        return self._metric_type if self._ensure_metric_parsed() else None

    @property
    def inv_metric(self) -> np.ndarray | None:
        """
        Inverse mass matrix used by sampler for each chain.
        Returns a ``nchains x nparams`` array when metric_type is 'diag_e',
        a ``nchains x nparams x nparams`` array when metric_type is 'dense_e',
        or ``None`` when metric_type is 'unit_e', algorithm is 'fixed_param',
        or no metric files are available.
        """
        if not self._ensure_metric_parsed() or self._metric_type == 'unit_e':
            return None
        return self._metric

    @property
    def step_size(self) -> np.ndarray | None:
        """
        Step size used by sampler for each chain.
        Returns None when sampler algorithm 'fixed_param' is specified, or when
        no metric files are available (e.g. adaptation was disabled).
        """
        return self._step_size if self._ensure_metric_parsed() else None

    @property
    def thin(self) -> int:
        """
        Period between recorded iterations.  (Default is 1).
        """
        return self._thin

    @property
    def divergences(self) -> np.ndarray | None:
        """
        Per-chain total number of post-warmup divergent iterations.
        When sampler algorithm 'fixed_param' is specified, returns None.
        """
        return self._divergences if not self._is_fixed_param else None

    @property
    def max_treedepths(self) -> np.ndarray | None:
        """
        Per-chain total number of post-warmup iterations where the NUTS sampler
        reached the maximum allowed treedepth.
        When sampler algorithm 'fixed_param' is specified, returns None.
        """
        return self._max_treedepths if not self._is_fixed_param else None

    def draws(
        self, *, inc_warmup: bool = False, concat_chains: bool = False
    ) -> np.ndarray:
        """
        Returns a numpy.ndarray over all draws from all chains which is
        stored column major so that the values for a parameter are contiguous
        in memory, likewise all draws from a chain are contiguous.
        By default, returns a 3D array arranged (draws, chains, columns);
        parameter ``concat_chains=True`` will return a 2D array where all
        chains are flattened into a single column, preserving chain order,
        so that given M chains of N draws, the first N draws are from chain 1,
        up through the last N draws from chain M.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        :param concat_chains: When ``True`` return a 2D array flattening all
            all draws from all chains.  Default value is ``False``.

        See Also
        --------
        CmdStanMCMC.draws_pd
        CmdStanMCMC.draws_xr
        CmdStanGQ.draws
        """
        self._assemble()

        if inc_warmup and not self._save_warmup:
            get_logger().warning(
                "Sample doesn't contain draws from warmup iterations,"
                ' rerun sampler with "save_warmup=True".'
            )

        start_idx = 0
        if not inc_warmup and self._save_warmup:
            start_idx = self.num_draws_warmup

        if concat_chains:
            return flatten_chains(self._draws[start_idx:, :, :])
        return self._draws[start_idx:, :, :]

    @classmethod
    def _comparable_config(
        cls, config: StanConfig[SampleConfig]
    ) -> dict[str, Any]:
        """Returns configuration settings that must agree across chains"""
        method_config = config.method_config
        return {
            **super()._comparable_config(config),
            'algorithm': method_config.algorithm,
            'num_samples': method_config.num_samples,
            'num_warmup': method_config.num_warmup,
            'save_warmup': method_config.save_warmup,
            'thin': method_config.thin,
            'max_depth': method_config.max_depth,
        }

    def _validate_csv_files(self) -> None:
        """
        Checks that the draws in the Stan CSV output files are consistent
        with the run configuration, and tabulates per-chain counts of
        sampling iters which are divergent or at max treedepth.

        Raises exception when inconsistencies detected.
        """
        expected_sampling = math.ceil(self._iter_sampling / self._thin)
        expected_warmup = (
            math.ceil(self._iter_warmup / self._thin)
            if self._save_warmup
            else 0
        )
        max_depth = self.config.method_config.max_depth
        for i in range(self.chains):
            path = self.csv_files[i]
            _, header, draws = stancsv.parse_comments_header_and_draws(path)
            if header is None:
                raise ValueError(f'No header found in Stan CSV file {path}')
            stancsv.raise_on_inconsistent_draws_shape(header, draws)

            num_warmup, num_sampling = stancsv.count_warmup_and_sampling_draws(
                path
            )
            if num_sampling != expected_sampling:
                raise ValueError(
                    f'Bad Stan CSV file {path}, expected {expected_sampling} '
                    f'draws, found {num_sampling}'
                )
            if self._save_warmup and num_warmup != expected_warmup:
                raise ValueError(
                    f'Bad Stan CSV file {path}, expected {expected_warmup} '
                    f'warmup draws, found {num_warmup}'
                )
            if not self._is_fixed_param:
                # HMC runs always record max_depth; only fixed_param omits it
                assert max_depth is not None
                treedepths, divergences = (
                    stancsv.extract_max_treedepth_and_divergence_counts(
                        header, draws, max_depth, num_warmup
                    )
                )
                self._max_treedepths[i] = treedepths
                self._divergences[i] = divergences

    def _ensure_metric_parsed(self) -> bool:
        """Parse the metric info if available, returning whether it is.

        Returns False (and the metric properties then report None) for
        fixed-param runs or when no metric files were written.
        """
        if self._is_fixed_param:
            return False
        if self._metric_type is None:
            if not self._metric_available():
                return False
            self._parse_metric_info()
        return True

    def _metric_available(self) -> bool:
        """Whether a metric JSON is present on disk for every chain.

        Returns False when there are no metric files at all -- either none
        were passed, or CmdStan wrote none because adaptation was disabled --
        so the metric properties can report None rather than fail. Raises when
        some but not all chains have a metric file, since parsing a partial set
        would silently misalign the per-chain metric arrays.
        """
        if self.metric_files is None:
            return False
        existing = [os.path.exists(f) for f in self.metric_files]
        if not any(existing):
            return False
        if not all(existing):
            missing = [
                f for f, ok in zip(self.metric_files, existing) if not ok
            ]
            raise ValueError(
                'Metric files missing for some chains: ' + ', '.join(missing)
            )
        return True

    def _parse_metric_info(self) -> None:
        """Extracts metric type, inv_metric, and step size information from the
        parsed metric JSONs."""
        if self.metric_files is None:
            raise ValueError(
                'No metric files available; cannot read metric info.'
            )
        chain_metric_info: list[MetricInfo] = []
        for mf in self.metric_files:
            with open(mf) as f:
                chain_metric_info.append(
                    MetricInfo.model_validate_json(f.read())
                )

        metric_types = {cmi.metric_type for cmi in chain_metric_info}
        if len(metric_types) != 1:
            raise ValueError("Inconsistent metric types found across chains")
        self._metric_type = chain_metric_info[0].metric_type
        self._metric = np.asarray([cmi.inv_metric for cmi in chain_metric_info])
        self._step_size = np.asarray(
            [cmi.stepsize for cmi in chain_metric_info]
        )

    def _check_sampler_diagnostics(self) -> None:
        """
        Warn if any iterations ended in divergences or hit maxtreedepth.
        """
        if np.any(self._divergences) or np.any(self._max_treedepths):
            diagnostics = ['Some chains may have failed to converge.']
            ct_iters = self.config.method_config.num_samples
            for i in range(self.chains):
                if self._divergences[i] > 0:
                    diagnostics.append(
                        f'Chain {i + 1} had {self._divergences[i]} '
                        'divergent transitions '
                        f'({((self._divergences[i] / ct_iters) * 100):.1f}%)'
                    )
                if self._max_treedepths[i] > 0:
                    diagnostics.append(
                        f'Chain {i + 1} had {self._max_treedepths[i]} '
                        'iterations at max treedepth '
                        f'({((self._max_treedepths[i] / ct_iters) * 100):.1f}%)'
                    )
            diagnostics.append(
                'Use the "diagnose()" method on the CmdStanMCMC object'
                ' to see further information.'
            )
            get_logger().warning('\n\t'.join(diagnostics))

    def _assemble(self) -> None:
        """
        Allocates and populates the sample array by parsing the validated
        stan_csv files.
        """
        if self._draws.shape != (0,):
            return

        num_draws = self.num_draws_sampling
        if self._save_warmup:
            num_draws += self.num_draws_warmup
        self._draws = np.empty(
            (num_draws, self.chains, len(self.column_names)),
            dtype=np.float64,
            order='F',
        )

        for chain in range(self.chains):
            try:
                (
                    _,
                    header,
                    draws,
                ) = stancsv.parse_comments_header_and_draws(
                    self.csv_files[chain]
                )

                draws_np = stancsv.csv_bytes_list_to_numpy(draws)
                if draws_np.shape[0] == 0:
                    n_cols = header.count(",") + 1  # type: ignore
                    draws_np = np.empty((0, n_cols))

                self._draws[:, chain, :] = draws_np
            except Exception as exc:
                raise ValueError(
                    f"Parsing output from {self.csv_files[chain]} failed"
                ) from exc

    def summary(
        self,
        percentiles: Sequence[int] = (5, 50, 95),
        sig_figs: int = 6,
    ) -> pd.DataFrame:
        """
        Run cmdstan/bin/stansummary over all output CSV files, assemble
        summary into DataFrame object.  The first row contains statistics
        for the total joint log probability `lp__`, but is omitted when the
        Stan model has no parameters.  The remaining rows contain summary
        statistics for all parameters, transformed parameters, and generated
        quantities variables, in program declaration order.

        :param percentiles: Ordered non-empty sequence of percentiles to report.
            Must be integers from (1, 99), inclusive. Defaults to
            ``(5, 50, 95)``

        :param sig_figs: Number of significant figures to report.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.
            If precision above 6 is requested, sample must have been produced
            by CmdStan version 2.25 or later and sampler output precision
            must equal to or greater than the requested summary precision.

        :return: pandas.DataFrame
        """
        if len(percentiles) == 0:
            raise ValueError(
                'Invalid percentiles argument, must be ordered'
                ' non-empty list from (1, 99), inclusive.'
            )
        cur_pct = 0
        for pct in percentiles:
            if pct > 99 or not pct > cur_pct:
                raise ValueError(
                    'Invalid percentiles spec, must be ordered'
                    ' non-empty list from (1, 99), inclusive.'
                )
            cur_pct = pct
        percentiles_str = (
            f"--percentiles= {','.join(str(x) for x in percentiles)}"
        )

        if not isinstance(sig_figs, int) or sig_figs < 1 or sig_figs > 18:
            raise ValueError(
                'Keyword "sig_figs" must be an integer between 1 and 18,'
                ' found {}'.format(sig_figs)
            )
        csv_sig_figs = self.sig_figs or 6
        if sig_figs > csv_sig_figs:
            get_logger().warning(
                'Requesting %d significant digits of output, but CSV files'
                ' only have %d digits of precision.',
                sig_figs,
                csv_sig_figs,
            )
        sig_figs_str = f'--sig_figs={sig_figs}'
        cmd_path = os.path.join(
            cmdstan_path(), 'bin', 'stansummary' + EXTENSION
        )
        tmp_csv_file = f'stansummary-{self.model_name}-'
        tmp_csv_path = create_named_text_file(
            dir=_TMPDIR, prefix=tmp_csv_file, suffix='.csv', name_only=True
        )
        csv_str = f'--csv_filename={tmp_csv_path}'

        cmd = [
            cmd_path,
            percentiles_str,
            sig_figs_str,
            csv_str,
        ] + self.csv_files
        do_command(cmd, fd_out=None)
        with open(tmp_csv_path, 'rb') as fd:
            summary_data = pd.read_csv(
                fd,
                delimiter=',',
                header=0,
                index_col=0,
                comment='#',
                float_precision='high',
            )
        mask = (
            [not x.endswith('__') for x in summary_data.index]
            if self._is_fixed_param
            else [
                x == 'lp__' or not x.endswith('__') for x in summary_data.index
            ]
        )
        summary_data.index.name = None
        return summary_data[mask]

    def diagnose(self) -> str | None:
        """
        Run cmdstan/bin/diagnose over all output CSV files,
        return console output.

        The diagnose utility reads the outputs of all chains
        and checks for the following potential problems:

        + Transitions that hit the maximum treedepth
        + Divergent transitions
        + Low E-BFMI values (sampler transitions HMC potential energy)
        + Low effective sample sizes
        + High R-hat values
        """
        cmd_path = os.path.join(cmdstan_path(), 'bin', 'diagnose' + EXTENSION)
        cmd = [cmd_path] + self.csv_files
        result = StringIO()
        do_command(cmd=cmd, fd_out=result)
        return result.getvalue()

    def draws_pd(
        self,
        vars: list[str] | str | None = None,
        inc_warmup: bool = False,
    ) -> pd.DataFrame:
        """
        Returns the sample draws as a pandas DataFrame.
        Flattens all chains into single column.  Container variables
        (array, vector, matrix) will span multiple columns, one column
        per element. E.g. variable 'matrix[2,2] foo' spans 4 columns:
        'foo[1,1], ... foo[2,2]'.

        :param vars: optional list of variable names.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        See Also
        --------
        CmdStanMCMC.draws
        CmdStanMCMC.draws_xr
        CmdStanGQ.draws_pd
        """
        if vars is not None:
            if isinstance(vars, str):
                vars_list = [vars]
            else:
                vars_list = vars

        if inc_warmup and not self._save_warmup:
            get_logger().warning(
                'Draws from warmup iterations not available,'
                ' must run sampler with "save_warmup=True".'
            )

        self._assemble()
        cols = []
        if vars is not None:
            for var in dict.fromkeys(vars_list):
                if var in self.metadata.method_vars:
                    cols.append(var)
                elif var in self.metadata.stan_vars:
                    info = self.metadata.stan_vars[var]
                    cols.extend(
                        self.column_names[info.start_idx : info.end_idx]
                    )
                elif var in ['chain__', 'iter__', 'draw__']:
                    cols.append(var)
                else:
                    raise ValueError(f'Unknown variable: {var}')
        else:
            cols = ['chain__', 'iter__', 'draw__'] + list(self.column_names)

        draws = self.draws(inc_warmup=inc_warmup)
        # add long-form columns for chain, iteration, draw
        n_draws, n_chains, _ = draws.shape
        chains_col = (
            np.repeat(np.arange(1, n_chains + 1), n_draws)
            .reshape(1, n_chains, n_draws)
            .T
        )
        iter_col = (
            np.tile(np.arange(1, n_draws + 1), n_chains)
            .reshape(1, n_chains, n_draws)
            .T
        )
        draw_col = (
            np.arange(1, (n_draws * n_chains) + 1)
            .reshape(1, n_chains, n_draws)
            .T
        )
        draws = np.concatenate([chains_col, iter_col, draw_col, draws], axis=2)

        return pd.DataFrame(
            data=flatten_chains(draws),
            columns=['chain__', 'iter__', 'draw__'] + list(self.column_names),
        )[cols]

    def draws_xr(
        self, vars: str | list[str] | None = None, inc_warmup: bool = False
    ) -> xr.Dataset:
        """
        Returns the sampler draws as a xarray Dataset.

        :param vars: optional list of variable names.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        See Also
        --------
        CmdStanMCMC.draws
        CmdStanMCMC.draws_pd
        CmdStanGQ.draws_xr
        """
        if not XARRAY_INSTALLED:
            raise RuntimeError(
                'Package "xarray" is not installed, cannot produce draws array.'
            )
        if inc_warmup and not self._save_warmup:
            get_logger().warning(
                'Draws from warmup iterations not available,'
                ' must run sampler with "save_warmup=True".'
            )
        if vars is None:
            vars_list = list(self.metadata.stan_vars.keys())
        elif isinstance(vars, str):
            vars_list = [vars]
        else:
            vars_list = vars

        self._assemble()

        num_draws = self.num_draws_sampling
        attrs: MutableMapping[Hashable, Any] = {
            "stan_version": f"{self.config.stan_major_version}."
            f"{self.config.stan_minor_version}."
            f"{self.config.stan_patch_version}",
            "model": self.model_name,
            "num_draws_sampling": num_draws,
        }
        if inc_warmup and self._save_warmup:
            num_draws += self.num_draws_warmup
            attrs["num_draws_warmup"] = self.num_draws_warmup

        data: MutableMapping[Hashable, Any] = {}
        coordinates: MutableMapping[Hashable, Any] = {
            "chain": self.chain_ids,
            "draw": np.arange(num_draws),
        }

        for var in vars_list:
            build_xarray_data(
                data,
                self.metadata.stan_vars[var],
                self.draws(inc_warmup=inc_warmup),
            )
        return xr.Dataset(data, coords=coordinates, attrs=attrs).transpose(
            'chain', 'draw', ...
        )

    def stan_variable(
        self,
        var: str,
        inc_warmup: bool = False,
    ) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the set of draws
        for the named Stan program variable.  Flattens the chains,
        leaving the draws in chain order.  The first array dimension,
        corresponds to number of draws or post-warmup draws in the sample,
        per argument ``inc_warmup``.  The remaining dimensions correspond to
        the shape of the Stan program variable.

        Underlyingly draws are in chain order, i.e., for a sample with
        N chains of M draws each, the first M array elements are from chain 1,
        the next M are from chain 2, and the last M elements are from chain N.

        * If the variable is a scalar variable, the return array has shape
          ( draws * chains, 1).
        * If the variable is a vector, the return array has shape
          ( draws * chains, len(vector))
        * If the variable is a matrix, the return array has shape
          ( draws * chains, size(dim 1), size(dim 2) )
        * If the variable is an array with N dimensions, the return array
          has shape ( draws * chains, size(dim 1), ..., size(dim N))

        For example, if the Stan program variable ``theta`` is a 3x3 matrix,
        and the sample consists of 4 chains with 1000 post-warmup draws,
        this function will return a numpy.ndarray with shape (4000,3,3).

        This functionaltiy is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        See Also
        --------
        CmdStanMCMC.stan_variables
        CmdStanMLE.stan_variable
        CmdStanPathfinder.stan_variable
        CmdStanVB.stan_variable
        CmdStanGQ.stan_variable
        CmdStanLaplace.stan_variable
        """
        draws = self.draws(inc_warmup=inc_warmup, concat_chains=True)
        return self._extract_stan_var(var, draws)

    def method_variables(self) -> dict[str, np.ndarray]:
        """
        Returns a dictionary of all sampler variables, i.e., all
        output column names ending in `__`.  Assumes that all variables
        are scalar variables where column name is variable name.
        Maps each column name to a numpy.ndarray (draws x chains x 1)
        containing per-draw diagnostic values.
        """
        self._assemble()
        return {
            name: var.extract_reshape(self._draws)
            for name, var in self.metadata.method_vars.items()
        }
