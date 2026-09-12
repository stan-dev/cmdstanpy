"""Shared base classes for containers of CmdStan inference results."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generic

import numpy as np

from cmdstanpy.utils import stancsv

from .metadata import InferenceMetadata, MethodT, StanConfig


@dataclass(kw_only=True)
class StanFit(Generic[MethodT]):
    """
    Base container for the outputs of a CmdStan run: the metadata parsed
    from the Stan CSV header, the run configuration parsed from the config
    JSON, and the locations of the output files on disk.

    Draws are loaded lazily from the output CSV file(s) on first access.
    """

    metadata: InferenceMetadata
    model_name: str
    config: StanConfig[MethodT]

    # Attributes naming output files; each holds a path, an optional path,
    # or a list of paths. ``save_output_files`` moves every named file.
    _FILE_ATTRS: ClassVar[tuple[str, ...]] = ()

    def _assemble(self) -> None:
        """Load lazily-parsed outputs from the output files."""
        raise NotImplementedError

    def stan_variable(self, var: str) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the values for the named
        Stan program variable.
        """
        raise NotImplementedError

    def stan_variables(self) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        See Also
        --------
        StanFit.stan_variable
        """
        result = {}
        for name in self.metadata.stan_vars:
            result[name] = self.stan_variable(name)
        return result

    def _extract_stan_var(self, var: str, draws: np.ndarray) -> np.ndarray:
        """Reshape the draws for one Stan program variable, raising a
        ValueError naming the available variables when it is unknown."""
        try:
            out: np.ndarray = self.metadata.stan_vars[var].extract_reshape(
                draws
            )
            return out
        except KeyError:
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are '
                + ", ".join(self.metadata.stan_vars.keys())
            ) from None

    @property
    def column_names(self) -> tuple[str, ...]:
        """
        Names of all outputs from the method, comprising method diagnostics
        and all components of all model parameters, transformed parameters,
        and quantities of interest. Corresponds to Stan CSV file header row,
        with names munged to array notation, e.g. `beta[1]` not `beta.1`.
        """
        return self.metadata.column_names

    def __getattr__(self, attr: str) -> np.ndarray:
        """Synonymous with ``fit.stan_variable(attr)"""
        if attr.startswith("_"):
            raise AttributeError(f"Unknown variable name {attr}")
        try:
            return self.stan_variable(attr)
        except ValueError as e:
            # pylint: disable=raise-missing-from
            raise AttributeError(*e.args)

    def __getstate__(self) -> dict:
        # This function returns the mapping of objects to serialize with pickle.
        # See https://docs.python.org/3/library/pickle.html#object.__getstate__
        # for details. We call _assemble to ensure lazily-parsed outputs have
        # been loaded prior to serialization.
        self._assemble()
        return self.__dict__

    def save_output_files(self, dir: str | None = None) -> None:
        """
        Move the output CSV file(s), and any associated config, metric, and
        stdout files, to the specified directory. Updates the corresponding
        attributes on this object to point at the new locations.

        :param dir: directory path

        See Also
        --------
        cmdstanpy.from_output_files
        """
        dest = Path(dir) if dir is not None else Path.cwd()
        try:
            dest.mkdir(parents=True, exist_ok=True)
            probe = dest / f'.cmdstanpy-write-test-{os.getpid()}'
            probe.touch()
            probe.unlink()
        except OSError as exc:
            raise RuntimeError(f'Cannot save to path: {dest}') from exc

        def _move(src: str, is_csv: bool) -> str:
            if not os.path.exists(src):
                if is_csv:
                    raise ValueError(f'Cannot access CSV file {src}')
                return src  # leave e.g. a deleted stdout file as-is
            dst = dest / Path(src).name
            if dst.exists():
                raise ValueError(f'File exists, not overwriting: {dst}')
            shutil.move(src, dst)
            return os.fspath(dst)

        for attr in self._FILE_ATTRS:
            val = getattr(self, attr)
            is_csv = attr in ('csv_file', 'csv_files')
            if val is None:
                continue
            if isinstance(val, list):
                setattr(self, attr, [_move(f, is_csv) for f in val])
            else:
                setattr(self, attr, _move(val, is_csv))


# _assemble and stan_variable remain abstract here; the concrete
# per-method subclasses implement them. This requires the below
# pylint disable. Perhaps replace with a proper ABC in the future?
@dataclass(kw_only=True)
# pylint: disable-next=abstract-method
class MultiChainFit(StanFit[MethodT]):
    """
    Base container for the outputs of a CmdStan run whose draws are held
    in one Stan CSV file per chain.
    """

    csv_files: list[str]
    chain_ids: list[int]
    config_files: list[str] | None = None
    stdout_files: list[str] | None = None

    _FILE_ATTRS: ClassVar[tuple[str, ...]] = (
        'csv_files',
        'config_files',
        'stdout_files',
    )

    def __post_init__(self) -> None:
        self._validate_configs()

    @property
    def chains(self) -> int:
        """Number of chains."""
        return len(self.csv_files)

    @classmethod
    def _from_files_kwargs(
        cls,
        csv_files: Sequence[str | os.PathLike],
        config_files: Sequence[str | os.PathLike] | str | os.PathLike,
        stdout_files: Sequence[str | os.PathLike] | None,
        chain_ids: Sequence[int] | None,
        config_cls: type[StanConfig[MethodT]],
    ) -> dict[str, Any]:
        """Common constructor arguments parsed from the output files.

        ``config_files`` may be a single path (when CmdStan ran multiple
        chains in one process) or a per-chain list.
        """
        csv_files_list = [os.fspath(f) for f in csv_files]
        chains = len(csv_files_list)
        if chains == 0:
            raise ValueError('At least one CSV file is required.')

        if isinstance(config_files, (str, os.PathLike)):
            config_files_list = [os.fspath(config_files)]
        else:
            config_files_list = [os.fspath(f) for f in config_files]
        if not config_files_list:
            raise ValueError('At least one config file is required.')

        with open(config_files_list[0]) as f:
            stan_config = config_cls.from_json(f.read())

        stdout_files_list = (
            [os.fspath(f) for f in stdout_files]
            if stdout_files is not None
            else None
        )

        if chain_ids is None:
            chain_ids_list = list(range(1, chains + 1))
        else:
            chain_ids_list = list(chain_ids)
            if len(chain_ids_list) != chains:
                raise ValueError(
                    f'Got {chains} csv files but {len(chain_ids_list)} '
                    'chain ids'
                )

        return {
            'metadata': InferenceMetadata.from_csv(csv_files_list[0]),
            'model_name': stan_config.model_name,
            'csv_files': csv_files_list,
            'config': stan_config,
            'chain_ids': chain_ids_list,
            'config_files': config_files_list,
            'stdout_files': stdout_files_list,
        }

    @classmethod
    def _comparable_config(cls, config: StanConfig[MethodT]) -> dict[str, Any]:
        """The config settings which must agree across chains: the model
        name and Stan version. Subclasses extend this with the settings
        that affect how their draws are laid out."""
        return {
            'model': config.model_name,
            'stan_version_major': config.stan_major_version,
            'stan_version_minor': config.stan_minor_version,
            'stan_version_patch': config.stan_patch_version,
            'stanc_version': config.stanc_version,
        }

    def _validate_configs(self) -> None:
        """
        Checks that the CmdStan config JSONs for all chains agree on the
        settings named by ``_comparable_config``.

        When CmdStan ran all chains in a single process there is only one
        config file, so there is nothing to cross-check.

        Raises exception if inconsistencies are detected.
        """
        if self.config_files is None or len(self.config_files) < 2:
            return

        expected = self._comparable_config(self.config)
        for config_file in self.config_files[1:]:
            # Parse sibling configs with the same class as self.config so
            # the comparison sees identically-typed method configs, however
            # this object was constructed.
            with open(config_file) as f:
                other = self._comparable_config(
                    type(self.config).from_json(f.read())
                )
            for key, want in expected.items():
                if other[key] != want:
                    raise ValueError(
                        'CmdStan config mismatch in config file '
                        f'{config_file}: arg {key} is {other[key]}, '
                        f'expected {want}'
                    )


@dataclass(kw_only=True)
class SingleFileFit(StanFit[MethodT]):
    """
    Base container for the outputs of a CmdStan run whose draws are held
    in a single Stan CSV file.
    """

    csv_file: str
    config_file: str | None = None  # None if config object passed directly
    stdout_file: str | None = None

    _FILE_ATTRS: ClassVar[tuple[str, ...]] = (
        'csv_file',
        'config_file',
        'stdout_file',
    )

    _draws: np.ndarray = field(default_factory=lambda: np.array(()), init=False)

    @classmethod
    def _from_files_kwargs(
        cls,
        csv_file: str | os.PathLike,
        config_file: str | os.PathLike,
        stdout_file: str | os.PathLike | None,
        config_cls: type[StanConfig[MethodT]],
    ) -> dict[str, Any]:
        """Common constructor arguments parsed from the output files."""
        with open(config_file) as f:
            stan_config = config_cls.from_json(f.read())

        return {
            'metadata': InferenceMetadata.from_csv(csv_file),
            'model_name': stan_config.model_name,
            'csv_file': os.fspath(csv_file),
            'config': stan_config,
            'config_file': os.fspath(config_file),
            'stdout_file': (
                os.fspath(stdout_file) if stdout_file is not None else None
            ),
        }

    def _assemble(self) -> None:
        if self._draws.shape != (0,):
            return

        try:
            *_, draws = stancsv.parse_comments_header_and_draws(self.csv_file)
            self._draws = stancsv.csv_bytes_list_to_numpy(draws)
        except Exception as exc:
            raise ValueError(
                f"An error occurred when parsing Stan csv {self.csv_file}"
            ) from exc

    def stan_variable(self, var: str) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the values for the
        named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        This functionality is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        See Also
        --------
        StanFit.stan_variables
        """
        self._assemble()
        return self._extract_stan_var(var, self._draws)

    def method_variables(self) -> dict[str, np.ndarray]:
        """
        Returns a dictionary of all method variables, i.e., all
        output column names ending in `__`.  Assumes that all variables
        are scalar variables where column name is variable name.
        Maps each column name to a numpy.ndarray containing
        per-draw diagnostic values.
        """
        self._assemble()
        return {
            name: var.extract_reshape(self._draws)
            for name, var in self.metadata.method_vars.items()
        }

    def _draws_for_inits(self) -> np.ndarray:
        """Return the rows from which ``create_inits`` may select."""
        self._assemble()
        return self._draws

    def create_inits(
        self, seed: int | None = None, chains: int = 4
    ) -> list[dict[str, np.ndarray]] | dict[str, np.ndarray]:
        """
        Create initial values for the parameters of the model
        by randomly selecting draws from the fit.

        :param seed: Used for random selection, defaults to None
        :param chains: Number of initial values to return, defaults to 4
        :return: The initial values for the parameters of the model.

        If ``chains`` is 1, a dictionary is returned, otherwise a list
        of dictionaries is returned, in the format expected for the
        ``inits`` argument of :meth:`CmdStanModel.sample`.
        """
        draws = self._draws_for_inits()
        rng = np.random.default_rng(seed)
        idxs = rng.choice(draws.shape[0], size=chains, replace=False)
        if chains == 1:
            draw = draws[idxs[0]]
            return {
                name: var.extract_reshape(draw)
                for name, var in self.metadata.stan_vars.items()
            }
        return [
            {
                name: var.extract_reshape(draws[idx])
                for name, var in self.metadata.stan_vars.items()
            }
            for idx in idxs
        ]
