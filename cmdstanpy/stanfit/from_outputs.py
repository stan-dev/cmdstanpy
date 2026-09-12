"""Reconstruct fit objects from CmdStan output files."""

from __future__ import annotations

import glob
import os
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

from cmdstanpy.utils.filesystem import accompanying_json

from .laplace import CmdStanLaplace
from .mcmc import CmdStanMCMC
from .metadata import (
    AnyMethodConfig,
    LaplaceConfig,
    SampleConfig,
    StanConfig,
    parse_config,
)
from .mle import CmdStanMLE
from .pathfinder import CmdStanPathfinder
from .vb import CmdStanVB

AnyStanFit: TypeAlias = (
    CmdStanMCMC | CmdStanMLE | CmdStanVB | CmdStanPathfinder | CmdStanLaplace
)

_VALID_METHODS = ('sample', 'optimize', 'variational', 'laplace', 'pathfinder')
_CONFIG_SUFFIX = '_config.json'
_METRIC_SUFFIX = '_metric.json'
_CHAIN_ID_RE = re.compile(r'[_-](\d+)$')


def _split_chain_id(stem: str) -> tuple[str, int] | None:
    match = _CHAIN_ID_RE.search(stem)
    if match is None:
        return None
    return stem[: match.start()], int(match.group(1))


def _json_for_csv(csv_file: Path, kind: str) -> Path:
    return Path(accompanying_json(csv_file, kind))


@dataclass(frozen=True)
class ParsedConfig:
    """A config JSON and its parsed value."""

    path: Path
    value: StanConfig[AnyMethodConfig]


@dataclass(frozen=True)
class FitFiles:
    """Normalized files required to reconstruct one fit."""

    method: str
    csv_files: tuple[Path, ...]
    config_files: tuple[Path, ...]
    metric_files: tuple[Path, ...] = ()
    chain_ids: tuple[int, ...] = ()
    mode_files: FitFiles | None = None


def _read_config(config_file: Path) -> ParsedConfig:
    """Read one CmdStan config JSON, retaining its path for diagnostics."""
    try:
        return ParsedConfig(config_file, parse_config(config_file.read_bytes()))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f'Cannot parse CmdStan config JSON {config_file}'
        ) from exc


def _config_id(config: ParsedConfig) -> int:
    """Return a valid sample chain ID, defaulting to one."""
    chain_id = 1 if config.value.id is None else config.value.id
    if chain_id <= 0:
        raise ValueError(
            f'Config JSON {config.path} has a non-positive chain ID.'
        )
    return chain_id


def _num_chains(config: ParsedConfig) -> int:
    method = cast(SampleConfig, config.value.method_config)
    return method.num_chains


def _output_names(config: ParsedConfig) -> list[str]:
    names = (name.strip() for name in config.value.output.file.split(','))
    return [Path(name).name for name in names if name]


def _managed_csv_files(
    config: ParsedConfig, expected_count: int
) -> tuple[Path, ...]:
    """Resolve CSVs named by a config in a managed output bundle."""
    names = _output_names(config)
    if len(names) != expected_count:
        raise ValueError(
            f'Config JSON {config.path} records {len(names)} output CSVs, '
            f'expected {expected_count}.'
        )
    csv_files = tuple(config.path.parent / name for name in names)
    expected_config = _json_for_csv(csv_files[0], 'config')
    if expected_config.name != config.path.name:
        raise ValueError(
            f'Config JSON {config.path.name} is not named for its first '
            f'output CSV {csv_files[0].name}.'
        )
    missing = [str(path) for path in csv_files if not path.is_file()]
    if missing:
        raise ValueError('Output CSVs are missing: ' + ', '.join(missing))
    return csv_files


def _validate_compatible(configs: Sequence[ParsedConfig]) -> None:
    if not configs:
        return
    expected = CmdStanMCMC._comparable_config(configs[0].value)  # type: ignore
    for config in configs[1:]:
        actual = CmdStanMCMC._comparable_config(config.value)  # type: ignore
        if actual != expected:
            raise ValueError(
                'Config JSONs do not describe one compatible fit: '
                f'{configs[0].path} and {config.path}. '
                'Pass the files to the appropriate method-specific '
                'from_files() constructor if explicit associations are needed.'
            )


def _validate_mode_compatible(
    laplace: ParsedConfig, mode: ParsedConfig
) -> None:
    left = laplace.value
    right = mode.value
    if (
        left.model_name != right.model_name
        or left.stan_major_version != right.stan_major_version
        or left.stan_minor_version != right.stan_minor_version
        or left.stan_patch_version != right.stan_patch_version
    ):
        raise ValueError(
            f'Laplace config {laplace.path} and optimization mode config '
            f'{mode.path} do not describe the same model and Stan version.'
        )
    laplace_jacobian = getattr(left.method_config, 'jacobian', None)
    mode_jacobian = getattr(right.method_config, 'jacobian', None)
    if laplace_jacobian != mode_jacobian:
        raise ValueError(
            f'Laplace config {laplace.path} and optimization mode config '
            f'{mode.path} disagree on the jacobian setting.'
        )


def _discover_metric_files(csv_files: Sequence[Path]) -> tuple[Path, ...]:
    metrics = tuple(_json_for_csv(csv_file, 'metric') for csv_file in csv_files)
    return metrics if all(metric.exists() for metric in metrics) else ()


def _single_process_sample_candidate(config: ParsedConfig) -> FitFiles:
    count = _num_chains(config)
    csv_files = _managed_csv_files(config, count)
    first_id = _config_id(config)
    chain_ids = tuple(range(first_id, first_id + count))
    return FitFiles(
        method='sample',
        csv_files=csv_files,
        config_files=(config.path,),
        metric_files=_discover_metric_files(csv_files),
        chain_ids=chain_ids,
    )


def _per_process_sample_candidate(
    anchor: ParsedConfig,
    cache: dict[Path, ParsedConfig],
) -> FitFiles:
    chain_id = _config_id(anchor)
    anchor_csv = _managed_csv_files(anchor, 1)[0]
    split = _split_chain_id(anchor_csv.stem)
    if split is None or split[1] != chain_id:
        configs = [anchor]
    else:
        base = split[0]
        pattern = re.compile(
            re.escape(base) + r'[_-](\d+)' + re.escape(_CONFIG_SUFFIX)
        )
        configs = []
        for path in sorted(anchor.path.parent.glob(f'*{_CONFIG_SUFFIX}')):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            parsed = cache.get(path)
            if parsed is None:
                try:
                    parsed = _read_config(path)
                except ValueError as exc:
                    raise ValueError(
                        f'Cannot validate sibling config JSON {path}'
                    ) from exc
                cache[path] = parsed
            if parsed.value.method_config.method != 'sample':
                continue
            sibling_id = _config_id(parsed)
            if sibling_id != int(match.group(1)):
                continue
            if _num_chains(parsed) != 1:
                raise ValueError(
                    f'Config JSON {path} mixes per-chain naming with '
                    'a multi-chain process.'
                )
            configs.append(parsed)
    _validate_compatible(configs)
    by_id = sorted((_config_id(config), config) for config in configs)
    ids = tuple(chain_id for chain_id, _ in by_id)
    if len(set(ids)) != len(ids):
        raise ValueError('Sample config JSONs have duplicate chain IDs.')
    ordered = tuple(config for _, config in by_id)
    csv_files = tuple(_managed_csv_files(config, 1)[0] for config in ordered)
    return FitFiles(
        method='sample',
        csv_files=csv_files,
        config_files=tuple(config.path for config in ordered),
        metric_files=_discover_metric_files(csv_files),
        chain_ids=ids,
    )


def _laplace_mode_name(config: ParsedConfig) -> str:
    method_config = config.value.method_config
    if not isinstance(method_config, LaplaceConfig):
        raise ValueError(f'Config JSON {config.path} is not from Laplace.')
    return Path(method_config.mode).name


def _laplace_candidate(
    config: ParsedConfig,
    csv_file: Path,
    cache: dict[Path, ParsedConfig],
) -> FitFiles:
    mode_name = _laplace_mode_name(config)
    mode_csv = config.path.parent / mode_name
    mode_config_path = _json_for_csv(mode_csv, 'config')
    missing = [
        str(path)
        for path in (csv_file, mode_csv, mode_config_path)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(
            f'Laplace fit for {config.path} is missing its Laplace '
            'CSV or optimization mode CSV/config JSON: '
            + ', '.join(missing)
            + '. Pass the Laplace and mode files explicitly, or call '
            'CmdStanLaplace.from_files().'
        )
    mode_config = cache.get(mode_config_path)
    if mode_config is None:
        mode_config = _read_config(mode_config_path)
        cache[mode_config_path] = mode_config
    if mode_config.value.method_config.method != 'optimize':
        raise ValueError(
            f'Laplace mode config JSON {mode_config_path} is not from optimize.'
        )
    configured_mode_csv = _managed_csv_files(mode_config, 1)[0]
    if configured_mode_csv != mode_csv:
        raise ValueError(
            f'Laplace mode config {mode_config_path} names output CSV '
            f'{configured_mode_csv.name}, expected {mode_csv.name}.'
        )
    _validate_mode_compatible(config, mode_config)
    mode = FitFiles(
        method='optimize',
        csv_files=(mode_csv,),
        config_files=(mode_config_path,),
    )
    return FitFiles(
        method='laplace',
        csv_files=(csv_file,),
        config_files=(config.path,),
        mode_files=mode,
    )


def _candidate_from_config(
    config: ParsedConfig, parsed_configs: dict[Path, ParsedConfig]
) -> FitFiles:
    """Resolve one fit candidate anchored on a parsed config."""
    method = config.value.method_config.method
    if method not in _VALID_METHODS:
        raise ValueError(
            f'Unsupported CmdStan method {method} in {config.path}.'
        )
    if method == 'sample':
        if _num_chains(config) > 1:
            return _single_process_sample_candidate(config)
        return _per_process_sample_candidate(config, parsed_configs)
    csv_file = _managed_csv_files(config, 1)[0]
    if method == 'laplace':
        return _laplace_candidate(config, csv_file, parsed_configs)
    return FitFiles(
        method=method, csv_files=(csv_file,), config_files=(config.path,)
    )


def _discover_directory(directory: Path) -> FitFiles:
    cache: dict[Path, ParsedConfig] = {}
    for path in sorted(directory.glob(f'*{_CONFIG_SUFFIX}')):
        try:
            cache[path] = _read_config(path)
        except ValueError:
            # Directories commonly contain unrelated JSON; only valid,
            # self-contained candidates matter during a scan.
            continue

    candidates: set[FitFiles] = set()
    covered_configs: set[Path] = set()
    for config in list(cache.values()):
        if config.path in covered_configs:
            continue
        try:
            candidate = _candidate_from_config(config, cache)
        except ValueError:
            continue
        candidates.add(candidate)
        covered_configs.update(candidate.config_files)
        if candidate.mode_files is not None:
            covered_configs.update(candidate.mode_files.config_files)

    # The optimization output used by Laplace is part of that fit, rather
    # than a second top-level candidate in the same output directory.
    nested_modes = {
        candidate.mode_files
        for candidate in candidates
        if candidate.mode_files is not None
    }
    candidates.difference_update(nested_modes)
    if not candidates:
        raise ValueError(
            f'No discoverable fit was found in {directory}. No CmdStan config '
            'files found there describe a complete fit. Discovery requires the '
            'config JSONs and associated CSVs. Pass all required output files '
            'explicitly as a sequence for another layout.'
        )
    if len(candidates) > 1:
        labels = sorted(
            ', '.join(path.name for path in candidate.config_files)
            for candidate in candidates
        )
        raise ValueError(
            'Discovery found more than one fit in directory '
            f'{directory}: '
            + '; '.join(labels)
            + '. Pass a config JSON for the desired fit.'
        )
    return next(iter(candidates))


def _config_for_csv(
    csv_file: Path,
) -> tuple[ParsedConfig, dict[Path, ParsedConfig]]:
    config_path = _json_for_csv(csv_file, 'config')
    if config_path.is_file():
        config = _read_config(config_path)
        return config, {config_path: config}

    cache: dict[Path, ParsedConfig] = {}
    matches: list[ParsedConfig] = []
    for path in sorted(csv_file.parent.glob(f'*{_CONFIG_SUFFIX}')):
        try:
            config = _read_config(path)
        except ValueError:
            continue
        cache[path] = config
        if csv_file.name in _output_names(config):
            matches.append(config)
    if len(matches) != 1:
        raise ValueError(
            f'Cannot identify one config JSON for output CSV {csv_file}; '
            f'found {len(matches)}.'
        )
    return matches[0], cache


def _discover_fit(path: Path) -> FitFiles:
    """Discover a complete fit from a directory or anchor file."""
    if path.is_dir():
        return _discover_directory(path)
    if not path.is_file():
        raise ValueError(f'Invalid path specification: {path}')
    if path.name.endswith(_CONFIG_SUFFIX):
        config = _read_config(path)
        parsed_configs = {path: config}
    elif path.suffix == '.csv':
        config, parsed_configs = _config_for_csv(path)
    else:
        raise ValueError(
            f'Invalid path specification: {path}. Expected a directory, a '
            'CmdStan config JSON, or a Stan CSV.'
        )
    try:
        return _candidate_from_config(config, parsed_configs)
    except ValueError as exc:
        raise ValueError(
            f'Cannot discover a fit from {path} using output conventions: '
            f'{exc}'
        ) from exc


def _config_csv_matches(config: ParsedConfig, csv_file: Path) -> bool:
    return csv_file.name in _output_names(config)


def _associate_per_chain_configs(
    configs: Sequence[ParsedConfig], csv_files: Sequence[Path]
) -> list[tuple[int, ParsedConfig, Path]]:
    result: list[tuple[int, ParsedConfig, Path]] = []
    remaining = set(csv_files)
    for config in configs:
        matches = [
            path for path in remaining if _config_csv_matches(config, path)
        ]
        if len(matches) != 1:
            raise ValueError(
                f'Cannot associate config JSON {config.path} with exactly one '
                'supplied draw CSV using its configured output name. Use '
                'CmdStanMCMC.from_files(..., chain_ids=...) for this layout.'
            )
        if config.value.id is None:
            raise ValueError(
                f'Config JSON {config.path} has no chain ID. Use '
                'CmdStanMCMC.from_files(..., chain_ids=...) for this layout.'
            )
        csv_file = matches[0]
        remaining.remove(csv_file)
        result.append((_config_id(config), config, csv_file))
    if remaining:
        raise ValueError(
            'Some supplied draw CSVs cannot be associated with a config JSON: '
            + ', '.join(str(path) for path in sorted(remaining))
        )
    return result


def _single_config_sample(
    config: ParsedConfig, csv_files: Sequence[Path]
) -> tuple[tuple[Path, ...], tuple[int, ...]]:
    count = _num_chains(config)
    if count != len(csv_files):
        raise ValueError(
            f'Sample config JSON {config.path} records {count} chains, but '
            f'{len(csv_files)} draw CSVs were supplied.'
        )
    first_id = _config_id(config)
    chain_ids = tuple(range(first_id, first_id + count))
    names = _output_names(config)
    by_name = {path.name: path for path in csv_files}
    if (
        len(names) == count
        and len(set(names)) == count
        and set(names) == set(by_name)
    ):
        ordered = tuple(by_name[name] for name in names)
    elif len(names) == 1:
        recorded = Path(names[0])
        expanded = [
            f'{recorded.stem}_{chain_id}{recorded.suffix}'
            for chain_id in chain_ids
        ]
        if set(expanded) == set(by_name):
            ordered = tuple(by_name[name] for name in expanded)
        else:
            raise ValueError(
                f'Configured output name {names[0]} does not identify the '
                'supplied draw CSVs. Use CmdStanMCMC.from_files() for an '
                'explicitly ordered layout.'
            )
    else:
        raise ValueError(
            f'Config JSON {config.path} does not identify the supplied draw '
            'CSVs by their configured output names. Use '
            'CmdStanMCMC.from_files() for an explicitly ordered layout.'
        )
    return ordered, chain_ids


def _associate_metrics(
    metric_files: Sequence[Path], csv_files: Sequence[Path]
) -> tuple[Path, ...]:
    if not metric_files:
        return ()
    if len(metric_files) != len(csv_files):
        raise ValueError(
            f'Explicit sample files contain {len(metric_files)} metric JSONs '
            f'for {len(csv_files)} chains. Supply one per chain or none.'
        )
    by_name = {path.name: path for path in metric_files}
    wanted = [_json_for_csv(path, 'metric').name for path in csv_files]
    if len(by_name) == len(metric_files) and set(wanted) == set(by_name):
        return tuple(by_name[name] for name in wanted)
    raise ValueError(
        'Non-standard metric JSONs names found. Call '
        'CmdStanMCMC.from_files() with an explicitly ordered '
        'metric_files list instead.'
    )


def _explicit_sample(
    configs: Sequence[ParsedConfig],
    csv_files: Sequence[Path],
    metric_files: Sequence[Path],
) -> FitFiles:
    ordered_csvs: tuple[Path, ...]
    ordered_configs: tuple[Path, ...]
    chain_ids: tuple[int, ...]
    if len(configs) == 1:
        ordered_csvs, chain_ids = _single_config_sample(configs[0], csv_files)
        ordered_configs = (configs[0].path,)
    elif len(configs) == len(csv_files):
        if any(_num_chains(config) != 1 for config in configs):
            raise ValueError(
                'Explicit per-chain sample configs must each record '
                'num_chains=1.'
            )
        associated = _associate_per_chain_configs(configs, csv_files)
        ids = [chain_id for chain_id, _, _ in associated]
        if len(set(ids)) != len(ids):
            raise ValueError(
                'Explicit sample config JSONs have duplicate chain IDs.'
            )
        associated.sort(key=lambda value: value[0])
        chain_ids = tuple(value[0] for value in associated)
        ordered_configs = tuple(value[1].path for value in associated)
        ordered_csvs = tuple(value[2] for value in associated)
    else:
        raise ValueError(
            f'Explicit sample loading requires either one multi-chain config '
            f'JSON or one config per draw CSV; found {len(configs)} config '
            f'JSONs and {len(csv_files)} draw CSVs. Alternatively call '
            'CmdStanMCMC.from_files().'
        )
    return FitFiles(
        method='sample',
        csv_files=ordered_csvs,
        config_files=ordered_configs,
        metric_files=_associate_metrics(metric_files, ordered_csvs),
        chain_ids=chain_ids,
    )


def _unique_csv_for_config(
    config: ParsedConfig, csv_files: Sequence[Path], role: str
) -> Path:
    matches = [path for path in csv_files if _config_csv_matches(config, path)]
    if len(matches) != 1:
        raise ValueError(
            f'Cannot identify the {role} CSV associated with config JSON '
            f'{config.path} among the explicitly supplied files. Use the '
            'method-specific from_files() constructor for explicit association.'
        )
    return matches[0]


def _explicit_laplace(
    configs: Sequence[ParsedConfig], csv_files: Sequence[Path]
) -> FitFiles:
    laplace_configs = [
        config
        for config in configs
        if config.value.method_config.method == 'laplace'
    ]
    optimize_configs = [
        config
        for config in configs
        if config.value.method_config.method == 'optimize'
    ]
    if (
        len(laplace_configs) != 1
        or len(optimize_configs) != 1
        or len(configs) != 2
    ):
        raise ValueError(
            'Explicit Laplace loading requires exactly one Laplace config JSON '
            'and the optimization mode config JSON. Alternatively call '
            'CmdStanLaplace.from_files() with an explicit mode object.'
        )
    if len(csv_files) != 2:
        raise ValueError(
            'Explicit Laplace loading requires the Laplace CSV and the '
            f'optimization mode CSV; found {len(csv_files)} CSV files.'
        )
    laplace = laplace_configs[0]
    optimize = optimize_configs[0]
    _validate_mode_compatible(laplace, optimize)
    laplace_csv = _unique_csv_for_config(laplace, csv_files, 'Laplace')
    mode_name = _laplace_mode_name(laplace)
    mode_matches = [path for path in csv_files if path.name == mode_name]
    if len(mode_matches) == 1:
        mode_csv = mode_matches[0]
    else:
        mode_csv = _unique_csv_for_config(optimize, csv_files, 'mode')
    if mode_csv == laplace_csv:
        raise ValueError('Laplace CSV and optimization mode CSV must differ.')
    mode = FitFiles(
        method='optimize',
        csv_files=(mode_csv,),
        config_files=(optimize.path,),
    )
    return FitFiles(
        method='laplace',
        csv_files=(laplace_csv,),
        config_files=(laplace.path,),
        mode_files=mode,
    )


def _parse_explicit_fit(files: Sequence[Path]) -> FitFiles:
    """Parse an explicit fit without inspecting any undeclared siblings."""
    if not files:
        raise ValueError('No output files provided.')
    csv_files: list[Path] = []
    config_files: list[Path] = []
    metric_files: list[Path] = []
    for path in files:
        if not path.is_file():
            raise ValueError(f'Output file not found: {path}')
        if path.name.endswith(_METRIC_SUFFIX):
            metric_files.append(path)
        elif path.suffix == '.json':
            config_files.append(path)
        elif path.suffix == '.csv':
            csv_files.append(path)
        else:
            raise ValueError(
                f'Unrecognized output file: {path}. Supply only draw CSVs, '
                'config JSONs, and optional metric JSONs.'
            )
    if len(set(files)) != len(files):
        raise ValueError('Explicit output files contain duplicate paths.')
    if not csv_files:
        raise ValueError('No draw CSV files were supplied.')
    if not config_files:
        raise ValueError('No CmdStan config JSON was supplied.')
    configs = [_read_config(path) for path in config_files]
    methods = {config.value.method_config.method for config in configs}
    if 'laplace' in methods:
        if metric_files:
            raise ValueError(
                'Metric JSONs are not Laplace reconstruction files.'
            )
        return _explicit_laplace(configs, csv_files)
    if len(methods) != 1:
        details = ', '.join(
            f'{config.path} ({config.value.method_config.method})'
            for config in configs
        )
        raise ValueError(
            'Explicit config JSONs use different methods: ' + details
        )
    method = methods.pop()
    if method not in _VALID_METHODS:
        raise ValueError(
            f'Unsupported CmdStan method in explicit files: {method}'
        )
    if method == 'sample':
        return _explicit_sample(configs, csv_files, metric_files)
    if len(csv_files) != 1 or len(configs) != 1 or metric_files:
        raise ValueError(
            f'Explicit {method} loading requires one output CSV and one config '
            f'JSON, with no metric JSONs; found {len(csv_files)} CSVs, '
            f'{len(configs)} configs, and {len(metric_files)} metrics.'
        )
    return FitFiles(
        method=method,
        csv_files=(csv_files[0],),
        config_files=(configs[0].path,),
    )


def from_output_files(
    path: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
    method: str | None = None,
) -> AnyStanFit:
    """
    Instantiate a fit from the output files of a CmdStan run.

    A string or path-like argument discovers a fit using CmdStanPy's output
    conventions. It may name a directory containing exactly one fit, a config
    JSON, or a Stan CSV with its config JSON alongside it. Output filenames are
    taken from the config and resolved beside it, allowing a bundle to be moved
    but not renamed. A sampling chain anchor discovers the complete fit.

    A non-string sequence uses **explicit loading**. Every required draw CSV
    and config JSON must be listed (plus both the Laplace and optimization-mode
    files for Laplace). Optional sampler metric JSONs may also be listed. No
    parent directory is searched and no unlisted sibling is opened.

    Glob strings and CSV-only explicit lists are not supported. For layouts
    whose files cannot be associated unambiguously, call the method-specific
    ``from_files()`` constructor and provide ordering or ``chain_ids`` there.

    Examples
    --------
    Discover a fit from a CmdStanPy output bundle::

        fit = from_output_files("results/model_1_config.json")

    Explicit nonstandard or direct-CmdStan bundle::

        fit = from_output_files([
            "results/output_1.csv",
            "results/output_2.csv",
            "results/output_config.json",
        ])

    :param path: directory/anchor path, or an explicit sequence of files
    :param method: optional method name assertion
    :return: fit object corresponding to the recorded CmdStan method
    """
    if path is None:
        raise ValueError('Must specify path to the CmdStan output files.')
    if method is not None and method not in _VALID_METHODS:
        raise ValueError(
            f'Bad method argument {method}, must be one of: '
            + ', '.join(f'"{name}"' for name in _VALID_METHODS)
        )
    try:
        if isinstance(path, (str, os.PathLike)):
            manifest = _discover_fit(Path(path))
        else:
            manifest = _parse_explicit_fit(tuple(Path(item) for item in path))
        if method is not None and manifest.method != method:
            raise ValueError(
                f'Expecting CmdStan output files from method {method}, found '
                f'outputs from method {manifest.method}'
            )
        match manifest.method:
            case 'sample':
                return CmdStanMCMC.from_files(
                    csv_files=manifest.csv_files,
                    config_files=manifest.config_files,
                    metric_files=manifest.metric_files or None,
                    chain_ids=manifest.chain_ids,
                )
            case 'optimize':
                return CmdStanMLE.from_files(
                    manifest.csv_files[0], manifest.config_files[0]
                )
            case 'variational':
                return CmdStanVB.from_files(
                    manifest.csv_files[0], manifest.config_files[0]
                )
            case 'pathfinder':
                return CmdStanPathfinder.from_files(
                    manifest.csv_files[0], manifest.config_files[0]
                )
            case 'laplace':
                assert manifest.mode_files is not None
                mode_fit = CmdStanMLE.from_files(
                    manifest.mode_files.csv_files[0],
                    manifest.mode_files.config_files[0],
                )
                return CmdStanLaplace.from_files(
                    manifest.csv_files[0],
                    manifest.config_files[0],
                    mode=mode_fit,
                )
            case _:
                raise ValueError(
                    f'Unsupported CmdStan method: {manifest.method}'
                )
    except OSError as exc:
        raise ValueError(
            f'An error occurred processing the output files:\n\t{exc}'
        ) from exc


def from_csv(
    path: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
    method: str | None = None,
) -> AnyStanFit:
    """Load a fit from Stan CSV files; deprecated alias for from_output_files.

    In addition to the path forms supported by :func:`from_output_files`, this
    compatibility wrapper accepts legacy CSV-only lists and glob patterns. CSV
    lists are supplemented with their adjacent config JSON files before being
    passed to :func:`from_output_files`.
    """
    warnings.warn(
        "from_csv is deprecated and will be removed in a future release; "
        "use from_output_files instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    csv_files: tuple[Path, ...] | None = None
    if isinstance(path, str) and '*' in path:
        csv_files = tuple(Path(file) for file in glob.glob(path))
    elif isinstance(path, Sequence) and not isinstance(
        path, (str, os.PathLike)
    ):
        csv_files = tuple(Path(file) for file in path)

    if csv_files is None:
        return from_output_files(path, method)
    if len(csv_files) == 1:
        return from_output_files(csv_files[0], method)
    return from_output_files(
        (*csv_files, *(_json_for_csv(file, 'config') for file in csv_files)),
        method,
    )
