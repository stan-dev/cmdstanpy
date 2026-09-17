"""Laplace output association tests using saved fixtures (no CmdStan needed)."""

import json
import shutil
from pathlib import Path

import pytest

from cmdstanpy.stanfit import CmdStanLaplace, from_output_files


@pytest.fixture(name='bundle')
def bundle_fixture(tmp_path: Path) -> Path:
    source = Path(__file__).parent / 'data' / 'laplace'
    for path in source.iterdir():
        shutil.copy(path, tmp_path / path.name)
    return tmp_path


def _update_config(bundle: Path, name: str, field: str, value: object) -> None:
    path = bundle / f'rosenbrock_{name}_config.json'
    config = json.loads(path.read_text())
    if field == 'jacobian':
        config['method']['optimize']['jacobian'] = value
    elif field == 'mode':
        config['method']['laplace']['mode'] = value
    else:
        config[field] = value
    path.write_text(json.dumps(config))


@pytest.mark.parametrize(
    'anchor', ['', 'rosenbrock_laplace.csv', 'rosenbrock_laplace_config.json']
)
def test_discover_relocated_bundle(bundle: Path, anchor: str) -> None:
    fit = from_output_files(bundle / anchor)
    assert isinstance(fit, CmdStanLaplace)
    assert Path(fit.mode.csv_file) == bundle / 'rosenbrock_opt.csv'


def test_explicit_files_do_not_discover_siblings(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = sorted(bundle.iterdir())
    # Nonstandard config filenames must work when supplied explicitly.
    for index, path in enumerate(files):
        if path.suffix == '.json':
            files[index] = path.rename(path.with_name(f'config_{index}.json'))
    allowed = set(files)
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        assert path in allowed, f'Opened unlisted file: {path}'
        return original_read_bytes(path)

    def no_glob(*_args: object, **_kwargs: object) -> None:
        pytest.fail('Explicit loading must not search for sibling files')

    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    monkeypatch.setattr(Path, 'glob', no_glob)
    fit = from_output_files(files)
    assert isinstance(fit, CmdStanLaplace)
    assert Path(fit.mode.csv_file) == bundle / 'rosenbrock_opt.csv'


def test_explicit_mode_falls_back_to_optimize_output_name(bundle: Path) -> None:
    _update_config(bundle, 'laplace', 'mode', '/stale/renamed_mode.csv')
    fit = from_output_files(sorted(bundle.iterdir()))
    assert isinstance(fit, CmdStanLaplace)
    assert Path(fit.mode.csv_file) == bundle / 'rosenbrock_opt.csv'
    with pytest.raises(ValueError, match='optimization mode CSV/config JSON'):
        from_output_files(bundle / 'rosenbrock_laplace.csv')


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize(
    'field,value,message',
    [
        ('model_name', 'other_model', 'same model and Stan version'),
        ('stan_major_version', '3', 'same model and Stan version'),
        ('stan_minor_version', '38', 'same model and Stan version'),
        ('stan_patch_version', '1', 'same model and Stan version'),
        ('jacobian', False, 'jacobian setting'),
    ],
)
def test_incompatible_mode(
    bundle: Path, explicit: bool, field: str, value: object, message: str
) -> None:
    _update_config(bundle, 'opt', field, value)
    with pytest.raises(ValueError, match=message):
        from_output_files(
            sorted(bundle.iterdir())
            if explicit
            else bundle / 'rosenbrock_laplace_config.json'
        )


def test_explicit_mode_must_differ_from_laplace_csv(bundle: Path) -> None:
    _update_config(bundle, 'laplace', 'mode', 'rosenbrock_laplace.csv')
    with pytest.raises(ValueError, match='must differ'):
        from_output_files(sorted(bundle.iterdir()))


@pytest.mark.parametrize(
    'missing', ['rosenbrock_opt.csv', 'rosenbrock_opt_config.json']
)
def test_explicit_missing_mode_file(bundle: Path, missing: str) -> None:
    files = [path for path in bundle.iterdir() if path.name != missing]
    with pytest.raises(ValueError, match='Explicit Laplace loading requires'):
        from_output_files(files)
