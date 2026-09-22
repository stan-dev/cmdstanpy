"""Metadata tests"""

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from cmdstanpy.stanfit import InferenceMetadata
from cmdstanpy.stanfit.metadata import (
    GeneratedQuantitiesConfig,
    LaplaceConfig,
    MetricInfo,
    OptimizeConfig,
    PathfinderConfig,
    SampleConfig,
    StanConfig,
    VariationalConfig,
    parse_config,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')

DATAFILES_PATH = os.path.join(HERE, 'data')
GOODFILES_PATH = os.path.join(DATAFILES_PATH, 'runset-good')
BADFILES_PATH = os.path.join(DATAFILES_PATH, 'runset-bad')


def make_config_output(
    method_name: str, method_body: dict[str, Any]
) -> dict[str, Any]:
    return {
        'stan_major_version': '2',
        'stan_minor_version': '37',
        'stan_patch_version': '0',
        'model_name': 'mu_model',
        'start_datetime': '2026-01-23 22:20:01 UTC',
        'method': {
            'value': method_name,
            method_name: method_body,
        },
        'id': 1,
        'data': {'file': ''},
        'init': '2',
        'random': {'seed': 12345},
        'output': {
            'file': '/tmp/mu.csv',
            'diagnostic_file': '',
            'refresh': 100,
            'sig_figs': 8,
            'profile_file': 'profile.csv',
            'save_cmdstan_config': True,
        },
        'num_threads': 4,
        'mpi_enabled': False,
        'stanc_version': 'stanc3 v2.37.0',
        'stancflags': '--filename-in-msg=mu.stan',
    }


CONFIG_OUTPUT_CASES: list[
    tuple[str, dict[str, Any], type[Any], dict[str, Any]]
] = [
    (
        'sample',
        make_config_output(
            'sample',
            {
                'num_samples': 1000,
                'num_warmup': 1000,
                'save_warmup': False,
                'thin': 1,
                'adapt': {
                    'engaged': True,
                    'gamma': 0.05,
                    'delta': 0.8,
                    'kappa': 0.75,
                    't0': 10,
                    'init_buffer': 75,
                    'term_buffer': 50,
                    'window': 25,
                    'save_metric': True,
                },
                'algorithm': {
                    'value': 'hmc',
                    'hmc': {
                        'engine': {
                            'value': 'nuts',
                            'nuts': {'max_depth': 10},
                        },
                        'metric': {'value': 'diag_e'},
                        'metric_file': '',
                        'stepsize': 1,
                        'stepsize_jitter': 0,
                    },
                },
                'num_chains': 4,
            },
        ),
        SampleConfig,
        {
            'algorithm': 'hmc',
            'num_samples': 1000,
            'num_warmup': 1000,
            'save_warmup': False,
            'thin': 1,
            'max_depth': 10,
        },
    ),
    (
        'optimize',
        make_config_output(
            'optimize',
            {
                'algorithm': {
                    'value': 'lbfgs',
                    'lbfgs': {
                        'init_alpha': 0.001,
                        'tol_obj': 1e-12,
                        'tol_rel_obj': 10000,
                        'tol_grad': 1e-08,
                        'tol_rel_grad': 10000000,
                        'tol_param': 1e-08,
                        'history_size': 5,
                    },
                },
                'jacobian': False,
                'iter': 2000,
                'save_iterations': False,
            },
        ),
        OptimizeConfig,
        {
            'algorithm': 'lbfgs',
            'jacobian': False,
            'save_iterations': False,
        },
    ),
    (
        'variational',
        make_config_output(
            'variational',
            {
                'algorithm': {
                    'value': 'meanfield',
                    'meanfield': {},
                },
                'iter': 10000,
                'grad_samples': 1,
                'elbo_samples': 100,
                'eta': 1,
                'adapt': {
                    'engaged': True,
                    'iter': 50,
                },
                'tol_rel_obj': 0.01,
                'eval_elbo': 100,
                'output_samples': 1000,
            },
        ),
        VariationalConfig,
        {
            'algorithm': 'meanfield',
            'iter': 10000,
            'grad_samples': 1,
            'elbo_samples': 100,
            'eta': 1.0,
        },
    ),
    (
        'pathfinder',
        make_config_output(
            'pathfinder',
            {
                'init_alpha': 0.001,
                'tol_obj': 1e-12,
                'tol_rel_obj': 10000,
                'tol_grad': 1e-08,
                'tol_rel_grad': 10000000,
                'tol_param': 1e-08,
                'history_size': 5,
                'num_psis_draws': 1000,
                'num_paths': 4,
                'save_single_paths': False,
                'psis_resample': True,
                'calculate_lp': True,
                'max_lbfgs_iters': 1000,
                'num_draws': 1000,
                'num_elbo_draws': 25,
            },
        ),
        PathfinderConfig,
        {
            'num_draws': 1000,
            'num_paths': 4,
            'psis_resample': True,
            'calculate_lp': True,
        },
    ),
    (
        'laplace',
        make_config_output(
            'laplace',
            {
                'mode': '/tmp/mu-opt.csv',
                'jacobian': True,
                'draws': 1000,
                'calculate_lp': True,
            },
        ),
        LaplaceConfig,
        {
            'mode': '/tmp/mu-opt.csv',
            'draws': 1000,
            'jacobian': True,
        },
    ),
    (
        'generate_quantities',
        make_config_output(
            'generate_quantities',
            {
                'fitted_params': '/tmp/mu-fit.csv',
                'num_chains': 1,
            },
        ),
        GeneratedQuantitiesConfig,
        {
            'fitted_params': '/tmp/mu-fit.csv',
            'num_chains': 1,
        },
    ),
]


def test_good() -> None:
    csv_file = os.path.join(DATAFILES_PATH, 'runset-good', 'bern-1.csv')
    metadata = InferenceMetadata.from_csv(csv_file)

    hmc_vars = {
        'lp__',
        'accept_stat__',
        'stepsize__',
        'treedepth__',
        'n_leapfrog__',
        'divergent__',
        'energy__',
    }
    assert hmc_vars == metadata.method_vars.keys()
    assert {'theta'} == metadata.stan_vars.keys()
    assert metadata.column_names == (
        'lp__',
        'accept_stat__',
        'stepsize__',
        'treedepth__',
        'n_leapfrog__',
        'divergent__',
        'energy__',
        'theta',
    )


@pytest.mark.parametrize(
    'method_name, config_json, expected_type, expected_fields',
    CONFIG_OUTPUT_CASES,
)
def test_parse_config_outputs(
    method_name: str,
    config_json: dict[str, Any],
    expected_type: type[Any],
    expected_fields: dict[str, Any],
) -> None:
    parsed = parse_config(json.dumps(config_json))

    assert isinstance(parsed, StanConfig)
    assert parsed.model_name == 'mu_model'
    assert parsed.stan_major_version == '2'
    assert parsed.stan_minor_version == '37'
    assert parsed.stan_patch_version == '0'
    assert isinstance(parsed.method_config, expected_type)
    assert parsed.method_config.method == method_name

    dumped = parsed.method_config.model_dump()
    for key, expected in expected_fields.items():
        assert dumped[key] == expected


def test_parse_config_accepts_bytes() -> None:
    config_json = make_config_output(
        'generate_quantities',
        {
            'fitted_params': '/tmp/mu-fit.csv',
            'num_chains': 1,
        },
    )

    parsed = parse_config(json.dumps(config_json).encode())

    assert isinstance(parsed.method_config, GeneratedQuantitiesConfig)
    assert parsed.method_config.fitted_params == '/tmp/mu-fit.csv'


class TestMetricInfoValidators:
    """Test custom validators for MetricInfo model"""

    def test_valid_diag_e_metric(self) -> None:
        """Test valid diag_e metric with 1D list"""
        metric = MetricInfo(
            stepsize=0.5,
            metric_type="diag_e",
            inv_metric=[1.0, 2.0, 3.0],
        )
        assert metric.stepsize == 0.5
        assert isinstance(metric.inv_metric, list)
        assert isinstance(metric.inv_metric[0], float)
        assert len(metric.inv_metric) == 3

    def test_valid_unit_e_metric(self) -> None:
        """Test valid unit_e metric with 1D list"""
        metric = MetricInfo(
            stepsize=0.1,
            metric_type="unit_e",
            inv_metric=[1.0, 1.0, 1.0],
        )
        assert metric.metric_type == "unit_e"
        assert isinstance(metric.inv_metric[0], float)
        assert len(metric.inv_metric) == 3

    def test_valid_dense_e_metric(self) -> None:
        """Test valid dense_e metric with 2D square list"""
        metric = MetricInfo(
            stepsize=0.3,
            metric_type="dense_e",
            inv_metric=[[1.0, 0.5], [0.5, 1.0]],
        )
        assert metric.metric_type == "dense_e"
        assert isinstance(metric.inv_metric[0], list)
        assert len(metric.inv_metric) == 2
        assert len(metric.inv_metric[0]) == 2

    def test_inv_metric_stays_as_list(self) -> None:
        """Test that inv_metric remains as list type"""
        metric = MetricInfo(
            stepsize=0.5,
            metric_type="diag_e",
            inv_metric=[1.0, 2.0, 3.0],
        )
        assert isinstance(metric.inv_metric, list)

    def test_inv_metric_nested_list(self) -> None:
        """Test that inv_metric handles nested lists correctly"""
        metric = MetricInfo(
            stepsize=0.5,
            metric_type="dense_e",
            inv_metric=[[1.0, 0.0], [0.0, 1.0]],
        )
        assert isinstance(metric.inv_metric, list)
        assert isinstance(metric.inv_metric[0], list)

    def test_stepsize_positive(self) -> None:
        """Test valid positive stepsize"""
        metric = MetricInfo(
            stepsize=0.5,
            metric_type="diag_e",
            inv_metric=[1.0],
        )
        assert metric.stepsize == 0.5

    def test_stepsize_nan_allowed(self) -> None:
        """Test that NaN stepsize is allowed"""
        metric = MetricInfo(
            stepsize=math.nan,
            metric_type="diag_e",
            inv_metric=[1.0],
        )
        assert math.isnan(metric.stepsize)

    def test_stepsize_zero_raises_error(self) -> None:
        """Test that zero stepsize raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=0.0,
                metric_type="diag_e",
                inv_metric=[1.0],
            )
        assert "stepsize must be greater than 0 or NaN" in str(exc_info.value)

    def test_stepsize_negative_raises_error(self) -> None:
        """Test that negative stepsize raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=-0.5,
                metric_type="diag_e",
                inv_metric=[1.0],
            )
        assert "stepsize must be greater than 0 or NaN" in str(exc_info.value)

    def test_diag_e_with_2d_list_raises_error(self) -> None:
        """Test that diag_e with 2D list raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=0.5,
                metric_type="diag_e",
                inv_metric=[[1.0, 2.0]],
            )
        assert "inv_metric must be 1D for diag_e and unit_e" in str(
            exc_info.value
        )

    def test_unit_e_with_2d_list_raises_error(self) -> None:
        """Test that unit_e with 2D list raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=0.5,
                metric_type="unit_e",
                inv_metric=[[1.0], [1.0]],
            )
        assert "inv_metric must be 1D for diag_e and unit_e" in str(
            exc_info.value
        )

    def test_dense_e_with_1d_list_raises_error(self) -> None:
        """Test that dense_e with 1D list raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=0.5,
                metric_type="dense_e",
                inv_metric=[1.0, 2.0],
            )
        assert "Dense inv_metric must be 2D" in str(exc_info.value)

    def test_dense_e_non_square_raises_error(self) -> None:
        """Test that dense_e with non-square list raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=0.5,
                metric_type="dense_e",
                inv_metric=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            )
        assert "Dense inv_metric must be square" in str(exc_info.value)


class TestMetricInfoModelValidateJson:
    """Test model_validate_json class method"""

    def test_from_json_diag_e(self) -> None:
        """Test loading diag_e metric from JSON file"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(
                {
                    'stepsize': 0.5,
                    'metric_type': 'diag_e',
                    'inv_metric': [1.0, 2.0, 3.0],
                },
                f,
            )
            temp_path = f.name

        try:
            with open(temp_path) as f:
                metric = MetricInfo.model_validate_json(f.read())
            assert metric.stepsize == 0.5
            assert metric.metric_type == "diag_e"
            assert metric.inv_metric == [1.0, 2.0, 3.0]
        finally:
            Path(temp_path).unlink()

    def test_from_json_dense_e(self) -> None:
        """Test loading dense_e metric from JSON file"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(
                {
                    'stepsize': 0.3,
                    'metric_type': 'dense_e',
                    'inv_metric': [[1.0, 0.5], [0.5, 1.0]],
                },
                f,
            )
            temp_path = f.name

        try:
            with open(temp_path) as f:
                metric = MetricInfo.model_validate_json(f.read())
            assert metric.stepsize == 0.3
            assert metric.metric_type == "dense_e"
            assert len(metric.inv_metric) == 2
            assert len(metric.inv_metric[0]) == 2  # type: ignore
        finally:
            Path(temp_path).unlink()

    def test_from_json_invalid_data_raises_error(self) -> None:
        """Test that invalid data in JSON raises ValidationError"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(
                {
                    'stepsize': -0.5,  # Invalid: negative stepsize
                    'metric_type': 'diag_e',
                    'inv_metric': [1.0, 2.0, 3.0],
                },
                f,
            )
            temp_path = f.name

        try:
            with pytest.raises(ValidationError):
                with open(temp_path) as f:
                    MetricInfo.model_validate_json(f.read())
        finally:
            Path(temp_path).unlink()

    def test_from_json_pathlike(self) -> None:
        """Test from_json works with PathLike objects"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(
                {
                    'stepsize': 0.5,
                    'metric_type': 'unit_e',
                    'inv_metric': [1.0, 1.0],
                },
                f,
            )
            temp_path = Path(f.name)

        try:
            with open(temp_path) as f:
                metric = MetricInfo.model_validate_json(f.read())
            assert metric.metric_type == "unit_e"
        finally:
            temp_path.unlink()

    def test_invalid_metric_type_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            MetricInfo(
                stepsize=0.5,
                metric_type="not_a_metric",  # type: ignore
                inv_metric=[1.0],
            )

    def test_from_json_invalid_metric_type_raises_error(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(
                {
                    'stepsize': 0.5,
                    'metric_type': 'not_a_metric',
                    'inv_metric': [1.0, 2.0, 3.0],
                },
                f,
            )
            temp_path = f.name

        try:
            with pytest.raises(ValidationError):
                with open(temp_path) as fh:
                    MetricInfo.model_validate_json(fh.read())
        finally:
            Path(temp_path).unlink()
