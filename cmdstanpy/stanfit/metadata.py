"""Container for metadata parsed from the output of a CmdStan run"""

from __future__ import annotations

import json
import math
import os
from typing import Annotated, Any, Generic, Iterator, Literal

import stanio
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self, TypeVar

from cmdstanpy.utils import stancsv


class InferenceMetadata:
    """
    Names and structure of the output variables of a CmdStan run, parsed
    from the Stan CSV file's column header. Assumes valid CSV files.
    """

    def __init__(self, header: str) -> None:
        """Initialize object from a Stan CSV header line."""
        self._column_names = stancsv.parse_header(header)

        vars = stanio.parse_header(header)
        self._method_vars = {
            k: v for (k, v) in vars.items() if k.endswith('__')
        }
        self._stan_vars = {
            k: v for (k, v) in vars.items() if not k.endswith('__')
        }

    @classmethod
    def from_csv(
        cls, stan_csv: str | os.PathLike | Iterator[bytes]
    ) -> InferenceMetadata:
        try:
            header = stancsv.scan_header(stan_csv)
            if header is None:
                raise ValueError("No header line found")
            return cls(header)
        except Exception as exc:
            raise ValueError(
                f"An error occurred when parsing Stan csv {stan_csv}"
            ) from exc

    def __repr__(self) -> str:
        return 'Metadata:\n column_names={}\n'.format(self._column_names)

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._column_names

    @property
    def method_vars(self) -> dict[str, stanio.Variable]:
        """
        Method variable names always end in `__`, e.g. `lp__`.
        """
        return self._method_vars

    @property
    def stan_vars(self) -> dict[str, stanio.Variable]:
        """
        These are the user-defined variables in the Stan program.
        """
        return self._stan_vars


class MetricInfo(BaseModel):
    """Structured representation of HMC-NUTS metric information,
    as output by CmdStan"""

    stepsize: float
    metric_type: Literal["diag_e", "dense_e", "unit_e"]
    inv_metric: list[float] | list[list[float]]

    @field_validator("stepsize")
    @classmethod
    def validate_stepsize(cls, v: float) -> float:
        if not math.isnan(v) and v <= 0:
            raise ValueError("stepsize must be greater than 0 or NaN")
        return v

    @model_validator(mode="after")
    def validate_inv_metric_shape(self) -> MetricInfo:
        if not self.inv_metric:  # Empty inv_metric, e.g. from no parameters
            return self

        is_1d = isinstance(self.inv_metric[0], float)

        if self.metric_type in ("diag_e", "unit_e") and not is_1d:
            raise ValueError(
                "inv_metric must be 1D for diag_e and unit_e metric type"
            )
        if self.metric_type == "dense_e":
            if is_1d:
                raise ValueError("Dense inv_metric must be 2D")

            if any(not row for row in self.inv_metric):
                raise ValueError("Dense inv_metric cannot contain empty rows")

            n_rows = len(self.inv_metric)
            if not all(
                len(row) == n_rows for row in self.inv_metric  # type: ignore
            ):
                raise ValueError("Dense inv_metric must be square")

        return self


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["sample"] = "sample"
    algorithm: str
    num_samples: int
    num_warmup: int
    num_chains: int = Field(default=1, gt=0)
    save_warmup: bool = False
    thin: int = 1
    max_depth: int | None = None


class OptimizeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["optimize"] = "optimize"
    algorithm: str
    save_iterations: bool = False
    jacobian: bool = False


class PathfinderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["pathfinder"] = "pathfinder"
    num_draws: int = 1000
    num_paths: int = 4
    psis_resample: bool = True
    calculate_lp: bool = True


class LaplaceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["laplace"] = "laplace"
    mode: str
    draws: int = 1000
    jacobian: bool = True


class VariationalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["variational"] = "variational"
    algorithm: str
    iter: int = 10000
    grad_samples: int = 1
    elbo_samples: int = 100
    eta: float = 1.0
    tol_rel_obj: float = 0.01
    eval_elbo: int = 100
    output_samples: int = 1000


class GeneratedQuantitiesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["generate_quantities"] = "generate_quantities"
    fitted_params: str
    num_chains: int = 1


AnyMethodConfig = Annotated[
    SampleConfig
    | OptimizeConfig
    | PathfinderConfig
    | LaplaceConfig
    | VariationalConfig
    | GeneratedQuantitiesConfig,
    Discriminator("method"),
]

MethodT = TypeVar("MethodT", bound=BaseModel, default=AnyMethodConfig)


class OutputConfig(BaseModel):
    """Common CmdStan output settings used when reconstructing fits."""

    model_config = ConfigDict(extra="allow")

    file: str = ''
    sig_figs: int = -1


class StanConfig(BaseModel, Generic[MethodT]):
    """Common representation of a config JSON file output as part of a
    Stan inference run. Separate method-specific config classes handle
    the variation of output between methods."""

    model_config = ConfigDict(extra="allow")

    model_name: str
    stan_major_version: str
    stan_minor_version: str
    stan_patch_version: str
    stanc_version: str | None = None
    id: int | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)

    method_config: MethodT

    @classmethod
    def from_json(cls, json_data: str | bytes) -> Self:
        """Parse a CmdStan config JSON string into this config class."""
        raw = json.loads(json_data)
        config: Self = cls.model_validate(flatten_config(raw))
        return config


# Named classes for each method - these can be pickled unlike the generic
# StanConfig[MethodConfig]


# pylint: disable-next=too-few-public-methods
class SampleRunConfig(StanConfig[SampleConfig]):
    """Run configuration of the sample method."""


# pylint: disable-next=too-few-public-methods
class OptimizeRunConfig(StanConfig[OptimizeConfig]):
    """Run configuration of the optimize method."""


# pylint: disable-next=too-few-public-methods
class VariationalRunConfig(StanConfig[VariationalConfig]):
    """Run configuration of the variational method."""


# pylint: disable-next=too-few-public-methods
class LaplaceRunConfig(StanConfig[LaplaceConfig]):
    """Run configuration of the laplace method."""


# pylint: disable-next=too-few-public-methods
class PathfinderRunConfig(StanConfig[PathfinderConfig]):
    """Run configuration of the pathfinder method."""


# pylint: disable-next=too-few-public-methods
class GeneratedQuantitiesRunConfig(StanConfig[GeneratedQuantitiesConfig]):
    """Run configuration of the generate_quantities method."""


def flatten_value_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively flatten CmdStan's nested value/subdict structure.

    CmdStan uses a pattern where a field contains:
        {"value": "val", "val": {"k1": v1, "k2": v2}}

    This flattens it to the parent level as:
        {"field": "val", "k1": v1, "k2": v2}

    The flattening is applied recursively to any nested dicts.
    """
    result: dict[str, Any] = {}

    for key, val in data.items():
        if not isinstance(val, dict):
            result[key] = val
            continue

        if "value" in val:
            value_name = val['value']
            result[key] = value_name

            # Get the nested dict matching the value name and flatten it
            nested = val.get(value_name, {})
            if isinstance(nested, dict):
                flattened_nested = flatten_value_dict(nested)
                for nested_key, nested_val in flattened_nested.items():
                    if nested_key not in result:
                        result[nested_key] = nested_val
        else:
            # Regular dict without value pattern - recurse into it
            result[key] = flatten_value_dict(val)

    return result


def flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested CmdStan config JSON structure.

    CmdStan outputs config JSON with deeply nested structure like:
        {"method": {"value": "sample", "sample": {"num_samples": 1000, ...}}}

    This flattens it to:
        {"method_config": {"method": "sample", "num_samples": 1000, ...}, ...}
    """
    method_data = data.get('method')
    if not isinstance(method_data, dict):
        return data

    result = {k: v for k, v in data.items() if k != "method"}
    method_name = method_data.get('value')

    # Build method_config from the method-specific nested dict
    nested_method = method_data.get(method_name, {})
    method_config = flatten_value_dict(nested_method)
    method_config['method'] = method_name

    result['method_config'] = method_config
    return result


def parse_config(json_data: str | bytes) -> StanConfig[AnyMethodConfig]:
    """Parse a CmdStan config JSON string into a StanConfig, auto-detecting
    the method from the JSON content via the discriminated union
    ``AnyMethodConfig``.

    When the method is known in advance, use the ``from_json`` constructor
    of the corresponding named config class (e.g. ``SampleRunConfig``)
    instead; unlike the parametrized generic returned here, instances of
    the named classes support pickling.
    """

    raw = json.loads(json_data)
    flat = flatten_config(raw)
    return StanConfig[AnyMethodConfig].model_validate(flat)  # type: ignore
