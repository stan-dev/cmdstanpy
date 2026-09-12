"""
Container for the result of running Pathfinder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from cmdstanpy.stanfit.base import SingleFileFit
from cmdstanpy.stanfit.metadata import PathfinderConfig, PathfinderRunConfig


@dataclass(kw_only=True)
class CmdStanPathfinder(SingleFileFit[PathfinderConfig]):
    """
    Container for outputs from the Pathfinder algorithm.
    Created by :meth:`CmdStanModel.pathfinder()`.
    """

    @classmethod
    def from_files(
        cls,
        csv_file: str | os.PathLike,
        config_file: str | os.PathLike,
        stdout_file: str | os.PathLike | None = None,
    ) -> CmdStanPathfinder:
        return cls(
            **cls._from_files_kwargs(
                csv_file, config_file, stdout_file, PathfinderRunConfig
            )
        )

    def __repr__(self) -> str:
        lines = [
            f'CmdStanPathfinder: model={self.model_name}',
            f' csv_file:\n\t{self.csv_file}',
        ]
        if self.config_file is not None:
            lines.append(f' config_file:\n\t{self.config_file}')
        if self.stdout_file is not None:
            lines.append(f' output_file:\n\t{self.stdout_file}')
        return '\n'.join(lines)

    def draws(self) -> np.ndarray:
        """
        Return a numpy.ndarray containing the draws from the
        approximate posterior distribution. This is a 2-D array
        of shape (draws, parameters).
        """
        self._assemble()
        return self._draws

    @property
    def is_resampled(self) -> bool:
        """
        Returns True if the draws were resampled from several Pathfinder
        approximations, False otherwise.
        """
        return (
            self.config.method_config.num_paths > 1
            and self.config.method_config.psis_resample
            and self.config.method_config.calculate_lp
        )
