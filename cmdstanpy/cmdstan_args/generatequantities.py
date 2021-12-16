"""Arguments for the generate quantities subcommand"""

import os
from typing import List

from cmdstanpy.cmdstan_args.cmdstan import RunConfiguration

from .cmdstan import CmdStanArgs
from .util import Method


class GenerateQuantitiesArgs:
    """Arguments needed for generate_quantities method."""

    def __init__(self, args: CmdStanArgs, csv_files: List[str]) -> None:
        """Initialize object."""
        self.cmdstan_args = args
        self.sample_csv_files = csv_files
        self.validate()

    def validate(self) -> None:
        """
        Check arguments correctness and consistency.

        * check that sample csv files exist
        """
        for csv in self.sample_csv_files:
            if not os.path.exists(csv):
                raise ValueError(
                    'Invalid path for sample csv file: {}'.format(csv)
                )

    @classmethod
    def method(cls) -> Method:
        return Method.GENERATE_QUANTITIES

    def compose_command(self, rs: RunConfiguration, idx: int) -> List[str]:
        """
        Compose CmdStan command for method-specific non-default arguments.
        """
        cmd = self.cmdstan_args.begin_command(rs, idx)

        cmd.append('method=generate_quantities')
        cmd.append('fitted_params={}'.format(self.sample_csv_files[idx]))
        return cmd
