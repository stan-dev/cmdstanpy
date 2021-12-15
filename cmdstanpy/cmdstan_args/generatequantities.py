"""Arguments for the generate quantities subcommand"""

import os
from typing import List, Optional


class GenerateQuantitiesArgs:
    """Arguments needed for generate_quantities method."""

    def __init__(self, csv_files: List[str]) -> None:
        """Initialize object."""
        self.sample_csv_files = csv_files

    def validate(
        self, chains: Optional[int] = None  # pylint: disable=unused-argument
    ) -> None:
        """
        Check arguments correctness and consistency.

        * check that sample csv files exist
        """
        for csv in self.sample_csv_files:
            if not os.path.exists(csv):
                raise ValueError(
                    'Invalid path for sample csv file: {}'.format(csv)
                )

    def compose(self, idx: int, cmd: List[str]) -> List[str]:
        """
        Compose CmdStan command for method-specific non-default arguments.
        """
        cmd.append('method=generate_quantities')
        cmd.append('fitted_params={}'.format(self.sample_csv_files[idx]))
        return cmd
