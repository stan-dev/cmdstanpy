"""
    Container for the result of running a laplace approximation.
"""

from cmdstanpy.cmdstan_args import Method

# from cmdstanpy.utils.data_munging import extract_reshape
# from cmdstanpy.cmdstan_args import LaplaceArgs,
# from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet


class CmdStanLaplace:
    def __init__(self, runset: RunSet, mode: CmdStanMLE) -> None:
        """Initialize object."""
        if not runset.method == Method.LAPLACE:
            raise ValueError(
                'Wrong runset method, expecting laplace runset, '
                'found method {}'.format(runset.method)
            )
        self.runset = runset
        self.mode = mode
