"""Top-level argument object for CmdStan"""

import os
from time import time
from typing import Any, List, Mapping, Optional, Protocol, Union

from numpy.random import RandomState

from cmdstanpy.utils import cmdstan_path, cmdstan_version_before, get_logger

from .util import Method


class RunConfiguration(Protocol):
    """
    A protocol mirroring the necessary parts of RunSet to make mypy happy.
    Cannot use RunSet itself due to circular import problem
    """

    @property
    def one_process_per_chain(self) -> bool:
        ...

    @property
    def chains(self) -> int:
        ...

    def get_csv_file(self, idx: int) -> str:
        ...

    def get_diagnostic_file(self, idx: int) -> str:
        ...

    def get_profile_file(self, idx: int) -> str:
        ...


class Args(Protocol):
    cmdstan_args: "CmdStanArgs"

    def compose_command(self, rs: RunConfiguration, idx: int) -> List[str]:
        ...

    @classmethod
    def method(cls) -> Method:
        ...


class CmdStanArgs:
    """
    Container for CmdStan command line arguments.
    Consists of arguments common to all methods and
    and an object which contains the method-specific arguments.
    """

    def __init__(
        self,
        model_name: str,
        model_exe: Optional[str],
        chain_ids: Union[List[int], None],
        data: Union[Mapping[str, Any], str, None] = None,
        seed: Union[int, List[int], None] = None,
        inits: Union[int, float, str, List[str], None] = None,
        output_dir: Optional[str] = None,
        sig_figs: Optional[int] = None,
        save_latent_dynamics: bool = False,
        save_profile: bool = False,
        refresh: Optional[int] = None,
    ) -> None:
        """Initialize object."""
        self.model_name = model_name
        self.model_exe = model_exe
        self.chain_ids = chain_ids
        self.data = data
        self.seed = seed
        self.inits = inits
        self.output_dir = output_dir
        self.sig_figs = sig_figs
        self.save_latent_dynamics = save_latent_dynamics
        self.save_profile = save_profile
        self.refresh = refresh

        self.validate()

    def validate(self) -> None:
        """
        Check arguments correctness and consistency.

        * input files must exist
        * output files must be in a writeable directory
        * if no seed specified, set random seed.
        * length of per-chain lists equals specified # of chains
        """
        if self.model_name is None:
            raise ValueError('no stan model specified')
        if self.model_exe is None:
            raise ValueError('model not compiled')

        if self.chain_ids is not None:
            for chain_id in self.chain_ids:
                if chain_id < 1:
                    raise ValueError('invalid chain_id {}'.format(chain_id))
        if self.output_dir is not None:
            self.output_dir = os.path.realpath(
                os.path.expanduser(self.output_dir)
            )
            if not os.path.exists(self.output_dir):
                try:
                    os.makedirs(self.output_dir)
                    get_logger().info(
                        'created output directory: %s', self.output_dir
                    )
                except (RuntimeError, PermissionError) as exc:
                    raise ValueError(
                        'Invalid path for output files, '
                        'no such dir: {}.'.format(self.output_dir)
                    ) from exc
            if not os.path.isdir(self.output_dir):
                raise ValueError(
                    'Specified output_dir is not a directory: {}.'.format(
                        self.output_dir
                    )
                )
            try:
                testpath = os.path.join(self.output_dir, str(time()))
                with open(testpath, 'w+'):
                    pass
                os.remove(testpath)  # cleanup
            except Exception as exc:
                raise ValueError(
                    'Invalid path for output files,'
                    ' cannot write to dir: {}.'.format(self.output_dir)
                ) from exc
        if self.refresh is not None:
            if not isinstance(self.refresh, int) or self.refresh < 1:
                raise ValueError(
                    'Argument "refresh" must be a positive integer value, '
                    'found {}.'.format(self.refresh)
                )

        if self.sig_figs is not None:
            if (
                not isinstance(self.sig_figs, int)
                or self.sig_figs < 1
                or self.sig_figs > 18
            ):
                raise ValueError(
                    'Argument "sig_figs" must be an integer between 1 and 18,'
                    ' found {}'.format(self.sig_figs)
                )
            # TODO: remove at some future release
            if cmdstan_version_before(2, 25):
                self.sig_figs = None
                get_logger().warning(
                    'Argument "sig_figs" invalid for CmdStan versions < 2.25, '
                    'using version %s in directory %s',
                    os.path.basename(cmdstan_path()),
                    os.path.dirname(cmdstan_path()),
                )

        if self.seed is None:
            rng = RandomState()
            self.seed = rng.randint(1, 99999 + 1)
        else:
            if not isinstance(self.seed, (int, list)):
                raise ValueError(
                    'Argument "seed" must be an integer between '
                    '0 and 2**32-1, found {}.'.format(self.seed)
                )
            if isinstance(self.seed, int):
                if self.seed < 0 or self.seed > 2 ** 32 - 1:
                    raise ValueError(
                        'Argument "seed" must be an integer between '
                        '0 and 2**32-1, found {}.'.format(self.seed)
                    )
            else:
                if self.chain_ids is None:
                    raise ValueError(
                        'List of per-chain seeds cannot be evaluated without '
                        'corresponding list of chain_ids.'
                    )
                if len(self.seed) != len(self.chain_ids):
                    raise ValueError(
                        'Number of seeds must match number of chains,'
                        ' found {} seed for {} chains.'.format(
                            len(self.seed), len(self.chain_ids)
                        )
                    )
                for seed in self.seed:
                    if seed < 0 or seed > 2 ** 32 - 1:
                        raise ValueError(
                            'Argument "seed" must be an integer value'
                            ' between 0 and 2**32-1,'
                            ' found {}'.format(seed)
                        )

        if isinstance(self.data, str):
            if not os.path.exists(self.data):
                raise ValueError('no such file {}'.format(self.data))
        elif self.data is not None and not isinstance(self.data, (str, dict)):
            raise ValueError('Argument "data" must be string or dict')

        if self.inits is not None:
            if isinstance(self.inits, (float, int)):
                if self.inits < 0:
                    raise ValueError(
                        'Argument "inits" must be > 0, found {}'.format(
                            self.inits
                        )
                    )
            elif isinstance(self.inits, str):
                if not os.path.exists(self.inits):
                    raise ValueError('no such file {}'.format(self.inits))
            elif isinstance(self.inits, list):
                if self.chain_ids is None:
                    raise ValueError(
                        'List of inits files cannot be evaluated without '
                        'corresponding list of chain_ids.'
                    )

                if len(self.inits) != len(self.chain_ids):
                    raise ValueError(
                        'Number of inits files must match number of chains,'
                        ' found {} inits files for {} chains.'.format(
                            len(self.inits), len(self.chain_ids)
                        )
                    )
                for inits in self.inits:
                    if not os.path.exists(inits):
                        raise ValueError('no such file {}'.format(inits))

    def begin_command(self, rs: RunConfiguration, idx: int) -> List[str]:
        """
        Compose CmdStan command for non-default arguments.
        """
        cmd: List[str] = []
        if idx is not None and self.chain_ids is not None:
            if idx < 0 or idx > len(self.chain_ids) - 1:
                raise ValueError(
                    'index ({}) exceeds number of chains ({})'.format(
                        idx, len(self.chain_ids)
                    )
                )
            cmd.append(self.model_exe)  # type: ignore # guaranteed by validate
            cmd.append('id={}'.format(self.chain_ids[idx]))
        else:
            cmd.append(self.model_exe)  # type: ignore # guaranteed by validate

        if self.seed is not None:
            if not isinstance(self.seed, list):
                cmd.append('random')
                cmd.append('seed={}'.format(self.seed))
            else:
                cmd.append('random')
                cmd.append('seed={}'.format(self.seed[idx]))
        if self.data is not None:
            cmd.append('data')
            cmd.append('file={}'.format(self.data))
        if self.inits is not None:
            if not isinstance(self.inits, list):
                cmd.append('init={}'.format(self.inits))
            else:
                cmd.append('init={}'.format(self.inits[idx]))
        cmd.append('output')
        # files taken from RunSet
        cmd.append('file={}'.format(rs.get_csv_file(idx)))
        if self.save_latent_dynamics:
            cmd.append('diagnostic_file={}'.format(rs.get_diagnostic_file(idx)))
        if self.save_profile:
            cmd.append('profile_file={}'.format(rs.get_profile_file(idx)))
        if self.refresh is not None:
            cmd.append('refresh={}'.format(self.refresh))
        if self.sig_figs is not None:
            cmd.append('sig_figs={}'.format(self.sig_figs))

        return cmd
