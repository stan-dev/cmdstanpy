"""Container objects for results of CmdStan run(s)."""

import glob
import os
from typing import Any, Dict, List, Optional, Union

from cmdstanpy.cmdstan_args import (
    CmdStanArgs,
    OptimizeArgs,
    SamplerArgs,
    VariationalArgs,
)
from cmdstanpy.utils import check_sampler_csv, get_logger, scan_config

from .gq import CmdStanGQ
from .mcmc import CmdStanMCMC
from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet
from .vb import CmdStanVB

__all__ = [
    "RunSet",
    "InferenceMetadata",
    "CmdStanMCMC",
    "CmdStanMLE",
    "CmdStanVB",
    "CmdStanGQ",
]


def from_csv(
    path: Union[str, List[str], os.PathLike, None] = None,
    method: Optional[str] = None,
) -> Union[CmdStanMCMC, CmdStanMLE, CmdStanVB, None]:
    """
    Instantiate a CmdStan object from a the Stan CSV files from a CmdStan run.
    CSV files are specified from either a list of Stan CSV files or a single
    filepath which can be either a directory name, a Stan CSV filename, or
    a pathname pattern (i.e., a Python glob).  The optional argument 'method'
    checks that the CSV files were produced by that method.
    Stan CSV files from CmdStan methods 'sample', 'optimize', and 'variational'
    result in objects of class CmdStanMCMC, CmdStanMLE, and CmdStanVB,
    respectively.

    :param path: directory path
    :param method: method name (optional)

    :return: either a CmdStanMCMC, CmdStanMLE, or CmdStanVB object
    """
    if path is None:
        raise ValueError('Must specify path to Stan CSV files.')
    if method is not None and method not in [
        'sample',
        'optimize',
        'variational',
    ]:
        raise ValueError(
            'Bad method argument {}, must be one of: '
            '"sample", "optimize", "variational"'.format(method)
        )

    csvfiles = []
    if isinstance(path, list):
        csvfiles = path
    elif isinstance(path, str) and '*' in path:
        splits = os.path.split(path)
        if splits[0] is not None:
            if not (os.path.exists(splits[0]) and os.path.isdir(splits[0])):
                raise ValueError(
                    'Invalid path specification, {} '
                    ' unknown directory: {}'.format(path, splits[0])
                )
        csvfiles = glob.glob(path)
    elif isinstance(path, (str, os.PathLike)):
        if os.path.exists(path) and os.path.isdir(path):
            for file in os.listdir(path):
                if os.path.splitext(file)[1] == ".csv":
                    csvfiles.append(os.path.join(path, file))
        elif os.path.exists(path):
            csvfiles.append(str(path))
        else:
            raise ValueError('Invalid path specification: {}'.format(path))
    else:
        raise ValueError('Invalid path specification: {}'.format(path))

    if len(csvfiles) == 0:
        raise ValueError('No CSV files found in directory {}'.format(path))
    for file in csvfiles:
        if not (os.path.exists(file) and os.path.splitext(file)[1] == ".csv"):
            raise ValueError(
                'Bad CSV file path spec,'
                ' includes non-csv file: {}'.format(file)
            )

    config_dict: Dict[str, Any] = {}
    try:
        with open(csvfiles[0], 'r') as fd:
            scan_config(fd, config_dict, 0)
    except (IOError, OSError, PermissionError) as e:
        raise ValueError('Cannot read CSV file: {}'.format(csvfiles[0])) from e
    if 'model' not in config_dict or 'method' not in config_dict:
        raise ValueError("File {} is not a Stan CSV file.".format(csvfiles[0]))
    if method is not None and method != config_dict['method']:
        raise ValueError(
            'Expecting Stan CSV output files from method {}, '
            ' found outputs from method {}'.format(
                method, config_dict['method']
            )
        )
    try:
        if config_dict['method'] == 'sample':
            chains = len(csvfiles)
            sampler_args = SamplerArgs(
                iter_sampling=config_dict['num_samples'],
                iter_warmup=config_dict['num_warmup'],
                thin=config_dict['thin'],
                save_warmup=config_dict['save_warmup'],
            )
            # bugfix 425, check for fixed_params output
            try:
                check_sampler_csv(
                    csvfiles[0],
                    iter_sampling=config_dict['num_samples'],
                    iter_warmup=config_dict['num_warmup'],
                    thin=config_dict['thin'],
                    save_warmup=config_dict['save_warmup'],
                )
            except ValueError:
                try:
                    check_sampler_csv(
                        csvfiles[0],
                        is_fixed_param=True,
                        iter_sampling=config_dict['num_samples'],
                        iter_warmup=config_dict['num_warmup'],
                        thin=config_dict['thin'],
                        save_warmup=config_dict['save_warmup'],
                    )
                    sampler_args = SamplerArgs(
                        iter_sampling=config_dict['num_samples'],
                        iter_warmup=config_dict['num_warmup'],
                        thin=config_dict['thin'],
                        save_warmup=config_dict['save_warmup'],
                        fixed_param=True,
                    )
                except (ValueError) as e:
                    raise ValueError(
                        'Invalid or corrupt Stan CSV output file, '
                    ) from e

            cmdstan_args = CmdStanArgs(
                model_name=config_dict['model'],
                model_exe=config_dict['model'],
                chain_ids=[x + 1 for x in range(chains)],
                method_args=sampler_args,
            )
            runset = RunSet(args=cmdstan_args, chains=chains)
            runset._csv_files = csvfiles
            for i in range(len(runset._retcodes)):
                runset._set_retcode(i, 0)
            fit = CmdStanMCMC(runset)
            fit.draws()
            return fit
        elif config_dict['method'] == 'optimize':
            if 'algorithm' not in config_dict:
                raise ValueError(
                    "Cannot find optimization algorithm"
                    " in file {}.".format(csvfiles[0])
                )
            optimize_args = OptimizeArgs(
                algorithm=config_dict['algorithm'],
                save_iterations=config_dict['save_iterations'],
            )
            cmdstan_args = CmdStanArgs(
                model_name=config_dict['model'],
                model_exe=config_dict['model'],
                chain_ids=None,
                method_args=optimize_args,
            )
            runset = RunSet(args=cmdstan_args)
            runset._csv_files = csvfiles
            for i in range(len(runset._retcodes)):
                runset._set_retcode(i, 0)
            return CmdStanMLE(runset)
        elif config_dict['method'] == 'variational':
            if 'algorithm' not in config_dict:
                raise ValueError(
                    "Cannot find variational algorithm"
                    " in file {}.".format(csvfiles[0])
                )
            variational_args = VariationalArgs(
                algorithm=config_dict['algorithm'],
                iter=config_dict['iter'],
                grad_samples=config_dict['grad_samples'],
                elbo_samples=config_dict['elbo_samples'],
                eta=config_dict['eta'],
                tol_rel_obj=config_dict['tol_rel_obj'],
                eval_elbo=config_dict['eval_elbo'],
                output_samples=config_dict['output_samples'],
            )
            cmdstan_args = CmdStanArgs(
                model_name=config_dict['model'],
                model_exe=config_dict['model'],
                chain_ids=None,
                method_args=variational_args,
            )
            runset = RunSet(args=cmdstan_args)
            runset._csv_files = csvfiles
            for i in range(len(runset._retcodes)):
                runset._set_retcode(i, 0)
            return CmdStanVB(runset)
        else:
            get_logger().info(
                'Unable to process CSV output files from method %s.',
                (config_dict['method']),
            )
            return None
    except (IOError, OSError, PermissionError) as e:
        raise ValueError(
            'An error occurred processing the CSV files:\n\t{}'.format(str(e))
        ) from e
