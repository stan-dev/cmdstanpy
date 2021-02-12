import os
from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import Method, SamplerArgs, CmdStanArgs
from cmdstanpy.utils import EXTENSION
from cmdstanpy.stanfit import RunSet, CmdStanMCMC
from cmdstanpy.model import CmdStanModel

# HERE = os.path.dirname(os.path.abspath(__file__))
# DATAFILES_PATH = os.path.join(HERE, 'test', 'data')

DATAFILES_PATH = os.path.join('test', 'data')

stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
bern_model = CmdStanModel(stan_file=stan)

jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
bern_fit = bern_model.sample(
    data=jdata,
    chains=2,
    parallel_chains=2,
    seed=12345,
    iter_sampling=100,
    sig_figs=1,
)
