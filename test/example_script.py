"""Example script to be runned after tests, tqdm installed."""
import os
import sys

# explicit import to test if it is installed
# pylint: disable=E0401,W0611,C0411
import tqdm  # noqa

from cmdstanpy import CmdStanModel, cmdstan_path


def run_bernoulli_fit():
    # specify Stan file, create, compile CmdStanModel object
    bernoulli_path = os.path.join(
        cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan'
    )
    bernoulli_model = CmdStanModel(stan_file=bernoulli_path)
    bernoulli_model.compile(force=True)

    # specify data, fit the model
    bernoulli_data = {'N': 10, 'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    # Show progress
    bernoulli_fit = bernoulli_model.sample(
        chains=4, parallel_chains=2, data=bernoulli_data, show_progress=True
    )

    # summarize the results (wraps CmdStan `bin/stansummary`):
    print(bernoulli_fit.summary())


if __name__ == '__main__':
    run_bernoulli_fit()
    # exit explicitly
    sys.exit(0)
