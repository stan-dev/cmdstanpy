"""Example script to be runned after tests, tqdm installed."""
import os
import sys

from cmdstanpy import CmdStanModel, cmdstan_path

# explicit import to test if it is installed
# pylint: disable=import-error,unused-import,wrong-import-order
import tqdm # noqa



def run_bernoulli_fit():
    # specify Stan file, create, compile CmdStanModel object
    bernoulli_path = os.path.join(
        cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan'
    )
    bernoulli_model = CmdStanModel(stan_file=bernoulli_path)
    bernoulli_model.compile()

    # specify data, fit the model
    bernoulli_data = {'N': 10, 'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    # Show progress
    bernoulli_fit = bernoulli_model.sample(
        chains=4, cores=2, data=bernoulli_data, show_progress=True
    )

    # summarize the results (wraps CmdStan `bin/stansummary`):
    print(bernoulli_fit.summary())


if __name__ == '__main__':
    run_bernoulli_fit()
    # exit explicitly
    sys.exit(0)
