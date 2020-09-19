# CmdStanPy

[![codecov](https://codecov.io/gh/stan-dev/cmdstanpy/branch/master/graph/badge.svg)](https://codecov.io/gh/stan-dev/cmdstanpy)


CmdStanPy is a lightweight interface to Stan for Python users which
provides the necessary objects and functions to do Bayesian inference
given a probability model written as a Stan program and data.
Under the hood, CmdStanPy uses the CmdStan command line interface
to compile and run a Stan program.

### Goals

- Clean interface to Stan services so that CmdStanPy can keep up with Stan releases.

- Provide access to all CmdStan inference methods.

- Easy to install,
  + minimal Python library dependencies: numpy, pandas
  + Python code doesn't interface directly with c++, only calls compiled executables

- Modular - CmdStanPy produces a MCMC sample (or point estimate) from the posterior; other packages do analysis and visualization.

### Docs

See https://cmdstanpy.readthedocs.io/

### Source Repository

CmdStan's source-code repository is hosted here on GitHub.

### Licensing

The CmdStanPy, CmdStan, and the core Stan C++ code are licensed under new BSD.

### Example

::

    import os
    from cmdstanpy import cmdstan_path, CmdStanModel

    # specify locations of Stan program file and data
    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    bernoulli_data = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')

    # instantiate a model; compiles the Stan program by default
    bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)

    # obtain a posterior sample from the model conditioned on the data
    bernoulli_fit = bernoulli_model.sample(chains=4, data=bernoulli_data)

    # summarize the results (wraps CmdStan `bin/stansummary`):
    bernoulli_fit.summary()
