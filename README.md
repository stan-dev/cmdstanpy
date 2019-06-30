# CmdStanPy

CmdStanPy is a lightweight interface to Stan for Python users which
provides the necessary objects and functions to compile a Stan program
and run Stan's samplers.

### Goals

- Clean interface to Stan services so that CmdStanPy can keep up with Stan releases.

- Provides complete control - all sampler arguments have corresponding named argument
for CmdStanPy sampler function.

- Easy to install,
  + minimal Python library dependencies: numpy, pandas
  + Python code doesn't interface directly with c++, only calls compiled executables

- Modular - CmdStanPy produces a sample from the posterior, downstream modules do the analysis.


### Source Repository

CmdStan's source-code repository is hosted here on GitHub.

### Licensing

The CmdStanPy, CmdStan, and the core Stan C++ code are licensed under new BSD.

### Example

::

    import os
    from cmdstanpy import cmdstan_path, compile_model, RunSet, sample, summary

    # create model
    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    bernoulli_model = compile_model(bernoulli_stan)

    # setup data and fit the model
    bernoulli_data = { "N" : 10, "y" : [0,1,0,0,0,0,0,0,0,1] }
    bernoulli_fit = sample(bernoulli_model, data=bernoulli_data)

    # print summary
    print(summary(bern_fit))
