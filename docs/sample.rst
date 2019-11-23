MCMC Sampling
=============

The :ref:`class_cmdstanmodel` class method  ``sample`` invokes Stan's adaptive HMC-NUTS
sampler which uses the Hamiltonian Monte Carlo (HMC) algorithm
and its adaptive variant the no-U-turn sampler (NUTS) to produce a set of
draws from the posterior distribution of the model parameters conditioned on the data.

In order to evaluate the fit of the model to the data, it is necessary to run
several Monte Carlo chains and compare the set of draws returned by each.
By default, the ``sample`` command runs 4 sampler chains, i.e.,
CmdStanPy invokes CmdStan 4 times.
CmdStanPy uses Python's ``subprocess`` and ``multiprocessing`` libraries
to run these chains in separate processes.
This processing can be done in parallel, up to the number of
processor cores available.

NUTS-HMC sampler configuration
------------------------------

- ``chains``: Number of sampler chains.

- ``cores``: Number of processes to run in parallel.

- ``seed``: The seed or list of per-chain seeds for the sampler's random number generator.

- ``chain_ids``: The offset or list of per-chain offsets for the random number generator.

- ``inits``: Specifies how the sampler initializes parameter values.

- ``warmup_iters``: Number of warmup iterations for each chain.

- ``sampling_iters``: Number of draws from the posterior for each chain.

- ``save_warmup``: When True, sampler saves warmup draws as part of output csv file.

- ``thin``: Period between saved samples.

- ``max_treedepth``: Maximum depth of trees evaluated by NUTS sampler per iteration.

- ``metric``: Specification of the mass matrix.

- ``step_size``: Initial stepsize for HMC sampler.

- ``adapt_engaged``: When ``True``, adapt stepsize and metric.

- ``fixed_param``: When ``True``, call CmdStan with argument "algorithm=fixed_param".  

.. include:: common_config.rst

All of these arguments are optional; when unspecified, the CmdStan defaults will be used.


Example: fit model - sampler defaults
-------------------------------------

In this example we use the CmdStan example model
`bernoulli.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.stan>`__
and data file
`bernoulli.data.json <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.data.json>`__.

The :ref:`class_cmdstanmodel` class method  ``sample`` returns a ``CmdStanMCMC`` object
which provides properties to retrieve information about the sample, as well as methods
to run CmdStan's summary and diagnostics tools:

- ``chains``
- ``draws``
- ``columns``   
- ``column_names``
- ``metric_type``
- ``metric``
- ``sample``
- ``stepsize``

- ``get_drawset``
- ``summary()``
- ``diagnose()``
- ``save_csvfiles()``

  
By default the sampler runs 4 chains.
It will use 2 less than that number of cores available, as determined by Python's ``multiprocessing.cpu_count()`` function.  For example, on a dual-processor machine with 4 virtual cores, the defaults will result a run of 4 chains, but only 2 chains will be run parallel.

.. code-block:: python

    import os
    from cmdstanpy import cmdstan_path, CmdStanModel
    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')

    # instantiate, compile bernoulli model
    bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)

    # run CmdStan's sample method, returns object `CmdStanMCMC`
    bernoulli_data = { "N" : 10, "y" : [0,1,0,0,0,0,0,0,0,1] }
    bern_fit = bernoulli_model.sample(data=bernoulli_data)
    bern_fit.sample.shape
    bern_fit.summary()

    

Example: generate data - `fixed_param=True`
-------------------------------------------


The Stan programming language can be used to write Stan programs which generate
simulated data for a set of known parameter values by calling Stan's RNG functions.
Such programs don't need to declare parameters or model blocks because all
computation is done in the generated quantities block.

For example, the Stan program
`bernoulli.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli_datagen.stan>`__
can be used to generate a dataset of simulated data, where each row in the dataset consists of `N` draws from a Bernoulli distribution given probability `theta`:

.. code::

    transformed data { 
      int<lower=0> N = 10;
      real<lower=0,upper=1> theta = 0.35;
    } 
    generated quantities {
      int y_sim[N];
      for (n in 1:N)
        y_sim[n] = bernoulli_rng(theta);
    }

This program doesn't contain parameters or a model block, therefore
we run the sampler without ding any MCMC estimation by
specifying ``fixed_param=True``.
When ``fixed_param=True``, the ``sample`` method only runs 1 chain.
The sampler runs without updating the Markov Chain,
thus the values of all parameters and
transformed parameters are constant across all draws and
only those values in the generated quantities block that are
produced by RNG functions may change.

.. code-block:: python

    datagen_stan = os.path.join('..', '..', 'test', 'data', 'bernoulli_datagen.stan')
    datagen_model = CmdStanModel(stan_file=datagen_stan)

    sim_data = datagen_model.sample(fixed_param=True)
    sim_data.summary()

Compute, plot histogram of total successes for `N` Bernoulli trials with chance of success `theta`:

.. code-block:: python

    drawset_pd = sim_data.get_drawset()
    drawset_pd.columns

    # extract new series of outcomes of N Bernoulli trials
    y_sims = drawset_pd.drop(columns=['lp__', 'accept_stat__'])

    # plot total number of successes per draw
    y_sums = y_sims.sum(axis=1)
    y_sums.astype('int32').plot.hist(range(0,datagen_data['N']+1))
