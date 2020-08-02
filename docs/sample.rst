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

- ``parallel_chains``: Number of processes to run in parallel.

- ``seed``: The seed or list of per-chain seeds for the sampler's random number generator.

- ``chain_ids``: The offset or list of per-chain offsets for the random number generator.

- ``inits``: Specifies how the sampler initializes parameter values.

- ``iter_warmup``: Number of warmup iterations for each chain.

- ``iter_sampling``: Number of draws from the posterior for each chain.

- ``save_warmup``: When ``True``, sampler saves warmup draws as part of output csv file.

- ``thin``: Period between saved samples.

- ``max_treedepth``: Maximum depth of trees evaluated by NUTS sampler per iteration.

- ``metric``: Specification of the mass matrix.

- ``step_size``: Initial stepsize for HMC sampler.

- ``adapt_engaged``: When ``True``, tune stepsize and metric during warmup. The default is ``True``.

- ``adapt_delta``: Adaptation target Metropolis acceptance rate. The default value is 0.8.  Increasing this value, which must be strictly less than 1, causes adaptation to use smaller step sizes. It improves the effective sample size, but may increase the time per iteration.

- ``adapt_init_phase``: Iterations for initial phase of adaptation during which step size is adjusted so that the chain converges towards the typical set.

- ``adapt_metric_window``: The second phase of adaptation tunes the metric and stepsize in a series of intervals.  This parameter specifies the number of iterations used for the first tuning interval; window size increases for each subsequent interval.

- ``adapt_step_size``: Number of iterations given over to adjusting the step size given the tuned metric during the final phase of adaptation.

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
to run CmdStan's summary and diagnostics tools.
Useful methods for information about the sample and the fit are:
  
- ``summary()`` - Run CmdStan's `stansummary <https://mc-stan.org/docs/cmdstan-guide/stansummary.html>`__ utility on the sample.
- ``diagnose()`` - Run CmdStan's `diagnose <https://mc-stan.org/docs/cmdstan-guide/diagnose.html>`__ utility on the sample.
- ``sampler_diagnostics()`` - Returns the sampler parameters as a map from sampler parameter names to a numpy.ndarray of dimensions draws X chains X 1.
- ``save_csvfiles(dir_name)`` - Move output csvfiles to specified directory.
- ``chains`` - Number of chains
- ``num_draws`` - Number of post-warmup draws (i.e., sampling iterations)
- ``num_warmup_draws`` - Number of warmup draws.
- ``metric`` - Per chain metric by the HMC sampler.
- ``stepsize`` - Per chain stepszie used by the HMC sampler.
- ``sample`` - A 3-D numpy.ndarray which contains all post-warmup draws across all chains arranged as (draws, chains, columns).
- ``warmup`` - A 3-D numpy.ndarray which contains all warmup draws across all chains arranged as (draws, chains, columns).
  
Useful methods for downstream analysis are:
  
- ``stan_variable(var_name)`` - Returns a numpy.ndarray which contains the set of draws in the sample for the named Stan program variable.
- ``stan_variables()`` - Return dictionary of all Stan program variables.
  
By default the sampler runs 4 chains, running as many chains in parallel as there
are available processors as determined by Python's ``multiprocessing.cpu_count()`` function.
For example, on a dual-processor machine with 4 virtual cores, all 4 chains will be run in parallel.
Continuing this example, specifying ``chains=6`` will result in 4 chains being run in parallel,
and as soon as 2 of them are finished, the remaining 2 chains will run.
Specifying ``chains=6, parallel_chains=6`` will run all 6 chains in parallel.

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


Example: high-level parallelization with `reduce_sum`
-----------------------------------------------------

    

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
