MCMC Sampling
=============

The :ref:`class_cmdstanmodel` class method  ``sample`` invokes Stan's adaptive HMC-NUTS
sampler which uses the Hamiltonian Monte Carlo (HMC) algorithm
and its adaptive variant the no-U-turn sampler (NUTS) to produce a set of
draws from the posterior distribution of the model parameters conditioned on the data.


Running the sampler
-------------------

In order to evaluate the fit of the model to the data, it is necessary to run
several Monte Carlo chains and compare the set of draws returned by each.
CmdStanPy uses Python's ``subprocess`` and ``multiprocessing`` libraries
to run these chains in separate processes.

Chains and parallelization
""""""""""""""""""""""""""

- ``chains``: Number of sampler chains. 

- ``parallel_chains``: Number of processes to run in parallel.

- ``threads_per_chains``: The number of threads to use in parallelized sections within an MCMC chain

By default the sampler runs 4 chains, running as many chains in parallel as there
are available processors as determined by Python's ``multiprocessing.cpu_count()`` function.
E.g., on a dual-processor machine with 4 virtual cores, 4 chains will be run in parallel.
Specifying ``chains=6`` will result in 4 chains being run in parallel,
and as soon as 2 of them are finished, the remaining 2 chains will run.
Specifying ``chains=6, parallel_chains=6`` will run all 6 chains in parallel.

Sample iterations and reported draws
""""""""""""""""""""""""""""""""""""

- ``iter_warmup``: Number of warmup iterations for each chain.

- ``iter_sampling``: Number of draws from the posterior for each chain.

- ``save_warmup``: When ``True``, sampler saves warmup draws as part of output csv file.

- ``thin``: Period between saved samples (draws).  Default is 1, i.e., save all iterations.

The terms `iterations` and `draws` are not synonymous.
The HMC sampler is configured to run a specified number of iterations.
By default, at the end of each iteration, the values of all sampler
and parameter variables are written to the Stan CSV output file.
Each reported set of estimates constitutes one row's worth of data,
each row of data is called a "draw".
The sampler argument ``thin`` controls the rate at which iterations are
recorded as draws.  By default, ``thin`` is 1, so every
iteration is recorded.  Increasing the thinning rate will reduce the
frequency with which the iterations are recorded, e.g., ``thin = 5``
will record every 5th iteration.

NUTS-HMC sampler configuration
""""""""""""""""""""""""""""""

- ``chain_ids``: The offset or list of per-chain offsets for the random number generator. 

- ``max_treedepth``: Maximum depth of trees evaluated by NUTS sampler per iteration.

- ``metric``: Specification of the mass matrix.

- ``step_size``: Initial step size for HMC sampler.

- ``adapt_engaged``: When ``True``, tune stepsize and metric during warmup. The default is ``True``.

- ``adapt_delta``: Adaptation target Metropolis acceptance rate. The default value is 0.8.  Increasing this value, which must be strictly less than 1, causes adaptation to use smaller step sizes. It improves the effective sample size, but may increase the time per iteration.

- ``adapt_init_phase``: Iterations for initial phase of adaptation during which step size is adjusted so that the chain converges towards the typical set.

- ``adapt_metric_window``: The second phase of adaptation tunes the metric and stepsize in a series of intervals.  This parameter specifies the number of iterations used for the first tuning interval; window size increases for each subsequent interval.

- ``adapt_step_size``: Number of iterations given over to adjusting the step size given the tuned metric during the final phase of adaptation.

- ``fixed_param``: When ``True``, call CmdStan with argument "algorithm=fixed_param".

.. include:: common_config.rst

All of these arguments are optional; when unspecified, the CmdStan defaults will be used.
See :meth:`~cmdstanpy.CmdStanModel.sample` for more details about the parameters.


Sampler results: the CmdStanMCMC object 
---------------------------------------

The :ref:`class_cmdstanmodel` method  ``sample`` returns a :ref:`class_cmdstanmcmc` object
which provides properties and methods to access and manage the sample; these fall into the following
following functional categories:

Fitted estimates
""""""""""""""""

The sampler returns a set of draws as a `Stan CSV file <https://mc-stan.org/docs/2_27/cmdstan-guide/stan-csv.html#mcmc-sampler-csv-output>`_.
The sample can be accessed either in terms of the CSV file column labels, or in terms of the underlying Stan sampler and model variables.

The following methods allow access in terms of the CSV file column labels: 

- ``draws`` - Returns the sample as a numpy.ndarray. By default, this is 3D array (draws, chains, columns); the argument ``concat_chains=True`` returns a 2D array which flattens the chains into a single set of draws.   If the sample contains saved warmup draws, these are not included by default; to get the warmup draws as well, use argument ``inc_warmup=True``.

- ``draws_pd`` - Returns the sample as a pandas.DataFrame.  By default, returns all output columns; the argument ``vars`` allows the specification of one or more variables of interest.   If the sample contains saved warmup draws, these are not included by default; to get the warmup draws as well, use argument ``inc_warmup=True``.

The following methods allow access in terms of the sampler and model variable names:
  
- ``draws_xr`` - Returns the sample as an xarray.DataSet.  By default, contains all method and Stan model variables; the argument ``vars`` allows the specification of one or more variables of interest.   If the sample contains saved warmup draws, these are not included by default; to get the warmup draws as well, use argument ``inc_warmup=True``.
  
- ``stan_variable(var=var_name)`` - Returns a numpy.ndarray which contains the set of draws in the sample for the named Stan program variable. 

- ``stan_variables()`` - Returns a Python dict, key: Stan program variable name, value: numpy.ndarray of draws.
- ``method_variables()`` - Returns a a Python dict, key: sampler variable name, value: numpy.ndarray of draws.

The following methods return the sampler tuning parameters:

- ``metric`` - List of per-chain metric values, metric is either a vector ('diag_e') or matrix ('dense_e')
- ``stepsize`` - List of per-chain step sizes.


Sample metadata and properties
""""""""""""""""""""""""""""""

The property ``metadata`` returns a :ref:`class_inferencemetadata` object which describes the inference engine configuration and outputs.

- ``metadata.method_vars_cols`` - Python dict, key: sampler parameter name, value: tuple of output column indices.
- ``metadata.stan_vars_cols`` - Python dict, key: Stan program variable name, value: tuple of output column indices.
- ``metadata.stan_vars_dims`` - Python dict, key: Stan program variable name, value: tuple of dimensions, or empty tuple, for scalar variables.

- ``metadata.cmdstan_config`` - Python dict, key: CmdStan argument name, value: value used for this sampler run, whether user-specified or CmdStan default. 

The :ref:`class_cmdstanmcmc` object also provides the following properties:

- ``column_names`` - List of column labels for one draw from the sampler. 
- ``chains`` - Number of chains 
- ``chains_ids`` - Chain ids

- ``num_draws_sampling`` - Number of sampling (post-warmup) draws per chain, i.e., sampling iterations, thinned.
- ``num_draws_warmup`` - Number of warmup draws per chain, i.e., thinned warmup iterations.
- ``thin`` - Period between recorded iterations. 

- ``metric_type`` - Metric type used for adaptation, either ``diag_e``
  or ``dense_e``, or ``None``, if the Stan program doesn't have any parameters.
- ``num_unconstrained_params`` - Count of `unconstrained` model parameters. For metric ``diag_e``, this is the length of the diagonal vector and for metric ``dense_e`` this is the size of the full covariance matrix.

Utilities to summarize and diagnose the sample
""""""""""""""""""""""""""""""""""""""""""""""

CmdStanPy wraps CmdStan utilities to summarize and diagnose the sample:


- ``summary()`` - Run CmdStan's `stansummary <https://mc-stan.org/docs/cmdstan-guide/stansummary.html>`__ utility on the sample.
- ``diagnose()`` - Run CmdStan's `diagnose <https://mc-stan.org/docs/cmdstan-guide/diagnose.html>`__ utility on the sample.

Save the Stan CSV output files
""""""""""""""""""""""""""""""

Underlyingly, the sample is a set of Stan CSV files.
By default, these are written to a temporary directory.
You can save them to a permanent location, from which they can
be reassembled into a CmdStanMCMC object.

- CmdStanMCMC class method ``save_csvfiles(dir_name)`` - Move output Stan CSV files to specified directory. 
- Utility ``from_csv(dir_name)`` - Given a set of Stan CSV files generated by the NUT-HMC sampler, returns a CmdStanMCMC object.



Example: fit model - sampler defaults
-------------------------------------

In this example we use the CmdStan example model
`bernoulli.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.stan>`__
and data file
`bernoulli.data.json <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.data.json>`__.


.. code-block:: python

    import os
    from cmdstanpy import cmdstan_path, CmdStanModel
    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    bernoulli_data = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')

    # instantiate, compile bernoulli model
    bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)

    # run the NUTS-HMC sampler 
    bern_fit = bernoulli_model.sample(data=bernoulli_data)

    # summarize the fit 
    bern_fit.summary()

    # instantiate, inspect the sample 
    bern_fit.draws.shape
    bern_fit.draws.column_names
    
    sampler_variables = bern_fit.sampler_vars_cols
    stan_variables = bern_fit.stan_vars_cols
    print('Sampler variables:\n{}'.format(sampler_variables)) 
    print('Stan variables:\n{}'.format(stan_variables)) 

    # get parameter variable estimates
    draws_theta = bern_fit.stan_variable(var='theta') 
    draws_theta.shape 



Example: high-level parallelization with **reduce_sum**
--------------------------------------------------------

Stan provides `high-level parallelization <https://mc-stan.org/docs/stan-users-guide/parallelization-chapter.html>`_
via multi-threading by use of the **reduce_sum** and **map_rect** functions in a Stan program.
To use this feature, a Stan program must be compiled with the C++ compiler flag **STAN_THREADS**
as described in the :ref:`model-compilation` section.

.. code-block:: python

    proc_parallel_model = CmdStanModel(
        stan_file='proc_parallel.stan',
        cpp_options={"STAN_THREADS": True}),
    )

When running the sampler with this model, you must explicitly specify the number
of threads to use via ``sample`` method argument **threads_per_chain**.
For example, to run 4 chains multi-threaded using 4 threads per chain:

.. code-block:: python

    proc_parallel_fit = proc_parallel_model.sample(data=proc_data,
        chains=4, threads_per_chain=4)

By default, the number of parallel chains will be equal to the number of
available cores on your machine, which may adversely affect overall
performance.  For example, on a machine with Intel's dual processor hardware,
i.e, 4 virtual cores, the above configuration will use 16 threads.
To limit this, specify the **parallel_chains** option so that
the maximum number of threads used will be **parallel_chains** X **threads_per_chain**

.. code-block:: python

    proc_parallel_fit = proc_parallel_model.sample(data=proc_data,
        chains=4, parallel_chains=1, threads_per_chain=4)



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
The sampler runs without updating the Markov Chain,
thus the values of all parameters and
transformed parameters are constant across all draws and
only those values in the generated quantities block that are
produced by RNG functions may change.

.. code-block:: python

    import os
    from cmdstanpy import CmdStanModel
    datagen_stan = os.path.join('..', '..', 'test', 'data', 'bernoulli_datagen.stan')
    datagen_model = CmdStanModel(stan_file=datagen_stan)
    sim_data = datagen_model.sample(fixed_param=True)
    sim_data.summary()

Each draw contains variable `y_sim`, a vector of `N` binary outcomes.
From this, we can compute the probability of new data given an estimate of
parameter `theta` - the chance of success of a single Bernoulli trial.
By plotting the histogram of the distribution of total number of successes
in `N` trials shows the **posterior predictive distribution** of `theta`.

.. code-block:: python

    # extract int array `y_sim` from the sampler output
    y_sims = sim_data.stan_variable(var='y_sim')
    y_sims.shape

    # each draw has 10 replicates of estimated parameter 'theta'
    y_sums = y_sims.sum(axis=1)
    # plot total number of successes per draw
    import pandas as pd
    y_sums_pd = pd.DataFrame(data=y_sums)
    y_sums_pd.plot.hist(range(0,datagen_data['N']+1))
