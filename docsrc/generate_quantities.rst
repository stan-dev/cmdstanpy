Generate Quantities of Interest
===============================

The Stan sampler executes the model's
`generated quantities block <https://mc-stan.org/docs/reference-manual/program-block-generated-quantities.html>`_
once per iteration, using the input data and current estimates of the
parameters and transformed parameters to compute 
*quantities of interest* (QOIs).
The generated quantities block is used to:

-  `generate simulated data for model testing by forward sampling <https://mc-stan.org/docs/stan-users-guide/stand-alone-generated-quantities-and-ongoing-prediction.html>`_
-  generate predictions for new data
-  calculate posterior event probabilities, including multiple comparisons, sign tests, etc.
-  calculating posterior expectations
-  transform parameters for reporting
-  apply full Bayesian decision theory
-  calculate log likelihoods, deviances, etc. for model comparison

The `CmdStan method `generate` <https://mc-stan.org/docs/cmdstan-guide/standalone-generate-quantities.html>`_
provides a way to bypass estimation of the parameters, and instead
use the estimates from a previous sample, along with input data, to compute the
variables defined in the generated quantities block.

The :ref:`class_cmdstanmodel` class ``generate_quantities`` method wraps the
call to CmdStan's `generate` method given a suitable Stan model,
input data, and an existing sample containing estimate for all model parameters.
It returns a ``CmdStanGQ`` object which provides properties to retrieve information
all variables and metadata in both the existing sample and new quantities of interest.


Running stand-alone generate quantities
---------------------------------------

The :ref:`class_cmdstanmodel` class ``generate_quantities`` method takes the following arguments:


- ``mcmc_sample``: (Required) A sample containing estimates for all model parameters, either a ``CmdStanMCMC`` object or a list of Stan CSV files.

- ``data``: (Required, unless model has no data variables) Values for all data variables in the model, specified either as a dictionary with entries matching the data variables, or as the path of a data file in JSON or Rdump format.

- ``seed``: (Optional) A list of per-chain seeds for the random number generator.
            
- ``gq_output_dir``: (Optional) A path or file name which will be used as the basename for the CmdStan output files.

- ``sig_figs``: (Optional) Numerical precision used for output CSV and text files. Must be an integer between 1 and 18.  If unspecified, the default precision for the system file I/O is used; the usual value is 6.
  
- ``refresh``: (Optional) Specifies the number of inference method iterations between progress messages. Default value is 100.  Value ``refresh = 0`` suppresses output of iteration number messages.



Results:  the CmdStanGQ object
------------------------------

The :ref:`class_cmdstanmodel` method  ``generate`` returns a :ref:`class_cmdstangq` object,
which provides access to the input sample containing the fitted parameters via property ``mcmc_sample``,
and properties and methods to access and manage the new set of generated quantities.

The generated quantities can be accessed either in terms of the CSV file column labels, or in terms of the underlying Stan model variables.
These methods have argument `inc_sample` which returns columns and variables from both the input sample and the new generated quantities.


Fitted estimates
""""""""""""""""

The following methods allow access in terms of the CSV file column labels: 

- ``draws`` - Returns the sample as a numpy.ndarray. By default, this is 3D array (draws, chains, columns); the argument ``concat_chains=True`` returns a 2D array which flattens the chains into a single set of draws.   If the sample contains saved warmup draws, these are not included by default; to get the warmup draws as well, use argument ``inc_warmup=True``.

- ``draws_pd`` - Returns the sample as a pandas.DataFrame.  By default, returns all output columns; the argument ``vars`` allows the specification of one or more variables of interest.   If the sample contains saved warmup draws, these are not included by default; to get the warmup draws as well, use argument ``inc_warmup=True``.


- ``stan_variable(name=var_name)`` - Returns a numpy.ndarray which contains the set of draws in the sample for the named Stan program variable.
- ``stan_variables()`` - Returns a Python dict, key: Stan program variable name, value: numpy.ndarray of draws.

The following methods allow access in terms of the sampler and model variable names:
  
- ``draws_xr`` - Returns the sample as an xarray.DataSet.  By default, all generated quantities block variables; the argument ``vars`` allows the specification of one or more variables of interest.   If the sample contains saved warmup draws, these are not included by default; to get the warmup draws as well, use argument ``inc_warmup=True``.  To get all variables from the input sample as well as the new generated quantities, use argument ``inc_sample=True``.
  
- ``stan_variable(var=var_name)`` - Returns a numpy.ndarray which contains the set of draws for a named generated quantities block variable.    If the sample contains saved warmup draws, these are not included by default; to get the warmup draws as well, use argument ``inc_warmup=True``.  To get a variable from the input sample, use argument ``inc_sample=True``.

- ``stan_variables()`` - Returns a Python dict, key: Stan program variable name, value: numpy.ndarray of draws. If the sample contains saved warmup draws, these are not included by default; to get the warmup draws as well, use argument ``inc_warmup=True``.  To get variable from the input sample as well as from the new generated quantities, use argument ``inc_sample=True``.

Sample metadata and properties
""""""""""""""""""""""""""""""

The property ``metadata`` returns a :ref:`class_inferencemetadata` object which describes the inference engine configuration and outputs.

- ``metadata.stan_vars_cols`` - Python dict, key: Stan program variable name, value: tuple of output column indices.
- ``metadata.stan_vars_dims`` - Python dict, key: Stan program variable name, value: tuple of dimensions, or empty tuple, for scalar variables.

- ``metadata.cmdstan_config`` - Python dict, key: CmdStan argument name, value: value used for this sampler run, whether user-specified or CmdStan default. 

The :ref:`class_cmdstangq` object also provides the following properties:

- ``column_names`` - List of column labels for one draw from the sampler. 
- ``chains`` - Number of chains 
- ``chains_ids`` - Chain ids


Example: add posterior predictive checks to ``bernoulli.stan``
--------------------------------------------------------------

In this example we use the CmdStan example model
`bernoulli.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.stan>`__
and data file
`bernoulli.data.json <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.data.json>`__
as our existing model and data.
We create the program
`bernoulli_ppc.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli_ppc.stan>`__
by adding a ``generated quantities`` block to bernoulli.stan
which generates a new data vector ``y_rep`` using the current estimate of theta.

.. code::

    generated quantities {
      int y_sim[N];
      real<lower=0,upper=1> theta_rep;
      for (n in 1:N)
        y_sim[n] = bernoulli_rng(theta);
      theta_rep = sum(y) / N;
    }


The first step is to fit model ``bernoulli`` to the data:

.. code:: python

    import os
    from cmdstanpy import CmdStanModel, cmdstan_path

    bernoulli_dir = os.path.join(cmdstan_path(), 'examples', 'bernoulli')
    bernoulli_path = os.path.join(bernoulli_dir, 'bernoulli.stan')
    bernoulli_data = os.path.join(bernoulli_dir, 'bernoulli.data.json')

    # instantiate, compile bernoulli model
    bernoulli_model = CmdStanModel(stan_file=bernoulli_path)

    # fit the model to the data
    bern_fit = bernoulli_model.sample(data=bernoulli_data)


Then we compile the model ``bernoulli_ppc`` and use the fit parameter estimates
to generate quantities of interest:


.. code:: python

    bernoulli_ppc_model = CmdStanModel(stan_file='bernoulli_ppc.stan')
    new_quantities = bernoulli_ppc_model.generate_quantities(data=bern_data, mcmc_sample=bern_fit)

The ``generate_quantities`` method returns a ``CmdStanGQ`` object which
contains the values for all variables in the generated quantities block
of the program ``bernoulli_ppc.stan``. Unlike the output from the
``sample`` method, it doesn’t contain any information on the joint log
probability density, sampler state, or parameters or transformed
parameter values.

.. code:: python

    new_quantities.column_names
    new_quantities.generated_quantities.shape
    for i in range(len(new_quantities.column_names)):
        print(new_quantities.generated_quantities[:,i].mean())


The method ``draws_pd(inc_sample=True)`` returns a pandas DataFrame which
combines the input drawset with the generated quantities.

.. code:: python

    sample_plus_pd = new_quantities.draws_pd(inc_sample=True)
    print(sample_plus.shape)
    print(sample_plus.columns)        
