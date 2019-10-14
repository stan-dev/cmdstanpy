Run Generated Quantities
========================

The `generated quantities block <https://mc-stan.org/docs/reference-manual/program-block-generated-quantities.html>`__
computes *quantities of interest* (QOIs) based on the data,
transformed data, parameters, and transformed parameters.
It can be used to:

-  generate simulated data for model testing by forward sampling
-  generate predictions for new data
-  calculate posterior event probabilities, including multiple
   comparisons, sign tests, etc.
-  calculating posterior expectations
-  transform parameters for reporting
-  apply full Bayesian decision theory
-  calculate log likelihoods, deviances, etc. for model comparison

The :ref:`class_model` class ``run_generated_quantities`` method is useful once you
have successfully fit a model to your data and have a valid
sample from the posterior.
If you need to compute additional quantities of interest,
you can do this using the existing parameter estimates.
It takes the existing sample as input, and for each draw it
runs the generated quantities block of the program using the
per-draw parameter estimates to compute the quantities of interest.
In this way you add more columns of information to an existing sample.

Configuration
-------------

- ``csv_files``: A list of sampler output csv files.

- ``data``: Values for all data variables in the model, specified either as a dictionary with entries matching the data variables, or as the path of a data file in JSON or Rdump format.

- ``seed``: The seed for random number generator.
            
- ``gq_csv_basename``:  A path or file name which will be used as the basename for the CmdStan output files.


Example: add posterior predictive checks to ``bernoulli.stan``
--------------------------------------------------------------

In this example we use the CmdStan example model
`bernoulli.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.stan>`__
and data file
`bernoulli.data.json <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.data.json>`__
as our existing model and data.
We create the program
`bernoulli_ppc.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli_ppc.stan>`__
by adding a ``generated quantities`` block to ``bernoulli.stan``
which generates a new data vector ``y_rep`` using the current estimate of theta.

.. code::

    generated quantities {
      int y_sim[N];
      real<lower=0,upper=1> theta_rep;
      for (n in 1:N)
        y_sim[n] = bernoulli_rng(theta);
      theta_rep = sum(y) / N;
    }


The :ref:`class_model` class method  ``run_generated_quantities`` returns a ``StanQuantities`` object
which provides properties to retrieve information about the sample:


- ``chains``
- ``column_names``
- ``generated_quantities``

- ``save_csvfiles()``


The arguments to the ``run_generated_quantities`` method are:

- the data used to fit the model (``bern_data``)
- the list of the resulting stan csv files (``bern_fit.csv_files``)


Therefore the first step is to fit the bernoulli model to the data:

.. code:: ipython3

    import os
    from cmdstanpy import Model, cmdstan_path
    bernoulli_path = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    
    bernoulli_model = Model(stan_file=bernoulli_path)
    bernoulli_model.compile()

    bern_data = os.path.join(bernoulli_dir, 'bernoulli.data.json')
    bern_fit = bernoulli_model.sample(data=bern_data)


.. code:: ipython3

    bernoulli_ppc_model = Model(stan_file='bernoulli_ppc.stan')
    bernoulli_ppc_model.compile()

    new_quantities = bernoulli_ppc_model.run_generated_quantities(data=bern_data, csv_files=bern_fit.csv_files)



The ``StanQuantities`` object contains the values for all variables in
the generated quantities block of the program ``bernoulli_ppc.stan``.
Unlike the output from the ``sample`` method, it doesn’t contain any
information on the joint log probability density, sampler state, or
parameters or transformed parameter values.

.. code:: ipython3

    new_quantities.column_names
    new_quantities.generated_quantities.shape
    for i in range(len(new_quantities.column_names)):
        print(new_quantities.generated_quantities[:,i].mean())





