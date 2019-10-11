Run Generated Quantities
========================

The `generated quantities block <https://mc-stan.org/docs/reference-manual/program-block-generated-quantities.html>`__
computes quantities of interest based on the data,
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

When you have already successfully fit a model to the data,
and are satisfied with the model but would like to compute
new quantities of interest, the :ref:`class_model` class method ``run_generated_quantities``
lets you do this without the expense of re-running the sampler.
It takes the existing sample as input, and for each draw it
runs the generated quantities block of the program using the
per-draw parameter estimates to compute the quantities of interest.


Example: add posterior predictive checks to ``bernoulli.stan``
--------------------------------------------------------------

In this example we use the CmdStan example model
`bernoulli.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.stan>`__
and data file
`bernoulli.data.json <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.data.json>`__
as our existing model and data.
We instantiate the model ``bernoulli``, as in the “Hello World”, and produce a sample
from the model conditioned on the data.


.. code:: ipython3

    import os
    from cmdstanpy import Model, cmdstan_path
    bernoulli_path = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    
    bernoulli_model = Model(stan_file=bernoulli_path)
    bernoulli_model.compile()

    bern_data = os.path.join(bernoulli_dir, 'bernoulli.data.json')
    bern_fit = bernoulli_model.sample(data=bern_data)



We create program
`bernoulli_ppc.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli_ppc.stan>`__
by adding a ``generated quantities`` block which generates a new data
vector ``y_rep`` using the current estimate of theta.


The arguments to the ``run_generated_quantities`` method are:

- the data used to fit the model (``bern_data``)
- the list of the resulting stan csv files (``bern_fit.csv_files``)

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





