Maximum Likelihood Estimation
=============================

Stan provides optimization algorithms which find modes of the density specified by a Stan program.
Three different algorithms are available:
a Newton optimizer, and two related quasi-Newton algorithms, BFGS and L-BFGS.
The L-BFGS algorithm is the default optimizer.
Newton’s method is the least efficient of the three, but has the advantage of setting its own stepsize.


Optimize configuration
----------------------

- ``algorithm``: Algorithm to use. One of: "BFGS", "LBFGS", "Newton".

- ``init_alpha``: Line search step size for first iteration.

- ``iter``: Total number of iterations.

.. include:: common_config.rst

All of these arguments are optional; when unspecified, the CmdStan defaults will be used.


Example: estamate MLE for model ``bernoulli.stan`` by optimization
------------------------------------------------------------------

In this example we use the CmdStan example model
`bernoulli.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.stan>`__
and data file
`bernoulli.data.json <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.data.json>`__.

The :ref:`class_model` class method  ``optimize`` returns a ``StanMLE`` object
which provides properties to retrieve the estimate of the
penalized maximum likelihood estaimate of all model parameters:

- ``column_names``
- ``optimized_params_dict``
- ``optimized_params_np``
- ``optimized_params_pd``

In the following example, we instantiate a model and do optimization using the default CmdStan settings:

.. code:: ipython3

    import os
    from cmdstanpy.model import Model
    from cmdstanpy.utils import cmdstan_path
    
    bernoulli_dir = os.path.join(cmdstan_path(), 'examples', 'bernoulli')
    bernoulli_path = os.path.join(bernoulli_dir, 'bernoulli.stan')
    bernoulli_data = os.path.join(bernoulli_dir, 'bernoulli.data.json')
    # instantiate bernoulli model, compile Stan program
    bernoulli_model = Model(stan_file=bernoulli_path)
    bernoulli_model.compile()
    # run CmdStan's otpimize method, returns object `StanMLE`
    mle = bernoulli_model.optimize(data=bernoulli_data)
    print(mle.column_names)
    print(mle.optimized_params_dict)



References
----------

- Stan manual:  https://mc-stan.org/docs/reference-manual/optimization-algorithms-chapter.html


