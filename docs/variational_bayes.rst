Variational Inference
=====================

Variational inference is a scalable technique for approximate Bayesian inference.
In the Stan ecosystem, the terms "VI" and "VB" ("variational Bayes")
are used synonymously.

Stan implements an automatic variational inference algorithm,
called Automatic Differentiation Variational Inference (ADVI)
which searches over a family of simple densities to find the best
approximate posterior density.
ADVI produces an estimate of the parameter means together with a sample
from the approximate posterior density.

ADVI approximates the variational objective function, the evidence lower bound or ELBO,
using stochastic gradient ascent.
The algorithm ascends these gradients using an adaptive stepsize sequence
that has one parameter ``eta`` which is adjusted during warmup.
The number of draws used to approximate the ELBO is denoted by ``elbo_samples``. 
ADVI heuristically determines a rolling window over which it computes
the average and the median change of the ELBO.
When this change falls below a threshold, denoted by ``tol_rel_obj``,
the algorithm is considered to have converged.


ADVI configuration
------------------

- ``algorithm``: Algorithm to use. One of: "meanfield", "fullrank".

- ``iter``: Maximum number of ADVI iterations.

- ``grad_samples``: Number of MC draws for computing the gradient.

- ``elbo_samples``: Number of MC draws for estimate of ELBO.

- ``eta``: Stepsize scaling parameter.

- ``adapt_iter``: Number of iterations for eta adaptation.

- ``tol_rel_obj``: Relative tolerance parameter for convergence.

- ``eval_elbo``: Number of interactions between ELBO evaluations.

- ``output_samples``: Number of approximate posterior output draws to save.

.. include:: common_config.rst

All of these arguments are optional; when unspecified, the CmdStan defaults will be used.


Example: variational inference for model ``bernoulli.stan``
-----------------------------------------------------------

In this example we use the CmdStan example model
`bernoulli.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.stan>`__
and data file
`bernoulli.data.json <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.data.json>`__.

The :ref:`class_cmdstanmodel` class method  ``variational`` returns a ``CmdStanVB`` object which provides properties
to retrieve the estimate of the
approximate posterior mean of all model parameters,
and the returned set of draws from this approximate posterior (if any):

- ``column_names``
- ``variational_params_dict``
- ``variational_params_np``
- ``variational_params_pd``
- ``variational_sample``

- ``save_csvfiles()``

In the following example, we instantiate a model and run variational inference using the default CmdStan settings:

.. code:: ipython3

    import os
    from cmdstanpy.model import CmdStanModel
    from cmdstanpy.utils import cmdstan_path

    # instantiate bernoulli model, compile Stan program
    bernoulli_path = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    bernoulli_model = CmdStanModel(stan_file=bernoulli_path)

    # run CmdStan's variational inference method, returns object `CmdStanVB`
    bern_data = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')
    bern_vb = bernoulli_model.variational(data=bernoulli_data)
    print(bern_vb.column_names)
    print(bern_vb.variational_params_dict)
    bern_vb.variational_sample.shape

These estimates are only valid if the algorithm has converged to a good
approximation. When the algorithm fails to do so, the ``variational``
method will throw a ``RuntimeError``.

.. code:: ipython3

    fail_stan = os.path.join(datafiles_path, 'variational', 'eta_should_fail.stan')
    fail_model = CmdStanModel(stan_file=fail_stan)
    model.compile()
    vb = model.variational()


References
----------


- Paper:  [Kucukelbir et al](http://arxiv.org/abs/1506.03431)
- Stan manual:  https://mc-stan.org/docs/reference-manual/vi-algorithms-chapter.html




