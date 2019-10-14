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

- ``eval_elbo``: Number of interations between ELBO evaluations.

- ``output_samples``: Number of approximate posterior output draws to save.

.. include:: common_config.rst

All of these arguments are optional; when unspecified, the CmdStan defaults will be used.


Example: variational inference for model ``bernoulli.stan``
-----------------------------------------------------------

In this example we use the CmdStan example model
`bernoulli.stan <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.stan>`__
and data file
`bernoulli.data.json <https://github.com/stan-dev/cmdstanpy/blob/master/test/data/bernoulli.data.json>`__.

The :ref:`class_model` class method  ``variational`` runs Cmdstan and returns a ``StanVariational`` object.
In the following example, we run the ADVI algorithm specifying only the argument ``method=variational``

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
    # run CmdStan's variational inference method, returns object `StanVariational`
    vi = bernoulli_model.variational(data=bernoulli_data)

The
```StanVariational`` 
class provides methods to retrieve the estimate of the
approximate posterior mean of all model parameters,
the returned set of draws from this approximate posterior (if any):

- ``column_names``
- ``variational_params_dict``
- ``variational_params_np``
- ``variational_params_pd``
- ``sample``

.. code:: ipython3

    print(vi.column_names)
    print(vi.variational_params_dict['theta'])
    print(vi.sample.shape)

These estimates are only valid if the algorithm has converged to a good
approximation. When the algorithm fails to do so, the ``variational``
method will throw a ``RuntimeError``.

.. code:: ipython3

    fail_stan = os.path.join(datafiles_path, 'variational', 'eta_should_fail.stan')
    fail_model = Model(stan_file=fail_stan)
    model.compile()
    vi = model.variational()


References
----------


- Paper:  [Kucukelbir et al](http://arxiv.org/abs/1506.03431)
- Stan manual:  https://mc-stan.org/docs/2_20/reference-manual/vi-algorithms-chapter.html




