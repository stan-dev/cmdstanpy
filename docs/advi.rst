Variational Inference
=====================

Variational inference is a scalable technique for approximate Bayesian inference.
Unlike Stan's HMC-NUTS sampler, which produces a set of draws from the joint log
probability density of the model conditioned on the data, Variational Inference
produces a set of draws from a simpler, computationally tractable, probability density.
Stan implements an automatic variational inference algorithm,
called Automatic Differentiation Variational Inference (ADVI)
which searches over a family of simple densities to find the best
approximate posterior density.

ADVI produces an estimate of the parameter means together with a sample
from the approximate posterior density.

ADVI approximates the variational objective function, the evidence lower bound or ELBO,
using stochastic gradient ascent.
The algorithm ascends these gradients using an adaptive stepsize sequence,
which has one parameter `eta`, which is adjusted during warmup.
The number of draws used to approximate the ELBO is denoted by `elbo_samples`. 
ADVI heuristically determines a rolling window over which it computes
the average and the median change of the ELBO.
When this change falls below a threshold, denoted by `tol_rel_obj`,
the algorithm is considered to have converged.

See:

 - Paper:  [Kucukelbir et al](http://arxiv.org/abs/1506.03431)
 - Stan manual:  https://mc-stan.org/docs/2_20/reference-manual/vi-algorithms-chapter.html












