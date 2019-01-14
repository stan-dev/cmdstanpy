Guide
=====

This tutorial will walk you through the different steps of using CmdStanPy. 

Install
-------

CmdStanPy is a pure-Python package which can be installed from
PyPI

.. code-block:: bash

	pip install --upgrade cmdstanpy

or from sources

.. code-block:: bash

	pip install -e git+https://gitlab.thevirtualbrain.org/tvb/cmdstanpy

Basics
------

.. code-block:: python

	import os
	os.environ['CMDSTAN'] = '~/src/cmdstan-2.17.1'
	from cmdstanpy import Model, Run

	model = Model('''
	data { vector[20] x; real mu; }
	parameters { real sig; }
	model { x ~ normal(mu, sig); }
	generate quantities {
	    vector[20] log_lik;
	    for (i in 1:20) log_lik[i] = normal_lpdf(x[i] | mu, sig);
	}
	''')

	runs = model.sample(
		data=dict(mu, **data),
		chains=4
	)
	assert runs.N_eff_per_iter.min() > 0.2
	assert runs.R_hat.max() < 1.2

	data = {'x': np.random.randn(20) + 5.0}
	loo = []
	mus = np.r_[1.0, 3.0, 5.0, 7.0, 9.0]
	for mu in mus:
	    run = model.sample(
	        data=dict(mu=mu, **data), num_warmup=200, num_samples=200)
	    loo.append(run['loo'])
	assert mus[np.argmin(loo)] == 5.0

CmdStan's command line arguments are structured as a tree, so while child parameters of the 
method argument can be passed directly,

.. code-block:: python

   model.sample(num_samples=500)

equivalent to

.. code-block:: bash

	./$model sample num_samples=500

A more complex case with nested parameters looks like

.. code-block:: bash

	./$model id=$i \
	    sample save_warmup=1 num_warmup=200 num_samples=200 \
	        adapt \
	            delta=0.8 \
	        algorithm=hmc \
	            engine=nuts \
	                max_depth=12

CmdStanPy doesn't do anything clever (yet), so full set of subarguments need to be
passed as equivalent strings

.. code-block:: python

	model.sample(
		save_warmup=1,
		num_warmup=200,
		num_samples=200,
		adapt_='delta=0.8',
		algorithm='hmc engine=nuts max_depth=12')

Here, the :code:`_` postfix on :code:`adapt_` means :code:`adapt` doesn't take a value, but subarguments. In doubt,
the command line used to call the model is available as an attribute of the `Run` instance,

.. code-block:: python

	run = model.sample(...)
	print(run.cmd)

Plots
-----

(todo)

Trace plots
^^^^^^^^^^^

.. image:: https://gitlab.thevirtualbrain.org/tvb/cmdstanpy/-/jobs/artifacts/master/raw/test_trace_nuts.png?job=test

Parallel coordinates plot
^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: https://gitlab.thevirtualbrain.org/tvb/cmdstanpy/-/jobs/artifacts/master/raw/test_plot_parallel_coordinates.png?job=test