.. py:currentmodule:: cmdstanpy

CmdStanPy Workflow
__________________

The statistical modeling enterprise has two principal modalities:
development and production.
The focus of development is model building, comparison, and validation.
Many models are written and fitted to many kinds of data.
The focus of production is using a trusted model on real-world data
to obtain estimates for decision-making.
In both modalities, the essential workflow remains the same:
compile a Stan model, assemble input data,
do inference on the model conditioned on the data,
and validate, access, and export the results.

Model development and testing is an
open-ended process, usually requiring many iterations of
developing a model, fitting the data, and evaluating the results.
Since more user time is spent in model development,
CmdStanPy defaults favor development mode.
CmdStan is file-based interface.
On the assumption that model development will require
many successive runs of a model, by default, outputs are written
to a temporary directory to avoid filling up the filesystem with
unneeded CmdStan output files.
Non-default options allow all filepaths to be fully specified
so that scripts can be used to distribute analysis jobs across
nodes and machines.

The Bayesian workflow for model comparison and model expansion
provides a framework for model development, much of which
also applies to monitoring model performance in production.
The following sections describe the process of building, running, and
managing the resulting inference for a single model and set of inputs.

.. _model-compilation:

Compile the Stan model
^^^^^^^^^^^^^^^^^^^^^^

The: :class:`CmdStanModel` class provides methods
to compile and run the Stan program.
A CmdStanModel object can be instantiated by specifying
either a Stan file or the executable file, or both.
If only the Stan file path is specified, the constructor will
check for the existence of a correspondingly named exe file in
the same directory.  If found, it will use this as the exe file path.

By default, when a CmdStanModel object is instantiated from a Stan file,
the constructor will compile the model as needed.
The constructor argument `compile` controls this behavior.

* ``compile=False``: never compile the Stan file.
* ``compile="Force"``: always compile the Stan file.
* ``compile=True``: (default) compile the Stan file as needed, i.e., if no exe file exists or if the Stan file is newer than the exe file.

.. code-block:: python

    import os
    from cmdstanpy import CmdStanModel

    my_stanfile = os.path.join('.', 'my_model.stan')
    my_model = CmdStanModel(stan_file=my_stanfile)
    my_model.name
    my_model.stan_file
    my_model.exe_file
    my_model.code()

The CmdStanModel class also provides the :meth:`~CmdStanModel.compile` method,
which can be called at any point to (re)compile the model as needed.

Model compilation is carried out via the GNU Make build tool.
The CmdStan ``makefile`` contains a set of general rules which
specify the dependencies between the Stan program and the
Stan platform components and low-level libraries.
Optional behaviors can be specified by use of variables
which are passed in to the ``make`` command as name, value pairs.

Model compilation is done in two steps:

* The ``stanc`` compiler translates the Stan program to C++.
* The C++ compiler compiles the generated code and links in
  the necessary supporting libraries.

Therefore, both the constructor and the ``compile`` method
allow optional arguments ``stanc_options`` and ``cpp_options`` which
specify options for each compilation step.
Options are specified as a Python dictionary mapping
compiler option names to appropriate values.

In order parallelize within-chain computations using the
Stan language ``reduce_sum`` function, or to parallelize
running the NUTS-HMC sampler across chains,
the Stan model must be compiled with
C++ compiler flag **STAN_THREADS**.
While any value can be used,
we recommend the value ``True``, e.g.:


.. code-block:: python

    import os
    from cmdstanpy import CmdStanModel

    my_stanfile = os.path.join('.', 'my_model.stan')
    my_model = CmdStanModel(stan_file=my_stanfile, cpp_options={'STAN_THREADS':'true'})


Assemble input and initialization data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStan is file-based interface, therefore all model input and
initialization data must be supplied as JSON files, as described in the
`CmdStan User's Guide
<https://mc-stan.org/docs/cmdstan-guide/json.html>`__.

CmdStanPy inference methods allow inputs and initializations
to be specified as in-memory Python dictionary objects
which are then converted to JSON via the utility function :func:`cmdstanpy.write_stan_json`.
This method should be used to create JSON input files whenever
these inputs contain either a collection compatible with
numpy arrays or pandas.Series.


Run the CmdStan inference engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each CmdStan inference method, there is a corresponding method on the :class:`CmdStanModel` class.
An example of each is provided in the `next section <examples.rst>`__

* The :meth:`~CmdStanModel.sample` method runs Stan's
  `HMC-NUTS sampler <https://mc-stan.org/docs/reference-manual/hamiltonian-monte-carlo.html>`_.

  It returns a :class:`CmdStanMCMC` object which contains
  a sample from the posterior distribution of the model conditioned on the data.

* The :meth:`~CmdStanModel.variational` method runs Stan's
  `Automatic Differentiation Variational Inference (ADVI) algorithm <https://mc-stan.org/docs/reference-manual/vi-algorithms-chapter.html>`_.

  It returns a :class:`CmdStanVB` object which contains
  an approximation the posterior distribution in the unconstrained variable space.

* The :meth:`~CmdStanModel.optimize` runs one of
  `Stan's optimization algorithms <https://mc-stan.org/docs/reference-manual/optimization-algorithms-chapter.html>`_
  to find a mode of the density specified by the Stan program.

  It returns a :class:`CmdStanMLE` object.

* The :meth:`~CmdStanModel.generate_quantities` method runs Stan's
  `generate_quantities method <https://mc-stan.org/docs/cmdstan-guide/standalone-generate-quantities.html>`_
  which generates additional quantities of interest from a mode. Its take an existing fit as input and
  uses the parameter estimates in the fit to run the Stan program's `generated quantities block <https://mc-stan.org/docs/reference-manual/program-block-generated-quantities.html>`__.

  It returns a :class:`CmdStanGQ` object.


Validate, view, export the inference engine outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The inference engine results objects
:class:`CmdStanMCMC`, :class:`CmdStanVB`, :class:`CmdStanMLE` and :class:`CmdStanGQ,`
contain the CmdStan method configuration information
and the location of all output files produced.
The provide a common set methods for accessing the inference results and metadata,
as well as method-specific informational properties and methods.objects

Output data
-----------

The resulting Stan CSV file or set of files are assembled into an inference result object.

+ :class:`CmdStanMCMC` object contains the :meth:`~CmdStanModel.sample` outputs
+ :class:`CmdStanVB` object contains the :meth:`~CmdStanModel.variational` outputs
+ :class:`CmdStanMLE` object contains the :meth:`~CmdStanModel.optimize` outputs
+ :class:`CmdStanGQ` object contains the :meth:`~CmdStanModel.generate_quantities` outputs


The objects provide accessor methods which return this information
either as tabular data (i.e., in terms of the per-chain CSV file rows and columns),
or as structured objects which correspond to the variables in the Stan model
and the individual diagnostics produced by the inference method.

The ``stan_variables`` method returns a Python dict over all Stan model variables,
see :meth:`~CmdStanMCMC.stan_variables`.

The ``stan_variable`` method returns a single model variable as a numpy.ndarray object
with the same structure (per draw) as the Stan program variable,
see :meth:`~CmdStanMCMC.stan_variable`.

The ``method_variables`` method returns a Python dict over all inference
method variables, cf :meth:`~CmdStanMCMC.method_variables`


The output from the methods :class:`CmdStanMCMC` and :class:`CmdStanGQ` return the sample contents
in tabular form, see :meth:`~CmdStanMCMC.draws` and :meth:`~CmdStanMCMC.draws_pd`.
Similarly, the :meth:`~CmdStanMCMC.draws_xr` method returns the sample
contents as an :py:class:`xarray.Dataset` which is a mapping from variable names to their respective values.
