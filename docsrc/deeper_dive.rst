CmdStanPy Deeper Dive
_____________________

The CmdStanPy interface provides one class which corresponds to the Stan program
and a set of method-specific classes which manage the Stan inference engine outputs,
and a few utilities to manage data and the local CmdStan installation.

The Bayesian workflow for model comparison and model expansion involves many iterations
of building, running, and analyzing the outputs of a Stan program.

Each iteration has two phases:  generation and analysis.
The estimate generation workflow is:

* Create a `CmdStanModel` object given a Stan program.

* Assemble the program inputs.  CmdStanPy provides the utility method `write_stan_json`
  which can be used to create both input data and parameter initializations.

* Do inference by calling the `CmdStanModel` object's inference method.


For each CmdStan inference method there is a CmdStanPy class which manages
the inference method outputs:

* method `sample` returns a `CmdStanMCMC` object

* method `optimize` returns a `CmdStanMLE` object

* method `variational` returns a `CmdStanVB` object

* method `generate` returns a `CmdStanGQ` object

The analysis workflow is open-ended.  The above classes provide
properties and methods for accessing and exporting the inference engine
outputs and metadata.

  
Instantiate and Run a Stan Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model Instantiation
"""""""""""""""""""
The :ref:`class_cmdstanmodel` class manages the Stan program and
It provides properties and functions to inspect the model code and filepaths.

.. code-block:: python

    # import packages
    import os
    from cmdstanpy import cmdstan_path, CmdStanModel

    # specify Stan program file 
    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')

    # instantiate the model; compiles the Stan program as needed.
    bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)

    # inspect model object 
    print(bernoulli_model)


Model Compilation
"""""""""""""""""
The Stan program is compiled on instantiation unless the keyword argument `compile=False` is specified.

* StanC options
* C++ options

  
Model inputs:  in-memory objects, JSON files
""""""""""""""""""""""""""""""""""""""""""""

Utility method `write_stan_json`
Link to Stan JSON spec
            
Inference methods
"""""""""""""""""

List of inference methods documentation, 1 sentence summary of each

  
Analyze and save the results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Common accessor methods


Per method outputs




Include under-the-hood here?
